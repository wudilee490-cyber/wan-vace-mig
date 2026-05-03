"""
SA-V MIG Dataset (适配 sav_pipeline 输出)
==========================================
读取 sav_pipeline 5 个阶段产出的 clip 缓存:
    {clip_id}/
    ├── video_latent.pt        [C_lat, F_lat, H_lat, W_lat]
    ├── obj_image_masks.pt     [K, H_lat, W_lat]
    ├── obj_volume_masks.pt    [K, F_p, H_p, W_p]
    ├── verb_embeddings.pt     [K, N_v, text_dim]
    ├── verb_masks.pt          [K, N_v] bool
    ├── global_context.pt      [L_g, text_dim]
    └── meta.json / captions.json

加载:
    ds = SAVMIGDataset(clips_root="/data/sav_cache", split="train")
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class SAVMIGDataset(Dataset):
    def __init__(
        self,
        clips_root: str,
        split: str = "train",                 # "train" | "val"
        max_objects: int = 3,
        min_objects: int = 1,
        random_object_subset: bool = True,
        seed: int = 42,
    ):
        self.clips_root = Path(clips_root)
        self.max_objects = max_objects
        self.min_objects = min_objects
        self.random_object_subset = random_object_subset
        self.rng = random.Random(seed)

        idx_file = self.clips_root / f"{split}.json"
        if not idx_file.exists():
            raise FileNotFoundError(
                f"{idx_file} not found. Did you run 05_build_index.py?"
            )
        with open(idx_file) as f:
            self.entries: List[Dict] = json.load(f)

        # 过滤物体数不够的
        self.entries = [e for e in self.entries
                        if e["n_objects"] >= min_objects]
        if len(self.entries) == 0:
            raise RuntimeError(f"no usable samples in {idx_file}")

        # 检查 grid 一致性 (训练时 batch 内必须同 shape)
        grids = set((e["F_p"], e["H_p"], e["W_p"]) for e in self.entries)
        if len(grids) > 1:
            print(f"[SAVMIGDataset] WARN: multiple grid shapes found: {grids}. "
                  f"Make sure your batch sampler groups same-shape clips.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Dict:
        e = self.entries[idx]
        cd = self.clips_root / e["clip_id"]

        video_latent     = torch.load(cd / "video_latent.pt", map_location="cpu",
                                      weights_only=True)
        obj_image_masks  = torch.load(cd / "obj_image_masks.pt", map_location="cpu",
                                      weights_only=True)
        obj_volume_masks = torch.load(cd / "obj_volume_masks.pt", map_location="cpu",
                                      weights_only=True)
        verb_emb         = torch.load(cd / "verb_embeddings.pt", map_location="cpu",
                                      weights_only=True)
        verb_mask        = torch.load(cd / "verb_masks.pt", map_location="cpu",
                                      weights_only=True)
        global_context   = torch.load(cd / "global_context.pt", map_location="cpu",
                                      weights_only=True)

        K_total = obj_image_masks.shape[0]
        if self.random_object_subset and K_total > self.max_objects:
            k = self.rng.randint(self.min_objects, self.max_objects)
            sel = sorted(self.rng.sample(range(K_total), k))
        else:
            sel = list(range(min(K_total, self.max_objects)))

        sel_t = torch.tensor(sel, dtype=torch.long)
        return {
            "video_latent":      video_latent,                              # [C, F_lat, H_lat, W_lat]
            "first_frame_latent": video_latent[:, 0],                       # [C, H_lat, W_lat]
            "obj_image_masks":   obj_image_masks.index_select(0, sel_t),
            "obj_volume_masks":  obj_volume_masks.index_select(0, sel_t),
            "verb_embeddings":   verb_emb.index_select(0, sel_t),
            "verb_masks":        verb_mask.index_select(0, sel_t),
            "global_context":    global_context,
            "obj_motion_phrases": [e["phrases"][i] for i in sel],
            "global_prompt":     e["global_prompt"],
            "clip_id":           e["clip_id"],
        }


def collate_mig_batch(batch: List[Dict]) -> Dict:
    """
    把可变物体数 / 可变文本长度的样本 padding 到 batch 内统一形状。

    注意 batch 内 video_latent 的 (F_lat, H_lat, W_lat) 必须一致 (同分辨率)。
    如果你的数据里 grid 不一致,需要用 GroupedBatchSampler 按 grid 分组采样,
    或者 Stage 1 时强制全部 resize 到同分辨率。
    """
    B = len(batch)
    # 物体数对齐
    n_obj_max = max(b["obj_image_masks"].shape[0] for b in batch)
    H_lat, W_lat = batch[0]["obj_image_masks"].shape[-2:]
    F_p, H_p, W_p = batch[0]["obj_volume_masks"].shape[-3:]
    text_dim = batch[0]["verb_embeddings"].shape[-1]
    N_v_max = max(b["verb_embeddings"].shape[1] for b in batch)
    L_g_max = max(b["global_context"].shape[0] for b in batch)

    out = {
        "video_latent":       torch.stack([b["video_latent"] for b in batch]),
        "first_frame_latent": torch.stack([b["first_frame_latent"] for b in batch]),
    }

    # 物体相关 padding
    obj_image_masks  = torch.zeros(B, n_obj_max, H_lat, W_lat)
    obj_volume_masks = torch.zeros(B, n_obj_max, F_p, H_p, W_p)
    verb_emb         = torch.zeros(B, n_obj_max, N_v_max, text_dim,
                                   dtype=batch[0]["verb_embeddings"].dtype)
    verb_mask        = torch.zeros(B, n_obj_max, N_v_max, dtype=torch.bool)
    obj_valid        = torch.zeros(B, n_obj_max, dtype=torch.bool)
    phrases_list = []
    for b, sample in enumerate(batch):
        K = sample["obj_image_masks"].shape[0]
        obj_image_masks[b, :K]  = sample["obj_image_masks"]
        obj_volume_masks[b, :K] = sample["obj_volume_masks"]
        Nv = sample["verb_embeddings"].shape[1]
        verb_emb[b, :K, :Nv]  = sample["verb_embeddings"]
        verb_mask[b, :K, :Nv] = sample["verb_masks"]
        obj_valid[b, :K] = True
        phrases = list(sample["obj_motion_phrases"])
        while len(phrases) < n_obj_max: phrases.append("")
        phrases_list.append(phrases)

    out["obj_image_masks"]  = obj_image_masks
    out["obj_volume_masks"] = obj_volume_masks
    out["verb_embeddings"]  = verb_emb
    out["verb_masks"]       = verb_mask
    out["obj_valid"]        = obj_valid
    out["obj_motion_phrases"] = phrases_list
    out["global_prompts"] = [b["global_prompt"] for b in batch]
    out["clip_ids"] = [b["clip_id"] for b in batch]

    # 全局 context padding
    g_ctx = torch.zeros(B, L_g_max, text_dim, dtype=batch[0]["global_context"].dtype)
    g_mask = torch.zeros(B, L_g_max, dtype=torch.bool)
    for b, sample in enumerate(batch):
        L = sample["global_context"].shape[0]
        g_ctx[b, :L] = sample["global_context"]
        g_mask[b, :L] = True
    out["global_context"] = g_ctx
    out["global_context_mask"] = g_mask
    return out
