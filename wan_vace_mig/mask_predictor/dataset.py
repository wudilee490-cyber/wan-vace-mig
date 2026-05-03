"""
Mask Predictor Dataset
=======================
读 sav_mig_data 缓存, 但只取 mask 训练相关的字段.

需要 sav_mig_data 至少跑到 Stage 4, 并且 Stage 3 时打开了 dense_masks 保存
(03_encode_latents.py 自动加了 obj_dense_masks_lat.pt).

输出每个样本:
    first_frame_mask:   [K, H_lat, W_lat]    binary
    target_mask_seq:    [K, F_lat, H_lat, W_lat] binary
    first_frame_latent: [C_lat, H_lat, W_lat] (用作 RGB 上下文 -- 是 latent 不是像素)
                                                 注: 这是 VAE 编码后的 latent 特征,
                                                     不是真实 RGB. 但能提供视觉语义.
    verb_embeddings:    [K, N_v, D_text]
    verb_masks:         [K, N_v]
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class MaskPredictorDataset(Dataset):
    def __init__(
        self,
        clips_root: str,
        split: str = "train",
        max_objects: int = 3,
        min_objects: int = 1,
        random_object_subset: bool = True,
        use_first_frame_latent_as_rgb: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            use_first_frame_latent_as_rgb:
                True (默认): 把 first_frame_latent (来自 VAE) 当 "RGB 上下文" 喂给
                            mask predictor. 这是 latent, 不是真正 RGB, 但分辨率匹配
                            mask 维度且包含视觉语义, 训练效果应优于无视觉信息.
                            模型 use_rgb=True 时需要这个.
                False: 不喂视觉上下文, 模型 use_rgb=False.
        """
        self.clips_root = Path(clips_root)
        self.max_objects = max_objects
        self.min_objects = min_objects
        self.random_object_subset = random_object_subset
        self.use_first_frame_latent_as_rgb = use_first_frame_latent_as_rgb
        self.rng = random.Random(seed)

        idx_file = self.clips_root / f"{split}.json"
        if not idx_file.exists():
            raise FileNotFoundError(
                f"{idx_file} not found. Did you run sav-mig-data 05_build_index.py?"
            )
        with open(idx_file) as f:
            entries: List[Dict] = json.load(f)
        # 过滤
        self.entries = [e for e in entries if e["n_objects"] >= min_objects]

        # 检查 dense mask 是否存在 (用一个 entry 试一下)
        if self.entries:
            cd = self.clips_root / self.entries[0]["clip_id"]
            if not (cd / "obj_dense_masks_lat.pt").exists():
                raise FileNotFoundError(
                    f"{cd}/obj_dense_masks_lat.pt not found.\n"
                    f"You need to re-run sav-mig-data Stage 3 with the latest "
                    f"03_encode_latents.py (which now saves obj_dense_masks_lat.pt)."
                )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Dict:
        e = self.entries[idx]
        cd = self.clips_root / e["clip_id"]

        dense_masks = torch.load(cd / "obj_dense_masks_lat.pt", map_location="cpu",
                                  weights_only=True).float()           # [K, F_lat, H_lat, W_lat]
        verb_emb = torch.load(cd / "verb_embeddings.pt", map_location="cpu",
                               weights_only=True)                       # [K, N_v, D_text]
        verb_mask = torch.load(cd / "verb_masks.pt", map_location="cpu",
                                weights_only=True)                      # [K, N_v]

        # 物体随机子采样 (跟 wan_vace_mig 一致)
        K_total = dense_masks.shape[0]
        if self.random_object_subset and K_total > self.max_objects:
            k = self.rng.randint(self.min_objects, self.max_objects)
            sel = sorted(self.rng.sample(range(K_total), k))
        else:
            sel = list(range(min(K_total, self.max_objects)))
        sel_t = torch.tensor(sel, dtype=torch.long)

        dense_masks = dense_masks.index_select(0, sel_t)
        verb_emb = verb_emb.index_select(0, sel_t)
        verb_mask = verb_mask.index_select(0, sel_t)

        # 首帧 mask (输入) + 目标序列 (gt)
        first_frame_mask = dense_masks[:, 0]                            # [K, H_lat, W_lat]
        target_mask_seq = dense_masks                                   # [K, F_lat, H_lat, W_lat]

        out = {
            "first_frame_mask": first_frame_mask,
            "target_mask_seq": target_mask_seq,
            "verb_embeddings": verb_emb,
            "verb_masks": verb_mask,
            "clip_id": e["clip_id"],
        }

        if self.use_first_frame_latent_as_rgb:
            # 用 video latent 第 0 帧作为视觉上下文 (是 latent 不是 RGB,但够用)
            v_latent = torch.load(cd / "video_latent.pt", map_location="cpu",
                                   weights_only=True)                   # [C, F_lat, H_lat, W_lat]
            # 取前 3 个 channel 当伪 RGB
            first_latent = v_latent[:3, 0]                              # [3, H_lat, W_lat]
            out["first_frame_rgb"] = first_latent.float()

        return out


def collate_mask_batch(batch: List[Dict]) -> Dict:
    """
    Pad 物体数 + verb 长度.
    
    Batch 内必须 (H_lat, W_lat, F_lat) 一致 — 即同一分辨率/帧数的 clip.
    若数据集有多种 grid, 需要 GroupedBatchSampler.
    """
    B = len(batch)
    n_obj_max = max(b["first_frame_mask"].shape[0] for b in batch)
    H, W = batch[0]["first_frame_mask"].shape[-2:]
    F_lat = batch[0]["target_mask_seq"].shape[-3]
    text_dim = batch[0]["verb_embeddings"].shape[-1]
    N_v_max = max(b["verb_embeddings"].shape[1] for b in batch)

    first_frame_mask = torch.zeros(B, n_obj_max, H, W)
    target_mask_seq = torch.zeros(B, n_obj_max, F_lat, H, W)
    verb_emb = torch.zeros(B, n_obj_max, N_v_max, text_dim,
                            dtype=batch[0]["verb_embeddings"].dtype)
    verb_mask = torch.zeros(B, n_obj_max, N_v_max, dtype=torch.bool)
    obj_valid = torch.zeros(B, n_obj_max, dtype=torch.bool)

    for b, sample in enumerate(batch):
        K = sample["first_frame_mask"].shape[0]
        first_frame_mask[b, :K] = sample["first_frame_mask"]
        target_mask_seq[b, :K] = sample["target_mask_seq"]
        Nv = sample["verb_embeddings"].shape[1]
        verb_emb[b, :K, :Nv] = sample["verb_embeddings"]
        verb_mask[b, :K, :Nv] = sample["verb_masks"]
        obj_valid[b, :K] = True

    out = {
        "first_frame_mask": first_frame_mask,
        "target_mask_seq": target_mask_seq,
        "verb_embeddings": verb_emb,
        "verb_masks": verb_mask,
        "obj_valid": obj_valid,
        "clip_ids": [b["clip_id"] for b in batch],
    }

    if "first_frame_rgb" in batch[0]:
        out["first_frame_rgb"] = torch.stack([b["first_frame_rgb"] for b in batch])

    return out
