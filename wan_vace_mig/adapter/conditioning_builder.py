"""
Conditioning Builder (Phase-Aware compatible)
=============================================
负责把 agent pipeline 的产出转为 DecoupledMIGAdapter 需要的张量。

身份特征(global KV,无相位):
    用 VACE 自带的 vace_patch_embedding 把首帧 latent 嵌成 token,按 obj_image_masks
    抽取每个物体的 token,作为 identity_kv_list[i]。

动作特征(per-frame KV 由 adapter 内部生成):
    Builder 只输出每个物体的 verb embedding [B, N_v_i, text_dim]。
    PhaseAwareMotionEncoder(在 adapter 内)负责拼上 PE(f/T) 并 MLP 得到每帧 K/V。

这样把"文本 → 嵌入"(builder 职责)和"嵌入 → per-frame K/V"(可学习模块)解耦,
T5 编码可以一次性算完缓存,相位展开由轻量 MLP 在 GPU 上跑。
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditioningBuilder:
    def __init__(
        self,
        vace_model: nn.Module,
        text_encoder,                                  # 复用 Wan pipeline 里的 T5 / umT5
        identity_proj: Optional[nn.Module] = None,     # 可选,把外部 image feat 投到 dim
    ):
        self.vace = vace_model
        self.text_encoder = text_encoder
        self.identity_proj = identity_proj

    # ------------------------------------------------------------------
    # 身份分支:首帧 latent + 首帧 mask  →  per-object identity KV
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_identity_kv_from_first_frame(
        self,
        first_frame_latent: torch.Tensor,    # [B, C_lat, H_lat, W_lat]
        obj_image_masks: torch.Tensor,       # [B, n_obj, H_lat, W_lat] 0/1
        max_tokens_per_obj: int = 256,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        返回:
            identity_kv_list:  List[Tensor [B, N_id_i, dim]]
            identity_kv_masks: List[Tensor [B, N_id_i] bool]
        """
        B, C, H, W = first_frame_latent.shape
        n_obj = obj_image_masks.shape[1]

        x = first_frame_latent.unsqueeze(2)                  # [B, C, 1, H, W]
        tokens = self.vace.vace_patch_embedding(x)           # [B, dim, 1, H_p, W_p]
        _, dim, _, H_p, W_p = tokens.shape
        tokens = tokens.squeeze(2).permute(0, 2, 3, 1)       # [B, H_p, W_p, dim]

        ph, pw = self.vace.patch_size[1], self.vace.patch_size[2]
        m = obj_image_masks.float().view(B * n_obj, 1, H, W)
        m = F.max_pool2d(m, kernel_size=(ph, pw), stride=(ph, pw))
        m = m.view(B, n_obj, H_p, W_p).bool()

        identity_kv_list, identity_kv_masks = [], []
        for i in range(n_obj):
            kv_b, mask_b = [], []
            for b in range(B):
                sel = m[b, i]
                feats = tokens[b][sel]
                if feats.shape[0] == 0:
                    feats = tokens.new_zeros(1, dim)
                if feats.shape[0] > max_tokens_per_obj:
                    idx = torch.randperm(feats.shape[0], device=feats.device)[:max_tokens_per_obj]
                    feats = feats[idx]
                kv_b.append(feats)
                mask_b.append(torch.ones(feats.shape[0], dtype=torch.bool, device=feats.device))
            kv_padded, mask_padded = self._pad_var_len(kv_b, mask_b)
            identity_kv_list.append(kv_padded)
            identity_kv_masks.append(mask_padded)
        return identity_kv_list, identity_kv_masks

    # ------------------------------------------------------------------
    # 动作分支:每物体动词短语  →  verb 嵌入(相位展开由 adapter 内部完成)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_verb_embeddings(
        self,
        obj_motion_phrases: List[List[str]],   # [B][n_obj] 动作短语,如 "jumping over a fence"
        max_len: int = 32,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        每个物体跑一次 text_encoder,得到 [B, N_v_i, text_dim] + [B, N_v_i] mask。

        N_v_i 通常很小(几个到几十个 token,取决于动词短语长度)。
        text_dim 应当与 WanModel 的 text_dim(默认 4096)一致,这样 adapter 内
        PhaseAwareMotionEncoder 输入维度对得上。
        """
        B = len(obj_motion_phrases)
        n_obj = len(obj_motion_phrases[0])
        for row in obj_motion_phrases:
            assert len(row) == n_obj, "每个 batch 的物体数必须一致"

        verb_emb_list, verb_mask_list = [], []
        for i in range(n_obj):
            phrases_i = [obj_motion_phrases[b][i] for b in range(B)]
            embeds = self.text_encoder(phrases_i)
            kv_padded, mask_padded = self._normalize_text_embeds(embeds, max_len)
            verb_emb_list.append(kv_padded)
            verb_mask_list.append(mask_padded)
        return verb_emb_list, verb_mask_list

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_var_len(tensors: List[torch.Tensor], masks: List[torch.Tensor]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_n = max(t.shape[0] for t in tensors)
        dim = tensors[0].shape[-1]
        device, dtype = tensors[0].device, tensors[0].dtype
        out = torch.zeros(len(tensors), max_n, dim, device=device, dtype=dtype)
        out_m = torch.zeros(len(tensors), max_n, dtype=torch.bool, device=device)
        for b, (t, m) in enumerate(zip(tensors, masks)):
            n = t.shape[0]
            out[b, :n] = t
            out_m[b, :n] = m
        return out, out_m

    @staticmethod
    def _normalize_text_embeds(embeds, max_len: int):
        """List[Tensor [L_b, D]] 或 Tensor [B, L, D] → ([B, L', D], [B, L'] mask)"""
        if isinstance(embeds, (list, tuple)):
            tensors = list(embeds)
            masks = [torch.ones(t.shape[0], dtype=torch.bool, device=t.device) for t in tensors]
            tensors = [t[:max_len] for t in tensors]
            masks = [m[:max_len] for m in masks]
            return ConditioningBuilder._pad_var_len(tensors, masks)
        else:
            B, L, D = embeds.shape
            mask = torch.ones(B, L, dtype=torch.bool, device=embeds.device)
            return embeds[:, :max_len], mask[:, :max_len]
