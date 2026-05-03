"""
Motion Mask Predictor
======================
从首帧 mask + 动作短语 → 预测后续所有帧的 mask 序列.

输入:
    first_frame_mask:  [B, K, H, W]    user-provided (来自 3DIS / SAM2 等)
    first_frame_rgb:   [B, 3, H, W]    可选, 提供视觉上下文
    verb_emb:          [B, K, N_v, D]  动作 phrase 的 T5 embedding
    verb_mask:         [B, K, N_v]     padding mask
    F_target:          int             要预测的帧数

输出:
    mask_logits:       [B, K, F_target, H, W]   逐帧 mask logit
                                                 sigmoid → 0..1 概率

设计要点 (复用 wan_vace_mig 的设计哲学):
    1. **每个物体独立 forward** — 共享网络权重, 物体间互不干扰.
       通过把 (B, K) 摊平成 (B*K) 一次喂网络, 共享权重无开销.
    2. **DiT 风格** — patch embed → spatial+temporal+cross attn × N → unpatchify
    3. **Phase-aware verb conditioning** — 复用 PhaseAwareMotionEncoder 思想:
       K_act^f = MLP([verb_emb; PE(f/F)])  让同一动作在不同帧有不同表达
    4. **Bypass connection (关键)** — 网络输出加上首帧 mask 的 inverse-sigmoid,
       零初始化 unpatchify 后, sigmoid 输出 ≈ 首帧 mask repeat.
       让模型先学会 "保留首帧", 再学 "演化它", 训练稳定.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 0. 工具
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def sinusoidal_pe(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal PE for scalar t. t [...], returns [..., dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.unsqueeze(-1).float() * freqs
    pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if pe.shape[-1] < dim:
        pe = F.pad(pe, (0, dim - pe.shape[-1]))
    return pe


def _to_logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """概率 → logit. 用于 bypass connection."""
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


# ============================================================================
# 1. Phase-aware verb conditioning
# ============================================================================
class PhaseVerbEncoder(nn.Module):
    """[B', N_v, D_text] → [B', F, N_v, dim] (phase-aware)"""
    def __init__(self, text_dim: int, dim: int, pe_dim: int = 64):
        super().__init__()
        self.pe_dim = pe_dim
        self.proj = nn.Sequential(
            nn.Linear(text_dim + pe_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, verb: torch.Tensor, F_target: int) -> torch.Tensor:
        B_, N_v, _ = verb.shape
        device = verb.device
        f_idx = torch.arange(F_target, device=device).float() / max(F_target - 1, 1)
        pe = sinusoidal_pe(f_idx, self.pe_dim)                   # [F, pe_dim]
        verb_exp = verb.unsqueeze(1).expand(-1, F_target, -1, -1)
        pe_exp = pe.view(1, F_target, 1, -1).expand(B_, -1, N_v, -1)
        cat = torch.cat([verb_exp, pe_exp], dim=-1)
        return self.proj(cat)


# ============================================================================
# 2. Attention 模块
# ============================================================================
class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, kv_dim: Optional[int] = None,
                 qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        kv_dim = kv_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(kv_dim, dim)
        self.v_proj = nn.Linear(kv_dim, dim)
        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.out = nn.Linear(dim, dim)

    def forward(self, x, kv, kv_mask=None):
        B, Lq, D = x.shape
        H = self.num_heads
        q = self.q_proj(x).view(B, Lq, H, -1).transpose(1, 2)
        k = self.k_proj(kv).view(B, kv.shape[1], H, -1).transpose(1, 2)
        v = self.v_proj(kv).view(B, kv.shape[1], H, -1).transpose(1, 2)
        q = self.norm_q(q); k = self.norm_k(k)
        attn_mask = None
        if kv_mask is not None:
            attn_mask = kv_mask.view(B, 1, 1, -1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.out(out)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x).view(B, L, 3, H, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.norm_q(q); k = self.norm_k(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class FFN(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hidden = int(dim * mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))


# ============================================================================
# 3. DiT block
# ============================================================================
class MaskDiTBlock(nn.Module):
    """
    输入 x: [B', F, S, dim] (B' = B*K, F 帧, S = H_p*W_p tokens)
    
    一个 block = spatial self-attn + temporal self-attn + cross-attn(verb) + FFN
    """
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.spatial_attn = SelfAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.temporal_attn = SelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = RMSNorm(dim, eps=eps)
        self.cross_attn = CrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.norm4 = RMSNorm(dim, eps=eps)
        self.ffn = FFN(dim)

    def forward(self, x: torch.Tensor, verb_tokens: torch.Tensor,
                verb_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, F_, S, D = x.shape

        # --- 1. Spatial self-attn (within-frame) ---
        x_flat = x.reshape(B_ * F_, S, D)
        x_flat = x_flat + self.spatial_attn(self.norm1(x_flat))

        # --- 2. Temporal self-attn (across-frame, per spatial pos) ---
        x = x_flat.reshape(B_, F_, S, D).permute(0, 2, 1, 3).reshape(B_ * S, F_, D)
        x = x + self.temporal_attn(self.norm2(x))
        x = x.reshape(B_, S, F_, D).permute(0, 2, 1, 3).contiguous()

        # --- 3. Cross-attn to per-frame verb tokens ---
        x_flat = x.reshape(B_ * F_, S, D)
        v_flat = verb_tokens.reshape(B_ * F_, verb_tokens.shape[-2], D)
        m_flat = (verb_mask.reshape(B_ * F_, -1) if verb_mask is not None else None)
        x_flat = x_flat + self.cross_attn(self.norm3(x_flat), v_flat, m_flat)

        # --- 4. FFN ---
        x_flat = x_flat + self.ffn(self.norm4(x_flat))

        return x_flat.reshape(B_, F_, S, D)


# ============================================================================
# 4. 顶层模型
# ============================================================================
class MotionMaskPredictor(nn.Module):
    """
    输入:
        first_frame_mask:  [B, K, H, W]
        verb_emb:          [B, K, N_v, D_text]
        verb_mask:         [B, K, N_v] bool
        F_target:          int
        first_frame_rgb:   [B, 3, H, W]  optional

    输出:
        mask_logits:       [B, K, F_target, H, W]
                           sigmoid 后 > 0.5 即为前景
    """
    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        num_blocks: int = 6,
        text_dim: int = 4096,
        patch_size: int = 4,
        pe_dim: int = 64,
        use_rgb: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.use_rgb = use_rgb
        self.patch_size = patch_size

        in_channels = 1 + (3 if use_rgb else 0)

        # patch embed (mask + optional rgb)
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

        # phase-aware verb encoder
        self.verb_encoder = PhaseVerbEncoder(text_dim=text_dim, dim=dim, pe_dim=pe_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            MaskDiTBlock(dim=dim, num_heads=num_heads, qk_norm=qk_norm, eps=eps)
            for _ in range(num_blocks)
        ])
        self.final_norm = RMSNorm(dim, eps=eps)

        # unpatchify (ConvTranspose2d 像素级 upsample, 输出 1 channel logit)
        self.unpatchify = nn.ConvTranspose2d(
            dim, 1, kernel_size=patch_size, stride=patch_size,
        )

        # 关键: 零初始化 unpatchify, 让网络初始输出 ≈ 0
        # 配合 bypass connection (forward 里加首帧 mask 的 logit), 初始输出 ≈ 首帧 mask repeat
        nn.init.zeros_(self.unpatchify.weight)
        nn.init.zeros_(self.unpatchify.bias)

    def forward(
        self,
        first_frame_mask: torch.Tensor,
        verb_emb: torch.Tensor,
        verb_mask: Optional[torch.Tensor] = None,
        F_target: int = 21,
        first_frame_rgb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K, H, W = first_frame_mask.shape
        device = first_frame_mask.device

        BK = B * K
        first_mask_flat = first_frame_mask.reshape(BK, 1, H, W).float()

        # 把首帧 mask 复制 F 次作为输入,网络在此基础上学"演化"
        x_per_frame = first_mask_flat.unsqueeze(1).expand(-1, F_target, -1, -1, -1).contiguous()
        # → [BK, F, 1, H, W]

        # 拼接 RGB
        if self.use_rgb:
            if first_frame_rgb is not None:
                rgb_flat = first_frame_rgb.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(BK, 3, H, W).float()
            else:
                rgb_flat = torch.zeros(BK, 3, H, W, device=device, dtype=x_per_frame.dtype)
            rgb_per_frame = rgb_flat.unsqueeze(1).expand(-1, F_target, -1, -1, -1).contiguous()
            x_per_frame = torch.cat([x_per_frame, rgb_per_frame], dim=2)

        # patch embed: [BK, F, C, H, W] → [BK, F, dim, H_p, W_p] → [BK, F, S, dim]
        BK_, F_, C_, H_, W_ = x_per_frame.shape
        x_for_embed = x_per_frame.reshape(BK_ * F_, C_, H_, W_)
        x_emb = self.patch_embed(x_for_embed)
        H_p, W_p = x_emb.shape[-2:]
        x = x_emb.flatten(2).transpose(1, 2)
        x = x.reshape(BK_, F_, H_p * W_p, self.dim)

        # verb tokens: [BK, F, N_v, dim]
        verb_flat = verb_emb.reshape(BK, verb_emb.shape[-2], verb_emb.shape[-1])
        verb_tokens = self.verb_encoder(verb_flat, F_target)
        if verb_mask is not None:
            verb_mask_flat = (verb_mask.reshape(BK, verb_mask.shape[-1])
                              .unsqueeze(1).expand(-1, F_target, -1).contiguous())
        else:
            verb_mask_flat = None

        # DiT blocks
        for block in self.blocks:
            x = block(x, verb_tokens, verb_mask_flat)

        x = self.final_norm(x)

        # unpatchify → mask logits [BK, F, H, W]
        BK_F, S, D = x.reshape(-1, x.shape[-2], x.shape[-1]).shape
        x_for_unpatch = x.reshape(BK_ * F_, H_p, W_p, self.dim).permute(0, 3, 1, 2).contiguous()
        mask_logits = self.unpatchify(x_for_unpatch)        # [BK*F, 1, H, W]
        mask_logits = mask_logits.reshape(BK_, F_, mask_logits.shape[-2], mask_logits.shape[-1])

        # **Bypass**: 加上首帧 mask 的 logit, sigmoid 后初始 ≈ 首帧 repeat
        first_logit = _to_logit(first_mask_flat.squeeze(1))      # [BK, H, W]
        mask_logits = mask_logits + first_logit.unsqueeze(1)      # broadcast over F

        return mask_logits.reshape(B, K, F_target, H, W)

    def trainable_parameters(self):
        return list(self.parameters())

    @torch.no_grad()
    def predict(
        self,
        first_frame_mask: torch.Tensor,
        verb_emb: torch.Tensor,
        verb_mask: Optional[torch.Tensor] = None,
        F_target: int = 21,
        first_frame_rgb: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """推理: 返回二值 mask [B, K, F_target, H, W] uint8"""
        logits = self(first_frame_mask, verb_emb, verb_mask, F_target, first_frame_rgb)
        prob = torch.sigmoid(logits)
        return (prob > threshold).to(torch.uint8)
