"""
Training losses for DecoupledMIGAdapter
========================================
我们的总损失 = 主损失 (扩散去噪) + 两个辅助损失。
公式:
    L_total = L_diffusion  +  λ_id · L_identity  +  λ_phase · L_phase

下面逐项解释设计动机和实现。
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. 主损失:Flow Matching (Wan2.1 训练目标)
# =========================================================================
def flow_matching_loss(
    transformer_pred: torch.Tensor,         # [B, C, F, H, W] 模型预测的速度场
    x0: torch.Tensor,                        # [B, C, F, H, W] 干净 latent
    noise: torch.Tensor,                     # [B, C, F, H, W] 与 x0 同形状的噪声
    t: torch.Tensor,                         # [B] in (0, 1) 的时间
    weight_scheme: str = "logit_normal",
) -> torch.Tensor:
    """
    Wan2.1 用的是 rectified flow / flow matching 训练目标:
        x_t = (1 - t) * x0 + t * noise
        target_velocity = noise - x0
        loss = E_t [ w(t) · ||model(x_t, t) - target_velocity||^2 ]

    weight_scheme:
        "uniform"      — w(t) = 1
        "logit_normal" — w(t) ∝ pdf of logit-normal(0,1)(t),
                         强调中间时间步,这是 Stable Diffusion 3 / Wan2.1 用的
                         (不依赖 t 的具体值,这里把它折叠到采样阶段去做,
                         loss 计算保持 uniform,简化实现)
    """
    target = noise - x0
    diff = (transformer_pred - target) ** 2                     # [B, C, F, H, W]
    # mean over all dims (per-sample 之后再 batch 平均)
    return diff.mean()


def sample_flow_matching_batch(
    x0: torch.Tensor,
    t_schedule: str = "logit_normal",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    采样 (t, noise, x_t) 用于 flow matching 训练。
    t_schedule:
        "uniform"      — t ~ U(0, 1)
        "logit_normal" — t = sigmoid(N(0, 1))  (Wan2.1 训练偏好中间时间步)
    """
    B = x0.shape[0]
    device = x0.device
    if t_schedule == "uniform":
        t = torch.rand(B, device=device, dtype=torch.float32)
    elif t_schedule == "logit_normal":
        u = torch.randn(B, device=device, dtype=torch.float32)
        t = torch.sigmoid(u)
    else:
        raise ValueError(f"unknown t_schedule: {t_schedule}")
    noise = torch.randn_like(x0)
    t_b = t.view(B, *([1] * (x0.ndim - 1)))
    x_t = (1.0 - t_b) * x0 + t_b * noise
    return t, noise, x_t


# =========================================================================
# 2. 辅助损失 1:Identity Preservation Loss
# =========================================================================
def identity_preservation_loss(
    pred_velocity: torch.Tensor,             # [B, C, F, H, W]
    target_velocity: torch.Tensor,           # [B, C, F, H, W]
    obj_volume_masks: torch.Tensor,          # [B, K, F_p, H_p, W_p]
    obj_valid: torch.Tensor,                 # [B, K] bool
    p_t: int = 1, p_h: int = 2, p_w: int = 2,
    weight_first_frame: float = 2.0,
) -> torch.Tensor:
    """
    动机:
        主损失是全图 MSE,但物体区域只占视频很小一部分 —— 如果只用主损失,
        梯度会被背景主导,adapter 学不到对物体区域的精细控制。
    
    做法:
        在物体覆盖的 token 位置上额外加一份 weighted MSE,等于把"物体区域"的
        梯度信号放大,鼓励 adapter 在这些位置真正起作用。
    
        first frame 权重再加倍,因为 adapter 的核心能力是"把首帧物体身份保留到
        后续帧",首帧本身的重建质量决定后续注入的身份特征质量。

    数学:
        把 obj_volume_masks 上采样到 latent 分辨率 (per-frame patch → per-pixel),
        合并所有 valid 物体的并集,得到 mask_lat [B, F, H, W]。
        L_id = mean( mask_lat * (pred - target)^2 ) / max(mean(mask_lat), eps)
    """
    B, C, F_lat, H_lat, W_lat = pred_velocity.shape
    K = obj_volume_masks.shape[1]
    F_p, H_p, W_p = obj_volume_masks.shape[-3:]

    # 把 valid 维度乘进去,无效物体 mask 全 0
    masks = obj_volume_masks * obj_valid.float().view(B, K, 1, 1, 1)

    # K 个物体取 union
    union = masks.amax(dim=1)                                   # [B, F_p, H_p, W_p]

    # patch → pixel 上采样 (nearest)
    union_lat = union.repeat_interleave(p_t, dim=1)              # [B, F_lat, H_p, W_p]
    union_lat = union_lat.repeat_interleave(p_h, dim=2)          # [B, F_lat, H_lat, W_p]
    union_lat = union_lat.repeat_interleave(p_w, dim=3)          # [B, F_lat, H_lat, W_lat]
    union_lat = union_lat.unsqueeze(1)                           # [B, 1, F, H, W]

    # 首帧加权
    weight = torch.ones_like(union_lat)
    weight[:, :, 0] = weight_first_frame
    weight = weight * union_lat

    sq_err = (pred_velocity - target_velocity) ** 2              # [B, C, F, H, W]
    weighted = sq_err * weight
    denom = weight.sum().clamp(min=1.0)
    return weighted.sum() / denom


# =========================================================================
# 3. 辅助损失 2:Phase Consistency Loss
# =========================================================================
class PhaseConsistencyLoss(nn.Module):
    """
    动机:
        相位编码器 K_act = MLP([verb_emb; PE(f/T)]) 想让模型学到"同一动词在不同帧
        产生不同的 K/V"。但纯靠扩散损失,模型可能偷懒 —— 让 PE(f/T) 的权重接近 0,
        让 K/V 退化成静态 (因为静态 K/V 也能通过物体区域注入部分动作信息)。
    
    解决:
        加一个对比损失,显式鼓励 PE 学到平滑的相位差异:
            - 相邻帧的 K_act 应当相似 (动作连续)
            - 远帧的 K_act 应当显著不同 (动作演化)
        
    实现:
        从 adapter 内部抠出 phase_motion_encoder 在不同帧的输出,
        计算"相邻帧相似度 - 远帧相似度"作为 margin loss:
            L_phase = max(0, m - (sim_far_to_close_diff))
        其中 m 是期望的差距 margin,推荐 0.1。

    使用前提:
        phase_motion_encoder 的输出已经在 forward 中被缓存 (cond["_motion_cache_*"]),
        训练步里调一下 adapter._get_motion_kv_for_layer(0) 拿到 [B, F_p, N_v, dim]
        就行。
    """

    def __init__(self, margin: float = 0.1, far_distance_ratio: float = 0.5):
        """
        margin: 相邻帧相似度 vs 远帧相似度的期望差距
        far_distance_ratio: 远帧定义 = 相距 ≥ ratio * F_p 帧数
        """
        super().__init__()
        self.margin = margin
        self.far_distance_ratio = far_distance_ratio

    def forward(self, motion_kv_perframe: torch.Tensor,
                obj_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        motion_kv_perframe: [B, F_p, N_v, dim]
        obj_valid:          [B] 是否参与 (这个 batch slot 的物体是否有效)
        """
        B, F_p, N_v, D = motion_kv_perframe.shape
        if F_p < 4:        # 太少帧无法测远近
            return motion_kv_perframe.new_zeros(())

        # 平均一下 verb 维度,得到 [B, F_p, dim]
        kv = motion_kv_perframe.mean(dim=2)
        kv = F.normalize(kv, dim=-1)                               # 单位化便于求 cos

        # 相邻帧相似度: cos(kv[f], kv[f+1])
        sim_close = (kv[:, :-1] * kv[:, 1:]).sum(-1)               # [B, F_p-1]
        sim_close_mean = sim_close.mean(dim=1)                     # [B]

        # 远帧相似度: cos(kv[0], kv[F_p - far_dist:])
        far_dist = max(2, int(F_p * self.far_distance_ratio))
        sim_far = (kv[:, 0:1] * kv[:, far_dist:]).sum(-1)          # [B, F_p-far_dist]
        sim_far_mean = sim_far.mean(dim=1)                         # [B]

        # 我们想要 sim_close > sim_far + margin
        gap = sim_close_mean - sim_far_mean                        # 越大越好
        loss_per_b = F.relu(self.margin - gap)                     # 不到 margin 才惩罚

        if obj_valid is not None:
            mask = obj_valid.float()
            return (loss_per_b * mask).sum() / mask.sum().clamp(min=1)
        return loss_per_b.mean()


# =========================================================================
# 4. 总损失装配
# =========================================================================
class MIGTrainingLoss(nn.Module):
    def __init__(
        self,
        lambda_identity: float = 0.5,
        lambda_phase: float = 0.05,
        weight_first_frame: float = 2.0,
        phase_margin: float = 0.1,
    ):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.lambda_phase = lambda_phase
        self.weight_first_frame = weight_first_frame
        self.phase_loss = PhaseConsistencyLoss(margin=phase_margin)

    def forward(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        obj_volume_masks: torch.Tensor,
        obj_valid: torch.Tensor,
        motion_kv_perframe_per_obj: Optional[List[torch.Tensor]] = None,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ) -> Dict[str, torch.Tensor]:
        # 主损失
        l_diff = ((pred_velocity - target_velocity) ** 2).mean()

        # 身份保持
        p_t, p_h, p_w = patch_size
        l_id = identity_preservation_loss(
            pred_velocity, target_velocity, obj_volume_masks, obj_valid,
            p_t=p_t, p_h=p_h, p_w=p_w,
            weight_first_frame=self.weight_first_frame,
        )

        # 相位一致
        if motion_kv_perframe_per_obj is not None and len(motion_kv_perframe_per_obj) > 0:
            phase_losses = []
            K = len(motion_kv_perframe_per_obj)
            for k in range(K):
                kv_k = motion_kv_perframe_per_obj[k]              # [B, F_p, N_v, dim]
                phase_losses.append(self.phase_loss(kv_k, obj_valid[:, k]))
            l_phase = torch.stack(phase_losses).mean()
        else:
            l_phase = pred_velocity.new_zeros(())

        total = l_diff + self.lambda_identity * l_id + self.lambda_phase * l_phase

        return {
            "loss": total,
            "loss_diffusion": l_diff.detach(),
            "loss_identity": l_id.detach(),
            "loss_phase": l_phase.detach(),
        }
