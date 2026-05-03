"""
Mask Predictor Losses
======================
组合 BCE + Dice + (可选) Temporal smoothness.

为什么不用纯 BCE:
    Mask 通常正样本远少于负样本 (前景占画面 5-30%),
    纯 BCE 的梯度被负样本主导,模型容易输出全 0.
    
    Dice loss 直接优化 IoU,对类别不均衡天然鲁棒.
    
    BCE + Dice 组合是分割任务的成熟选择 (来自 nnU-Net / SAM 训练实践).

Temporal smoothness 是可选项:
    L_smooth = mean( |M_t - M_{t-1}| ) — 鼓励相邻帧 mask 平滑变化.
    数据集本身已经平滑时这个 loss 几乎为 0; 但小数据集时能防止跳变.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,         # [..., H, W]
    target: torch.Tensor,         # [..., H, W] in {0, 1} or float
    smooth: float = 1.0,
    spatial_dims: Tuple[int, int] = (-2, -1),
) -> torch.Tensor:
    """Multi-dim Dice loss. 在最后两维 (H, W) 求并集/交集, 其他维度求平均."""
    prob = torch.sigmoid(logits)
    target = target.float()
    intersection = (prob * target).sum(dim=spatial_dims)
    union = prob.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    obj_valid: Optional[torch.Tensor] = None,    # [B, K] bool
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """BCE loss, 只在 obj_valid=True 的物体上算."""
    target = target.float()
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=logits.device)
        loss_per_pixel = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pw, reduction="none"
        )
    else:
        loss_per_pixel = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
    # logits/target: [B, K, F, H, W]
    # 平均: [B, K, F, H, W] → [B, K]
    loss_per_obj = loss_per_pixel.mean(dim=(2, 3, 4))    # [B, K]
    if obj_valid is not None:
        loss_per_obj = loss_per_obj * obj_valid.float()
        denom = obj_valid.float().sum().clamp(min=1)
        return loss_per_obj.sum() / denom
    return loss_per_obj.mean()


def temporal_smoothness(
    logits: torch.Tensor,                # [B, K, F, H, W]
    obj_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 between adjacent frames, 鼓励相邻帧 mask 平滑."""
    prob = torch.sigmoid(logits)
    diff = (prob[:, :, 1:] - prob[:, :, :-1]).abs()       # [B, K, F-1, H, W]
    loss_per_obj = diff.mean(dim=(2, 3, 4))                # [B, K]
    if obj_valid is not None:
        loss_per_obj = loss_per_obj * obj_valid.float()
        denom = obj_valid.float().sum().clamp(min=1)
        return loss_per_obj.sum() / denom
    return loss_per_obj.mean()


class MaskPredictorLoss(nn.Module):
    """
    L = w_bce * BCE + w_dice * Dice + w_smooth * Temporal
    
    默认权重: BCE 1.0, Dice 1.0, Temporal 0.0 (关闭 — 数据本身已平滑)
    """
    def __init__(
        self,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_smooth: float = 0.0,
        bce_pos_weight: Optional[float] = None,
        first_frame_loss_only: bool = False,
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_smooth = w_smooth
        self.bce_pos_weight = bce_pos_weight
        self.first_frame_loss_only = first_frame_loss_only

    def forward(
        self,
        mask_logits: torch.Tensor,        # [B, K, F, H, W]
        target_mask: torch.Tensor,        # [B, K, F, H, W]
        obj_valid: Optional[torch.Tensor] = None,    # [B, K]
    ) -> dict:
        """
        Returns dict with 'loss' (total) and breakdown for logging.
        """
        if self.first_frame_loss_only:
            mask_logits = mask_logits[:, :, :1]
            target_mask = target_mask[:, :, :1]

        bce = masked_bce(mask_logits, target_mask, obj_valid, self.bce_pos_weight)

        # Dice 在 (F, H, W) 三维上算交并 (整段 mask 序列作一个体)
        # 这样能避免某些"空帧" (物体看不见时) 给 Dice 拉空
        if obj_valid is not None:
            # 把 invalid 物体的 logit/target 屏蔽
            valid_5d = obj_valid.view(*obj_valid.shape, 1, 1, 1)
            ml = mask_logits * valid_5d
            tg = target_mask * valid_5d
        else:
            ml, tg = mask_logits, target_mask
        dice = soft_dice_loss(ml, tg, spatial_dims=(-3, -2, -1))

        loss = self.w_bce * bce + self.w_dice * dice
        log = {"bce": bce.detach(), "dice": dice.detach()}

        if self.w_smooth > 0:
            smooth = temporal_smoothness(mask_logits, obj_valid)
            loss = loss + self.w_smooth * smooth
            log["smooth"] = smooth.detach()

        log["loss"] = loss.detach()
        return {"loss": loss, **log}


# ============================================================================
# 评估指标 (训练时打印, 不参与反向)
# ============================================================================
@torch.no_grad()
def compute_iou(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    obj_valid: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """[B, K, F, H, W] → scalar mean IoU"""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    target = target.float()
    inter = (pred * target).sum(dim=(2, 3, 4))            # [B, K]
    union = ((pred + target) > 0).float().sum(dim=(2, 3, 4))
    iou = inter / union.clamp(min=1)                       # [B, K]
    if obj_valid is not None:
        iou = iou * obj_valid.float()
        denom = obj_valid.float().sum().clamp(min=1)
        return iou.sum() / denom
    return iou.mean()
