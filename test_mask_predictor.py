"""
Smoke test for Motion Mask Predictor
=====================================
1. forward 不挂, 形状正确
2. 零初始化时, sigmoid(logits) ≈ 首帧 mask 复制 F 次 (bypass + zero-init)
3. backward 能跑通, 5 步训练 loss 不 NaN
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wan_vace_mig.mask_predictor.model import MotionMaskPredictor
from wan_vace_mig.mask_predictor.losses import MaskPredictorLoss, compute_iou


def test_forward_shape():
    print("=" * 65)
    print("[Test 1] forward shape & no-NaN")
    print("=" * 65)
    B, K, H, W, F = 2, 3, 32, 48, 8
    text_dim = 512

    model = MotionMaskPredictor(
        dim=128, num_heads=4, num_blocks=2,
        text_dim=text_dim, patch_size=4, use_rgb=True,
    ).eval()

    first_mask = (torch.rand(B, K, H, W) > 0.7).float()
    rgb = torch.randn(B, 3, H, W)
    verb_emb = torch.randn(B, K, 6, text_dim)
    verb_mask = torch.ones(B, K, 6, dtype=torch.bool)

    with torch.no_grad():
        logits = model(first_mask, verb_emb, verb_mask, F_target=F, first_frame_rgb=rgb)
    assert logits.shape == (B, K, F, H, W), f"got {logits.shape}"
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    print(f"  ✓ shape ok: {tuple(logits.shape)}")
    print(f"  ✓ no NaN/Inf")
    return True


def test_zero_init_equivalence():
    print("\n" + "=" * 65)
    print("[Test 2] zero-init: sigmoid(logits) ≈ first_mask repeated")
    print("=" * 65)
    print("    (unpatchify 零初始化 + bypass connection 应当让初始预测 = 首帧 mask)")

    B, K, H, W, F = 1, 2, 32, 48, 5
    text_dim = 256
    model = MotionMaskPredictor(
        dim=64, num_heads=4, num_blocks=2,
        text_dim=text_dim, patch_size=4, use_rgb=False,
    ).eval()

    first_mask = (torch.rand(B, K, H, W) > 0.7).float()
    verb_emb = torch.randn(B, K, 4, text_dim)
    verb_mask = torch.ones(B, K, 4, dtype=torch.bool)

    with torch.no_grad():
        logits = model(first_mask, verb_emb, verb_mask, F_target=F)
        prob = torch.sigmoid(logits)

    # prob 在所有帧应该 ≈ first_mask
    target = first_mask.unsqueeze(2).expand(-1, -1, F, -1, -1)
    diff = (prob - target).abs().max().item()
    mean_diff = (prob - target).abs().mean().item()
    print(f"  max |sigmoid(logit) - first_mask|: {diff:.4f}")
    print(f"  mean |sigmoid(logit) - first_mask|: {mean_diff:.6f}")
    # 由于 _to_logit 用了 eps=1e-4, sigmoid 后会有 ~1e-4 误差
    assert diff < 0.01, f"zero-init bypass broken! diff={diff}"
    print(f"  ✓ PASS (< 0.01 max diff, expected by eps=1e-4 in _to_logit)")
    return True


def test_training_step():
    print("\n" + "=" * 65)
    print("[Test 3] 5-step training, loss 应该下降, 无 NaN")
    print("=" * 65)
    B, K, H, W, F = 2, 3, 32, 48, 8
    text_dim = 256
    model = MotionMaskPredictor(
        dim=128, num_heads=4, num_blocks=2,
        text_dim=text_dim, patch_size=4, use_rgb=True,
    )

    first_mask = (torch.rand(B, K, H, W) > 0.7).float()
    rgb = torch.randn(B, 3, H, W)
    verb_emb = torch.randn(B, K, 6, text_dim)
    verb_mask = torch.ones(B, K, 6, dtype=torch.bool)
    target = (torch.rand(B, K, F, H, W) > 0.7).float()
    obj_valid = torch.ones(B, K, dtype=torch.bool)

    loss_fn = MaskPredictorLoss(w_bce=1.0, w_dice=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(5):
        opt.zero_grad()
        logits = model(first_mask, verb_emb, verb_mask, F_target=F, first_frame_rgb=rgb)
        l = loss_fn(logits, target, obj_valid)
        l["loss"].backward()
        opt.step()
        losses.append(l["loss"].item())
        print(f"  step {step+1}: loss={l['loss'].item():.4f}, "
              f"bce={l['bce'].item():.4f}, dice={l['dice'].item():.4f}")

    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "NaN in loss"
    # loss 至少要有下降趋势 (5 步太少不强求严格单调, 但末尾应 ≤ 起始)
    assert losses[-1] <= losses[0] * 1.05, f"loss not decreasing: {losses}"
    print(f"  ✓ PASS (loss {losses[0]:.3f} → {losses[-1]:.3f})")
    return True


def test_param_count():
    print("\n" + "=" * 65)
    print("[Test 4] 模型参数量 (sanity)")
    print("=" * 65)
    model = MotionMaskPredictor(
        dim=384, num_heads=6, num_blocks=6,
        text_dim=4096, patch_size=4, use_rgb=True,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  default config (dim=384, blocks=6): {n_params:,} ({n_params/1e6:.1f}M)")
    # 期望 ~20M
    assert 15e6 < n_params < 30e6, f"param count {n_params} 不符合预期"
    print(f"  ✓ PASS (in 15M-30M range)")
    return True


if __name__ == "__main__":
    ok1 = test_forward_shape()
    ok2 = test_zero_init_equivalence()
    ok3 = test_training_step()
    ok4 = test_param_count()
    print()
    if all([ok1, ok2, ok3, ok4]):
        print("✓ All tests passed")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
