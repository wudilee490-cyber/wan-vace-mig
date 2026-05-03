"""
Self-check for DecoupledMIGAdapter
==================================
不依赖真实 VACE 权重,用一个假的 mock VaceWanModel 来验证两件事:

1) 未训练时,attach adapter 后模型输出与 attach 前严格相同(数值等价)
   —— 也就是回答"未训练时是否和原VACE效果相同"

2) 训练能跑通(能算 loss、能反向、梯度不为 None、参数会被更新)
   —— 也就是回答"可以训练吗"
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from wan_vace_mig.adapter.decoupled_mig_adapter import (
    DecoupledMIGAdapter,
)


# --------------------------------------------------------------
# Mock VaceWanModel (足够小,但完全模拟原始 stack 协议)
# --------------------------------------------------------------
class MockVaceBlock(nn.Module):
    """模拟 VaceWanAttentionBlock:首层 c = before_proj(c)+x,后续层从 stack 里 pop。
       super().forward 用一个简单的 Linear 替代;after_proj 也是一个 Linear。"""
    def __init__(self, dim, block_id):
        super().__init__()
        self.block_id = block_id
        self.dim = dim
        self.body = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)

    def forward(self, c, x=None, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = self.body(c)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        return torch.stack(all_c)


class MockBaseBlock(nn.Module):
    """模拟 BaseWanAttentionBlock: x = body(x) + hints[block_id] * scale"""
    def __init__(self, dim, block_id):
        super().__init__()
        self.block_id = block_id
        self.body = nn.Linear(dim, dim)

    def forward(self, x, hints=None, context_scale=1.0, **kwargs):
        x = self.body(x)
        if self.block_id is not None and hints is not None:
            x = x + hints[self.block_id] * context_scale
        return x


class MockVaceWanModel(nn.Module):
    """足够小但完整模拟 VaceWanModel 的关键协议: vace_blocks/blocks/dim/patch_size。"""
    def __init__(self, dim=64, num_layers=4, vace_layers=(0, 2)):
        super().__init__()
        self.dim = dim
        self.patch_size = (1, 2, 2)
        self.vace_layers = list(vace_layers)
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        self.vace_blocks = nn.ModuleList([
            MockVaceBlock(dim, block_id=k) for k in range(len(self.vace_layers))
        ])
        self.blocks = nn.ModuleList([
            MockBaseBlock(dim, block_id=self.vace_layers_mapping.get(i))
            for i in range(num_layers)
        ])
        # 给 ConditioningBuilder 用(本测试不会用到)
        self.vace_in_dim = dim
        self.vace_patch_embedding = nn.Conv3d(8, dim,
                                              kernel_size=(1, 2, 2),
                                              stride=(1, 2, 2))

    def forward_vace(self, x, c0):
        """模拟原 forward_vace 的核心循环。"""
        c = c0
        for block in self.vace_blocks:
            c = block(c, x=x) if block.block_id == 0 else block(c)
        hints = torch.unbind(c)[:-1]
        return hints

    def forward(self, x, c0, context_scale=1.0):
        hints = self.forward_vace(x, c0)
        for block in self.blocks:
            x = block(x, hints=hints, context_scale=context_scale)
        return x


# ==================================================================
# Test 1: 等价性 —— 未训练时挂上 adapter 是否改变输出
# ==================================================================
def test_equivalence():
    print("=" * 65)
    print("[Test 1] 未训练等价性: attach adapter 后输出是否完全相同")
    print("=" * 65)

    torch.manual_seed(42)
    dim = 64
    model = MockVaceWanModel(dim=dim, num_layers=4, vace_layers=(0, 2))
    model.eval()

    # 构造一个固定输入
    B, L = 1, 6   # F_p=3, H_p*W_p=2 → 6 个 token
    x  = torch.randn(B, L, dim)
    c0 = torch.randn(B, L, dim)

    # ---- baseline: 不挂 adapter 的输出 ----
    with torch.no_grad():
        out_baseline = model(x, c0)

    # ---- 创建 adapter (零初始化的 to_out) ----
    adapter = DecoupledMIGAdapter(
        vace_model=model,
        num_heads=4,
        text_dim=16,
        identity_dim=dim,
        pe_dim=8,
    )
    adapter.eval()

    # ---- Case A: 挂 adapter 但不设 conditioning(_cond is None) ----
    with adapter.attached():
        with torch.no_grad():
            out_no_cond = model(x, c0)
    diff_a = (out_no_cond - out_baseline).abs().max().item()
    print(f"A) attach 但不 set_conditioning: max |diff| = {diff_a:.2e}")

    # ---- Case B: 挂 adapter 并设了 conditioning(零初始化保证 Δ=0) ----
    n_obj = 2
    F_p, H_p, W_p = 3, 1, 2
    obj_volume_masks = torch.zeros(B, n_obj, F_p, H_p, W_p)
    obj_volume_masks[0, 0, :, 0, 0] = 1   # 物体 0 占左半
    obj_volume_masks[0, 1, :, 0, 1] = 1   # 物体 1 占右半
    grid_sizes = torch.tensor([[F_p, H_p, W_p]], dtype=torch.long)

    identity_kv_list = [torch.randn(B, 4, dim) for _ in range(n_obj)]
    verb_emb_list    = [torch.randn(B, 3, 16) for _ in range(n_obj)]

    adapter.set_conditioning(
        identity_kv_list=identity_kv_list,
        verb_emb_list=verb_emb_list,
        obj_volume_masks=obj_volume_masks,
        seq_len=L,
        grid_sizes=grid_sizes,
        F_p=F_p,
        alpha_id=1.0, alpha_mo=1.0,
    )
    with adapter.attached():
        with torch.no_grad():
            out_zero_init = model(x, c0)
    diff_b = (out_zero_init - out_baseline).abs().max().item()
    print(f"B) attach + set_conditioning (零初始化未训练): max |diff| = {diff_b:.2e}")

    # ---- Case C: detach 后又跑一次,确认 hook 被清理干净 ----
    adapter.detach()
    with torch.no_grad():
        out_after_detach = model(x, c0)
    diff_c = (out_after_detach - out_baseline).abs().max().item()
    print(f"C) detach 之后再跑: max |diff| = {diff_c:.2e}")

    # 严格判定
    tol = 1e-6
    pass_a = diff_a < tol
    pass_b = diff_b < tol
    pass_c = diff_c < tol
    all_pass = pass_a and pass_b and pass_c
    status = "✓ PASS" if all_pass else "✗ FAIL"
    print(f"\n  → 严格等价 (tol=1e-6): A={pass_a}, B={pass_b}, C={pass_c}  {status}")
    return all_pass


# ==================================================================
# Test 2: 训练可行性 —— 能算 loss、能反向、参数会更新
# ==================================================================
def test_trainable():
    print()
    print("=" * 65)
    print("[Test 2] 训练可行性: 反向传播能否打通到 adapter 参数")
    print("=" * 65)

    torch.manual_seed(0)
    dim = 64
    model = MockVaceWanModel(dim=dim, num_layers=4, vace_layers=(0, 2))

    adapter = DecoupledMIGAdapter(
        vace_model=model,
        num_heads=4,
        text_dim=16,
        identity_dim=dim,
        pe_dim=8,
    )

    # 冻结主干,只训 adapter
    adapter.freeze_base()
    n_train = sum(p.numel() for p in adapter.trainable_parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  可训练参数: {n_train:,}")
    print(f"  已冻结参数 (vace_model): {n_frozen:,}")

    # 输入
    B, L, dim = 1, 6, 64
    F_p, H_p, W_p = 3, 1, 2
    x  = torch.randn(B, L, dim, requires_grad=False)
    c0 = torch.randn(B, L, dim, requires_grad=False)

    n_obj = 2
    obj_volume_masks = torch.zeros(B, n_obj, F_p, H_p, W_p)
    obj_volume_masks[0, 0, :, 0, 0] = 1
    obj_volume_masks[0, 1, :, 0, 1] = 1
    grid_sizes = torch.tensor([[F_p, H_p, W_p]], dtype=torch.long)

    identity_kv_list = [torch.randn(B, 4, dim) for _ in range(n_obj)]
    verb_emb_list    = [torch.randn(B, 3, 16) for _ in range(n_obj)]

    target = torch.randn(B, L, dim)

    # ---- 走训练 step ----
    optimizer = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-3)

    # 因为零初始化,first step Δ=0,output 可能正好等于无 adapter 的输出 → grad 也可以正常流
    adapter.set_conditioning(
        identity_kv_list=identity_kv_list,
        verb_emb_list=verb_emb_list,
        obj_volume_masks=obj_volume_masks,
        seq_len=L,
        grid_sizes=grid_sizes,
        F_p=F_p,
        alpha_id=1.0, alpha_mo=1.0,
    )

    # 记录初始参数状态(取一个采样)
    sample_param = next(iter(adapter.injectors[0].id_attn.to_q.parameters()))
    init_val = sample_param.detach().clone()
    sample_pe_param = next(iter(adapter.phase_motion_encoder.mlp.parameters()))
    init_pe_val = sample_pe_param.detach().clone()

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        with adapter.attached():
            out = model(x, c0)
        loss = ((out - target) ** 2).mean()
        loss.backward()

        # 检查 adapter 参数有梯度,VACE 主干没梯度
        any_adapter_grad = False
        any_main_grad = False
        for p in adapter.trainable_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                any_adapter_grad = True
                break
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                any_main_grad = True
                break

        optimizer.step()
        losses.append(loss.item())
        if step == 0:
            first_step_grad_check = (any_adapter_grad, any_main_grad)

    # ---- 判定 ----
    moved_id = (sample_param - init_val).abs().max().item()
    moved_pe = (sample_pe_param - init_pe_val).abs().max().item()
    has_grad, no_main = first_step_grad_check

    print(f"  Loss 序列: {[f'{l:.4f}' for l in losses]}")
    print(f"  Step1: adapter 有梯度={has_grad}, vace_model 有梯度={no_main}")
    print(f"  采样参数变化量: id_attn.to_q={moved_id:.2e}, phase_mlp={moved_pe:.2e}")

    pass_grad_flow = has_grad and (not no_main)
    pass_param_moved = (moved_id > 1e-9) and (moved_pe > 1e-9)
    pass_loss_finite = all([torch.isfinite(torch.tensor(l)).item() for l in losses])

    print(f"\n  梯度只流向 adapter:  {'✓' if pass_grad_flow else '✗'}")
    print(f"  参数确实被更新:      {'✓' if pass_param_moved else '✗'}")
    print(f"  Loss 数值正常无 NaN: {'✓' if pass_loss_finite else '✗'}")

    all_pass = pass_grad_flow and pass_param_moved and pass_loss_finite
    return all_pass


# ==================================================================
# Test 3: 注入位置正确性 —— 改变 alpha 应该影响输出
# ==================================================================
def test_injection_takes_effect():
    print()
    print("=" * 65)
    print("[Test 3] 注入有效性: 训过的 adapter (用 alpha 模拟非零 Δ) 改变输出")
    print("=" * 65)

    torch.manual_seed(7)
    dim = 64
    model = MockVaceWanModel(dim=dim, num_layers=4, vace_layers=(0, 2))
    model.eval()

    adapter = DecoupledMIGAdapter(
        vace_model=model, num_heads=4, text_dim=16, identity_dim=dim, pe_dim=8,
    )
    # 手动把 to_out 的零权重改成非零,模拟"训完之后"
    for inj in adapter.injectors:
        nn.init.normal_(inj.id_attn.to_out.weight, std=0.01)
        nn.init.normal_(inj.mo_attn.to_out.weight, std=0.01)
    adapter.eval()

    B, L, dim = 1, 6, 64
    F_p, H_p, W_p = 3, 1, 2
    x  = torch.randn(B, L, dim)
    c0 = torch.randn(B, L, dim)

    n_obj = 2
    obj_volume_masks = torch.zeros(B, n_obj, F_p, H_p, W_p)
    obj_volume_masks[0, 0, :, 0, 0] = 1
    obj_volume_masks[0, 1, :, 0, 1] = 1
    grid_sizes = torch.tensor([[F_p, H_p, W_p]], dtype=torch.long)

    identity_kv_list = [torch.randn(B, 4, dim) for _ in range(n_obj)]
    verb_emb_list    = [torch.randn(B, 3, 16) for _ in range(n_obj)]

    with torch.no_grad():
        out_baseline = model(x, c0)

    # alpha=0 → 仍应等于 baseline
    adapter.set_conditioning(
        identity_kv_list=identity_kv_list, verb_emb_list=verb_emb_list,
        obj_volume_masks=obj_volume_masks, seq_len=L,
        grid_sizes=grid_sizes, F_p=F_p,
        alpha_id=0.0, alpha_mo=0.0,
    )
    with adapter.attached():
        with torch.no_grad():
            out_alpha0 = model(x, c0)
    diff_alpha0 = (out_alpha0 - out_baseline).abs().max().item()

    # alpha=1 → 应当不同(adapter 起作用了)
    adapter.update_alphas = lambda **kw: None  # noqa: prevent surprises
    adapter._cond["alpha_id"] = 1.0
    adapter._cond["alpha_mo"] = 1.0
    # 清缓存因为 alpha 不影响 motion KV,但保险一下
    adapter._cond["_motion_cache_shared"] = None
    adapter._cond["_motion_cache_per_layer"] = [None] * len(adapter.layers)

    with adapter.attached():
        with torch.no_grad():
            out_alpha1 = model(x, c0)
    diff_alpha1 = (out_alpha1 - out_baseline).abs().max().item()

    print(f"  alpha=0 时 max|diff| = {diff_alpha0:.2e}  (期望 ~0)")
    print(f"  alpha=1 时 max|diff| = {diff_alpha1:.2e}  (期望 >> 0)")
    pass_a0 = diff_alpha0 < 1e-6
    pass_a1 = diff_alpha1 > 1e-3
    return pass_a0 and pass_a1


if __name__ == "__main__":
    r1 = test_equivalence()
    r2 = test_trainable()
    r3 = test_injection_takes_effect()

    print()
    print("=" * 65)
    print("汇总:")
    print(f"  [1] 未训练等价性:    {'✓ PASS' if r1 else '✗ FAIL'}")
    print(f"  [2] 训练可行性:      {'✓ PASS' if r2 else '✗ FAIL'}")
    print(f"  [3] 注入生效性:      {'✓ PASS' if r3 else '✗ FAIL'}")
    print("=" * 65)
    sys.exit(0 if (r1 and r2 and r3) else 1)
