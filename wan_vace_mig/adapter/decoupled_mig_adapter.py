"""
Decoupled MIG Adapter (Phase-Aware) — ControlNet-style外挂模块
================================================================

相对上一版的关键升级:
    motion 分支从「整段视频共享一份 K/V」升级为「每个 patch 帧独立一份 K/V」,
    让动词在时间上展开成连续相位(起跳→腾空→落地),解决"running 在第 5 帧和第
    20 帧没区别"的问题。

公式(per-object i, per-frame f):
    K_act^{i,f} = V_act^{i,f} = MLP_i( [verb_emb_i ; PE(f/T)] )
其中 PE 是 sin/cos 位置编码,T 是总 patch 帧数,f 是当前帧 index。

实现技巧:
    把所有帧的相位 K/V 拼成一个长 K/V 序列 [B, F_p * N_v, dim],然后构造
    block-diagonal attention mask: query token 在第 f 帧时,只允许看 K/V 中
    第 f*N_v ~ (f+1)*N_v 这一段。一次 SDPA 完成所有帧,GPU 友好。

挂载位置仍然是 VaceWanAttentionBlock 的输出(hints 通路),不改 VACE 主干。
"""

from typing import List, Optional, Tuple, Dict, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 0. 位置编码 & RMSNorm 兜底
# ============================================================================
def sinusoidal_phase_pe(phases: torch.Tensor, dim: int) -> torch.Tensor:
    """
    将 [0,1] 区间的相位 phases 编码成 sin/cos 位置编码,最后一维 = dim。

    Args:
        phases: 任意形状的 float tensor,值域建议 [0,1]
        dim:    输出最后一维大小,需为偶数
    Returns:
        与 phases 同形状,最后一维多了 dim
    """
    assert dim % 2 == 0, "phase PE dim must be even"
    half = dim // 2
    device, dtype = phases.device, torch.float32
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / half
    )                                                                     # [half]
    angles = phases.to(dtype).unsqueeze(-1) * freq * (2 * math.pi)        # [..., half]
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)        # [..., dim]
    return pe


class _FallbackRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rmsnorm(dim: int, eps: float = 1e-6):
    return nn.RMSNorm(dim, eps=eps) if hasattr(nn, "RMSNorm") else _FallbackRMSNorm(dim, eps)


# ============================================================================
# 1. 相位感知动作编码器: K_act = V_act = MLP([verb_emb ; PE(f/T)])
# ============================================================================
class PhaseAwareMotionEncoder(nn.Module):
    """
    输入:
        verb_emb: [B, N_v, text_dim]    动词短语的 T5 嵌入(per object 已经在外面分好)
        F_p:      int                   总 patch 帧数 T
    输出:
        kv_per_frame: [B, F_p, N_v, dim]   每帧每动词 token 一份独立 K/V

    结构:
        concat([verb_emb broadcast, PE(f/T) broadcast]) → MLP → kv_per_frame
        其中 PE(f/T) 是逐帧的相位编码,广播到每个 verb token 上。

    物理意义:
        把抽象动词拆解成连续"相位"。"jump" 在 f/T=0.1 时是起跳,在 0.5 时是腾空,
        在 0.85 时是落地;同一个 verb_emb 经过不同 PE 调制后,MLP 输出截然不同
        的姿态语义,正好提供给 cross-attn 当作"该帧应有的动作特征"。
    """

    def __init__(
        self,
        text_dim: int,
        out_dim: int,
        pe_dim: int = 64,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.pe_dim = pe_dim

        in_dim = text_dim + pe_dim
        hidden_dim = hidden_dim if hidden_dim is not None else max(out_dim, in_dim)

        layers = []
        prev = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU()]
            prev = hidden_dim
        layers += [nn.Linear(prev, out_dim)]
        self.mlp = nn.Sequential(*layers)

        self.out_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        verb_emb: torch.Tensor,            # [B, N_v, text_dim]
        F_p: int,
    ) -> torch.Tensor:                     # [B, F_p, N_v, out_dim]
        B, N_v, _ = verb_emb.shape
        device = verb_emb.device

        # 用 (f + 0.5) / F_p 让两端不退化
        f_idx = torch.arange(F_p, device=device, dtype=torch.float32)
        phases = (f_idx + 0.5) / max(F_p, 1)                     # [F_p] in (0,1)
        pe = sinusoidal_phase_pe(phases, self.pe_dim)            # [F_p, pe_dim]

        verb = verb_emb.unsqueeze(1).expand(B, F_p, N_v, self.text_dim)
        pe_b = pe.view(1, F_p, 1, self.pe_dim).expand(B, F_p, N_v, self.pe_dim)
        cat = torch.cat([verb, pe_b.to(verb.dtype)], dim=-1)

        kv = self.mlp(cat)                                       # [B, F_p, N_v, out_dim]
        kv = self.out_norm(kv)
        return kv


# ============================================================================
# 2. Cross-Attn:支持「全局 K/V」和「per-frame K/V」两种模式
# ============================================================================
class MaskedCrossAttention(nn.Module):
    """
    身份分支:K/V 全局共享(身份是首帧固定特征,无相位概念)
    动作分支:K/V 按帧切片,query 在第 f 帧时只能看 K/V 第 f 个切片

    通过 frame_block_size 区分:
        None  → 传统全局 cross-attn
        N_v   → K/V 长度 = F_p * N_v,query 第 f 帧只看 [f*N_v, (f+1)*N_v)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        kv_dim = kv_dim if kv_dim is not None else dim
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(kv_dim, dim, bias=True)
        self.to_v = nn.Linear(kv_dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.norm_q = _rmsnorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.norm_k = _rmsnorm(self.head_dim, eps) if qk_norm else nn.Identity()

        # 零初始化:挂上外挂时模型行为不变,训练初期梯度集中在新模块
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(
        self,
        x: torch.Tensor,                         # [B, L, dim]
        kv: torch.Tensor,                        # [B, N_kv, kv_dim]
        q_mask: Optional[torch.Tensor],          # [B, L] bool
        kv_mask: Optional[torch.Tensor] = None,  # [B, N_kv] bool
        token_frame_idx: Optional[torch.Tensor] = None,  # [B, L] long
        frame_block_size: Optional[int] = None,  # 每帧 K/V 占多少 token
    ) -> torch.Tensor:
        B, L, _ = x.shape
        N = kv.shape[1]
        H, D = self.num_heads, self.head_dim

        if q_mask is not None and not q_mask.any():
            return x.new_zeros(x.shape)

        q = self.norm_q(self.to_q(x).view(B, L, H, D)).transpose(1, 2)   # [B,H,L,D]
        k = self.norm_k(self.to_k(kv).view(B, N, H, D)).transpose(1, 2)
        v = self.to_v(kv).view(B, N, H, D).transpose(1, 2)

        attn_mask = None
        if frame_block_size is not None:
            assert token_frame_idx is not None, \
                "per-frame K/V 模式必须提供 token_frame_idx"
            kv_frame = (torch.arange(N, device=kv.device) // frame_block_size)   # [N]
            allow = (token_frame_idx[:, None, :, None] ==
                     kv_frame.view(1, 1, 1, N))                          # [B,1,L,N]
            allow = allow.expand(B, H, L, N)
            attn_mask = allow
            if kv_mask is not None:
                attn_mask = attn_mask & kv_mask[:, None, None, :].expand(B, H, L, N)
        else:
            if kv_mask is not None:
                attn_mask = kv_mask[:, None, None, :].expand(B, H, L, N)

        # 防御:若某 query 行 attn_mask 全 False,SDPA 会出 NaN。允许它看第 0 个 K/V token
        # (q_mask gate 会把 Δ 屏蔽,具体值不影响结果)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            row_any = attn_mask.any(dim=-1, keepdim=True)
            fix = (~row_any) & (
                torch.arange(N, device=kv.device).view(1, 1, 1, N) == 0
            )
            attn_mask = attn_mask | fix

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, L, H * D)
        out = self.to_out(out)

        if q_mask is not None:
            out = out * q_mask.to(out.dtype).unsqueeze(-1)
        return out


# ============================================================================
# 3. 单层解耦注入块:身份(全局 KV) + 动作(per-frame KV)
# ============================================================================
class DecoupledInjectionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        text_dim: Optional[int] = None,
        identity_dim: Optional[int] = None,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        identity_dim = identity_dim if identity_dim is not None else dim

        self.norm_x_id = nn.LayerNorm(dim, eps=eps)
        self.norm_x_mo = nn.LayerNorm(dim, eps=eps)

        # 身份分支:全局 cross-attn
        self.id_attn = MaskedCrossAttention(
            dim=dim, num_heads=num_heads, kv_dim=identity_dim,
            qk_norm=qk_norm, eps=eps,
        )
        # 动作分支:per-frame cross-attn
        # 注意 kv_dim = dim,因为 PhaseAwareMotionEncoder 已经把 K/V 投到 dim
        self.mo_attn = MaskedCrossAttention(
            dim=dim, num_heads=num_heads, kv_dim=dim,
            qk_norm=qk_norm, eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,                                   # [B, L, dim]
        identity_kv_list: List[torch.Tensor],
        identity_kv_masks: Optional[List[torch.Tensor]],
        motion_kv_perframe_list: List[torch.Tensor],       # 每项 [B, F_p, N_v, dim]
        motion_kv_masks: Optional[List[torch.Tensor]],     # 每项 [B, F_p, N_v]
        obj_query_masks: List[torch.Tensor],
        token_frame_idx: torch.Tensor,
        alpha_id=1.0,                       # float | List[float] | Tensor[K]
        alpha_mo=1.0,                       # float | List[float] | Tensor[K]
        first_frame_protect_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        alpha_id / alpha_mo 接受三种形式:
            float          —— 全局, 所有物体相同强度 (向后兼容)
            List[float]    —— per object, 长度 = n_obj
            Tensor[K]      —— per object, 形状 [n_obj]
        长度不足时(< n_obj)自动用最后一个值 broadcast 到剩余物体.
        """
        delta = torch.zeros_like(x)

        x_id = self.norm_x_id(x)
        x_mo = self.norm_x_mo(x)

        n_obj = len(obj_query_masks)
        # 把 alpha 归一化到长度 n_obj 的列表 (per-object scalars)
        a_id_list = _broadcast_alpha(alpha_id, n_obj)
        a_mo_list = _broadcast_alpha(alpha_mo, n_obj)

        for i in range(n_obj):
            q_mask = obj_query_masks[i]

            # —— identity: 全局 KV ——
            id_kv = identity_kv_list[i]
            id_kvm = identity_kv_masks[i] if identity_kv_masks else None
            d_id = self.id_attn(x_id, id_kv, q_mask, id_kvm,
                                token_frame_idx=None, frame_block_size=None)

            # —— motion: per-frame KV ——
            mo_kv_pf = motion_kv_perframe_list[i]              # [B,F_p,N_v,dim]
            B_, F_p, N_v, D_ = mo_kv_pf.shape
            mo_kv = mo_kv_pf.reshape(B_, F_p * N_v, D_)
            mo_kvm = (motion_kv_masks[i].reshape(B_, F_p * N_v)
                      if motion_kv_masks is not None else None)

            d_mo = self.mo_attn(
                x_mo, mo_kv, q_mask, mo_kvm,
                token_frame_idx=token_frame_idx,
                frame_block_size=N_v,
            )

            delta = delta + a_id_list[i] * d_id + a_mo_list[i] * d_mo

        if first_frame_protect_mask is not None:
            keep = (~first_frame_protect_mask).to(delta.dtype).unsqueeze(-1)
            delta = delta * keep
        return delta


def _broadcast_alpha(alpha, n_obj: int) -> List:
    """
    把 alpha (float | list | tensor) 归一化成长度 = n_obj 的列表.

    规则:
        - 标量: 广播到 [α] * n_obj
        - 列表/张量长度 == n_obj: 直接用
        - 列表/张量长度 == 1: 取那个值广播
        - 列表/张量长度 < n_obj: 用最后一个值补齐 (允许"前 K 个物体精细控制,
          剩余用默认"的便利写法)
        - 长度 > n_obj: 截断 (不报错,生产环境更鲁棒)
    """
    # 标量 (int/float)
    if isinstance(alpha, (int, float)):
        return [float(alpha)] * n_obj

    # Tensor → list of float
    if isinstance(alpha, torch.Tensor):
        if alpha.ndim == 0:
            return [alpha.item()] * n_obj
        alpha = alpha.flatten().tolist()

    # list/tuple
    if not isinstance(alpha, (list, tuple)):
        raise TypeError(f"alpha must be float/list/Tensor, got {type(alpha)}")

    alpha = list(alpha)
    if len(alpha) == 0:
        return [1.0] * n_obj
    if len(alpha) == n_obj:
        return [float(a) for a in alpha]
    if len(alpha) == 1:
        return [float(alpha[0])] * n_obj
    if len(alpha) > n_obj:
        return [float(a) for a in alpha[:n_obj]]
    # < n_obj: 补最后一个
    return [float(a) for a in alpha] + [float(alpha[-1])] * (n_obj - len(alpha))


# ============================================================================
# 4. Mask / 索引工具
# ============================================================================
def build_obj_query_masks(
    obj_volume_masks: torch.Tensor,    # [B, n_obj, F_p, H_p, W_p]
    seq_len: int,
    grid_sizes: torch.Tensor,          # [B, 3]
) -> List[torch.Tensor]:
    """每个物体一份 [B, seq_len] bool query mask。"""
    B, n_obj = obj_volume_masks.shape[:2]
    device = obj_volume_masks.device
    out: List[torch.Tensor] = []
    for i in range(n_obj):
        m_i = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        for b in range(B):
            f, h, w = grid_sizes[b].tolist()
            real_len = f * h * w
            assert real_len <= seq_len
            flat = obj_volume_masks[b, i, :f, :h, :w].reshape(-1).bool()
            m_i[b, :real_len] = flat
        out.append(m_i)
    return out


def build_first_frame_protect_mask(
    seq_len: int, grid_sizes: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    """首帧 token 保护 mask,True 处不接受 Δ。"""
    B = grid_sizes.shape[0]
    mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    for b in range(B):
        f, h, w = grid_sizes[b].tolist()
        mask[b, : h * w] = True
    return mask


def build_token_frame_index(
    seq_len: int, grid_sizes: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    """
    每个序列位置属于哪一帧。pad 区域填 0(q_mask 会屏蔽,具体值不影响计算)。
    输出: [B, seq_len] long,值域 [0, F_p)。
    """
    B = grid_sizes.shape[0]
    out = torch.zeros(B, seq_len, dtype=torch.long, device=device)
    for b in range(B):
        f_p, h_p, w_p = grid_sizes[b].tolist()
        per_frame = h_p * w_p
        real_len = f_p * per_frame
        idx = torch.arange(real_len, device=device) // per_frame
        out[b, :real_len] = idx
    return out


# ============================================================================
# 5. 顶层 Adapter
# ============================================================================
class DecoupledMIGAdapter(nn.Module):
    """
    用法:
        adapter = DecoupledMIGAdapter(
            vace_model, num_heads=16, text_dim=4096,
        )
        adapter.set_conditioning(
            identity_kv_list=...,         # 每物体 [B, N_id_i, identity_dim]
            verb_emb_list=...,            # 每物体 [B, N_v_i, text_dim]
            obj_volume_masks=...,         # [B, n_obj, F_p, H_p, W_p]
            seq_len=..., grid_sizes=...,
            ...,
        )
        with adapter.attached():
            video = vace_model(...)

    关键点:
        - vace_model 不被注册为子模块,不会被 self.parameters() 暴露,避免被 freeze 影响
        - share_phase_encoder_across_layers=True 时所有层共用一份相位编码器,
          per-layer motion KV 只算一次缓存,省算力
    """

    def __init__(
        self,
        vace_model: nn.Module,
        num_heads: int,
        text_dim: int,
        identity_dim: Optional[int] = None,
        pe_dim: int = 64,
        phase_encoder_layers: int = 2,
        share_phase_encoder_across_layers: bool = True,
        share_injection_block: bool = False,
        qk_norm: bool = True,
        eps: float = 1e-6,
        layers: Optional[List[int]] = None,
    ):
        """
        Args:
            share_injection_block:
                False (默认): 每层一份独立的 DecoupledInjectionBlock (~70M params).
                              表达力强,允许浅/中/深层学不同的注入语义.
                True:         所有层共享同一个 block (~2-3M params).
                              省 95% adapter 参数 + 训练显存,但牺牲分层语义.
                              建议仅在显存严重不够时使用.
        """
        super().__init__()
        # 不通过 self.xxx = vace_model 注册,避免 .parameters() 把主干带进来
        object.__setattr__(self, "vace_model", vace_model)
        self.dim = vace_model.dim

        if layers is None:
            layers = list(range(len(vace_model.vace_blocks)))
        self.layers = layers

        self.share_injection_block = share_injection_block
        if share_injection_block:
            # 创建 1 个 injection block, 所有层 hook 共享同一个实例.
            # 注意: ModuleList 仍然必须有 len(layers) 个引用,否则 hook 找不到.
            #       用 [block] * N 共享同一对象,Python 引用,不会复制权重.
            shared = DecoupledInjectionBlock(
                dim=self.dim, num_heads=num_heads,
                text_dim=text_dim, identity_dim=identity_dim,
                qk_norm=qk_norm, eps=eps,
            )
            # 但 nn.ModuleList 不允许重复模块 (它会去重 parameters() 但 forward 仍正常),
            # 为安全,用普通 list,在 attach 时显式 hook 每层
            self._shared_injector = shared
            self.injectors = None    # 标记: 走共享路径
        else:
            self._shared_injector = None
            self.injectors = nn.ModuleList([
                DecoupledInjectionBlock(
                    dim=self.dim, num_heads=num_heads,
                    text_dim=text_dim, identity_dim=identity_dim,
                    qk_norm=qk_norm, eps=eps,
                )
                for _ in layers
            ])

        self.share_phase_encoder = share_phase_encoder_across_layers
        if share_phase_encoder_across_layers:
            self.phase_motion_encoder = PhaseAwareMotionEncoder(
                text_dim=text_dim, out_dim=self.dim, pe_dim=pe_dim,
                num_layers=phase_encoder_layers,
            )
            self.phase_motion_encoders = None
        else:
            self.phase_motion_encoder = None
            self.phase_motion_encoders = nn.ModuleList([
                PhaseAwareMotionEncoder(
                    text_dim=text_dim, out_dim=self.dim, pe_dim=pe_dim,
                    num_layers=phase_encoder_layers,
                )
                for _ in layers
            ])

        self._cond: Optional[Dict] = None
        self._hook_handles: List = []
        # 训练时的 per-forward-step 缓存 (避免 layer 间 phase_encoder 重复计算).
        # 在第 0 层 hook 调用前由 _reset_step_cache() 清空,保证每次 forward
        # 都用新的 graph 但同次 forward 内 layer 间共享.
        self._step_motion_cache_shared = None
        self._step_motion_cache_per_layer: List = [None] * len(self.layers)

    # -------- 条件设置 --------
    def set_conditioning(
        self,
        identity_kv_list: List[torch.Tensor],
        verb_emb_list: List[torch.Tensor],
        obj_volume_masks: torch.Tensor,
        seq_len: int,
        grid_sizes: torch.Tensor,
        F_p: Optional[int] = None,
        identity_kv_masks: Optional[List[torch.Tensor]] = None,
        verb_emb_masks: Optional[List[torch.Tensor]] = None,
        alpha_id: Union[float, List[float], torch.Tensor] = 1.0,
        alpha_mo: Union[float, List[float], torch.Tensor] = 1.0,
        protect_first_frame: bool = True,
        per_layer_alpha: Optional[List[Tuple[
            Union[float, List[float], torch.Tensor],
            Union[float, List[float], torch.Tensor],
        ]]] = None,
    ):
        """
        alpha_id / alpha_mo 接受三种形式:
            float            —— 全局,所有物体相同 (向后兼容)
            List[float] / Tensor[K]  —— per object 强度
        per_layer_alpha 接受 [(α_id_layer0, α_mo_layer0), ...],
            每个 α_id/α_mo 也可以是 float 或 per-object list/tensor.
        长度不足时自动用最后一个值补齐 (见 _broadcast_alpha 文档).
        """
        n_obj = obj_volume_masks.shape[1]
        assert len(identity_kv_list) == len(verb_emb_list) == n_obj
        device = obj_volume_masks.device

        if F_p is None:
            F_p = int(grid_sizes[:, 0].max().item())

        # Detach 用户传入的条件张量,确保它们不会把外部 graph 牵进 adapter 的 forward。
        # 这些张量都是"输入条件",梯度不应回传到它们的源头(VAE/T5/首帧 embedding 等)。
        # 不 detach 会在多步训练时触发 "backward through the graph a second time"。
        identity_kv_list = [t.detach() for t in identity_kv_list]
        verb_emb_list    = [t.detach() for t in verb_emb_list]
        if identity_kv_masks is not None:
            identity_kv_masks = [t.detach() for t in identity_kv_masks]
        if verb_emb_masks is not None:
            verb_emb_masks = [t.detach() for t in verb_emb_masks]

        obj_query_masks = build_obj_query_masks(obj_volume_masks, seq_len, grid_sizes)
        token_frame_idx = build_token_frame_index(seq_len, grid_sizes, device)
        protect = (build_first_frame_protect_mask(seq_len, grid_sizes, device)
                   if protect_first_frame else None)

        self._cond = dict(
            identity_kv_list=identity_kv_list,
            identity_kv_masks=identity_kv_masks,
            verb_emb_list=verb_emb_list,
            verb_emb_masks=verb_emb_masks,
            obj_query_masks=obj_query_masks,
            token_frame_idx=token_frame_idx,
            F_p=F_p,
            protect=protect,
            alpha_id=alpha_id,
            alpha_mo=alpha_mo,
            per_layer_alpha=per_layer_alpha,
            _motion_cache_shared=None,
            _motion_cache_per_layer=[None] * len(self.layers),
        )

    def clear_conditioning(self):
        self._cond = None

    # -------- 内部:取/算 per-frame motion KV --------
    def _get_motion_kv_for_layer(self, layer_pos: int):
        cond = self._cond
        verb_list = cond["verb_emb_list"]
        verb_masks = cond["verb_emb_masks"]
        F_p = cond["F_p"]

        def _build(encoder):
            kv_list, mask_list = [], ([] if verb_masks is not None else None)
            for i, verb in enumerate(verb_list):
                kv = encoder(verb, F_p=F_p)                    # [B,F_p,N_v,dim]
                kv_list.append(kv)
                if verb_masks is not None:
                    m = verb_masks[i].unsqueeze(1).expand(-1, F_p, -1)
                    mask_list.append(m)
            return kv_list, mask_list

        # 训练模式 vs 推理模式的缓存策略不同。
        # 训练: 不能跨 forward step 缓存 (graph buffer 在 backward 时释放),
        #       但可以在同一次 forward 内 layer 间复用 (避免 N×重算)。
        #       用 step_cache 实现: hook 在每个 vace_block 上调一次,
        #       同次 forward 内多 layer 共享同一个 phase_encoder 输出。
        # 推理: 跨多步采样缓存 (UniPC/DDIM 多步推理共享同一份条件),
        #       缓存进 self._cond,直到下次 set_conditioning() 才清空。
        is_training = self.training

        if self.share_phase_encoder:
            if is_training:
                # 同一 forward step 内复用 (step_cache 由 hook 重置)
                if self._step_motion_cache_shared is None:
                    self._step_motion_cache_shared = _build(self.phase_motion_encoder)
                return self._step_motion_cache_shared
            # eval/inference: 长期缓存
            if cond["_motion_cache_shared"] is None:
                cond["_motion_cache_shared"] = _build(self.phase_motion_encoder)
            return cond["_motion_cache_shared"]
        else:
            if is_training:
                if self._step_motion_cache_per_layer[layer_pos] is None:
                    enc = self.phase_motion_encoders[layer_pos]
                    self._step_motion_cache_per_layer[layer_pos] = _build(enc)
                return self._step_motion_cache_per_layer[layer_pos]
            # eval
            if cond["_motion_cache_per_layer"][layer_pos] is None:
                enc = self.phase_motion_encoders[layer_pos]
                cond["_motion_cache_per_layer"][layer_pos] = _build(enc)
            return cond["_motion_cache_per_layer"][layer_pos]

    def _reset_step_cache(self):
        """每次主模型 forward 开始时调用,清空 step-level cache."""
        self._step_motion_cache_shared = None
        self._step_motion_cache_per_layer = [None] * len(self.layers)

    # -------- Hook 管理 --------
    def attach(self):
        assert not self._hook_handles
        for layer_pos, vace_idx in enumerate(self.layers):
            block = self.vace_model.vace_blocks[vace_idx]
            # 共享模式: 所有层用同一个 injector 实例
            injector = (self._shared_injector if self.share_injection_block
                        else self.injectors[layer_pos])
            handle = block.register_forward_hook(self._make_hook(injector, layer_pos))
            self._hook_handles.append(handle)

    def detach(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def attached(self):
        adapter = self
        class _Ctx:
            def __enter__(self_inner):
                adapter.attach()
                return adapter
            def __exit__(self_inner, *a):
                adapter.detach()
                return False
        return _Ctx()

    def _make_hook(self, injector: DecoupledInjectionBlock, layer_pos: int):
        def hook(module, inputs, output):
            if self._cond is None:
                return output
            if output.dim() != 4 or output.shape[0] < 2:
                return output

            # 第一层 hook 触发时 = 一次新的 forward 开始,清空 step-level cache.
            # 这保证训练时每次 forward 都用新 graph,但同次 forward 内 layer
            # 共享同一个 phase_encoder 输出 (避免 N×重算).
            if layer_pos == 0:
                self._reset_step_cache()

            c_skip = output[-2]
            c_curr = output[-1]
            cond = self._cond

            if cond["per_layer_alpha"] is not None:
                a_id, a_mo = cond["per_layer_alpha"][layer_pos]
            else:
                a_id, a_mo = cond["alpha_id"], cond["alpha_mo"]

            mo_kv_list, mo_mask_list = self._get_motion_kv_for_layer(layer_pos)

            delta = injector(
                c_skip,
                identity_kv_list=cond["identity_kv_list"],
                identity_kv_masks=cond["identity_kv_masks"],
                motion_kv_perframe_list=mo_kv_list,
                motion_kv_masks=mo_mask_list,
                obj_query_masks=cond["obj_query_masks"],
                token_frame_idx=cond["token_frame_idx"],
                alpha_id=a_id, alpha_mo=a_mo,
                first_frame_protect_mask=cond["protect"],
            )

            new_c_skip = c_skip + delta
            new_output = torch.cat(
                [output[:-2], new_c_skip.unsqueeze(0), c_curr.unsqueeze(0)], dim=0
            )
            return new_output
        return hook

    # -------- 训练辅助 --------
    def freeze_base(self):
        """冻结 VACE 主干(和 WanModel 主干)。"""
        for p in self.vace_model.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        params = []
        if self.share_injection_block:
            params += list(self._shared_injector.parameters())
        else:
            params += list(self.injectors.parameters())
        if self.phase_motion_encoder is not None:
            params += list(self.phase_motion_encoder.parameters())
        if self.phase_motion_encoders is not None:
            params += list(self.phase_motion_encoders.parameters())
        return params
