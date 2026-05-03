"""
Agent feedback loop — phase-aware + per-object control
========================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union


@dataclass
class FeedbackItem:
    frame_range: Tuple[int, int]
    object_idx: int                 # 哪个物体出问题
    issue: str                      # "identity_drift" | "action_mismatch" | "shape_break"
                                    # | "phase_lag" | "phase_lead" | "leakage_to_other_object"
    severity: float = 1.0


@dataclass
class GenerationState:
    """
    Per-object 强度: 列表形式, 每物体独立 alpha.
    per_layer_alpha 也是 per object: 每层每物体一个 (id, mo) 对.
    """
    n_obj: int = 1
    # per object alpha (长度 = n_obj)
    alpha_id: List[float] = field(default_factory=lambda: [1.0])
    alpha_mo: List[float] = field(default_factory=lambda: [1.0])
    # per layer × per object (None = 用全局 alpha_id/alpha_mo)
    # 形式: [ (id_per_obj_list, mo_per_obj_list), ... ] 长度 = num_layers
    per_layer_alpha: Optional[List[Tuple[List[float], List[float]]]] = None

    def __post_init__(self):
        # 自动 broadcast 标量到 per-obj
        if isinstance(self.alpha_id, (int, float)):
            self.alpha_id = [float(self.alpha_id)] * self.n_obj
        if isinstance(self.alpha_mo, (int, float)):
            self.alpha_mo = [float(self.alpha_mo)] * self.n_obj
        # 长度对齐
        if len(self.alpha_id) < self.n_obj:
            self.alpha_id = self.alpha_id + [self.alpha_id[-1]] * (self.n_obj - len(self.alpha_id))
        if len(self.alpha_mo) < self.n_obj:
            self.alpha_mo = self.alpha_mo + [self.alpha_mo[-1]] * (self.n_obj - len(self.alpha_mo))

    @classmethod
    def init_uniform(cls, n_obj: int, alpha_id: float = 1.0, alpha_mo: float = 1.0):
        return cls(
            n_obj=n_obj,
            alpha_id=[alpha_id] * n_obj,
            alpha_mo=[alpha_mo] * n_obj,
        )


def update_alphas_from_feedback(state: GenerationState,
                                feedbacks: List[FeedbackItem],
                                num_layers: int) -> GenerationState:
    """
    评估反馈 → per-object alpha 调整。

    现在每条反馈带 object_idx,只调整对应物体的 alpha,其他物体不动。
    这是相比全局调节更精准的关键 — 比如:
        物体 0 身份漂移 → 只加大物体 0 的 alpha_id 中层
        物体 1 动作不准 → 只加大物体 1 的 alpha_mo 深层
    """
    # 初始化 per_layer_alpha (per layer, per object)
    if state.per_layer_alpha is None:
        state.per_layer_alpha = [
            (list(state.alpha_id), list(state.alpha_mo))
            for _ in range(num_layers)
        ]

    L = num_layers
    shallow = range(0, L // 3)
    middle  = range(L // 3, 2 * L // 3)
    deep    = range(2 * L // 3, L)

    def bump(li, obj_idx, d_id=0.0, d_mo=0.0):
        ids, mos = state.per_layer_alpha[li]
        # 防越界
        if obj_idx >= len(ids):
            return
        ids[obj_idx] = max(0.0, ids[obj_idx] + d_id)
        mos[obj_idx] = max(0.0, mos[obj_idx] + d_mo)

    for fb in feedbacks:
        s = fb.severity
        oi = fb.object_idx

        if fb.issue == "identity_drift":
            for li in middle: bump(li, oi, d_id=+0.3 * s)
        elif fb.issue == "action_mismatch":
            for li in deep:   bump(li, oi, d_mo=+0.3 * s)
        elif fb.issue == "shape_break":
            for li in shallow: bump(li, oi, d_id=+0.4 * s, d_mo=-0.1 * s)
        elif fb.issue == "phase_lag":
            for li in deep:   bump(li, oi, d_mo=+0.25 * s)
        elif fb.issue == "phase_lead":
            for li in deep:   bump(li, oi, d_mo=-0.2 * s)
        elif fb.issue == "leakage_to_other_object":
            # 这个物体的特征"溢出"到别的物体 → 降它的 alpha_id 全局
            if oi < len(state.alpha_id):
                state.alpha_id[oi] = max(0.0, state.alpha_id[oi] - 0.2 * s)
    return state


def run_mig_pipeline(prompt, llm, t3dis, evaluator, vace, adapter, builder, vae,
                     max_rounds: int = 3):
    """end-to-end 骨架 (per-object 控制版)."""
    structured = llm.parse_and_optimize(prompt)
    # structured = {global_caption,
    #               objects: [{name, attrs, action_phrase, layout_box}, ...]}

    # ---- 首帧 ----
    for _ in range(3):
        first_frame = t3dis.generate(structured)
        ok, _ = evaluator.check_first_frame(first_frame, structured)
        if ok: break
    first_latent = vae.encode(first_frame)
    obj_image_masks = evaluator.extract_first_frame_masks(first_frame, structured)

    n_obj = len(structured["objects"])

    # ---- 运动 mask ----
    obj_volume_masks = build_motion_volume_masks(structured, obj_image_masks)
    grid_sizes, seq_len = compute_grid(obj_volume_masks, vace.patch_size)
    F_p = int(grid_sizes[:, 0].max().item())

    # ---- 条件特征 ----
    id_kv, id_m = builder.build_identity_kv_from_first_frame(first_latent, obj_image_masks)
    verb_emb, verb_m = builder.build_verb_embeddings(
        [[o["action_phrase"] for o in structured["objects"]]]
    )

    state = GenerationState.init_uniform(n_obj=n_obj)

    for _ in range(max_rounds):
        adapter.set_conditioning(
            identity_kv_list=id_kv, identity_kv_masks=id_m,
            verb_emb_list=verb_emb, verb_emb_masks=verb_m,
            obj_volume_masks=obj_volume_masks,
            seq_len=seq_len, grid_sizes=grid_sizes, F_p=F_p,
            alpha_id=state.alpha_id,        # 现在是 per-object list
            alpha_mo=state.alpha_mo,
            per_layer_alpha=state.per_layer_alpha,
            protect_first_frame=True,
        )
        with adapter.attached():
            video = vace_sample(vace, first_latent, structured)

        feedbacks = evaluator.evaluate_video(video, structured)
        if not feedbacks:
            return video
        state = update_alphas_from_feedback(state, feedbacks,
                                            num_layers=len(adapter.injectors))
    return video


# ---- 占位函数 ----
def build_motion_volume_masks(*args, **kwargs):
    raise NotImplementedError("由 layout 插值/SAM-Track/光流先验生成 [B, n_obj, F_p, H_p, W_p]")

def compute_grid(obj_volume_masks, patch_size):
    raise NotImplementedError

def vace_sample(vace, first_latent, structured):
    raise NotImplementedError("你已有的 VACE 采样器")
