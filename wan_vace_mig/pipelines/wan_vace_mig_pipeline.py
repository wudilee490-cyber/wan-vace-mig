"""
WanVaceMIGPipeline — Wan-VACE 推理与 MIG adapter 的粘合层
=========================================================

设计目标:
    复用官方 vace_wan_inference.py 的所有加载逻辑(模型权重、scheduler、VAE、T5)
    然后在调用底层 transformer.forward 之前,把 DecoupledMIGAdapter attach 上去。

不要做的事:
    - 不修改 vace/models/wan/wan_vace_model.py 的任何代码
    - 不修改 wan 包的任何代码
    - 不重新实现采样循环 —— 复用官方 WanVace.generate(...)

如何接入:
    在 vace_wan_inference.py 的等价位置(载入完模型后,sample 之前)做一次:
        adapter = DecoupledMIGAdapter(wan_vace.model, ...)
        adapter.set_conditioning(...)
        with adapter.attached():
            video = wan_vace.generate(...)
    本文件就是把这套流程封装成一个类。

注意:
    官方 WanVace 的 transformer 实例通常是 wan_vace.model 或 wan_vace.transformer,
    不同版本属性名可能不同;下面用 _resolve_transformer 兼容若干常见命名。
"""

from typing import List, Optional, Dict, Any, Union
import torch
import torch.nn as nn

from ..adapter import DecoupledMIGAdapter, ConditioningBuilder


def _resolve_transformer(wan_vace) -> nn.Module:
    """
    从官方 WanVace 容器对象中找到底层 VaceWanModel transformer。
    兼容: wan_vace.model / wan_vace.transformer / wan_vace.dit
    """
    for attr in ("model", "transformer", "dit"):
        m = getattr(wan_vace, attr, None)
        if m is not None and hasattr(m, "vace_blocks"):
            return m
    raise AttributeError(
        "无法从 wan_vace 中定位 VaceWanModel transformer. "
        "请检查官方 WanVace 类的属性名,在 _resolve_transformer 里加上对应分支."
    )


class WanVaceMIGPipeline:
    """
    用法概览
    ----------
    >>> from models.wan import WanVace
    >>> from wan_vace_mig.pipelines import WanVaceMIGPipeline
    >>>
    >>> wan_vace = WanVace(config=..., checkpoint_dir=..., device_id=0)   # 官方加载
    >>> pipe = WanVaceMIGPipeline(wan_vace, num_heads=16, text_dim=4096)
    >>>
    >>> # —— 准备 MIG 条件 ——
    >>> pipe.set_mig_conditioning(
    ...     first_frame_latent=...,         # [B, C, H, W]   3DIS 首帧经 VAE encode
    ...     obj_image_masks=...,            # [B, n_obj, H, W]
    ...     obj_motion_phrases=[["jumping", "running"]],   # [B][n_obj]
    ...     obj_volume_masks=...,           # [B, n_obj, F_p, H_p, W_p]
    ...     grid_sizes=...,                 # [B, 3]
    ...     seq_len=...,
    ...     alpha_id=1.0, alpha_mo=1.0,
    ... )
    >>>
    >>> # —— 推理(MIG 注入只在 with 块内生效) ——
    >>> with pipe.mig_attached():
    ...     video = wan_vace.generate(
    ...         input_prompt=..., src_video=..., src_mask=...,
    ...         src_ref_images=..., size=..., ...,
    ...     )
    """

    def __init__(
        self,
        wan_vace,                            # 官方 WanVace 实例(已加载权重)
        num_heads: int,
        text_dim: int = 4096,
        identity_dim: Optional[int] = None,
        pe_dim: int = 64,
        phase_encoder_layers: int = 2,
        share_phase_encoder_across_layers: bool = True,
        share_injection_block: bool = False,
        layers: Optional[List[int]] = None,
        adapter_state_dict: Optional[Dict[str, Any]] = None,
        adapter_dtype: Optional[torch.dtype] = None,
        adapter_device: Optional[torch.device] = None,
    ):
        self.wan_vace = wan_vace
        self.transformer = _resolve_transformer(wan_vace)

        # T5 文本编码器引用(供 ConditioningBuilder 用)
        # 官方 WanVace 一般有 self.text_encoder;按需调整
        self.text_encoder = getattr(wan_vace, "text_encoder", None)

        # ---- 创建 adapter ----
        self.adapter = DecoupledMIGAdapter(
            vace_model=self.transformer,
            num_heads=num_heads,
            text_dim=text_dim,
            identity_dim=identity_dim,
            pe_dim=pe_dim,
            phase_encoder_layers=phase_encoder_layers,
            share_phase_encoder_across_layers=share_phase_encoder_across_layers,
            share_injection_block=share_injection_block,
            layers=layers,
        )

        # 加载训好的 adapter 权重(若有)
        if adapter_state_dict is not None:
            missing, unexpected = self.adapter.load_state_dict(
                adapter_state_dict, strict=False
            )
            if missing or unexpected:
                print(f"[WanVaceMIGPipeline] adapter load_state_dict: "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")

        # 移到与 transformer 一致的 device/dtype
        if adapter_device is None:
            adapter_device = self.transformer.patch_embedding.weight.device
        if adapter_dtype is None:
            adapter_dtype = self.transformer.patch_embedding.weight.dtype
        self.adapter.to(device=adapter_device, dtype=adapter_dtype)

        # ---- 条件构造器(仅在用到 build_* 方法时需要 text_encoder) ----
        self.conditioning_builder = ConditioningBuilder(
            vace_model=self.transformer,
            text_encoder=self.text_encoder,
        )

    # ------------------------------------------------------------------
    # 一站式条件设置:从首帧/mask/动作短语 → adapter cond
    # ------------------------------------------------------------------
    def set_mig_conditioning(
        self,
        first_frame_latent: torch.Tensor,           # [B, C_lat, H_lat, W_lat]
        obj_image_masks: torch.Tensor,              # [B, n_obj, H_lat, W_lat]
        obj_motion_phrases: List[List[str]],        # [B][n_obj]
        obj_volume_masks: torch.Tensor,             # [B, n_obj, F_p, H_p, W_p]
        grid_sizes: torch.Tensor,                   # [B, 3]
        seq_len: int,
        F_p: Optional[int] = None,
        alpha_id: Union[float, List[float], torch.Tensor] = 1.0,
        alpha_mo: Union[float, List[float], torch.Tensor] = 1.0,
        protect_first_frame: bool = True,
        per_layer_alpha: Optional[List[tuple]] = None,
        max_id_tokens_per_obj: int = 256,
        max_verb_tokens: int = 32,
    ):
        """
        alpha_id / alpha_mo:
            float           — 全局,所有物体相同
            List[float]     — per object,长度 = n_obj
            Tensor[K]       — per object 张量
        Examples:
            # 全局: 1.0
            set_mig_conditioning(..., alpha_id=1.0, alpha_mo=1.0)
            # 物体 0 强身份,物体 1 强动作
            set_mig_conditioning(..., alpha_id=[1.5, 0.8], alpha_mo=[0.8, 1.5])
        """
        # 1) 身份 KV(全局,无相位)
        id_kv, id_m = self.conditioning_builder.build_identity_kv_from_first_frame(
            first_frame_latent, obj_image_masks, max_tokens_per_obj=max_id_tokens_per_obj,
        )
        # 2) 动作 verb 嵌入(相位展开由 adapter 内部完成)
        verb_emb, verb_m = self.conditioning_builder.build_verb_embeddings(
            obj_motion_phrases, max_len=max_verb_tokens,
        )
        # 3) 写入 adapter
        self.adapter.set_conditioning(
            identity_kv_list=id_kv, identity_kv_masks=id_m,
            verb_emb_list=verb_emb, verb_emb_masks=verb_m,
            obj_volume_masks=obj_volume_masks,
            seq_len=seq_len, grid_sizes=grid_sizes, F_p=F_p,
            alpha_id=alpha_id, alpha_mo=alpha_mo,
            protect_first_frame=protect_first_frame,
            per_layer_alpha=per_layer_alpha,
        )

    def update_alphas(
        self,
        alpha_id: Optional[Union[float, List[float], torch.Tensor]] = None,
        alpha_mo: Optional[Union[float, List[float], torch.Tensor]] = None,
        per_layer_alpha: Optional[List[tuple]] = None,
    ):
        """评估 agent 反馈后,无需重建条件,只调强度即可下一轮生成。
        
        alpha_id / alpha_mo: 接受 float 或 per-object list/tensor
        """
        cond = self.adapter._cond
        assert cond is not None, "请先调用 set_mig_conditioning"
        if alpha_id is not None: cond["alpha_id"] = alpha_id
        if alpha_mo is not None: cond["alpha_mo"] = alpha_mo
        if per_layer_alpha is not None: cond["per_layer_alpha"] = per_layer_alpha

    # ------------------------------------------------------------------
    # 上下文管理:进入则 attach,退出则 detach
    # ------------------------------------------------------------------
    def mig_attached(self):
        return self.adapter.attached()

    # ------------------------------------------------------------------
    # 训练辅助
    # ------------------------------------------------------------------
    def freeze_base(self):
        self.adapter.freeze_base()

    def trainable_parameters(self):
        return self.adapter.trainable_parameters()

    def save_adapter(self, path: str):
        torch.save(self.adapter.state_dict(), path)

    def load_adapter(self, path: str, strict: bool = False):
        sd = torch.load(path, map_location="cpu")
        return self.adapter.load_state_dict(sd, strict=strict)
