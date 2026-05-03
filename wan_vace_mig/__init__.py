"""
MIG (Multi-Instance Generation) extension for Wan-VACE
======================================================
ControlNet-style 外挂模块,在不修改 VACE 主干的前提下,通过解耦交叉注意力
把首帧物体身份特征 + 相位感知动作特征注入后续帧对应物体。

公开接口:
    DecoupledMIGAdapter  — 主 adapter 类
    ConditioningBuilder  — 把首帧 + 动作短语转成 KV 张量
    WanVaceMIGPipeline   — 一键推理封装(VACE + adapter)
    MotionMaskPredictor  — 从首帧 mask + 动作短语预测后续帧 mask 的子模块
"""

from .adapter.decoupled_mig_adapter import (
    DecoupledMIGAdapter,
    PhaseAwareMotionEncoder,
    DecoupledInjectionBlock,
    MaskedCrossAttention,
)
from .adapter.conditioning_builder import ConditioningBuilder
from .pipelines.wan_vace_mig_pipeline import WanVaceMIGPipeline

# Mask predictor (lazy-import 友好, 不影响主路径)
from .mask_predictor import MotionMaskPredictor

__all__ = [
    "DecoupledMIGAdapter",
    "PhaseAwareMotionEncoder",
    "DecoupledInjectionBlock",
    "MaskedCrossAttention",
    "ConditioningBuilder",
    "WanVaceMIGPipeline",
    "MotionMaskPredictor",
]
