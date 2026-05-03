# wan-vace-mig

**Multi-Instance Generation (MIG)** ControlNet-style adapter for [Wan-VACE](https://github.com/ali-vilab/VACE).

把 layout/instance-aware 控制能力外挂到 Wan-VACE 主干,**不改动主干权重**:
- 首帧物体身份特征 (从 first-frame VAE patch tokens) 通过解耦 cross-attention 注入后续帧
- 动作短语 (e.g. `"jumping over a fence"`) 经过 phase-aware 编码,逐帧注入对应物体
- 每个物体可独立调节 `alpha_id` (身份强度) 和 `alpha_mo` (动作强度)
- 训练只更新 adapter 参数 (~70M),VACE 主干完全冻结

## 依赖关系 (重要)

本仓库**核心代码** (`wan_vace_mig/` 包) 是纯 PyTorch,不直接 import wan/VACE。
但**训练/推理脚本** (`scripts/`) 需要两个上游项目:

```
wan-vace-mig/                       ← 本仓库 (你 git clone 这个)
  │
  ├── wan_vace_mig/                 纯 PyTorch 包,不依赖 wan/VACE 源码
  │   └── (adapter, mask_predictor, train, agent, ...)
  │
  └── scripts/                      训练/推理 CLI 脚本
       ├── 用 import wan            ← Wan2.1 主包 (pip install)
       └── 用 from models.wan       ← VACE 仓库 (git clone)
            import WanVace

外部依赖:
  ┌─ Wan-Video/Wan2.1 (pip 包)         由 install.sh 自动 pip install
  │  提供 wan.modules.* / wan.utils.*    安装位置: site-packages/wan/
  │
  └─ ali-vilab/VACE (git 仓库)          由 install.sh 自动 git clone
     提供 models.wan.WanVace 容器类      位置: third_party/VACE/
                                         路径注入: scripts 顶部 sys.path.insert
```

**install.sh 会自动**:
1. `pip install wan@git+https://github.com/Wan-Video/Wan2.1`
2. `git clone https://github.com/ali-vilab/VACE.git third_party/VACE`
3. 把 `third_party/VACE` 写入 `wan_vace_mig.pth` 让 conda env 自动注入 sys.path

## 关键性质

- ✅ **零初始化等价性**: 未训练的 adapter 加载到 VACE 上,生成结果与原版 VACE 位级一致
- ✅ **per-object alpha 控制**: 每个物体独立调身份/动作强度,支持 layer-wise schedule
- ✅ **相位感知动作编码**: `K_act^f = MLP([verb_emb; PE(f/T)])` 让同一动作在不同帧有不同表达
- ✅ **首帧保护**: 首帧 (来自 3DIS 等 layout 生成器) 不被 adapter 修改

## 仓库结构

```
wan-vace-mig/
├── wan_vace_mig/           核心 Python 包
│   ├── adapter/            DecoupledMIGAdapter, PhaseAwareMotionEncoder
│   ├── pipelines/          WanVaceMIGPipeline 推理封装
│   ├── train/              dataset, losses, training utils
│   ├── agent/              feedback-loop agent (per-object alpha 自动调节)
│   ├── mask_predictor/     Motion Mask Predictor (从首帧mask+phrase预测后续帧mask)
│   └── configs/            训练/推理配置
├── scripts/
│   ├── mig_train_adapter.py        DDP 训练 (主adapter)
│   ├── mig_inference.py            推理 CLI
│   ├── train_mask_predictor.py     mask predictor 训练
│   └── infer_mask_predictor.py     mask predictor 推理
├── setup/
│   ├── install.sh                  一键环境安装 (CUDA 自动检测)
│   ├── verify_env.py               环境验证 (诊断 GPU 兼容/import 链路)
│   └── requirements_*.txt
├── test_adapter.py                 adapter smoke test
├── test_mask_predictor.py          mask predictor smoke test
└── pyproject.toml
```

## 快速开始

### 1. 环境安装
```bash
git clone https://github.com/your-org/wan-vace-mig.git
cd wan-vace-mig
bash setup/install.sh                                   # 自动检测 GPU + CUDA
# bash setup/install.sh --cuda 13.0                     # RTX 5090 / Blackwell
# bash setup/install.sh --cuda 12.4 --legacy            # 老 GPU
```

### 2. 验证安装
```bash
conda activate vace
python setup/verify_env.py     # 期望全 ✓
python test_adapter.py         # 关键: 零初始化等价性 + 训练可行性
```

### 3. 准备训练数据

使用配套的 [sav-mig-data](https://github.com/your-org/sav-mig-data) 项目从 SA-V 生成训练缓存。

### 4. 训练
```bash
# 单卡 smoke test (5 步)
python scripts/mig_train_adapter.py \
    --cache_dir /path/to/sav_mig_cache \
    --wan_ckpt models/Wan2.1-VACE-1.3B \
    --max_steps 5

# 全量 DDP
torchrun --nproc_per_node=2 scripts/mig_train_adapter.py \
    --cache_dir /path/to/sav_mig_cache \
    --wan_ckpt models/Wan2.1-VACE-1.3B \
    --max_steps 50000
```

### 5. 推理
```bash
python scripts/mig_inference.py \
    --wan_ckpt models/Wan2.1-VACE-1.3B \
    --adapter_ckpt checkpoints/mig_full/last.pt \
    --first_frame /path/to/first.png \
    --layout config.yaml \
    --output out.mp4
```

## per-object alpha 控制

```python
# 全局 (老用法,完全兼容)
pipe.set_mig_conditioning(..., alpha_id=1.0, alpha_mo=1.0)

# per object (新用法)
# 物体 0: 强身份保持 (像 reference image)
# 物体 1: 强动作遵循 (动作精度优先)
pipe.set_mig_conditioning(..., alpha_id=[1.5, 0.8], alpha_mo=[0.8, 1.5])

# per layer × per object schedule
schedule = []
for li in range(num_layers):
    if li < 10:        schedule.append(([1.5, 1.0], [0.5, 1.0]))   # 浅层
    elif li < 20:      schedule.append(([1.2, 0.8], [0.8, 1.2]))   # 中层
    else:              schedule.append(([0.8, 0.5], [1.0, 1.8]))   # 深层
pipe.set_mig_conditioning(..., per_layer_alpha=schedule)
```

## 显存吃紧?共享 injection block

```python
# 默认: 每层独立 injector (~70M params)
pipe = WanVaceMIGPipeline(..., share_injection_block=False)

# 显存紧时: 所有层共享同 1 个 injector (~3M params)
pipe = WanVaceMIGPipeline(..., share_injection_block=True)
# 牺牲分层语义,但能省 95% adapter 显存
```

## Motion Mask Predictor (子模块)

输入: **首帧 mask** + **per-object 动作短语** → 输出: **后续所有帧 mask**

这是 MIG 推理 pipeline 的辅助组件,让用户只给首帧标注就能跑全流程,
不必逐帧标注 mask。模型架构是 Trajectory Transformer:

- **首帧 mask + 动作 phrase** → 物体级 token
- **F 帧 query tokens** + cross-attn 到物体 token  
- 输出每帧每物体的 7-D 几何参数 (Δcx, Δcy, log_w/h, log_scale, theta, deform_alpha)
- 用 `grid_sample` warp 首帧 mask 得到后续帧 mask

### 训练

```bash
# smoke test (5 step, 验证打通)
python scripts/train_mask_predictor.py \
    --cache_dir /path/to/sav_mig_cache \
    --max_steps 5 --batch_size 2

# 全量
python scripts/train_mask_predictor.py \
    --cache_dir /path/to/sav_mig_cache \
    --max_steps 30000 --batch_size 8 \
    --save_dir checkpoints/mask_predictor

# DDP
torchrun --nproc_per_node=2 scripts/train_mask_predictor.py ...
```

### 推理

```bash
python scripts/infer_mask_predictor.py \
    --ckpt checkpoints/mask_predictor/last.pt \
    --first_mask first_mask.png \
    --phrases "running" "jumping" \
    --F 21 \
    --output pred_masks.npy
```

### 关键性质

- **零初始化等价 identity**: 未训练时 head 输出全 0, warp 退化为 identity, 所有帧 = 首帧 (合理 baseline)
- **轻量**: dim=384, depth=6 → ~25M 参数; 单卡 4090 也能跑训练
- **数据复用**: 直接用 sav-mig-data 缓存的 obj_volume_masks + verb_embeddings,无需重做数据
- **可解释**: 输出是 7-D 几何参数,可视化轨迹方便 debug

## 引用与致谢

- [Wan-VACE](https://github.com/ali-vilab/VACE) — 主干视频生成模型
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — 基础 DiT 架构
- [SA-V](https://ai.meta.com/datasets/segment-anything-video) — 训练数据集

## License

Apache 2.0 — 与上游 Wan-VACE 一致。
