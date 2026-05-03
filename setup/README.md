# Environment Setup for VACE-MIG Pipeline

完整环境配置 —— 数据处理 + 训练 + 推理 + Agent 调试,**单一 conda 环境覆盖所有阶段**。

## 硬件要求

| 阶段 | 最低配置 | 推荐配置 |
|---|---|---|
| **Stage 1** (extract clips) | 16 GB RAM, 8 cores | 64 GB RAM, 32 cores |
| **Stage 2** (VLM caption) | 1× GPU 16GB (Qwen3-VL-2B) | 1× A100-80GB |
| **Stage 3** (VAE encode) | 1× GPU 24GB | 1× A100-80GB |
| **Stage 4** (T5 encode) | 1× GPU 24GB | — |
| **Stage 5** (build index) | CPU only | — |
| **Training** (1.3B + adapter) | 1× GPU 40GB | 8× A100-80GB |
| **Training** (14B + adapter) | 8× GPU 40GB | 8× A100-80GB DDP |
| **Inference** (1.3B) | 1× GPU 24GB | — |
| **Inference** (14B) | 1× GPU 80GB | — |

**磁盘**:SA-V 原始数据 ~600 GB,缓存(Stage 1-4 全产出)~7 TB(可瘦身到 500 GB)

## 软件要求

| 组件 | 推荐版本 | 备注 |
|---|---|---|
| OS | Ubuntu 20.04 / 22.04 | 其他 Linux 发行版应该也行 |
| CUDA Toolkit | **12.4** | 与 PyTorch 2.5.1 + flash-attn 2.7+ 最稳的组合 |
| Python | **3.10** | VACE 官方 `requires-python = >=3.10,<4.0` |
| PyTorch | **2.5.1** | VACE 官方 `torch>=2.5.1`,这个版本最稳 |
| Driver | ≥ 535 | CUDA 12.4 需要 |

## 一键安装

```bash
# 假设你在 VACE 仓库根目录
cd VACE
bash setup/install.sh
```

如果中途挂了,**安装脚本是幂等的**,直接重跑会跳过已完成步骤。

## 分步安装(推荐生产环境)

按下面的顺序,每一步独立检查通过再继续。

### Step 0: 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
    git git-lfs build-essential \
    ffmpeg libsm6 libxext6 \
    libgl1-mesa-glx \
    wget curl
git lfs install
```

### Step 1: 创建 conda env

```bash
# 假设已装 miniconda 或 anaconda
conda create -n vace_mig python=3.10 -y
conda activate vace_mig

# 升级基础工具
pip install --upgrade pip setuptools wheel
pip install ninja packaging        # flash-attn 编译需要
```

### Step 2: PyTorch (重要 — CUDA 版本对齐)

```bash
# CUDA 12.4 (推荐)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# 期望输出: 2.5.1+cu124 True 12.4
```

如果你机器上的 CUDA Driver 是 12.1/12.2,改成 cu121 索引也行。**不要用 cu118,
flash-attn 在 CUDA 11.8 + PyTorch 2.5+ 上编译会出问题。**

### Step 3: VACE 主依赖

```bash
# 一次性装齐 VACE 官方需要的包(除 flash-attn)
pip install -r setup/requirements_vace_main.txt

# Wan2.1 主包 (从 GitHub 装)
pip install wan@git+https://github.com/Wan-Video/Wan2.1
```

### Step 4: Flash Attention(慢,5-30 min)

```bash
# 优先尝试 wheel(秒装)
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 如果 wheel 找不到匹配,从源码编译(5-30 min)
# pip install flash-attn --no-build-isolation
```

如果编译爆 OOM:
```bash
MAX_JOBS=2 pip install flash-attn --no-build-isolation
```

### Step 5: MIG pipeline 额外依赖

```bash
pip install -r setup/requirements_mig.txt
```

### Step 6: 下载模型权重

```bash
mkdir -p models

# Wan2.1-VACE-1.3B (~6 GB)
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B \
    --local-dir models/Wan2.1-VACE-1.3B

# Wan2.1-VACE-14B (~28 GB,可选,如果你打算用 14B 模型训)
# huggingface-cli download Wan-AI/Wan2.1-VACE-14B \
#     --local-dir models/Wan2.1-VACE-14B

# Qwen3-VL-2B-Instruct VLM (~5 GB,Stage 2 caption 用)
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
    --local-dir models/Qwen3-VL-2B-Instruct
```

### Step 7: 验证完整环境

```bash
python setup/verify_env.py
```

期望输出:
```
✓ Python 3.10.x
✓ PyTorch 2.5.1 with CUDA 12.4
✓ flash-attn 2.7.x available
✓ Wan-VACE imports OK
✓ pycocotools, decord, einops, transformers OK
✓ MIG adapter imports OK
✓ VLM caller (qwen3_vl backend) OK
✓ Test adapter zero-init equivalence: PASS
```

## 不同后端的可选依赖

### 用本地 Qwen3-VL 做 Stage 2 caption(默认)
已经在 `requirements_mig.txt` 里 (transformers >= 4.49)。

### 用 vLLM 远程服务做 Stage 2 caption(高吞吐)
```bash
# 注意: vLLM 装在另一个环境里,跟训练环境隔离
conda create -n vllm_serving python=3.10 -y
conda activate vllm_serving
pip install vllm>=0.6.0
# 起服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct --port 8000
```

### DDP 多卡训练
PyTorch 内置支持,无需额外安装,只需:
```bash
torchrun --nproc_per_node=8 scripts/mig_train_adapter.py ...
```

## 数据集下载

### SA-V (主要训练数据)
```bash
# Meta 提供的官方下载链接,需要先在 ai.meta.com/datasets/segment-anything-video 申请
# 申请通过后会得到一个签名 URL,然后:
mkdir -p /data/SA-V
cd /data/SA-V
# 用官方 download script 或 wget 拉视频和 json
# 文档: https://github.com/facebookresearch/sam2/tree/main/sav_dataset
```

完整 SA-V manual + auto 大约 600 GB。

### YouTube-VIS 2021(可选,小数据集试水)
```bash
# 从 https://youtube-vos.org/dataset/vis/ 申请下载
# 大约 25 GB
```

## 常见错误处理

### "ImportError: libGL.so.1: cannot open shared object file"
```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### "RuntimeError: CUDA error: no kernel image is available for execution"
你的 GPU 显卡驱动太旧,升级到 535+ 或者降 CUDA 到 11.8。

### "fatal error: cuda_runtime.h: No such file or directory" (装 flash-attn 时)
没装 CUDA Toolkit(只有 driver),装一下:
```bash
# 检查
nvcc --version  # 如果报 command not found 就是没装

# Ubuntu 装 CUDA 12.4 toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run
# 设环境变量
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Python 3.11+ 上 pycocotools 装不上
```bash
pip install pycocotools-fix     # 替代品,接口完全兼容
# 或者降到 Python 3.10
```

### "decord 装不上" 或者 "GLIBCXX_3.4.29 not found"
```bash
pip uninstall -y decord
pip install eva-decord            # 一个维护得更好的 fork
```

`eva-decord` 接口和官方 `decord` 一模一样,代码不用改,直接 `import decord` 即可。

### 多卡 NCCL 连接失败
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1          # 单机训练把 InfiniBand 关掉
export NCCL_P2P_DISABLE=1         # 如果机器没有 NVLink
```

## 磁盘空间规划

```
$HOME/                        # ~5 GB conda + pip
├── miniconda3/               # 用户家目录
└── ...

/data/                        # 大数据放这,需要至少 10TB
├── SA-V/                     # ~600 GB 原始
│   ├── sav_train/
│   └── sav_val/
├── sav_mig_cache/            # ~7 TB 全产出 / ~500 GB 瘦身后
└── checkpoints/
    └── mig_adapter/          # ~1 GB 每个 checkpoint

/path/to/VACE/                # 代码 ~100 MB
├── models/                   # 模型权重
│   ├── Wan2.1-VACE-1.3B/    # ~6 GB
│   ├── Wan2.1-VACE-14B/     # ~28 GB (可选)
│   └── Qwen3-VL-2B-Instruct/ # ~5 GB
├── vace/
└── scripts/
```

## 一些性能调优建议

**Stage 1 (CPU)**: 
设 `--num_workers` = `$(nproc) - 4`,留一些核给 IO/系统。

**Stage 2 (VLM)**:  
- 本地 Qwen3-VL-2B 在 A100-40G 上约 0.5-1 clip/s
- 远程 vLLM 同模型并发 16 时约 4 clip/s,**13 倍提速**
- 推荐用远程 vLLM,只要你有专门的服务器跑 inference

**Stage 3 (VAE)**:  
- 1× A100-80G 约 0.8 clip/s (81 帧 480p)
- 多卡需要手动分片(把 clips 分成 N 份分别跑)

**训练**:  
- 1.3B + adapter 在 8× A100-80G 上 batch_size=1, grad_accum=4, 50K steps ≈ 4-5 天
- 推荐先用 5K 步冒烟,看 loss 曲线,确认方向对再上完整训练

## 离线无网环境

如果你的训练机器没有外网(常见于 GPU 集群),按下面策略:

1. 在有网的机器上完成 Step 1-5 后,把整个 conda env 打包:
```bash
conda pack -n vace_mig -o vace_mig_env.tar.gz
```

2. 上传到训练机器,解压并激活:
```bash
mkdir -p ~/envs/vace_mig
tar -xzf vace_mig_env.tar.gz -C ~/envs/vace_mig
source ~/envs/vace_mig/bin/activate
conda-unpack
```

3. 模型权重也要提前下载到训练机器。

## 验证清单

最终安装完了,跑一遍这个列表确认全过:

```bash
conda activate vace_mig
python setup/verify_env.py
```

如果全部 ✓,就可以开始处理数据了:
```bash
bash scripts/sav_pipeline/run_all.sh \
    --sav_root /data/SA-V/sav_train \
    --out_root /data/sav_mig_cache \
    --wan_ckpt models/Wan2.1-VACE-1.3B \
    --max_videos 10                    # 先用 10 个视频走通流程
```
