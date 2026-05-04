#!/usr/bin/env bash
# =====================================================================
# VACE-MIG 一键环境安装 (RTX 5090 / Blackwell sm_120 适配版)
# =====================================================================
# 默认配置: CUDA 13.0 + PyTorch 2.8 + flash-attn 4
# 适用于: RTX 5090 / Blackwell 架构 GPU
#
# 用法:
#   cd VACE/                                    # 在 VACE 仓库根目录运行
#   bash setup/install.sh                       # 默认 (RTX 5090 推荐)
#   bash setup/install.sh --cuda 12.8           # RTX 5090 with CUDA 12.8
#   bash setup/install.sh --cuda 12.4 --legacy  # 老硬件 (sm_70~sm_90)
#   bash setup/install.sh --skip_models         # 不下载模型权重
#
# 设计:
#   - 幂等: 失败可重跑,自动跳过已完成步骤
#   - GPU 自动检测: RTX 50-series 自动选 cuda 13,老硬件自动选 12.4
#   - 严格 GPU 兼容性: 装完 PyTorch 立即验证 sm_xx 在编译列表里
# =====================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# 默认参数
ENV_NAME="vace"                 # 与你实际使用的 env 一致
PY_VERSION="3.10"
CUDA_VERSION=""                  # 留空 = 根据 GPU 自动选
LEGACY=0
SKIP_MODELS=0
SKIP_FLASH_ATTN=0
SKIP_VLM=1                       # 模型项目默认不装 VLM (那是数据项目的事)

while [[ $# -gt 0 ]]; do
    case $1 in
        --env_name) ENV_NAME="$2"; shift 2 ;;
        --py_version) PY_VERSION="$2"; shift 2 ;;
        --cuda) CUDA_VERSION="$2"; shift 2 ;;
        --legacy) LEGACY=1; shift ;;
        --skip_models) SKIP_MODELS=1; shift ;;
        --skip_flash_attn) SKIP_FLASH_ATTN=1; shift ;;
        --skip_vlm) SKIP_VLM=1; shift ;;
        --with_vlm) SKIP_VLM=0; shift ;;
        -h|--help)
            cat <<USAGE
Usage: bash setup/install.sh [OPTIONS]

Options:
  --env_name NAME       conda env name (default: vace)
  --cuda VER            CUDA version (12.4|12.8|13.0|auto). Default: auto-detect
  --legacy              Force CUDA 12.4 + PyTorch 2.5.1 (for sm_70~sm_90 GPUs)
  --skip_models         Don't download model weights
  --skip_flash_attn     Don't install flash-attn
  --with_vlm            Also download Qwen3-VL-2B-Instruct (default: NO,
                        it's the data project's responsibility)
USAGE
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# 颜色输出
GREEN="\033[0;32m"; YELLOW="\033[0;33m"; RED="\033[0;31m"; CYAN="\033[0;36m"; NC="\033[0m"
say()  { echo -e "${GREEN}[install]${NC} $*"; }
warn() { echo -e "${YELLOW}[install]${NC} $*"; }
err()  { echo -e "${RED}[install]${NC} $*"; }
info() { echo -e "${CYAN}[install]${NC} $*"; }

say "============================================================"
say " VACE-MIG environment setup"
say "============================================================"

# -----------------------------------------------------------------
# Step 0: 系统检查 + GPU 自动检测
# -----------------------------------------------------------------
say "Step 0: system + GPU detection"

if ! command -v conda &> /dev/null; then
    err "conda not found. Install miniconda first."
    exit 1
fi
say "  ✓ conda: $(conda --version)"

if ! command -v nvidia-smi &> /dev/null; then
    err "nvidia-smi not found. NVIDIA driver required."
    exit 1
fi

DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' || echo "")
say "  ✓ driver: $DRIVER_VER"
say "  ✓ GPU: $GPU_NAME (sm_${GPU_CAP})"

# 自动选 CUDA 版本
if [[ -z "$CUDA_VERSION" ]]; then
    if [[ $LEGACY -eq 1 ]]; then
        CUDA_VERSION="12.4"
        info "  --legacy mode: using CUDA 12.4"
    elif [[ "$GPU_CAP" == "120" ]] || [[ "$GPU_CAP" == "100" ]]; then
        # RTX 50-series (sm_120) / H200 (sm_100) — 推荐 CUDA 13
        CUDA_VERSION="13.0"
        info "  GPU sm_${GPU_CAP} (Blackwell) → auto-selected CUDA 13.0"
    elif [[ "$GPU_CAP" == "90" ]]; then
        # H100 — CUDA 12.4 ok,但 12.8 也支持
        CUDA_VERSION="12.4"
        info "  GPU sm_90 (Hopper) → using CUDA 12.4"
    else
        # 老 GPU (sm_70/75/80/86/89)
        CUDA_VERSION="12.4"
        info "  GPU sm_${GPU_CAP} → using CUDA 12.4"
    fi
fi

# 验证选择的 CUDA 与 GPU 兼容
case $CUDA_VERSION in
    11.8|12.1|12.4|12.6|12.8|13.0) ;;
    13|130) CUDA_VERSION="13.0" ;;
    *) err "Unsupported CUDA: $CUDA_VERSION (supported: 11.8/12.1/12.4/12.6/12.8/13.0)"; exit 1 ;;
esac

# RTX 50-series + 老 CUDA → 升级
if [[ "$GPU_CAP" == "120" ]] && \
   [[ "$CUDA_VERSION" != "13.0" ]] && [[ "$CUDA_VERSION" != "12.8" ]]; then
    warn "  RTX 50-series (sm_120) detected but CUDA $CUDA_VERSION is too old"
    warn "  Auto-upgrading to CUDA 13.0"
    CUDA_VERSION="13.0"
fi

CU_TAG="cu${CUDA_VERSION//./}"
TORCH_INDEX="https://download.pytorch.org/whl/$CU_TAG"
say "  → using CUDA $CUDA_VERSION ($CU_TAG), index: $TORCH_INDEX"

# 选择 torch 版本配套
case $CUDA_VERSION in
    13.0)        TORCH_TARGET="auto"; TORCH_DESC="latest (>=2.8) for cu130" ;;
    12.8)        TORCH_TARGET="auto"; TORCH_DESC="latest (>=2.7) for cu128" ;;
    12.6)        TORCH_TARGET="2.6.0"; TORCH_DESC="2.6.0 for cu126" ;;
    12.4|12.1)   TORCH_TARGET="2.5.1"; TORCH_DESC="2.5.1 for $CU_TAG (legacy)" ;;
    11.8)        TORCH_TARGET="2.5.1"; TORCH_DESC="2.5.1 for cu118 (legacy)" ;;
esac

# nvcc 检查 (flash-attn 编译需要)
if ! command -v nvcc &> /dev/null; then
    warn "  nvcc not found. flash-attn build will fail unless you install CUDA Toolkit."
    warn "  If you have prebuilt flash-attn wheel, install will continue."
fi

# -----------------------------------------------------------------
# Step 1: conda 环境
# -----------------------------------------------------------------
say "Step 1: conda env '$ENV_NAME'"
if conda env list | grep -q "^${ENV_NAME} "; then
    say "  env exists, will reuse"
else
    say "  creating env with Python $PY_VERSION..."
    conda create -n $ENV_NAME python=$PY_VERSION -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
say "  ✓ activated $ENV_NAME ($(python --version))"

pip install --upgrade pip setuptools wheel ninja packaging -q

# -----------------------------------------------------------------
# Step 2: PyTorch (严格 GPU 兼容性验证)
# -----------------------------------------------------------------
say "Step 2: PyTorch ($TORCH_DESC)"

TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")

# 检查现有 torch 是否兼容当前 GPU
NEED_REINSTALL=0
if [[ -n "$TORCH_INSTALLED" ]]; then
    # 通过 arch list 判断
    if python -c "
import torch, sys
arch_list = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
gpu_cap = torch.cuda.get_device_capability(0) if torch.cuda.device_count() > 0 else None
sm = f'sm_{gpu_cap[0]}{gpu_cap[1]}' if gpu_cap else None
sys.exit(0 if (sm in arch_list or not arch_list) else 1)
" 2>/dev/null; then
        say "  PyTorch $TORCH_INSTALLED is compatible with GPU, keeping it"
    else
        warn "  PyTorch $TORCH_INSTALLED does NOT support sm_${GPU_CAP}, reinstalling..."
        NEED_REINSTALL=1
    fi
else
    NEED_REINSTALL=1
fi

if [[ $NEED_REINSTALL -eq 1 ]]; then
    [[ -n "$TORCH_INSTALLED" ]] && pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

    if [[ "$TORCH_TARGET" == "auto" ]]; then
        # CUDA 13 / 12.8 — 装 index 里最新版
        say "  installing latest torch from $TORCH_INDEX..."
        pip install torch torchvision --index-url $TORCH_INDEX
    else
        say "  installing torch==$TORCH_TARGET..."
        pip install "torch==$TORCH_TARGET" --index-url $TORCH_INDEX
        pip install torchvision --index-url $TORCH_INDEX
    fi
fi

# 严格验证: 实际 GPU 在 PyTorch 编译列表里
python -c "
import torch
import sys
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  ✓ torch {torch.__version__}, cuda {torch.version.cuda}, devices: {torch.cuda.device_count()}')

archs = torch.cuda.get_arch_list()
print(f'  PyTorch compiled archs: {archs}')

failed = []
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    name = torch.cuda.get_device_name(i)
    sm = f'sm_{cap[0]}{cap[1]}'
    if archs and sm not in archs:
        failed.append(f'GPU{i} {name} ({sm})')
    else:
        print(f'  ✓ GPU{i} {name} ({sm}) supported')

if failed:
    print(f'  ✗ Incompatible GPUs: {failed}')
    print(f'  This means PyTorch was compiled without support for your hardware.')
    print(f'  For RTX 50-series, try: --cuda 13.0  (or --cuda 12.8)')
    sys.exit(1)
"

# -----------------------------------------------------------------
# Step 3: VACE 主依赖 (matplotlib 在 requirements 里)
# -----------------------------------------------------------------
say "Step 3: VACE main dependencies"
pip install -r setup/requirements_vace_main.txt

# Wan2.1 主包 (Wan-Video/Wan2.1, 提供 wan.modules / wan.utils 等)
if python -c "import wan" 2>/dev/null; then
    WAN_PATH=$(python -c "import wan; print(wan.__file__)")
    say "  ✓ wan already installed at $WAN_PATH"
else
    say "  installing wan@git+https://github.com/Wan-Video/Wan2.1 ..."
    pip install --no-build-isolation "wan@git+https://github.com/Wan-Video/Wan2.1"
fi

# VACE 仓库 (ali-vilab/VACE, 提供 models.wan.WanVace 容器类)
# 注意: VACE 仓库不支持 pip install, 我们 git clone 到 third_party/ 然后 sys.path 注入.
# 训练/推理脚本顶部会自动把 third_party/VACE 加到 sys.path.
VACE_LOCAL="$REPO_ROOT/third_party/VACE"
if [[ -d "$VACE_LOCAL" ]] && [[ -d "$VACE_LOCAL/models/wan" ]]; then
    say "  ✓ VACE repo already at $VACE_LOCAL"
else
    say "  cloning VACE repo to third_party/VACE..."
    mkdir -p "$REPO_ROOT/third_party"
    if [[ ! -d "$VACE_LOCAL" ]]; then
        git clone --depth 1 https://github.com/ali-vilab/VACE.git "$VACE_LOCAL"
    fi
    # 验证结构
    if [[ ! -d "$VACE_LOCAL/models/wan" ]]; then
        warn "  VACE clone seems incomplete. Check $VACE_LOCAL/models/wan exists."
    else
        say "  ✓ VACE cloned to $VACE_LOCAL"
    fi
fi
# 写一个 .pth 文件让 conda env 自动加 sys.path (备选方案, 不强求)
SITE_DIR="$(python -c 'import site; print(site.getsitepackages()[0])')"
if [[ -d "$SITE_DIR" ]]; then
    echo "$VACE_LOCAL" > "$SITE_DIR/wan_vace_mig.pth"
    say "  ✓ added $VACE_LOCAL to python sys.path via $SITE_DIR/wan_vace_mig.pth"
fi

# -----------------------------------------------------------------
# Step 4: Flash Attention
# -----------------------------------------------------------------
if [[ $SKIP_FLASH_ATTN -eq 1 ]]; then
    warn "Step 4: Flash Attention (SKIPPED)"
else
    say "Step 4: Flash Attention"
    if python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null; then
        FA_VER=$(python -c "import flash_attn; print(flash_attn.__version__)")
        say "  ✓ flash-attn $FA_VER already installed"
    else
        say "  installing flash-attn (may take 5-30 min if compiling from source)..."

        # CUDA 13 / RTX 5090: 必须装 flash-attn 4.x (新 API,支持 Blackwell)
        # CUDA 12.x / 老硬件: flash-attn 2.7.x 即可
        if [[ "$CUDA_VERSION" == "13.0" ]] || [[ "$GPU_CAP" == "120" ]]; then
            info "  Blackwell / CUDA 13 detected, installing flash-attn 4.x..."
            # 4.x 通常需要从源码或预构建wheel装
            if pip install flash-attn --no-build-isolation 2>&1 | tee /tmp/fa_install.log; then
                say "  ✓ flash-attn installed"
            else
                warn "  flash-attn 4 install failed; will fall back to PyTorch SDPA"
                warn "  log: /tmp/fa_install.log"
                warn "  This is non-blocking — training/inference still works (slower)"
            fi
        else
            if pip install flash-attn==2.7.4.post1 --no-build-isolation 2>/dev/null; then
                say "  ✓ flash-attn 2.7.4.post1 installed (prebuilt wheel)"
            else
                warn "  prebuilt wheel not found, compiling from source..."
                pip install flash-attn --no-build-isolation || \
                    warn "  flash-attn install failed, fallback to SDPA"
            fi
        fi
    fi
fi

# -----------------------------------------------------------------
# Step 5: MIG 扩展依赖
# -----------------------------------------------------------------
say "Step 5: MIG-specific dependencies"
pip install -r setup/requirements_mig.txt

# transformers 升级到 4.57+ (Qwen3-VL/UMT5 需要)
say "  ensuring transformers >= 4.57.0..."
pip install -U "transformers>=4.57.0,<4.60.0"

# onnxruntime: transformers 5.x / diffusers 0.35+ 间接依赖
say "  installing onnxruntime (defensive, avoids ImportError)..."
pip install onnxruntime || warn "  onnxruntime install failed (non-blocking)"

# 二次校验
python -c "
import transformers
v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert v >= (4, 57), f'transformers {transformers.__version__} too old'
print(f'  ✓ transformers {transformers.__version__}')
try:
    import wan
    print(f'  ✓ wan importable from {wan.__file__}')
except ImportError as e:
    print(f'  ⚠️  wan not importable: {e}')
"

# -----------------------------------------------------------------
# Step 6: 模型权重(可跳过)
# -----------------------------------------------------------------
if [[ $SKIP_MODELS -eq 1 ]]; then
    warn "Step 6: model weights (SKIPPED)"
else
    say "Step 6: model weights"
    mkdir -p models

    if [[ -d "models/Wan2.1-VACE-1.3B" ]] && \
       [[ -n "$(ls -A models/Wan2.1-VACE-1.3B 2>/dev/null)" ]]; then
        say "  ✓ Wan2.1-VACE-1.3B already downloaded"
    else
        say "  downloading Wan2.1-VACE-1.3B (~6 GB)..."
        huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B \
            --local-dir models/Wan2.1-VACE-1.3B
    fi

    if [[ $SKIP_VLM -eq 0 ]]; then
        if [[ -d "models/Qwen3-VL-2B-Instruct" ]] && \
           [[ -n "$(ls -A models/Qwen3-VL-2B-Instruct 2>/dev/null)" ]]; then
            say "  ✓ Qwen3-VL-2B-Instruct already downloaded"
        else
            say "  downloading Qwen3-VL-2B-Instruct (~5 GB)..."
            huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
                --local-dir models/Qwen3-VL-2B-Instruct
        fi
    else
        info "  VLM (Qwen3-VL) NOT downloaded by default."
        info "  VLM is the data project's responsibility (sav_mig_data)."
        info "  Use --with_vlm if you need it for agent loop inference."
    fi
fi

# -----------------------------------------------------------------
# Step 7: 验证
# -----------------------------------------------------------------
say "Step 7: verification"
if python setup/verify_env.py; then
    say "============================================================"
    say " ✓ Installation complete"
    say "============================================================"
    say ""
    say "Next steps:"
    say "  conda activate $ENV_NAME"
    say "  python test_adapter.py     # Sanity test on adapter zero-init"
else
    err "verification reported issues — check above for details"
    exit 1
fi
