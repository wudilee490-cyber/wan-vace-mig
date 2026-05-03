"""
Verify VACE-MIG environment installation
=========================================
精确诊断版,适配 CUDA 12.x / 13.x、RTX 5090 (sm_120) 等新硬件。

退出码:
    0 = 全部通过
    1 = 关键检查失败 (训练/推理无法进行)
    2 = 非关键检查失败 (功能受限,但可以继续)
"""

import importlib
import os
import sys
import traceback
from pathlib import Path

# ANSI 颜色
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; CYAN = "\033[96m"; NC = "\033[0m"


class Checker:
    def __init__(self):
        self.crit_fail = []
        self.opt_fail = []
        self.passed = []

    def check(self, name, fn, critical=True, hint=""):
        try:
            ret = fn()
            if ret is False:
                raise RuntimeError("returned False")
            print(f"  {GREEN}✓{NC} {name}" + (f": {ret}" if isinstance(ret, str) else ""))
            self.passed.append(name)
            return True
        except Exception as e:
            tag = "CRITICAL" if critical else "OPTIONAL"
            print(f"  {RED}✗{NC} {name} [{tag}]: {type(e).__name__}: {e}")
            if hint:
                print(f"    {CYAN}→ Hint: {hint}{NC}")
            (self.crit_fail if critical else self.opt_fail).append(name)
            return False

    def section(self, title):
        print(f"\n{YELLOW}── {title} ──{NC}")

    def report(self):
        print()
        print("=" * 60)
        print(f"  Passed:        {len(self.passed)}")
        print(f"  Critical fail: {len(self.crit_fail)}")
        print(f"  Optional fail: {len(self.opt_fail)}")
        print("=" * 60)
        if self.crit_fail:
            print(f"\n{RED}Critical checks failed:{NC}")
            for f in self.crit_fail:
                print(f"  - {f}")
            return 1
        if self.opt_fail:
            print(f"\n{YELLOW}Optional checks failed (non-blocking):{NC}")
            for f in self.opt_fail:
                print(f"  - {f}")
            return 2
        print(f"\n{GREEN}All checks passed!{NC}")
        return 0


def main():
    c = Checker()
    repo_root = Path(__file__).resolve().parent.parent

    # ============ Python 版本 ============
    c.section("Python")
    def _py():
        v = sys.version_info
        assert v.major == 3 and v.minor >= 10, f"need 3.10+, got {v.major}.{v.minor}"
        return f"{sys.version.split()[0]}"
    c.check("Python 3.10+", _py)

    # ============ PyTorch + CUDA + GPU 兼容性 ============
    c.section("PyTorch + CUDA + GPU compatibility")

    def _torch_basic():
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
        return f"{torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}"
    c.check("PyTorch with CUDA available", _torch_basic)

    # 严格的 GPU 兼容性检查: 实际 sm_xx 必须在 PyTorch 编译列表里
    def _gpu_compat():
        import torch
        if torch.cuda.device_count() == 0:
            raise RuntimeError("no CUDA device")

        # PyTorch 编译时支持的 archs
        try:
            supported = torch.cuda.get_arch_list()  # ['sm_50', 'sm_60', ...]
        except Exception:
            supported = []
        supported_caps = set()
        for arch in supported:
            if arch.startswith("sm_"):
                num = arch[3:]
                if len(num) >= 2:
                    supported_caps.add((int(num[:-1]) if len(num) > 1 else int(num),
                                         int(num[-1])))

        problems = []
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)  # e.g. (12, 0) for sm_120
            name = torch.cuda.get_device_name(i)
            sm_str = f"sm_{cap[0]}{cap[1]}"
            gpu_info.append(f"GPU{i}={name} ({sm_str})")

            if cap not in supported_caps and supported_caps:
                problems.append(f"GPU{i} ({name}, {sm_str}) NOT in supported archs {supported}")

        if problems:
            raise RuntimeError("; ".join(problems))
        return f"compat ok — {', '.join(gpu_info)}"

    rtx_5090_hint = (
        "RTX 5090/Blackwell (sm_120) needs PyTorch >= 2.7.0 with CUDA 12.8+ or CUDA 13.\n"
        "    Install:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
        "    Or:       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130\n"
        "    Verify:   python -c \"import torch; print(torch.cuda.get_arch_list())\""
    )
    c.check("GPU compute capability matches PyTorch arch list",
            _gpu_compat, hint=rtx_5090_hint)

    # ============ VACE 主依赖 ============
    c.section("VACE main dependencies")
    for pkg in [
        "diffusers", "transformers", "tokenizers", "accelerate",
        "einops", "decord", "pycocotools", "PIL", "cv2", "numpy",
    ]:
        def _imp(p=pkg):
            mod = importlib.import_module(p)
            return getattr(mod, "__version__", "ok")
        c.check(f"import {pkg}", _imp)

    # transformers 版本必须在 Qwen3-VL 兼容范围内
    def _tf_version():
        import transformers
        v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
        # 4.57+ 可用,但 5.x 是大版本变动,需要警告(已知 5.7 是有bug的过渡版本)
        if v >= (5, 0):
            print(f"    {YELLOW}WARN: transformers {transformers.__version__} 是 5.x,"
                  f"VACE 的 wan 模块原本针对 4.x。可能存在 API 不兼容,如出问题降级到 4.57-4.59{NC}")
        if v < (4, 57):
            raise RuntimeError(f"need >= 4.57 for Qwen3-VL, got {transformers.__version__}")
        return f"{transformers.__version__} (Qwen3-VL compat)"
    c.check("transformers >= 4.57 (Qwen3-VL requirement)", _tf_version)

    # ============ Wan-VACE (双依赖: pip wan 包 + 本地 VACE 仓库) ============
    c.section("Wan-VACE (Wan2.1 pip + VACE repo)")

    # 先把 repo 根加到 sys.path (兼容 wan_vace_mig 源码加载)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # ---- 1. Wan2.1 主包 (pip 安装在 site-packages) ----
    wan_pip_hint = (
        "Wan2.1 主包应通过 pip 安装:\n"
        "    pip install 'wan@git+https://github.com/Wan-Video/Wan2.1'\n"
        "    或 setup/install.sh 自动装"
    )

    def _wan():
        import wan
        path = getattr(wan, "__file__", "<no __file__>")
        return f"version={getattr(wan, '__version__', 'unknown')}, path={path}"
    c.check("import wan (Wan2.1 main package)", _wan, hint=wan_pip_hint)

    def _wan_model():
        from wan.modules.model import WanModel  # noqa
        return "WanModel imported"
    c.check("wan.modules.model.WanModel", _wan_model, hint=wan_pip_hint)

    def _wan_attn():
        from wan.modules.attention import flash_attention  # noqa
        return "wan.modules.attention.flash_attention imported"
    c.check("wan.modules.attention.flash_attention", _wan_attn, hint=wan_pip_hint)

    # ---- 2. VACE 仓库 (ali-vilab/VACE) - 提供 models.wan.WanVace 容器 ----
    vace_hint = (
        "VACE 仓库 (ali-vilab/VACE) 不是 pip 包, 必须 git clone 到本地:\n"
        "    setup/install.sh 默认 clone 到 <repo_root>/third_party/VACE/\n"
        "    或显式: export VACE_REPO=/path/to/VACE\n"
        "    验证: ls <VACE_REPO>/models/wan/ 应有 wan_vace.py 等"
    )

    def _vace_repo():
        # 找 VACE 仓库
        candidates = []
        if os.environ.get("VACE_REPO"):
            candidates.append(Path(os.environ["VACE_REPO"]))
        candidates.append(repo_root / "third_party" / "VACE")
        candidates.append(Path.cwd())
        found = None
        for p in candidates:
            if (p / "models" / "wan").is_dir():
                found = p
                break
        if not found:
            raise FileNotFoundError(
                f"VACE repo not found. Checked: {[str(c) for c in candidates]}"
            )
        if str(found) not in sys.path:
            sys.path.insert(0, str(found))
        return f"located at {found}"
    c.check("VACE repo (models.wan/) located", _vace_repo, hint=vace_hint)

    def _wan_vace_class():
        from models.wan import WanVace  # noqa: F401
        from models.wan.configs import WAN_CONFIGS  # noqa: F401
        return "WanVace + WAN_CONFIGS imported from models.wan"
    c.check("models.wan.WanVace can import", _wan_vace_class, hint=vace_hint)

    # ============ Flash Attention ============
    c.section("Flash Attention (optional but recommended)")
    def _fa():
        import flash_attn
        return flash_attn.__version__
    fa_hint = (
        "RTX 5090 (sm_120) 需要 flash-attn >= 2.7.4.post1 或 flash-attn 3+\n"
        "    新硬件先尝试: pip install flash-attn --no-build-isolation\n"
        "    若编译失败,设 MAX_JOBS=4 限制并行编译进程数"
    )
    c.check("flash-attn installed", _fa, critical=False, hint=fa_hint)

    # ============ MIG 自身 (隔离每个子模块,避免一个挂全挂) ============
    c.section("MIG adapter (granular sub-module check)")

    # adapter 子模块: 纯 PyTorch,最干净
    def _mig_adapter():
        from wan_vace_mig.adapter import (
            DecoupledMIGAdapter, ConditioningBuilder,
            PhaseAwareMotionEncoder, MaskedCrossAttention,
        )
        return "all adapter classes imported"
    c.check("wan_vace_mig.adapter imports (core MIG model)", _mig_adapter)

    # train 子模块: 不含 data, 只有 dataset/losses
    def _mig_train():
        from wan_vace_mig.train.dataset import SAVMIGDataset, collate_mig_batch  # noqa
        from wan_vace_mig.train.losses import MIGTrainingLoss, sample_flow_matching_batch  # noqa
        return "train modules imported"
    c.check("wan_vace_mig.train imports", _mig_train)

    def _mask_predictor():
        from wan_vace_mig.mask_predictor import (  # noqa: F401
            MotionMaskPredictor, MaskPredictorDataset, MaskPredictorLoss,
        )
        return "mask predictor module imported"
    c.check("wan_vace_mig.mask_predictor imports", _mask_predictor)

    # pipelines 依赖 wan,所以单独check并明确依赖关系
    def _mig_pipelines():
        from wan_vace_mig.pipelines import WanVaceMIGPipeline  # noqa
        return "WanVaceMIGPipeline imported"
    c.check("wan_vace_mig.pipelines imports (depends on wan)",
            _mig_pipelines,
            hint="若上面 'import wan' 失败,这里也会失败,优先解决 wan 模块路径问题")

    # 顶层 init 是聚合,放最后,失败时给具体追溯
    def _mig_top():
        # 单独 reimport 看具体哪个子模块失败
        try:
            from wan_vace_mig import (
                DecoupledMIGAdapter, ConditioningBuilder, WanVaceMIGPipeline,
            )
        except Exception as e:
            # 给出更详细的错误来源
            raise RuntimeError(f"top-level import failed: {e}") from e
        return "top-level imports ok"
    c.check("wan_vace_mig (top-level aggregation)", _mig_top)

    # ============ Adapter 零初始化等价性 ============
    c.section("Adapter zero-init equivalence (smoke test)")

    def _zero_init():
        # 这个测试只用 torch + adapter 自身,不依赖 wan/transformers
        import torch
        from wan_vace_mig.adapter.decoupled_mig_adapter import MaskedCrossAttention

        m = MaskedCrossAttention(dim=64, num_heads=4, kv_dim=64)
        assert m.to_out.weight.abs().max().item() == 0
        assert m.to_out.bias.abs().max().item() == 0

        x = torch.randn(1, 8, 64)
        kv = torch.randn(1, 4, 64)
        q_mask = torch.ones(1, 8, dtype=torch.bool)
        out = m(x, kv, q_mask)
        assert out.abs().max().item() == 0, "zero-init Linear should produce 0 output"
        return "PASS (delta=0 confirmed) — adapter loaded but untrained ≡ vanilla VACE"
    c.check("MaskedCrossAttention zero-init produces zero output",
            _zero_init,
            hint="Adapter 内部测试,仅依赖 PyTorch。失败说明 adapter 实现有问题")

    # ============ 模型权重 ============
    c.section("Model weights (optional)")

    def _wan_ckpt():
        path = repo_root / "models" / "Wan2.1-VACE-1.3B"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        files = list(path.glob("*.safetensors")) + list(path.glob("*.pt")) + list(path.glob("*.pth"))
        if not files:
            raise FileNotFoundError(f"no weight files in {path}")
        return f"{len(files)} weight files in models/Wan2.1-VACE-1.3B"
    c.check("Wan2.1-VACE-1.3B weights", _wan_ckpt, critical=False,
            hint="huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/Wan2.1-VACE-1.3B")

    # 注意: 模型项目不再需要 Qwen3-VL,那是数据生成项目的事
    # 这里只检查 caption 是否能在 inference 时用 (可选)
    def _vlm_ckpt():
        # 模型项目不下载 VLM,但允许已经在的 (e.g. 用户从 sav_mig_data 项目软链过来)
        path = repo_root / "models" / "Qwen3-VL-2B-Instruct"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found "
                "(only needed if you do agent-loop inference with online VLM eval)"
            )
        return f"present at {path}"
    c.check("Qwen3-VL-2B-Instruct weights (optional, for agent loop)",
            _vlm_ckpt, critical=False,
            hint="模型项目本身不需要 VLM。VLM 是数据项目 sav_mig_data 的事。\n"
                 "    只有跑评估 agent loop 时才需要")

    # ============ DDP 准备 ============
    c.section("Distributed training prerequisites")
    def _torchrun():
        import shutil
        if shutil.which("torchrun") is None:
            raise FileNotFoundError("torchrun not found")
        return "torchrun available"
    c.check("torchrun available", _torchrun, critical=False)

    def _nccl():
        import torch.distributed as dist
        if not dist.is_available():
            raise RuntimeError("torch.distributed not available")
        return "torch.distributed available"
    c.check("torch.distributed available", _nccl, critical=False)

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
