"""
MIG inference entry point
=========================
这是 vace_wan_inference.py 的 MIG 版本。它在官方推理流程中插入两个步骤:
    1. 加载 adapter 权重并构造 WanVaceMIGPipeline
    2. 在 wan_vace.generate(...) 之前调用 set_mig_conditioning + with mig_attached()

调用方式与官方推理脚本几乎一样,新增几个参数:
    --adapter_ckpt           训好的 adapter 权重路径(必需)
    --first_frame            3DIS 生成的首帧 PNG/JPG
    --first_frame_masks_dir  目录,内含 obj_0.png, obj_1.png, ...(每个物体的首帧 mask)
    --motion_phrases         JSON 字符串或文件,如:["jumping over fence", "running"]
    --motion_masks_dir       目录,内含 volume mask 数组(每个物体一份 [F, H, W])
    --alpha_id, --alpha_mo   注入强度

用法:
    python scripts/mig_inference.py \
        --ckpt_dir models/Wan2.1-VACE-1.3B \
        --adapter_ckpt models/mig_adapter/adapter.pt \
        --first_frame outputs/3dis_first_frame.png \
        --first_frame_masks_dir outputs/first_frame_masks/ \
        --motion_phrases '["a person jumping over a fence","a dog running"]' \
        --motion_masks_dir outputs/motion_masks/ \
        --prompt "A person jumps over a fence while a dog runs alongside" \
        --alpha_id 1.0 --alpha_mo 1.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---- 定位 VACE 仓库 (ali-vilab/VACE) ----
def _ensure_vace_repo_in_path():
    candidates = []
    if os.environ.get("VACE_REPO"):
        candidates.append(Path(os.environ["VACE_REPO"]))
    candidates.append(Path(__file__).resolve().parent.parent / "third_party" / "VACE")
    candidates.append(Path.cwd())
    for p in candidates:
        if (p / "models" / "wan").is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
    raise ImportError(
        "Cannot locate VACE repo. Re-run setup/install.sh (clones VACE to "
        "third_party/VACE), or set VACE_REPO env var."
    )

_ensure_vace_repo_in_path()

# ---- 复用官方 import 链 ----
import wan
from wan.utils.utils import cache_video, str2bool
from models.wan import WanVace
from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

# ---- 我们自己的扩展 ----
from wan_vace_mig import WanVaceMIGPipeline


def parse_args():
    p = argparse.ArgumentParser()

    # —— 沿用官方推理脚本的参数(精简版) ——
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--model_name", default="vace-1.3B")
    p.add_argument("--size", default="480p")
    p.add_argument("--prompt", required=True)
    p.add_argument("--src_video", default=None)
    p.add_argument("--src_mask", default=None)
    p.add_argument("--src_ref_images", default=None)
    p.add_argument("--save_dir", default="./results")
    p.add_argument("--seed", type=int, default=42)

    # —— MIG 专属参数 ——
    p.add_argument("--adapter_ckpt", required=True,
                   help="训好的 DecoupledMIGAdapter state_dict 路径")
    p.add_argument("--first_frame", required=True,
                   help="3DIS 生成的首帧图(PNG/JPG)")
    p.add_argument("--first_frame_masks_dir", required=True,
                   help="目录,含每个物体的首帧 mask: obj_0.png, obj_1.png, ...")
    p.add_argument("--motion_phrases", required=True,
                   help="JSON: 每个物体的动作短语,如 '[\"jumping\",\"running\"]'")
    p.add_argument("--motion_masks", required=True,
                   help=".npy 文件,形状 [n_obj, F_p, H_p, W_p] 的运动 mask volume")
    p.add_argument("--alpha_id", type=float, default=1.0)
    p.add_argument("--alpha_mo", type=float, default=1.0)
    p.add_argument("--num_heads", type=int, default=16,
                   help="WanModel 的 num_heads(1.3B=12, 14B=40 视模型而定,见 configs)")
    p.add_argument("--text_dim", type=int, default=4096)
    return p.parse_args()


# ---- helpers (按你的实际 VAE/encoder pipeline 替换) ----
def encode_first_frame_to_latent(image_path: str, vae, device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)   # [1,3,H,W]
    # 当成单帧视频送 VAE: [1, 3, 1, H, W]
    with torch.no_grad():
        latent = vae.encode(t.unsqueeze(2))
    if isinstance(latent, (tuple, list)):
        latent = latent[0]
    # 期望 [1, C_lat, 1, H_lat, W_lat] → 取出帧维度
    if latent.dim() == 5:
        latent = latent.squeeze(2)
    return latent  # [1, C_lat, H_lat, W_lat]


def load_first_frame_masks(masks_dir: str, target_hw, device) -> torch.Tensor:
    """加载首帧 per-object mask,resize 到 latent 分辨率。
    输出 [1, n_obj, H_lat, W_lat]."""
    files = sorted([f for f in os.listdir(masks_dir) if f.startswith("obj_")])
    masks = []
    H_lat, W_lat = target_hw
    for f in files:
        m = np.array(Image.open(os.path.join(masks_dir, f)).convert("L"))
        m = (m > 127).astype(np.float32)
        m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float().to(device)
        m_t = torch.nn.functional.interpolate(m_t, size=(H_lat, W_lat), mode="nearest")
        masks.append(m_t.squeeze(0).squeeze(0))
    return torch.stack(masks, dim=0).unsqueeze(0)  # [1, n_obj, H_lat, W_lat]


def main():
    args = parse_args()
    device = torch.device("cuda")

    # ============ 1. 用官方流程加载 WanVace ============
    cfg = WAN_CONFIGS[args.model_name]
    wan_vace = WanVace(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0,
        rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False,
    )

    # ============ 2. 构建 MIG pipeline 并加载 adapter ============
    adapter_sd = torch.load(args.adapter_ckpt, map_location="cpu")
    pipe = WanVaceMIGPipeline(
        wan_vace=wan_vace,
        num_heads=args.num_heads,
        text_dim=args.text_dim,
        adapter_state_dict=adapter_sd,
    )

    # ============ 3. 准备 MIG 条件 ============
    # 动作短语
    if os.path.isfile(args.motion_phrases):
        with open(args.motion_phrases) as f:
            phrases = json.load(f)
    else:
        phrases = json.loads(args.motion_phrases)

    # 运动 mask volume
    motion_volume = np.load(args.motion_masks)              # [n_obj, F_p, H_p, W_p]
    if motion_volume.ndim != 4:
        raise ValueError(f"motion_masks 期望 4 维 [n_obj, F_p, H_p, W_p],"
                         f"实际 {motion_volume.shape}")
    obj_volume_masks = torch.from_numpy(motion_volume).unsqueeze(0).to(device)  # [1, n_obj, F_p, H_p, W_p]
    _, n_obj, F_p, H_p, W_p = obj_volume_masks.shape

    # grid_sizes & seq_len(VACE patch 化的网格)
    grid_sizes = torch.tensor([[F_p, H_p, W_p]], dtype=torch.long, device=device)
    seq_len = F_p * H_p * W_p

    # 首帧 latent + 首帧 per-object mask
    vae = getattr(wan_vace, "vae", None)
    first_latent = encode_first_frame_to_latent(args.first_frame, vae, device)
    H_lat, W_lat = first_latent.shape[-2:]
    obj_image_masks = load_first_frame_masks(
        args.first_frame_masks_dir, (H_lat, W_lat), device,
    )

    # 灌进 adapter
    pipe.set_mig_conditioning(
        first_frame_latent=first_latent,
        obj_image_masks=obj_image_masks,
        obj_motion_phrases=[phrases],            # batch=1
        obj_volume_masks=obj_volume_masks,
        grid_sizes=grid_sizes,
        seq_len=seq_len,
        F_p=F_p,
        alpha_id=args.alpha_id,
        alpha_mo=args.alpha_mo,
        protect_first_frame=True,
    )

    # ============ 4. 推理:在 with 块内 adapter 生效 ============
    save_path = Path(args.save_dir) / "mig_output.mp4"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with pipe.mig_attached():
        video = wan_vace.generate(
            input_prompt=args.prompt,
            src_video=args.src_video,
            src_mask=args.src_mask,
            src_ref_images=args.src_ref_images,
            size=args.size,
            seed=args.seed,
        )

    cache_video(tensor=video[None], save_file=str(save_path), fps=cfg.sample_fps,
                nrow=1, normalize=True, value_range=(-1, 1))
    print(f"✓ saved to {save_path}")


if __name__ == "__main__":
    main()
