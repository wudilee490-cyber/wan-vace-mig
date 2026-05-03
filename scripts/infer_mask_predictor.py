"""
Motion Mask Predictor 推理脚本
==============================
输入: 首帧 mask (per-object) + 动作短语
输出: 后续帧 mask 序列

Usage:
    python scripts/infer_mask_predictor.py \
        --ckpt checkpoints/mask_pred/last.pt \
        --first_frame_mask first_mask.npy \
        --phrases "running across grass" "standing still" \
        --F_target 21 \
        --output predicted_masks.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wan_vace_mig.mask_predictor.model import MotionMaskPredictor


def encode_phrases(phrases, text_model_id, device, dtype, max_length=64):
    """用 UMT5 编码动作短语, 返回 [K, N_v, D_text]"""
    from transformers import UMT5EncoderModel, AutoTokenizer
    encoder = UMT5EncoderModel.from_pretrained(
        text_model_id, subfolder="text_encoder",
        torch_dtype=dtype,
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(text_model_id, subfolder="tokenizer")

    enc = tok(phrases, return_tensors="pt", padding=True,
              truncation=True, max_length=max_length)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device))
    hidden = out.last_hidden_state                          # [K, L_max, D]
    mask = enc["attention_mask"].to(device).bool()
    return hidden, mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="训练好的 mask_predictor checkpoint")
    p.add_argument("--first_frame_mask", required=True,
                   help=".npy [K, H, W] 首帧 mask")
    p.add_argument("--phrases", nargs="+", required=True,
                   help="每个物体的动作短语")
    p.add_argument("--first_frame_rgb", default=None,
                   help="可选 .npy [3, H, W] 首帧 RGB latent (如果训练时 use_rgb=True)")
    p.add_argument("--F_target", type=int, default=21)
    p.add_argument("--text_model_id", default="Wan-AI/Wan2.1-VACE-1.3B-diffusers")
    p.add_argument("--output", required=True, help=".npy [K, F, H, W] uint8")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    # ---- 读 ckpt 拿训练时的模型超参 ----
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt["args"]
    print(f"[Init] Loading model trained at step {ckpt['step']}, "
          f"args: dim={train_args['dim']}, blocks={train_args['num_blocks']}, "
          f"use_rgb={train_args['use_rgb']}")

    # ---- 模型 ----
    model = MotionMaskPredictor(
        dim=train_args["dim"],
        num_heads=train_args["num_heads"],
        num_blocks=train_args["num_blocks"],
        text_dim=train_args["text_dim"],
        patch_size=train_args["patch_size"],
        use_rgb=train_args["use_rgb"],
    ).to(device).to(dtype).eval()
    model.load_state_dict(ckpt["model"])

    # ---- 输入数据 ----
    first_mask = np.load(args.first_frame_mask)               # [K, H, W] uint8
    K, H, W = first_mask.shape
    if len(args.phrases) != K:
        raise ValueError(f"# phrases ({len(args.phrases)}) != # objects ({K})")
    print(f"[Init] {K} objects, mask shape ({H}, {W}), F_target={args.F_target}")
    print(f"[Init] phrases: {args.phrases}")

    fm = torch.from_numpy(first_mask).float().unsqueeze(0).to(device).to(dtype)  # [1, K, H, W]

    # 编码动作
    verb_emb, verb_mask = encode_phrases(
        args.phrases, args.text_model_id, device, dtype,
    )
    verb_emb = verb_emb.unsqueeze(0).to(dtype)                # [1, K, N_v, D]
    verb_mask = verb_mask.unsqueeze(0)                         # [1, K, N_v]

    # 可选 RGB
    fr = None
    if args.first_frame_rgb is not None:
        fr_arr = np.load(args.first_frame_rgb)                # [3, H, W]
        fr = torch.from_numpy(fr_arr).float().unsqueeze(0).to(device).to(dtype)

    # ---- 推理 ----
    with torch.no_grad():
        binary = model.predict(
            first_frame_mask=fm,
            verb_emb=verb_emb,
            verb_mask=verb_mask,
            F_target=args.F_target,
            first_frame_rgb=fr,
            threshold=args.threshold,
        )                                                      # [1, K, F, H, W] uint8

    out = binary.squeeze(0).cpu().numpy().astype(np.uint8)
    np.save(args.output, out)
    print(f"[Done] saved {out.shape} → {args.output}")
    print(f"  per-object pixel coverage:")
    for k in range(K):
        cov = out[k].mean(axis=(1, 2))                         # [F]
        print(f"    obj{k} ({args.phrases[k]}): start={cov[0]*100:.1f}% "
              f"end={cov[-1]*100:.1f}% mean={cov.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
