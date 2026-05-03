"""
Train Motion Mask Predictor
============================
独立训练脚本, 不依赖 wan-vace-mig 的主 adapter.

Usage:
    # 单卡 smoke test (5 步)
    python scripts/train_mask_predictor.py \
        --cache_dir /path/to/sav_mig_cache \
        --max_steps 5

    # 短跑 (5K 步, ~1 hour)
    python scripts/train_mask_predictor.py \
        --cache_dir /path/to/sav_mig_cache \
        --max_steps 5000 --batch_size 4 \
        --save_dir checkpoints/mask_pred_short

    # DDP 全量
    torchrun --nproc_per_node=2 scripts/train_mask_predictor.py \
        --cache_dir /path/to/sav_mig_cache \
        --max_steps 30000 --batch_size 4 \
        --save_dir checkpoints/mask_pred_full

模型大小:
    dim=384, blocks=6 → ~30M params
    在 H_lat × W_lat = 60 × 104 (480p latent) 上训, batch=4 单卡显存 ~12GB
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wan_vace_mig.mask_predictor.model import MotionMaskPredictor
from wan_vace_mig.mask_predictor.dataset import MaskPredictorDataset, collate_mask_batch
from wan_vace_mig.mask_predictor.losses import MaskPredictorLoss, compute_iou


# ============================================================================
# DDP 工具
# ============================================================================
def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def is_main(rank): return rank == 0


def to_device(batch, device, dtype):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
        else:
            out[k] = v
    return out


# ============================================================================
# Args
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cache_dir", required=True, help="sav-mig-data 输出目录")

    # 模型
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--use_rgb", action="store_true",
                   help="把 video latent 第 0 帧当视觉上下文喂入 (推荐)")
    p.add_argument("--text_dim", type=int, default=4096)

    # 数据
    p.add_argument("--max_objects", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)

    # 训练
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])

    # Loss
    p.add_argument("--w_bce", type=float, default=1.0)
    p.add_argument("--w_dice", type=float, default=1.0)
    p.add_argument("--w_smooth", type=float, default=0.0)
    p.add_argument("--bce_pos_weight", type=float, default=None,
                   help="BCE 正样本权重, 前景占 ~10% 时建议 9.0")

    # 输出
    p.add_argument("--save_dir", default="checkpoints/mask_predictor")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--resume", default=None)

    return p.parse_args()


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    rank, world, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    if is_main(rank):
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        print(f"[Init] world={world}, batch={args.batch_size}, "
              f"effective_batch={args.batch_size*world*args.grad_accum}, "
              f"dim={args.dim}, blocks={args.num_blocks}, dtype={args.dtype}")

    # ---- Dataset ----
    train_ds = MaskPredictorDataset(
        clips_root=args.cache_dir, split="train",
        max_objects=args.max_objects,
        use_first_frame_latent_as_rgb=args.use_rgb,
    )
    val_ds = MaskPredictorDataset(
        clips_root=args.cache_dir, split="val",
        max_objects=args.max_objects,
        random_object_subset=False,
        use_first_frame_latent_as_rgb=args.use_rgb,
    )

    if world > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
    else:
        train_sampler = None; val_sampler = None

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=collate_mask_batch,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, collate_fn=collate_mask_batch,
        num_workers=args.num_workers, pin_memory=True,
    )
    if is_main(rank):
        print(f"[Init] train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    # ---- Model ----
    model = MotionMaskPredictor(
        dim=args.dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        text_dim=args.text_dim,
        patch_size=args.patch_size,
        use_rgb=args.use_rgb,
    ).to(device).to(dtype)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main(rank):
        print(f"[Init] trainable params: {n_params:,} ({n_params/1e6:.1f}M)")

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False,
        )

    # ---- Optimizer / Scheduler ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay, betas=(0.9, 0.999))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---- Loss ----
    loss_fn = MaskPredictorLoss(
        w_bce=args.w_bce, w_dice=args.w_dice, w_smooth=args.w_smooth,
        bce_pos_weight=args.bce_pos_weight,
    )

    # ---- Resume ----
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        if is_main(rank):
            print(f"[Resume] from {args.resume}, step={start_step}")

    # ---- Training loop ----
    model.train()
    t0 = time.time()
    step = start_step
    train_iter = iter(train_dl)
    accumulator = {}

    while step < args.max_steps:
        opt.zero_grad()
        accum_loss = 0.0
        for ga in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(step // max(1, len(train_dl)))
                train_iter = iter(train_dl)
                batch = next(train_iter)

            batch_dev = to_device(batch, device, dtype)
            F_target = batch_dev["target_mask_seq"].shape[2]
            mask_logits = model(
                first_frame_mask=batch_dev["first_frame_mask"],
                verb_emb=batch_dev["verb_embeddings"],
                verb_mask=batch_dev["verb_masks"],
                F_target=F_target,
                first_frame_rgb=batch_dev.get("first_frame_rgb"),
            )

            losses = loss_fn(
                mask_logits=mask_logits,
                target_mask=batch_dev["target_mask_seq"],
                obj_valid=batch_dev["obj_valid"],
            )
            loss = losses["loss"] / args.grad_accum
            loss.backward()
            accum_loss += loss.item() * args.grad_accum

            for k, v in losses.items():
                if k == "loss": continue
                accumulator[k] = accumulator.get(k, 0.0) + float(v)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        scheduler.step()

        step += 1
        accumulator["loss"] = accumulator.get("loss", 0.0) + accum_loss / args.grad_accum

        # ---- Logging ----
        if is_main(rank) and step % args.log_every == 0:
            avg = {k: v / args.log_every for k, v in accumulator.items()}
            dt = time.time() - t0
            rate = step / max(dt, 1)
            eta_min = (args.max_steps - step) / max(rate, 0.01) / 60
            log_str = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
            lr = scheduler.get_last_lr()[0]
            print(f"[step {step}/{args.max_steps}] {log_str} lr={lr:.2e} "
                  f"{rate:.2f} st/s ETA={eta_min:.0f}m")
            accumulator = {}

        # ---- Validation ----
        if step % args.val_every == 0:
            model.eval()
            val_loss = 0.0; val_iou = 0.0; n_val = 0
            with torch.no_grad():
                for vb_idx, vb in enumerate(val_dl):
                    if vb_idx >= args.val_batches: break
                    vb_dev = to_device(vb, device, dtype)
                    F_target = vb_dev["target_mask_seq"].shape[2]
                    out = model(
                        first_frame_mask=vb_dev["first_frame_mask"],
                        verb_emb=vb_dev["verb_embeddings"],
                        verb_mask=vb_dev["verb_masks"],
                        F_target=F_target,
                        first_frame_rgb=vb_dev.get("first_frame_rgb"),
                    )
                    losses = loss_fn(out, vb_dev["target_mask_seq"], vb_dev["obj_valid"])
                    val_loss += float(losses["loss"])
                    val_iou += float(compute_iou(out, vb_dev["target_mask_seq"], vb_dev["obj_valid"]))
                    n_val += 1
            if is_main(rank) and n_val > 0:
                print(f"  [val @ step {step}] loss={val_loss/n_val:.4f} "
                      f"IoU={val_iou/n_val:.3f}  (over {n_val} batches)")
            model.train()

        # ---- Save ----
        if is_main(rank) and (step % args.save_every == 0 or step == args.max_steps):
            ckpt_path = Path(args.save_dir) / f"step_{step:06d}.pt"
            m = model.module if hasattr(model, "module") else model
            torch.save({
                "step": step,
                "model": m.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            last_link = Path(args.save_dir) / "last.pt"
            if last_link.exists() or last_link.is_symlink(): last_link.unlink()
            try:
                last_link.symlink_to(ckpt_path.name)
            except Exception:
                import shutil
                shutil.copy(ckpt_path, last_link)
            print(f"  [save] {ckpt_path}")

    if is_main(rank):
        print(f"[Done] {args.max_steps} steps in {(time.time()-t0)/60:.1f} min")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
