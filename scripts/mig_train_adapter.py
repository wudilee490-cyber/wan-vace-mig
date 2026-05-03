"""
Training entry: DecoupledMIGAdapter on SA-V cache
==================================================

前置:
    1. 已经跑完 scripts/sav_preprocess.py 生成 cache_dir
    2. cache_dir/index.json 至少有 ~5K 个有效样本(SA-V manual + auto 都用)

启动:
    # 单卡冒烟测试 (5 步)
    python scripts/mig_train_adapter.py \
        --cache_dir /data/sa_v_mig_cache \
        --wan_ckpt models/Wan2.1-VACE-1.3B \
        --max_steps 5 --batch_size 1

    # 8 卡 DDP 完整训练
    torchrun --nproc_per_node=8 scripts/mig_train_adapter.py \
        --cache_dir /data/sa_v_mig_cache \
        --wan_ckpt models/Wan2.1-VACE-1.3B \
        --batch_size 1 --grad_accum 4 \
        --max_steps 50000 \
        --save_dir checkpoints/mig_adapter_sav

训练参数选择依据见文档底部的 "Training Recipe"。
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# 定位 VACE 仓库 (ali-vilab/VACE) - 提供 models.wan.WanVace 容器类
# ---------------------------------------------------------------------------
# 优先级:
#   1. 环境变量 VACE_REPO (用户显式指定)
#   2. <wan-vace-mig 仓库>/third_party/VACE (install.sh 默认 clone 位置)
#   3. cwd 下找 models/wan (兼容用户在 VACE 仓库根跑)
# 找不到就报错并提示如何修.
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
        "Cannot locate VACE repo (ali-vilab/VACE).\n"
        "  Looked at:\n"
        "    1. $VACE_REPO env var\n"
        "    2. <repo_root>/third_party/VACE/\n"
        "    3. current working directory\n"
        "  Fix: re-run setup/install.sh (which clones VACE), or set\n"
        "       export VACE_REPO=/path/to/VACE\n"
        "       and verify <VACE_REPO>/models/wan exists."
    )


_ensure_vace_repo_in_path()

# 官方 Wan-VACE  (Wan2.1 主包在 site-packages, VACE 容器在 third_party/VACE)
import wan
from models.wan import WanVace
from models.wan.configs import WAN_CONFIGS

# 我们的扩展
from wan_vace_mig import WanVaceMIGPipeline
from wan_vace_mig.train.dataset import SAVMIGDataset, collate_mig_batch
from wan_vace_mig.train.losses import MIGTrainingLoss, sample_flow_matching_batch


# =========================================================================
# DDP utilities
# =========================================================================
def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world, local_rank
    return False, 0, 1, 0


def is_main(rank): return rank == 0


# =========================================================================
# vace_context 构造:
# 训练时我们用"首帧 + per-object union mask"作为 vace_context 的语义信号。
# 这与 VACE 的 mask-conditioned generation 一致。
# =========================================================================
def build_vace_context(
    first_frame_latent: torch.Tensor,        # [B, C_lat, H_lat, W_lat]
    obj_volume_masks: torch.Tensor,          # [B, K, F_p, H_p, W_p]
    obj_valid: torch.Tensor,                 # [B, K]
    F_lat: int, H_lat: int, W_lat: int,
    p_t: int, p_h: int, p_w: int,
    vace_in_dim: int,
) -> List[torch.Tensor]:
    """
    返回 List[Tensor [vace_in_dim, F_lat, H_lat, W_lat]] (per batch)。

    设计:
        ch[:C_lat]      = 首帧 latent (broadcast 到所有帧)
        ch[C_lat:C_lat+1] = per-frame union mask (上采样到 latent 分辨率)
        其余通道  零 (让 vace_in_dim 对得上)
    """
    B, C_lat, H, W = first_frame_latent.shape
    K = obj_volume_masks.shape[1]
    F_p = obj_volume_masks.shape[2]

    # 把 union mask 上采样到 latent 分辨率
    union = (obj_volume_masks * obj_valid.float().view(B, K, 1, 1, 1)).amax(dim=1)
    # patch → pixel
    union = union.repeat_interleave(p_t, dim=1)
    union = union.repeat_interleave(p_h, dim=2)
    union = union.repeat_interleave(p_w, dim=3)                  # [B, F_lat, H_lat, W_lat]
    # 截到目标尺寸
    union = union[:, :F_lat, :H_lat, :W_lat]

    # 首帧广播到所有帧
    ff_broadcast = first_frame_latent.unsqueeze(2).expand(B, C_lat, F_lat, H, W)

    # 拼通道
    parts = [ff_broadcast, union.unsqueeze(1)]                   # [B, C_lat+1, F, H, W]
    cur = torch.cat(parts, dim=1)
    if cur.shape[1] < vace_in_dim:
        pad = torch.zeros(B, vace_in_dim - cur.shape[1], F_lat, H, W,
                          device=cur.device, dtype=cur.dtype)
        cur = torch.cat([cur, pad], dim=1)
    elif cur.shape[1] > vace_in_dim:
        cur = cur[:, :vace_in_dim]

    return [cur[b] for b in range(B)]


# =========================================================================
# 主训练循环
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--wan_ckpt", required=True)
    p.add_argument("--model_name", default="vace-1.3B")
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--text_dim", type=int, default=4096)

    p.add_argument("--max_objects", type=int, default=3,
                   help="每段视频最多用几个物体训练 (复杂度上限)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--lambda_id", type=float, default=0.5)
    p.add_argument("--lambda_phase", type=float, default=0.05)
    p.add_argument("--weight_first_frame", type=float, default=2.0)

    p.add_argument("--identity_dropout", type=float, default=0.10,
                   help="训练时随机把 alpha_id 置 0 的概率,鼓励泛化")
    p.add_argument("--motion_dropout", type=float, default=0.10)

    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="主干 vace_blocks 每层 checkpoint, 节省 ~3-4x activation 显存. "
                        "代价: 每 step 多 30-40% 时间. RTX 5090 32G 跑 1.3B batch>=2 必开.")
    p.add_argument("--save_every", type=int, default=2500)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_dir", default="checkpoints/mig_adapter")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    is_ddp, rank, world, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    torch.manual_seed(args.seed + rank)

    # --------- 1. 加载 Wan-VACE + 构造 pipeline ---------
    cfg = WAN_CONFIGS[args.model_name]
    wan_vace = WanVace(config=cfg, checkpoint_dir=args.wan_ckpt, device_id=local_rank,
                       rank=rank, t5_fsdp=False, dit_fsdp=False, use_usp=False)
    pipe = WanVaceMIGPipeline(
        wan_vace=wan_vace,
        num_heads=args.num_heads, text_dim=args.text_dim,
        adapter_dtype=dtype, adapter_device=device,
    )
    pipe.freeze_base()
    transformer = pipe.transformer
    p_t, p_h, p_w = transformer.patch_size
    vace_in_dim = transformer.vace_in_dim

    # 启用 gradient checkpointing — 主干 forward 不存中间 activation,
    # backward 时按需重算. 显存节省 ~3-4x, 时间多 ~30-40%.
    # RTX 5090 32G 跑 1.3B batch >= 2 几乎必须开.
    if args.gradient_checkpointing:
        # 分别尝试常见的几种 API
        ckpt_enabled = False
        for module in [transformer, wan_vace.model]:
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
                ckpt_enabled = True
                break
            elif hasattr(module, "enable_gradient_checkpointing"):
                module.enable_gradient_checkpointing()
                ckpt_enabled = True
                break
        if not ckpt_enabled:
            # 手动给 vace_blocks 包装 checkpoint
            import torch.utils.checkpoint as ckpt_util
            for block in transformer.vace_blocks:
                orig_forward = block.forward
                def make_ckpt_forward(orig_fn):
                    def new_fn(*args, **kwargs):
                        return ckpt_util.checkpoint(orig_fn, *args, use_reentrant=False, **kwargs)
                    return new_fn
                block.forward = make_ckpt_forward(orig_forward)
            ckpt_enabled = True
        if is_main(rank):
            print(f"[Init] gradient checkpointing: {'enabled' if ckpt_enabled else 'failed'}")

    if is_main(rank):
        n_train = sum(p.numel() for p in pipe.trainable_parameters())
        print(f"[Init] adapter trainable params: {n_train:,}")

    # --------- 2. 数据 ---------
    ds = SAVMIGDataset(args.cache_dir, max_objects=args.max_objects)
    sampler = DistributedSampler(ds) if is_ddp else None
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers,
        collate_fn=collate_mig_batch, pin_memory=True, drop_last=True,
    )
    if is_main(rank):
        print(f"[Init] dataset: {len(ds)} videos, {len(dl)} steps/epoch")

    # --------- 3. Optimizer + LR scheduler ---------
    params = pipe.trainable_parameters()
    if is_ddp:
        # adapter 用 DDP 包装 (主干已冻结无需包)
        pipe.adapter = torch.nn.parallel.DistributedDataParallel(
            pipe.adapter, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
        params = pipe.adapter.parameters()

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                              betas=(0.9, 0.999))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        # 余弦退火到 10% LR
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    loss_fn = MIGTrainingLoss(
        lambda_identity=args.lambda_id,
        lambda_phase=args.lambda_phase,
        weight_first_frame=args.weight_first_frame,
    )

    # --------- 4. Resume (可选) ---------
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        # adapter 权重
        sd = ck["adapter"]
        target = pipe.adapter.module if is_ddp else pipe.adapter
        target.load_state_dict(sd, strict=False)
        optim.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["scheduler"])
        step = ck["step"]
        if is_main(rank):
            print(f"[Resume] from step {step}")

    # --------- 5. 训练循环 ---------
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    transformer.train()
    pipe.adapter.train() if not is_ddp else None
    micro_step = 0
    accum = args.grad_accum
    t_log = time.time()
    log_buf = {"loss": 0.0, "diff": 0.0, "id": 0.0, "phase": 0.0, "n": 0}

    while step < args.max_steps:
        if sampler: sampler.set_epoch(step)
        for batch in dl:
            # ---- batch → device ----
            x0 = batch["video_latent"].to(device, dtype)               # [B, C, F_p, H_lat, W_lat]
            ff_lat = batch["first_frame_latent"].to(device, dtype)     # [B, C, H_lat, W_lat]
            obj_image_masks = batch["obj_image_masks"].to(device)
            obj_volume_masks = batch["obj_volume_masks"].to(device)
            verb_emb = batch["verb_embeddings"].to(device, dtype)      # [B, K, N_v, text_dim]
            verb_mask = batch["verb_masks"].to(device)
            obj_valid = batch["obj_valid"].to(device)
            global_ctx = batch["global_context"].to(device, dtype)     # [B, L_g, text_dim]
            global_ctx_mask = batch["global_context_mask"].to(device)

            B, _, F_lat, H_lat, W_lat = x0.shape
            K = obj_volume_masks.shape[1]
            F_p, H_p, W_p = obj_volume_masks.shape[-3:]
            grid_sizes = torch.tensor([[F_p, H_p, W_p]] * B, dtype=torch.long, device=device)
            seq_len = F_p * H_p * W_p

            # ---- flow matching: 采 t / noise / x_t ----
            t, noise, x_t = sample_flow_matching_batch(x0, t_schedule="logit_normal")

            # ---- 构造 vace_context (列表形式) ----
            vace_context = build_vace_context(
                ff_lat, obj_volume_masks, obj_valid,
                F_lat, H_lat, W_lat, p_t, p_h, p_w, vace_in_dim,
            )

            # ---- text context: 用 global_context (per-batch list, 长度=B) ----
            context_list = []
            for b in range(B):
                L = int(global_ctx_mask[b].sum().item())
                context_list.append(global_ctx[b, :L])

            # ---- 计算 first_frame mask (从首帧 image masks 拿到 latent 分辨率) ----
            # SA-V cache 已经存了 [B, K, H_lat, W_lat],直接用
            # ---- 设置 adapter conditioning ----
            # 把 batched verb_emb [B,K,N_v,D] 转成 per-object list
            verb_emb_list  = [verb_emb[:, k] for k in range(K)]
            verb_mask_list = [verb_mask[:, k] for k in range(K)]

            # 构造 identity KV (从 first_frame_latent 抠 per-object token)
            id_kv_list, id_mask_list = pipe.conditioning_builder.build_identity_kv_from_first_frame(
                ff_lat, obj_image_masks,
            )

            # 训练时随机丢弃 (提升鲁棒性)
            do_drop_id = torch.rand(()) < args.identity_dropout
            do_drop_mo = torch.rand(()) < args.motion_dropout
            alpha_id = 0.0 if do_drop_id else 1.0
            alpha_mo = 0.0 if do_drop_mo else 1.0

            pipe.adapter._cond = None    # 清缓存
            pipe.adapter.set_conditioning(
                identity_kv_list=id_kv_list, identity_kv_masks=id_mask_list,
                verb_emb_list=verb_emb_list, verb_emb_masks=verb_mask_list,
                obj_volume_masks=obj_volume_masks,
                seq_len=seq_len, grid_sizes=grid_sizes, F_p=F_p,
                alpha_id=alpha_id, alpha_mo=alpha_mo,
                protect_first_frame=True,
            ) if (alpha_id > 0 or alpha_mo > 0) else None

            # ---- forward ----
            x_t_list = [x_t[b] for b in range(B)]
            t_for_model = (t * 1000.0).to(dtype)

            ctx_mgr = pipe.mig_attached() if (alpha_id > 0 or alpha_mo > 0) else nullcontext()
            with ctx_mgr:
                pred_list = transformer(
                    x=x_t_list, t=t_for_model,
                    vace_context=vace_context, context=context_list,
                    seq_len=seq_len,
                )
            pred = torch.stack(pred_list, dim=0)
            target = noise - x0

            # ---- 取出相位 KV 给 phase loss 用 ----
            motion_kv_per_obj = None
            if (alpha_mo > 0) and pipe.adapter._cond is not None:
                kv_list, _ = pipe.adapter._get_motion_kv_for_layer(0)
                motion_kv_per_obj = kv_list                      # List[K] of [B, F_p, N_v, dim]

            # ---- 计算损失 ----
            losses = loss_fn(
                pred_velocity=pred.float(), target_velocity=target.float(),
                obj_volume_masks=obj_volume_masks,
                obj_valid=obj_valid,
                motion_kv_perframe_per_obj=motion_kv_per_obj,
                patch_size=(p_t, p_h, p_w),
            )
            loss = losses["loss"] / accum
            loss.backward()

            log_buf["loss"]  += losses["loss"].item()
            log_buf["diff"]  += losses["loss_diffusion"].item()
            log_buf["id"]    += losses["loss_identity"].item()
            log_buf["phase"] += losses["loss_phase"].item()
            log_buf["n"]     += 1
            micro_step += 1

            # ---- gradient accumulation ----
            if micro_step % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if is_main(rank) and step % args.log_every == 0:
                    n = max(log_buf["n"], 1)
                    dt = time.time() - t_log
                    lr = scheduler.get_last_lr()[0]
                    print(f"[step {step:6d}] "
                          f"loss={log_buf['loss']/n:.4f} "
                          f"(diff={log_buf['diff']/n:.4f} "
                          f"id={log_buf['id']/n:.4f} "
                          f"phase={log_buf['phase']/n:.4f}) "
                          f"lr={lr:.2e} "
                          f"{dt:.1f}s/{args.log_every} steps")
                    log_buf = {k: 0.0 for k in log_buf}
                    log_buf["n"] = 0
                    t_log = time.time()

                if is_main(rank) and step % args.save_every == 0:
                    save_path = Path(args.save_dir) / f"adapter_step{step}.pt"
                    target_module = pipe.adapter.module if is_ddp else pipe.adapter
                    torch.save({
                        "step": step,
                        "adapter": target_module.state_dict(),
                        "optim": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args),
                    }, save_path)
                    print(f"  saved -> {save_path}")

                if step >= args.max_steps:
                    break

    if is_main(rank):
        final = Path(args.save_dir) / "adapter_final.pt"
        target_module = pipe.adapter.module if is_ddp else pipe.adapter
        torch.save({"step": step, "adapter": target_module.state_dict()}, final)
        print(f"\n✓ done; saved -> {final}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


# ============================================================
# Training Recipe (设计依据)
# ============================================================
"""
帧数 / 分辨率:
    81 帧 (Wan2.1 训练默认), 480p (480x832), patch_size=(1,2,2) → F_p=81 H_p=60 W_p=104
    seq_len ≈ 504K? 不,patch_size=(1,2,2) 实际是 81*60*104 = 506K,这是 1.3B 模型的训练序列长度。
    实际上 Wan 用的是 latent 帧数,VAE 时间下采样 4x → F_p ≈ 21,seq_len ≈ 21*60*104 ≈ 130K
    (具体看 VAE 的时间步幅,本代码里 F_p 由 video_latent.shape 确定,自动对齐)

batch / accum:
    单卡 A100-80G:1.3B 模型 + adapter,batch=1, accum=4 → 有效 batch=4
    8 卡:有效 batch = 32,符合 LoRA / adapter 训练惯例

LR schedule:
    1e-4 with 500 warmup + cosine 到 1e-5
    AdamW betas (0.9, 0.999),weight_decay 0.01
    依据:LoRA / IP-Adapter 等加载式训练在 1e-4 ~ 5e-5 表现最稳

总步数:
    SA-V 50K 视频 × 有效 batch 32 ≈ 1.5K steps/epoch
    50K steps ≈ 32 epochs,LoRA-style 训练 10-30 epoch 通常够用
    冒烟测试: 5 步看 loss 数值合理 + 梯度不为 NaN
    最小可用: 5K-10K 步 (能看到 identity preservation 起效)
    完整训练: 50K 步 (收敛)

损失权重:
    λ_id = 0.5    : 让物体区域损失约等于全图损失的 1.5x (经验: 0.3-1.0 都可)
    λ_phase = 0.05: 相位辅助是软引导,过大会让 PE 学得太尖锐导致动作生硬

dropout 用法:
    identity_dropout / motion_dropout 各 10%:
    随机让 alpha_id=0 或 alpha_mo=0,模型学会"在缺一边时也能合理生成"
    这对推理时 evaluator 反馈做 alpha 调节很关键 (alpha=0 时不能崩)
"""
