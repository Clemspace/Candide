#!/usr/bin/env python3
"""
train_candide18c.py — Pretraining script for Candide 18c historical French LM

Reads pre-tokenized binary shards from /data/pretrain_candide/shards/
Uses the domain-trained 32k BPE tokenizer from /data/pretrain_candide/tokenizer/

Usage:
    # Trial run (1B tokens, no wandb)
    python train_candide18c.py --config configs/candide18c.yaml --trial

    # Full run with wandb
    python train_candide18c.py --config configs/candide18c.yaml --wandb

    # Resume
    python train_candide18c.py --config configs/candide18c.yaml --resume checkpoints/step_5000.pt --wandb

    # Pilot sweep arm (overrides for LR / seed / budget / naming)
    python train_candide18c.py --config configs/pilot/angpt.yaml --wandb \
        --lr 2e-3 --seed 1 --max-steps 1526 --run-name angpt-lr2e-3-s1 --group lr-sweep
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from dataclasses import dataclass, field
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size: int = 32132
    d_model: int = 768
    n_layers: int = 8
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    norm_first: bool = True
    norm_type: str = "rms"
    attention_type: str = "gqa"
    ffn_type: str = "swiglu"
    use_rope: bool = True
    rope_theta: float = 10000.0
    bias: bool = False
    tie_word_embeddings: bool = False
    # Architecture family: "gpt" (RamanujanTransformer) or "angpt" (ANGPTTransformer)
    arch: str = "gpt"
    qk_norm: bool = False          # gpt only; True = the GPT+ baseline
    # anGPT only (ignored for arch="gpt")
    alpha_init: float = 0.05
    alpha_scale: float = 0.01
    residual_mode: str = "approx"
    gate_scale: float = 1.0

@dataclass
class TrainingConfig:
    max_steps: int = 10000
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    min_lr: float = 3e-5
    mixed_precision: str = "bf16"
    eval_every: int = 250
    eval_steps: int = 50        # batches to use for val loss estimate
    save_every: int = 1000
    log_every: int = 10
    seed: int = 42

@dataclass
class DataConfig:
    shards_dir: str = "/data/pretrain_candide/shards"
    tokenizer_path: str = "/data/pretrain_candide/tokenizer/tokenizer.json"
    sequence_length: int = 1024
    num_workers: int = 2
    prefetch_factor: int = 4
    pin_memory: bool = True
    # "global": windows shuffled across all shards every epoch (default).
    # "sequential": legacy order, contiguous text within a batch (runs before 2026-11-07).
    train_order: str = "global"
    # Fixed random validation sample, identical for every run (None = legacy stream).
    val_windows: Optional[int] = 1600
    val_seed: int = 0

@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "candide-18c"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "candide18c-118m"
    wandb_group: Optional[str] = None
    wandb_tags: list = field(default_factory=lambda: [
        "candide", "18c-french", "historical-lm", "modern-corpus"
    ])
    output_dir: str = "./runs/candide18c"

@dataclass
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str) -> FullConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = FullConfig()
    if "model" in raw:
        for k, v in raw["model"].items():
            setattr(cfg.model, k, v)
    if "training" in raw:
        for k, v in raw["training"].items():
            setattr(cfg.training, k, v)
    if "data" in raw:
        for k, v in raw["data"].items():
            setattr(cfg.data, k, v)
    if "logging" in raw:
        for k, v in raw["logging"].items():
            setattr(cfg.logging, k, v)
    return cfg


# ── Data ──────────────────────────────────────────────────────────────────────

# Import the shard loader (place shard_loader.py in same dir or adjust path)
sys.path.insert(0, str(Path(__file__).parent))
from shard_loader import ShardLoader


# ── Model ─────────────────────────────────────────────────────────────────────

def setup_model(cfg: ModelConfig) -> nn.Module:
    """Build model from Candide framework config."""
    if cfg.arch == "angpt":
        return setup_angpt(cfg)
    if cfg.arch != "gpt":
        raise ValueError(f"Unknown arch: {cfg.arch}. Available: 'gpt', 'angpt'")

    from ramanujan.models.architectures import RamanujanTransformer, TransformerConfig

    tc = TransformerConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        norm_first=cfg.norm_first,
        norm_type=cfg.norm_type,
        attention_type=cfg.attention_type,
        ffn_type=cfg.ffn_type,
        use_rope=cfg.use_rope,
        rope_theta=cfg.rope_theta,
        bias=cfg.bias,
        qk_norm=cfg.qk_norm,
        tie_word_embeddings=cfg.tie_word_embeddings,
    )
    model = RamanujanTransformer(tc)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable), "
             f"arch=gpt, qk_norm={cfg.qk_norm}")
    return model


def setup_angpt(cfg: ModelConfig) -> nn.Module:
    """Build the approximately normalized transformer (anGPT)."""
    from ramanujan.models.architectures.angpt import ANGPTConfig, ANGPTTransformer

    if cfg.dropout > 0:
        log.warning(f"anGPT with dropout={cfg.dropout}; the paper uses 0")
    ac = ANGPTConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        attention_type=cfg.attention_type,
        use_rope=cfg.use_rope,
        rope_theta=cfg.rope_theta,
        bias=cfg.bias,
        tie_word_embeddings=cfg.tie_word_embeddings,
        alpha_init=cfg.alpha_init,
        alpha_scale=cfg.alpha_scale,
        residual_mode=cfg.residual_mode,
        gate_scale=cfg.gate_scale,
    )
    model = ANGPTTransformer(ac)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {total/1e6:.1f}M params, arch=angpt, residual_mode={cfg.residual_mode}, "
             f"gate_scale={cfg.gate_scale}")
    return model


# ── Optimizer / Scheduler ─────────────────────────────────────────────────────

def setup_optimizer(model: nn.Module, cfg: TrainingConfig, arch: str = "gpt"):
    if arch == "angpt":
        # anGPT: no weight decay on anything (rows are bounded by projection;
        # decay would shrink alpha / s_z / attention temperature). Same AdamW
        # implementation and betas as the baseline so only the decay differs.
        if cfg.weight_decay > 0:
            log.warning(f"anGPT: ignoring weight_decay={cfg.weight_decay}, using 0")
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            fused=torch.cuda.is_available(),
        )

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name for nd in ["bias", "norm", "ln_"]):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),  # fused AdamW — faster on A40
    )
    return optimizer


def cosine_lr(step: int, cfg: TrainingConfig) -> float:
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.training_max_steps - cfg.warmup_steps)
    ratio = cfg.min_lr / cfg.learning_rate
    return ratio + 0.5 * (1 - ratio) * (1 + math.cos(math.pi * progress))


# ── WandB ─────────────────────────────────────────────────────────────────────

def setup_wandb(cfg: FullConfig, model: nn.Module):
    if not cfg.logging.use_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.logging.wandb_run_name,
            group=cfg.logging.wandb_group,
            tags=list(cfg.logging.wandb_tags) + [f"arch-{cfg.model.arch}"]
                 + (["qk-norm"] if cfg.model.qk_norm else []),
            config={
                "model": cfg.model.__dict__,
                "training": cfg.training.__dict__,
                "data": cfg.data.__dict__,
            },
        )
        wandb.watch(model, log="gradients", log_freq=200)
        log.info(f"WandB: {run.url}")
        return wandb
    except ImportError:
        log.warning("wandb not installed — pip install wandb")
        return None


def log_metrics(wandb_run, step: int, metrics: dict):
    if wandb_run:
        wandb_run.log(metrics, step=step)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler_step, step, cfg, tag=""):
    out_dir = Path(cfg.logging.output_dir) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"step_{step}{tag}.pt" if not tag else f"{tag}.pt"
    path = out_dir / name
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "config":    cfg.__dict__,
    }, path)
    log.info(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    log.info(f"Resumed from step {step}: {path}")
    return step


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, n_steps: Optional[int], scaler_ctx) -> dict:
    """
    Token-weighted mean validation loss.

    n_steps=None reads the whole (finite) validation loader: with the fixed
    validation sample, periodic and final evaluations then use the same
    windows and are on the same scale. With the legacy stream, n_steps caps
    the number of batches. The loss is divided by the tokens actually seen.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    for i, batch in enumerate(val_loader):
        if n_steps is not None and i >= n_steps:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)
        with scaler_ctx:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)).float(), labels.view(-1))
        total_loss += loss.item()
        total_tokens += int((labels != -100).sum())
    model.train()
    avg_loss = total_loss / max(1, total_tokens)
    return {
        "val/loss": avg_loss,
        "val/perplexity": math.exp(min(avg_loss, 20)),
        "val/eval_tokens": total_tokens,
    }


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(cfg: FullConfig, resume: Optional[str] = None):
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Data
    log.info("Setting up data loaders...")
    loader = ShardLoader(
        data_dir=cfg.data.shards_dir,
        seq_len=cfg.data.sequence_length,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        pin_memory=cfg.data.pin_memory,
        seed=cfg.training.seed,
        train_order=cfg.data.train_order,
        val_windows=cfg.data.val_windows,
        val_seed=cfg.data.val_seed,
    )
    train_loader, val_loader = loader.get_dataloaders()
    # Fixed sample: every evaluation reads all of it. Legacy stream: capped.
    eval_periodic = None if cfg.data.val_windows else cfg.training.eval_steps
    eval_final = None if cfg.data.val_windows else cfg.training.eval_steps * 4

    # Model
    log.info("Building model...")
    model = setup_model(cfg.model).to(device)

    # Optimizer
    optimizer = setup_optimizer(model, cfg.training, arch=cfg.model.arch)
    projects = hasattr(model, "project_parameters")

    # Resume
    global_step = 0
    if resume:
        global_step = load_checkpoint(resume, model, optimizer)
    if projects:
        model.project_parameters()   # weight rows into the unit ball before step 0

    # Mixed precision
    dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype, enabled=(device.type == "cuda"))

    # WandB
    wandb_run = setup_wandb(cfg, model)

    criterion = nn.CrossEntropyLoss()
    eff_batch  = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    tokens_per_step = eff_batch * cfg.data.sequence_length

    log.info(f"Effective batch size: {eff_batch} "
             f"({cfg.training.batch_size} × {cfg.training.gradient_accumulation_steps} accum)")
    log.info(f"Tokens per step: {tokens_per_step:,}")
    log.info(f"Training for {cfg.training.max_steps:,} steps "
             f"= {cfg.training.max_steps * tokens_per_step / 1e9:.2f}B tokens")
    log.info("Starting training...")

    # LR lambda — needs cfg accessible
    def get_lr(step):
        if step < cfg.training.warmup_steps:
            return step / max(1, cfg.training.warmup_steps)
        progress = (step - cfg.training.warmup_steps) / max(
            1, cfg.training.max_steps - cfg.training.warmup_steps)
        ratio = cfg.training.min_lr / cfg.training.learning_rate
        return ratio + 0.5 * (1 - ratio) * (1 + math.cos(math.pi * progress))

    running_loss = 0.0
    t0 = time.perf_counter()
    t_start = t0
    optimizer.zero_grad()

    train_iter = iter(train_loader)

    while global_step < cfg.training.max_steps:

        # LR update
        lr = cfg.training.learning_rate * get_lr(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        step_loss = 0.0
        for micro in range(cfg.training.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)

            with autocast_ctx:
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / cfg.training.gradient_accumulation_steps

            scaler.scale(loss).backward()
            step_loss += loss.item()

        # Gradient clip + optimizer step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        max_row_norm = None
        if projects:
            if (global_step + 1) % cfg.training.log_every == 0:
                max_row_norm = model.max_weight_row_norm()   # before projection
            model.project_parameters()

        global_step += 1
        running_loss += step_loss

        # Logging
        if global_step % cfg.training.log_every == 0:
            t1 = time.perf_counter()
            elapsed = t1 - t0
            tps = tokens_per_step * cfg.training.log_every / elapsed

            avg_loss = running_loss / cfg.training.log_every
            ppl = math.exp(min(avg_loss, 20))

            log.info(
                f"step {global_step:>6}/{cfg.training.max_steps} | "
                f"loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                f"lr {lr:.2e} | grad {grad_norm:.3f} | "
                f"{tps/1e3:.1f}K tok/s"
            )

            log_metrics(wandb_run, global_step, {
                "train/loss":       avg_loss,
                "train/perplexity": ppl,
                "train/lr":         lr,
                "train/grad_norm":  grad_norm.item(),
                "train/tokens_per_sec": tps,
                "train/step":       global_step,
                "train/tokens_seen": global_step * tokens_per_step,
                "train/elapsed_hours": (time.perf_counter() - t_start) / 3600,
                **({"train/max_row_norm_pre_proj": max_row_norm} if max_row_norm is not None else {}),
            })

            running_loss = 0.0
            t0 = time.perf_counter()

        # Evaluation
        if global_step % cfg.training.eval_every == 0:
            val_metrics = evaluate(
                model, val_loader, device, eval_periodic, autocast_ctx
            )
            log.info(f"  Val loss: {val_metrics['val/loss']:.4f} | "
                     f"ppl: {val_metrics['val/perplexity']:.1f}")
            log_metrics(wandb_run, global_step, val_metrics)

        # Checkpoint
        if global_step % cfg.training.save_every == 0:
            save_checkpoint(model, optimizer, global_step, global_step, cfg)

    # Final eval + checkpoint
    val_metrics = evaluate(model, val_loader, device, eval_final, autocast_ctx)
    wall_hours = (time.perf_counter() - t_start) / 3600
    log.info(f"Final val loss: {val_metrics['val/loss']:.4f} | "
             f"ppl: {val_metrics['val/perplexity']:.1f} | {wall_hours:.2f}h")
    final = {f"final/{k.split('/')[-1]}": v for k, v in val_metrics.items()}
    final["final/wall_hours"] = wall_hours
    final["final/tokens"] = global_step * tokens_per_step
    log_metrics(wandb_run, global_step, {**val_metrics, **final})
    if wandb_run:
        for k, v in final.items():
            wandb_run.run.summary[k] = v
    save_checkpoint(model, optimizer, global_step, global_step, cfg, tag="final")

    if wandb_run:
        wandb_run.finish()

    log.info("Training complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--trial", action="store_true",
                        help="Trial run: 1000 steps, no wandb, verbose")
    # Sweep overrides (applied after the YAML)
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak LR; min_lr is rescaled to keep the same ratio")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None, help="W&B group")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.lr is not None:
        cfg.training.min_lr = cfg.training.min_lr * (args.lr / cfg.training.learning_rate)
        cfg.training.learning_rate = args.lr
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.warmup_steps is not None:
        cfg.training.warmup_steps = args.warmup_steps
    if args.run_name is not None:
        cfg.logging.wandb_run_name = args.run_name
        if args.output_dir is None:
            cfg.logging.output_dir = str(Path(cfg.logging.output_dir).parent / args.run_name)
    if args.group is not None:
        cfg.logging.wandb_group = args.group
    if args.output_dir is not None:
        cfg.logging.output_dir = args.output_dir

    if args.wandb:
        cfg.logging.use_wandb = True
    if args.trial:
        cfg.training.max_steps = 1000
        cfg.training.eval_every = 100
        cfg.training.save_every = 500
        cfg.training.log_every = 10
        cfg.logging.use_wandb = False
        cfg.logging.output_dir = "./runs/candide18c_trial"
        log.info("Trial mode: 1000 steps, no wandb")

    Path(cfg.logging.output_dir).mkdir(parents=True, exist_ok=True)

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()