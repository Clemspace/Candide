#!/usr/bin/env python3
"""
train_encoder18c.py — MLM pretraining for CandideEncoder on 18c French corpus.

Same data pipeline as train_candide18c.py (ShardLoader + pre-tokenized shards).
Swaps in MLMCollator and CandideEncoder instead of the causal LM setup.

Usage:
    # Trial run
    python train_encoder18c.py --config configs/candide18c_encoder.yaml --trial

    # Full run
    python train_encoder18c.py --config configs/candide18c_encoder.yaml --wandb

    # Resume
    python train_encoder18c.py --config configs/candide18c_encoder.yaml \\
        --resume checkpoints/encoder_step_5000.pt --wandb
"""

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
# Mirrors train_candide18c.py config structure exactly.
# Only ModelConfig gains encoder-specific fields.

@dataclass
class EncoderModelConfig:
    # Architecture (same as decoder)
    vocab_size: int = 32132
    d_model: int = 768
    n_layers: int = 12          # 12 layers for encoder (more than decoder's 8)
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 3072
    max_seq_len: int = 512      # Encoders typically use shorter sequences
    dropout: float = 0.1
    norm_first: bool = True
    norm_type: str = "rms"
    attention_type: str = "gqa"
    ffn_type: str = "swiglu"
    use_rope: bool = True
    rope_theta: float = 10000.0
    bias: bool = False
    # Encoder-specific
    mask_token_id: int = 4      # [MASK] token in your BPE vocab
    add_pooler: bool = True
    classifier_dropout: float = 0.1


@dataclass
class EncoderTrainingConfig:
    max_steps: int = 20000
    batch_size: int = 16        # Encoder can use larger batches (no KV cache)
    gradient_accumulation_steps: int = 4   # effective batch = 64
    learning_rate: float = 1e-4            # Lower LR than decoder
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    min_lr: float = 1e-5
    mixed_precision: str = "bf16"
    eval_every: int = 500
    eval_steps: int = 50
    save_every: int = 2000
    log_every: int = 10
    seed: int = 42
    # MLM-specific
    mlm_probability: float = 0.15
    masking_strategy: str = "token"        # "token" or "span"
    mean_span_length: int = 3              # only used if strategy="span"


@dataclass
class DataConfig:
    shards_dir: str = "/data/pretrain_candide/shards"
    tokenizer_path: str = "/data/pretrain_candide/tokenizer/tokenizer.json"
    sequence_length: int = 512
    num_workers: int = 2
    prefetch_factor: int = 4
    pin_memory: bool = True


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "candide-18c"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "candide18c-encoder-mlm"
    wandb_tags: list = field(default_factory=lambda: [
        "candide", "18c-french", "encoder", "mlm", "modernbert-style"
    ])
    output_dir: str = "./runs/candide18c_encoder"


@dataclass
class FullEncoderConfig:
    model: EncoderModelConfig = field(default_factory=EncoderModelConfig)
    training: EncoderTrainingConfig = field(default_factory=EncoderTrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str) -> FullEncoderConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = FullEncoderConfig()
    for section, cls in [
        ("model", EncoderModelConfig),
        ("training", EncoderTrainingConfig),
        ("data", DataConfig),
        ("logging", LoggingConfig),
    ]:
        if section in raw:
            obj = getattr(cfg, section)
            for k, v in raw[section].items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
                else:
                    log.warning(f"Unknown config key [{section}].{k} — skipping")
    return cfg


# ── Model ─────────────────────────────────────────────────────────────────────

def setup_model(cfg: EncoderModelConfig) -> nn.Module:
    from ramanujan.models.architectures.encoder import CandideEncoder, CandideEncoderConfig

    enc_cfg = CandideEncoderConfig(
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
        mask_token_id=cfg.mask_token_id,
        add_pooler=cfg.add_pooler,
        classifier_dropout=cfg.classifier_dropout,
    )

    model = CandideEncoder(enc_cfg)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"CandideEncoder: {total/1e6:.1f}M params")
    log.info(f"  Architecture: {cfg.n_layers}L × {cfg.d_model}d, "
             f"{cfg.n_heads}H ({cfg.n_kv_heads} KV), {cfg.d_ff}d_ff")
    log.info(f"  Bidirectional attention (no causal mask)")
    log.info(f"  MLM head tied to token embeddings")
    return model


# ── Optimizer ─────────────────────────────────────────────────────────────────

def setup_optimizer(model: nn.Module, cfg: EncoderTrainingConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name for nd in ["bias", "norm", "ln_"]):
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate,
        betas=(0.9, 0.98),   # 0.98 beta2 is standard for BERT-style encoders
        eps=1e-6,
        fused=torch.cuda.is_available(),
    )


def get_lr(step: int, cfg: EncoderTrainingConfig) -> float:
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    ratio = cfg.min_lr / cfg.learning_rate
    return ratio + 0.5 * (1 - ratio) * (1 + math.cos(math.pi * progress))


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step, cfg, tag=""):
    out_dir = Path(cfg.logging.output_dir) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{tag}.pt" if tag else f"step_{step}.pt"
    path = out_dir / name
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.__dict__,
    }, path)
    log.info(f"Checkpoint: {path}")
    return path


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    log.info(f"Resumed from step {step}")
    return step


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, collator, device, n_steps, autocast_ctx) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, total_masked, total_correct = 0.0, 0, 0

    for i, batch in enumerate(val_loader):
        if i >= n_steps:
            break
        raw_ids = batch["input_ids"]
        mlm_batch = collator(raw_ids)

        input_ids = mlm_batch["input_ids"].to(device, non_blocking=True)
        labels    = mlm_batch["labels"].to(device, non_blocking=True)
        attn_mask = mlm_batch["attention_mask"].to(device, non_blocking=True)

        with autocast_ctx:
            logits, _ = model(input_ids, attention_mask=attn_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()

        # Masked token accuracy
        masked_pos = labels != -100
        if masked_pos.any():
            preds = logits.argmax(-1)
            total_correct += (preds[masked_pos] == labels[masked_pos]).sum().item()
            total_masked  += masked_pos.sum().item()

    model.train()
    n_seen = min(i + 1, n_steps) if total_loss else 0
    avg_loss = total_loss / max(1, n_seen)
    accuracy = total_correct / max(1, total_masked)
    return {
        "val/loss":          avg_loss,
        "val/perplexity":    math.exp(min(avg_loss, 20)),
        "val/masked_acc":    accuracy,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: FullEncoderConfig, resume: Optional[str] = None):
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Data — same ShardLoader as the decoder script
    sys.path.insert(0, str(Path(__file__).parent))
    from ramanujan.data.loaders.mlm_collator import MLMCollator

    from shard_loader import ShardLoader

    log.info("Setting up data loaders...")
    loader = ShardLoader(
        data_dir=cfg.data.shards_dir,
        seq_len=cfg.data.sequence_length,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        pin_memory=cfg.data.pin_memory,
        seed=cfg.training.seed,
    )
    train_loader, val_loader = loader.get_dataloaders()

    # MLM collator — applied on-the-fly each batch (not pre-processed)
    collator = MLMCollator(
        vocab_size=cfg.model.vocab_size,
        mask_token_id=cfg.model.mask_token_id,
        pad_token_id=1,
        mlm_probability=cfg.training.mlm_probability,
        strategy=cfg.training.masking_strategy,
        mean_span_length=cfg.training.mean_span_length,
    )
    log.info(f"MLM collator: strategy={cfg.training.masking_strategy}, "
             f"p={cfg.training.mlm_probability}")

    # Model + optimizer
    model = setup_model(cfg.model).to(device)
    optimizer = setup_optimizer(model, cfg.training)

    global_step = 0
    if resume:
        global_step = load_checkpoint(resume, model, optimizer)

    # Mixed precision
    dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype, enabled=(device.type == "cuda"))

    # WandB
    wandb_run = None
    if cfg.logging.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                name=cfg.logging.wandb_run_name,
                tags=cfg.logging.wandb_tags,
                config={
                    "model": cfg.model.__dict__,
                    "training": cfg.training.__dict__,
                },
            )
            wandb.watch(model, log="gradients", log_freq=200)
            log.info(f"WandB: {wandb_run.url}")
        except ImportError:
            log.warning("wandb not installed")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    eff_batch = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    tokens_per_step = eff_batch * cfg.data.sequence_length

    log.info(f"Effective batch: {eff_batch} "
             f"({cfg.training.batch_size} × {cfg.training.gradient_accumulation_steps})")
    log.info(f"Tokens/step: {tokens_per_step:,}")
    log.info(f"Training for {cfg.training.max_steps:,} steps "
             f"= {cfg.training.max_steps * tokens_per_step / 1e9:.2f}B tokens")

    running_loss = 0.0
    running_acc  = 0.0
    t0 = time.perf_counter()
    optimizer.zero_grad()
    train_iter = iter(train_loader)

    while global_step < cfg.training.max_steps:

        lr = cfg.training.learning_rate * get_lr(global_step, cfg.training)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        step_loss = 0.0
        step_masked, step_correct = 0, 0

        for micro in range(cfg.training.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Apply MLM masking on CPU before moving to device
            mlm_batch = collator(batch["input_ids"])

            input_ids = mlm_batch["input_ids"].to(device, non_blocking=True)
            labels    = mlm_batch["labels"].to(device, non_blocking=True)
            attn_mask = mlm_batch["attention_mask"].to(device, non_blocking=True)

            with autocast_ctx:
                logits, _ = model(input_ids, attention_mask=attn_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / cfg.training.gradient_accumulation_steps

            scaler.scale(loss).backward()
            step_loss += loss.item()

            # Track masked accuracy
            with torch.no_grad():
                masked_pos = labels != -100
                if masked_pos.any():
                    preds = logits.detach().argmax(-1)
                    step_correct += (preds[masked_pos] == labels[masked_pos]).sum().item()
                    step_masked  += masked_pos.sum().item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        running_loss += step_loss
        running_acc  += step_correct / max(1, step_masked)

        if global_step % cfg.training.log_every == 0:
            t1 = time.perf_counter()
            tps = tokens_per_step * cfg.training.log_every / (t1 - t0)
            avg_loss = running_loss / cfg.training.log_every
            avg_acc  = running_acc  / cfg.training.log_every
            ppl = math.exp(min(avg_loss, 20))

            log.info(
                f"step {global_step:>6}/{cfg.training.max_steps} | "
                f"loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                f"acc {avg_acc:.3f} | lr {lr:.2e} | "
                f"grad {grad_norm:.3f} | {tps/1e3:.1f}K tok/s"
            )

            if wandb_run:
                wandb_run.log({
                    "train/loss":           avg_loss,
                    "train/perplexity":     ppl,
                    "train/masked_acc":     avg_acc,
                    "train/lr":             lr,
                    "train/grad_norm":      grad_norm.item(),
                    "train/tokens_per_sec": tps,
                    "train/tokens_seen":    global_step * tokens_per_step,
                }, step=global_step)

            running_loss = running_acc = 0.0
            t0 = time.perf_counter()

        if global_step % cfg.training.eval_every == 0:
            val_metrics = evaluate(
                model, val_loader, collator, device,
                cfg.training.eval_steps, autocast_ctx
            )
            log.info(f"  val loss {val_metrics['val/loss']:.4f} | "
                     f"ppl {val_metrics['val/perplexity']:.1f} | "
                     f"acc {val_metrics['val/masked_acc']:.3f}")
            if wandb_run:
                wandb_run.log(val_metrics, step=global_step)

        if global_step % cfg.training.save_every == 0:
            save_checkpoint(model, optimizer, global_step, cfg)

    # Final
    val_metrics = evaluate(model, val_loader, collator, device,
                           cfg.training.eval_steps * 4, autocast_ctx)
    log.info(f"Final — val loss {val_metrics['val/loss']:.4f} | "
             f"ppl {val_metrics['val/perplexity']:.1f} | "
             f"acc {val_metrics['val/masked_acc']:.3f}")
    save_checkpoint(model, optimizer, global_step, cfg, tag="final")

    if wandb_run:
        wandb_run.finish()
    log.info("Encoder pretraining complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--trial", action="store_true",
                        help="1000 steps, no wandb, verbose logging")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.wandb:
        cfg.logging.use_wandb = True
    if args.trial:
        cfg.training.max_steps = 1000
        cfg.training.eval_every = 100
        cfg.training.save_every = 500
        cfg.training.log_every = 10
        cfg.logging.use_wandb = False
        cfg.logging.output_dir = "./runs/candide18c_encoder_trial"
        log.info("Trial mode: 1000 steps")

    Path(cfg.logging.output_dir).mkdir(parents=True, exist_ok=True)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()