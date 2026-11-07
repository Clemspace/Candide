#!/usr/bin/env python3
"""
train_encoder18c.py — MLM pretraining for CandideEncoder on 18c French corpus.

Same data pipeline as train_candide18c.py (ShardLoader + pre-tokenized shards).
Swaps in MLMCollator and CandideEncoder instead of the causal LM setup.

Usage (from the repository root, PYTHONPATH=.):
    # Trial run
    python ramanujan/training/train_encoder18c.py --config configs/encoder18c.yaml --trial

    # ModernBERT-architecture run
    python ramanujan/training/train_encoder18c.py --config configs/encoder_modernbert18c.yaml --wandb

    # Resume
    python ramanujan/training/train_encoder18c.py --config configs/encoder18c.yaml \\
        --resume runs/.../checkpoints/step_5000.pt --wandb
"""

import argparse
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
# Mirrors train_candide18c.py. Every EncoderModelConfig field is passed to
# CandideEncoderConfig, so a new architecture option is one field here plus
# one key in the YAML.

@dataclass
class EncoderModelConfig:
    # Architecture (same as decoder)
    vocab_size: int = 32132
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = 4
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    norm_first: bool = True
    norm_type: str = "rms"
    attention_type: str = "gqa"   # 'mha' | 'gqa' | 'mqa' | 'alternating' (legacy)
    ffn_type: str = "swiglu"      # 'swiglu' | 'geglu' | 'gelu' | ...
    use_rope: bool = True
    rope_theta: float = 10000.0   # global layers
    bias: bool = False
    qk_norm: bool = False
    # Encoder-specific
    mask_token_id: int = 4
    add_pooler: bool = True
    classifier_dropout: float = 0.1
    # Legacy alternating blocks
    local_window_size: int = 128
    # Layer pattern (ModernBERT); None keeps the legacy blocks
    global_attn_every_n_layers: Optional[int] = None
    local_attention: int = 128            # total window: |i - j| <= local_attention // 2
    local_rope_theta: Optional[float] = None
    # Normalization / head / init
    norm_eps: Optional[float] = None
    norm_bias: bool = True
    embedding_norm: bool = False
    head_bias: bool = True
    decoder_bias: bool = True
    init_scheme: str = "normal"           # 'normal' | 'megatron'


@dataclass
class EncoderTrainingConfig:
    max_steps: int = 20000
    batch_size: int = 16
    gradient_accumulation_steps: int = 4   # effective batch = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    min_lr: float = 1e-5
    mixed_precision: str = "bf16"
    eval_every: int = 500
    eval_steps: Optional[int] = None       # None = whole fixed validation sample
    save_every: int = 2000
    log_every: int = 10
    seed: int = 42
    # MLM-specific
    mlm_probability: float = 0.15
    masking_strategy: str = "token"        # "token" or "span"
    mean_span_length: int = 3              # only used if strategy="span"
    sparse_prediction: bool = False        # MLM head on masked positions only
    val_mask_seed: int = 0                 # same validation masks at every evaluation


@dataclass
class DataConfig:
    shards_dir: str = "/data/pretrain_candide/shards"
    tokenizer_path: str = "/data/pretrain_candide/tokenizer/tokenizer.json"
    sequence_length: int = 512
    num_workers: int = 2
    prefetch_factor: int = 4
    pin_memory: bool = True
    train_order: str = "global"            # "global" | "sequential" (legacy)
    val_windows: Optional[int] = 1600      # None = legacy validation stream
    val_seed: int = 0
    pad_token_id: int = 1
    never_mask_tokens: List[str] = field(default_factory=lambda: ["<|bos|>", "<|eos|>"])


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "candide-18c"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "candide18c-encoder-mlm"
    wandb_group: Optional[str] = None
    wandb_tags: list = field(default_factory=lambda: [
        "candide", "18c-french", "encoder", "mlm",
    ])
    output_dir: str = "./runs/candide18c_encoder"


@dataclass
class FullEncoderConfig:
    model: EncoderModelConfig = field(default_factory=EncoderModelConfig)
    training: EncoderTrainingConfig = field(default_factory=EncoderTrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    allow_cpu: bool = False


SECTIONS = [
    ("model", EncoderModelConfig),
    ("training", EncoderTrainingConfig),
    ("data", DataConfig),
    ("logging", LoggingConfig),
]


def unknown_keys(raw: dict) -> List[str]:
    """YAML keys that no config field receives (they would be silently ignored)."""
    out = []
    for section, cls in SECTIONS:
        names = set(cls.__dataclass_fields__)
        out += [f"{section}.{k}" for k in (raw.get(section) or {}) if k not in names]
    return out


def load_config(path: str) -> FullEncoderConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    bad = unknown_keys(raw)
    if bad:
        raise ValueError(f"Unknown config keys in {path}: {bad}")
    cfg = FullEncoderConfig()
    for section, _ in SECTIONS:
        obj = getattr(cfg, section)
        for k, v in (raw.get(section) or {}).items():
            setattr(obj, k, v)
    return cfg


# ── Tokenizer checks ──────────────────────────────────────────────────────────

def resolve_special_ids(cfg: FullEncoderConfig) -> List[int]:
    """
    Check mask_token_id and vocab_size against the tokenizer and return the ids
    of data.never_mask_tokens. Raises if the tokenizer contradicts the config;
    warns and returns [] if the tokenizer cannot be read.
    """
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(cfg.data.tokenizer_path)
    except Exception as e:  # missing file or library: cannot check
        log.warning(f"Tokenizer not checked ({e}); BOS/EOS may be masked")
        return []

    n_vocab = tok.get_vocab_size(with_added_tokens=True)
    if n_vocab > cfg.model.vocab_size:
        raise ValueError(f"Tokenizer has {n_vocab} tokens > model vocab_size {cfg.model.vocab_size}")
    mask_tok = tok.id_to_token(cfg.model.mask_token_id)
    if mask_tok is None or "mask" not in mask_tok.lower():
        raise ValueError(
            f"mask_token_id {cfg.model.mask_token_id} is {mask_tok!r} in the tokenizer, "
            f"not a mask token"
        )
    ids = []
    for t in cfg.data.never_mask_tokens:
        i = tok.token_to_id(t)
        if i is None:
            log.warning(f"never_mask token {t!r} not in tokenizer")
        else:
            ids.append(i)
    log.info(f"Tokenizer: {n_vocab} tokens, mask={mask_tok!r} ({cfg.model.mask_token_id}), "
             f"never masked: {dict(zip(cfg.data.never_mask_tokens, ids))}")
    return ids


# ── Model ─────────────────────────────────────────────────────────────────────

def setup_model(cfg: EncoderModelConfig) -> nn.Module:
    from ramanujan.models.architectures.encoder import CandideEncoder, CandideEncoderConfig

    model = CandideEncoder(CandideEncoderConfig(**asdict(cfg)))
    total = sum(p.numel() for p in model.parameters())
    pattern = (f"global every {cfg.global_attn_every_n_layers}, local window {cfg.local_attention}"
               if cfg.global_attn_every_n_layers else f"attention={cfg.attention_type}")
    log.info(f"CandideEncoder: {total/1e6:.1f}M params")
    log.info(f"  {cfg.n_layers}L × {cfg.d_model}d, {cfg.n_heads}H ({cfg.n_kv_heads} KV), "
             f"{cfg.ffn_type} d_ff={cfg.d_ff}, {cfg.norm_type} norm, {pattern}")
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
        betas=(0.9, 0.98),
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
    path = out_dir / (f"{tag}.pt" if tag else f"step_{step}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_config": model.config.to_dict(),   # CandideEncoder.from_pretrained reads this
        "config": asdict(cfg),                     # plain dicts: loads with weights_only=True
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


# ── MLM loss ──────────────────────────────────────────────────────────────────

def mlm_loss(model, mlm_batch: dict, device, sparse: bool, autocast_ctx):
    """
    Summed MLM cross-entropy over masked positions.

    Returns (loss_sum, n_masked, n_correct) as tensors. With sparse=True the
    head runs only on masked positions; the loss is the same as the dense path.
    """
    input_ids = mlm_batch["input_ids"].to(device, non_blocking=True)
    labels    = mlm_batch["labels"].to(device, non_blocking=True)
    attn_mask = mlm_batch["attention_mask"].to(device, non_blocking=True)

    with autocast_ctx:
        if sparse:
            masked = labels != -100
            logits, _ = model(input_ids, attention_mask=attn_mask, masked_positions=masked)
            targets = labels[masked]
        else:
            logits, _ = model(input_ids, attention_mask=attn_mask)
            logits = logits.reshape(-1, logits.size(-1))
            targets = labels.reshape(-1)
        loss_sum = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")

    valid = targets != -100
    n_correct = ((logits.detach().argmax(-1) == targets) & valid).sum()
    return loss_sum, valid.sum(), n_correct


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, collator, device, n_steps: Optional[int], autocast_ctx,
             sparse: bool = False, mask_seed: int = 0) -> dict:
    """
    Token-weighted MLM loss and accuracy over masked positions.

    Masks are drawn from a CPU generator reseeded with mask_seed at every call,
    so with the fixed validation sample every evaluation (and every run) scores
    the same windows with the same masks. n_steps=None reads the whole loader.
    The training RNG state is restored afterwards.
    """
    model.eval()
    total_loss, total_masked, total_correct = 0.0, 0, 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(mask_seed)
        for i, batch in enumerate(val_loader):
            if n_steps is not None and i >= n_steps:
                break
            loss_sum, n_masked, n_correct = mlm_loss(
                model, collator(batch["input_ids"]), device, sparse, autocast_ctx)
            total_loss += loss_sum.item()
            total_masked += int(n_masked)
            total_correct += int(n_correct)
    model.train()
    avg_loss = total_loss / max(1, total_masked)
    return {
        "val/loss":          avg_loss,
        "val/perplexity":    math.exp(min(avg_loss, 20)),
        "val/masked_acc":    total_correct / max(1, total_masked),
        "val/masked_tokens": total_masked,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: FullEncoderConfig, resume: Optional[str] = None):
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not cfg.allow_cpu:
        raise RuntimeError("CUDA unavailable (check nvidia-smi); pass --allow-cpu to run on CPU")
    log.info(f"Device: {device}")
    if cfg.data.sequence_length > cfg.model.max_seq_len:
        raise ValueError(f"sequence_length {cfg.data.sequence_length} > max_seq_len {cfg.model.max_seq_len}")

    from ramanujan.data.loaders.mlm_collator import MLMCollator
    from ramanujan.data.loaders.shard_loader import ShardLoader

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

    collator = MLMCollator(
        vocab_size=cfg.model.vocab_size,
        mask_token_id=cfg.model.mask_token_id,
        pad_token_id=cfg.data.pad_token_id,
        special_token_ids=set(resolve_special_ids(cfg)),
        mlm_probability=cfg.training.mlm_probability,
        strategy=cfg.training.masking_strategy,
        mean_span_length=cfg.training.mean_span_length,
    )
    log.info(f"MLM collator: strategy={cfg.training.masking_strategy}, "
             f"p={cfg.training.mlm_probability}, sparse_prediction={cfg.training.sparse_prediction}")

    model = setup_model(cfg.model).to(device)
    optimizer = setup_optimizer(model, cfg.training)

    global_step = 0
    if resume:
        global_step = load_checkpoint(resume, model, optimizer)

    dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    autocast_ctx = torch.amp.autocast(device.type, dtype=dtype, enabled=(device.type == "cuda"))

    wandb_run = None
    if cfg.logging.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                name=cfg.logging.wandb_run_name,
                group=cfg.logging.wandb_group,
                tags=cfg.logging.wandb_tags,
                config=asdict(cfg),
            )
            wandb.watch(model, log="gradients", log_freq=200)
            log.info(f"WandB: {wandb_run.url}")
        except ImportError:
            log.warning("wandb not installed")

    accum = cfg.training.gradient_accumulation_steps
    sparse = cfg.training.sparse_prediction
    eff_batch = cfg.training.batch_size * accum
    tokens_per_step = eff_batch * cfg.data.sequence_length

    log.info(f"Effective batch: {eff_batch} ({cfg.training.batch_size} × {accum})")
    log.info(f"Tokens/step: {tokens_per_step:,}")
    log.info(f"Training for {cfg.training.max_steps:,} steps "
             f"= {cfg.training.max_steps * tokens_per_step / 1e9:.2f}B tokens")

    def run_eval(n_steps):
        return evaluate(model, val_loader, collator, device, n_steps, autocast_ctx,
                        sparse=sparse, mask_seed=cfg.training.val_mask_seed)

    running_loss = running_acc = 0.0
    t_start = t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)

    while global_step < cfg.training.max_steps:
        lr = cfg.training.learning_rate * get_lr(global_step, cfg.training)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        step_loss = 0.0
        step_masked, step_correct = 0, 0

        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            loss_sum, n_masked, n_correct = mlm_loss(
                model, collator(batch["input_ids"]), device, sparse, autocast_ctx)
            # Per-micro-batch mean over masked tokens, averaged over micro-batches
            loss = loss_sum / n_masked.clamp(min=1) / accum
            scaler.scale(loss).backward()
            step_loss += loss.item()
            step_masked += int(n_masked)
            step_correct += int(n_correct)

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
                    "train/elapsed_hours":  (t1 - t_start) / 3600,
                }, step=global_step)
            running_loss = running_acc = 0.0
            t0 = time.perf_counter()

        if global_step % cfg.training.eval_every == 0:
            val = run_eval(cfg.training.eval_steps)
            log.info(f"  val loss {val['val/loss']:.4f} | ppl {val['val/perplexity']:.1f} | "
                     f"acc {val['val/masked_acc']:.3f} | {val['val/masked_tokens']:,} masked tokens")
            if wandb_run:
                wandb_run.log(val, step=global_step)

        if global_step % cfg.training.save_every == 0:
            save_checkpoint(model, optimizer, global_step, cfg)

    # Final: same sample and masks as the periodic evaluations when eval_steps is None
    val = run_eval(cfg.training.eval_steps * 4 if cfg.training.eval_steps else None)
    wall_hours = (time.perf_counter() - t_start) / 3600
    log.info(f"Final — val loss {val['val/loss']:.4f} | ppl {val['val/perplexity']:.1f} | "
             f"acc {val['val/masked_acc']:.3f} | {wall_hours:.2f}h")
    save_checkpoint(model, optimizer, global_step, cfg, tag="final")

    if wandb_run:
        wandb_run.log(val, step=global_step)
        wandb_run.summary.update({
            "final/loss": val["val/loss"],
            "final/masked_acc": val["val/masked_acc"],
            "final/wall_hours": wall_hours,
            "final/tokens": global_step * tokens_per_step,
        })
        wandb_run.finish()
    log.info("Encoder pretraining complete.")
    return val


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--trial", action="store_true",
                        help="1000 steps, no wandb")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--lr", type=float, default=None,
                        help="override learning_rate; min_lr keeps its ratio")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.allow_cpu = args.allow_cpu

    if args.lr is not None:
        ratio = cfg.training.min_lr / cfg.training.learning_rate
        cfg.training.learning_rate = args.lr
        cfg.training.min_lr = args.lr * ratio
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.run_name:
        cfg.logging.wandb_run_name = args.run_name
    if args.group:
        cfg.logging.wandb_group = args.group
    if args.output_dir:
        cfg.logging.output_dir = args.output_dir
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
