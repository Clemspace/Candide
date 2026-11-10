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

    # Resume (exact: same data order, RNG, scaler, W&B run); "auto" = latest step_*.pt of the run
    python train_candide18c.py --config configs/candide18c.yaml --resume checkpoints/step_5000.pt --wandb
    python train_candide18c.py --config configs/candide18c.yaml --resume auto --wandb

    # WSD cooldown branched off a stable-phase checkpoint (scaling-law points)
    python train_candide18c.py --config configs/full.yaml --schedule wsd \
        --resume runs/full/checkpoints/step_20000.pt --cooldown-start 20000 --max-steps 24000 \
        --output-dir runs/full-cool20k --run-name full-cool20k

    # Pilot sweep arm (overrides for LR / seed / budget / naming)
    python train_candide18c.py --config configs/pilot/angpt.yaml --wandb \
        --lr 2e-3 --seed 1 --max-steps 1526 --run-name angpt-lr2e-3-s1 --group lr-sweep
"""

import argparse
import logging
import math
import os
import re
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
    # LR schedule (ramanujan/training/schedules.py). "cosine" = the pilot runs.
    # "wsd": warmup, constant peak, then a cooldown over the last cooldown_frac
    # of the steps (or from cooldown_start) to final_lr_ratio * peak.
    schedule: str = "cosine"
    cooldown_frac: float = 0.2
    cooldown_shape: str = "1-sqrt"
    final_lr_ratio: float = 0.0
    cooldown_start: Optional[int] = None
    # Checkpoint retention (None = keep every step_N.pt, as before).
    # keep_last: newest step checkpoints kept for resuming; keep_steps: steps
    # never deleted (stable-phase branch points for cooldowns). pre_cooldown.pt
    # and final.pt are never deleted.
    keep_last: Optional[int] = None
    keep_steps: list = field(default_factory=list)
    # Optimizer: "adamw" (the pilots) or "muon" (hidden matrices; embeddings, head
    # and 1-D parameters stay on AdamW inside the same optimizer). With
    # muon_adjust_lr "match_rms" the Muon update has the RMS of an AdamW update,
    # so learning_rate and weight_decay are on the AdamW scale (Moonshot, 2025).
    optimizer: str = "adamw"
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_adjust_lr: str = "match_rms"
    # Throughput: torch.compile the model (checkpoints always hold plain weights)
    compile: bool = False

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
    # Mixtures (None = shards_dir/train alone, as in the pilots).
    # sources: name -> root dir with train/ shards; "main" defaults to shards_dir.
    # mix: weights of the stable phase; cooldown_mix: weights from the start of
    # the WSD cooldown (None = keep mix). Validation stays shards_dir/val.
    sources: Optional[dict] = None
    mix: Optional[dict] = None
    cooldown_mix: Optional[dict] = None
    # name -> root dir with val/ shards, each scored on its own fixed sample
    extra_val: Optional[dict] = None

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

from ramanujan.data.loaders.shard_loader import FixedWindowDataset, ShardLoader  # noqa: E402
from ramanujan.training.schedules import lr_multiplier  # noqa: E402


def resolve_sources(cfg: "DataConfig") -> dict:
    """Mixture sources, with "main" = shards_dir unless given explicitly."""
    sources = dict(cfg.sources or {})
    sources.setdefault("main", cfg.shards_dir)
    return sources


def phase_mix(cfg: "DataConfig", phase: int) -> Optional[dict]:
    """Mixture weights of a phase (0 stable, 1 cooldown); None = single shard set."""
    if phase == 1 and cfg.cooldown_mix:
        return cfg.cooldown_mix
    return cfg.mix


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

def setup_muon(model: nn.Module, cfg: TrainingConfig):
    """
    Muon for attention and FFN matrices; AdamW (inside MuonOptimizer) for the
    embedding / tied head (decayed, as in the AdamW baseline) and for 1-D
    parameters (not decayed).
    """
    from ramanujan.training.optimizers.muon import MuonOptimizer, muon_param_groups
    groups = muon_param_groups(model)
    muon = [g for g in groups if g["use_muon"]]
    adam = [p for g in groups if not g["use_muon"] for p in g["params"]]
    matrices = [p for p in adam if p.ndim >= 2]
    vectors = [p for p in adam if p.ndim < 2]
    if matrices:
        muon.append({"params": matrices, "use_muon": False, "adamw_weight_decay": cfg.weight_decay})
    if vectors:
        muon.append({"params": vectors, "use_muon": False, "adamw_weight_decay": 0.0})
    n_muon = sum(p.numel() for g in muon if g["use_muon"] for p in g["params"])
    n_all = sum(p.numel() for g in muon for p in g["params"])
    log.info(f"Muon: {n_muon/1e6:.1f}M params on Muon, {(n_all - n_muon)/1e6:.1f}M on AdamW "
             f"(adjust_lr={cfg.muon_adjust_lr})")
    return MuonOptimizer(muon, lr=cfg.learning_rate, momentum=cfg.muon_momentum,
                         ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay,
                         adjust_lr=cfg.muon_adjust_lr, adamw_betas=(0.9, 0.95), adamw_eps=1e-8)


def setup_optimizer(model: nn.Module, cfg: TrainingConfig, arch: str = "gpt"):
    if getattr(cfg, "optimizer", "adamw") == "muon":
        if arch != "gpt":
            raise ValueError("optimizer=muon is wired for arch=gpt only")
        return setup_muon(model, cfg)
    if getattr(cfg, "optimizer", "adamw") != "adamw":
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}. Available: 'adamw', 'muon'")
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


def cooldown_start_step(cfg: TrainingConfig) -> Optional[int]:
    """First step of the WSD cooldown (None for cosine); same rule as lr_multiplier."""
    if cfg.schedule != "wsd":
        return None
    start = cfg.cooldown_start
    if start is None:
        start = cfg.max_steps - int(round(cfg.cooldown_frac * cfg.max_steps))
    return max(start, cfg.warmup_steps)


def lr_factor(step: int, cfg: TrainingConfig) -> float:
    """Multiplier of the peak LR at `step` (cosine floor min_lr, WSD floor final_lr_ratio)."""
    return lr_multiplier(step, cfg.max_steps, cfg.warmup_steps, schedule=cfg.schedule,
                         min_ratio=cfg.min_lr / cfg.learning_rate,
                         cooldown_shape=cfg.cooldown_shape, final_ratio=cfg.final_lr_ratio,
                         cooldown_start=cooldown_start_step(cfg))


def phase_at(step: int, cfg: FullConfig) -> int:
    """Data phase at `step`: 1 from the WSD cooldown start when a cooldown mix is set."""
    cs = cooldown_start_step(cfg.training)
    return int(cs is not None and bool(cfg.data.cooldown_mix) and step >= cs)


# ── WandB ─────────────────────────────────────────────────────────────────────

def setup_wandb(cfg: FullConfig, model: nn.Module, run_id: Optional[str] = None):
    if not cfg.logging.use_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            id=run_id,                                   # resume: continue the same W&B run
            resume="allow" if run_id else None,
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

def save_checkpoint(model, optimizer, scheduler_step, step, cfg, tag="", extra: Optional[dict] = None):
    """
    Model, optimizer, step and config, plus `extra` (scaler, RNG states, W&B id)
    for an exact resume. Written to a temporary file then renamed, so a crash
    while saving never leaves a truncated checkpoint.
    """
    out_dir = Path(cfg.logging.output_dir) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"step_{step}{tag}.pt" if not tag else f"{tag}.pt"
    path = out_dir / name
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "config":    cfg.__dict__,
        **(extra or {}),
    }, tmp)
    os.replace(tmp, path)
    log.info(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(path, model, optimizer) -> tuple:
    """Restore model and optimizer; returns (step, full checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    log.info(f"Resumed from step {step}: {path}")
    return step, ckpt


def latest_checkpoint(output_dir: str) -> Optional[Path]:
    """The step_N.pt with the largest N in output_dir/checkpoints (None if there is none)."""
    found = []
    for f in (Path(output_dir) / "checkpoints").glob("step_*.pt"):
        m = re.fullmatch(r"step_(\d+)\.pt", f.name)
        if m:
            found.append((int(m.group(1)), f))
    return max(found)[1] if found else None


def prune_checkpoints(output_dir: str, keep_last: Optional[int], keep_steps=()) -> list:
    """Delete step_N.pt except the keep_last newest and those in keep_steps; returns deleted paths."""
    if keep_last is None:
        return []
    found = []
    for f in (Path(output_dir) / "checkpoints").glob("step_*.pt"):
        m = re.fullmatch(r"step_(\d+)\.pt", f.name)
        if m:
            found.append((int(m.group(1)), f))
    found.sort()
    keep = {step for step, _ in found[-keep_last:]} if keep_last > 0 else set()
    keep |= set(int(k) for k in keep_steps)
    deleted = []
    for step, f in found:
        if step not in keep:
            f.unlink()
            deleted.append(f)
    return deleted


def rng_state() -> dict:
    state = {"torch_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(ckpt: dict):
    if "torch_rng" in ckpt:
        torch.set_rng_state(ckpt["torch_rng"])
    if "cuda_rng" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])


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


def source_windows(cfg: FullConfig, names) -> dict:
    """Training windows available in each named source (from shard headers only)."""
    from ramanujan.data.loaders.shard_loader import read_shard_header, windows_per_shard
    sources = resolve_sources(cfg.data)
    out = {}
    for n in names:
        shards = sorted((Path(sources[n]) / "train").glob("shard_*.bin"))
        out[n] = sum(windows_per_shard(read_shard_header(f)["n_tokens"], cfg.data.sequence_length)
                     for f in shards)
    return out


def mixture_passes(cfg: FullConfig, windows: dict, tokens_per_step: int,
                   cooldown_start: Optional[int]) -> dict:
    """Expected passes over each source in each phase: windows drawn / windows available."""
    steps = {0: cfg.training.max_steps}
    if cfg.data.cooldown_mix and cooldown_start is not None:
        steps = {0: cooldown_start, 1: cfg.training.max_steps - cooldown_start}
    per_step = tokens_per_step // cfg.data.sequence_length
    out = {}
    for ph, n_steps in steps.items():
        mix = {n: w for n, w in phase_mix(cfg.data, ph).items() if w > 0}
        total = sum(mix.values())
        out[ph] = {n: w / total * n_steps * per_step / max(1, windows[n]) for n, w in mix.items()}
    return out


def log_mixture_passes(cfg: FullConfig, tokens_per_step: int, cooldown_start: Optional[int]):
    if not cfg.data.mix:
        return
    try:
        names = set(cfg.data.mix) | set(cfg.data.cooldown_mix or {})
        passes = mixture_passes(cfg, source_windows(cfg, names), tokens_per_step, cooldown_start)
        for ph, per in passes.items():
            log.info(f"Phase {ph} expected passes: " + ", ".join(f"{n} {x:.2f}x" for n, x in per.items()))
    except Exception as e:  # informative only: never block a run
        log.warning(f"could not compute mixture passes: {e}")


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(cfg: FullConfig, resume: Optional[str] = None):
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    # fp32 matmuls outside autocast (if any) on tensor cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    accum = cfg.training.gradient_accumulation_steps
    cooldown_start = cooldown_start_step(cfg.training)
    if cfg.data.cooldown_mix and cooldown_start is None:
        raise ValueError("data.cooldown_mix needs training.schedule: wsd")
    if cfg.data.mix or cfg.data.cooldown_mix:
        known = resolve_sources(cfg.data)
        bad = [n for n in {**(cfg.data.mix or {}), **(cfg.data.cooldown_mix or {})} if n not in known]
        if bad:
            raise ValueError(f"mix names unknown sources {bad}; data.sources has {list(known)}")
        if cfg.data.cooldown_mix and not cfg.data.mix:
            raise ValueError("data.cooldown_mix needs data.mix for the stable phase (e.g. {main: 1.0})")

    def make_loader(phase: int, start_batch: int):
        """Loaders of a data phase, resumed after start_batch micro-batches of that phase."""
        mix = phase_mix(cfg.data, phase)
        return ShardLoader(
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
            train_sources=resolve_sources(cfg.data) if mix else None,
            train_mix=mix,
            phase=phase,
            start_batch=start_batch,
        )

    # Model
    log.info("Building model...")
    model = setup_model(cfg.model).to(device)

    # Optimizer
    optimizer = setup_optimizer(model, cfg.training, arch=cfg.model.arch)
    projects = hasattr(model, "project_parameters")

    # Mixed precision
    dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype, enabled=(device.type == "cuda"))

    # Resume: weights, optimizer, scaler, RNG; the data position follows from the step
    global_step, ckpt = 0, {}
    if resume:
        global_step, ckpt = load_checkpoint(resume, model, optimizer)
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        set_rng_state(ckpt)
        if "rng_after_init" not in ckpt and global_step > 0:
            log.warning("checkpoint without RNG state (older run): dropout draws will differ")
    if projects:
        model.project_parameters()   # weight rows into the unit ball before step 0

    # torch.compile: `model` runs compiled; `raw_model` is what is saved and projected
    raw_model = model
    if cfg.training.compile:
        log.info("torch.compile(model)")
        model = torch.compile(raw_model)

    start_step = global_step

    # Data
    log.info("Setting up data loaders...")
    phase = phase_at(global_step, cfg)
    phase_start = cooldown_start if phase == 1 else 0
    loader = make_loader(phase, (global_step - phase_start) * accum)
    train_loader, val_loader = loader.get_dataloaders()
    # Fixed sample: every evaluation reads all of it. Legacy stream: capped.
    eval_periodic = None if cfg.data.val_windows else cfg.training.eval_steps
    eval_final = None if cfg.data.val_windows else cfg.training.eval_steps * 4
    extra_val = {}
    for name, root in (cfg.data.extra_val or {}).items():
        ds = FixedWindowDataset(Path(root) / "val", cfg.data.sequence_length,
                                cfg.data.val_windows or 1600, cfg.data.val_seed)
        extra_val[name] = torch.utils.data.DataLoader(
            ds, batch_size=cfg.training.batch_size, shuffle=False,
            generator=torch.Generator().manual_seed(cfg.data.val_seed))

    def evaluate_all(n_steps):
        metrics = evaluate(model, val_loader, device, n_steps, autocast_ctx)
        for name, vl in extra_val.items():
            m = evaluate(model, vl, device, None, autocast_ctx)
            metrics.update({k.replace("val/", f"val_{name}/"): v for k, v in m.items()})
        return metrics

    # WandB
    wandb_run = setup_wandb(cfg, raw_model, run_id=ckpt.get("wandb_id"))

    def checkpoint(tag=""):
        extra = {"scaler": scaler.state_dict(), "rng_after_init": True, **rng_state(),
                 "phase": phase, "schedule": cfg.training.schedule,
                 "cooldown_start": cooldown_start}
        if wandb_run is not None and getattr(wandb_run, "run", None) is not None:
            extra["wandb_id"] = wandb_run.run.id
        return save_checkpoint(raw_model, optimizer, global_step, global_step, cfg, tag=tag, extra=extra)

    criterion = nn.CrossEntropyLoss()
    eff_batch  = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    tokens_per_step = eff_batch * cfg.data.sequence_length

    log.info(f"Effective batch size: {eff_batch} "
             f"({cfg.training.batch_size} × {cfg.training.gradient_accumulation_steps} accum)")
    log.info(f"Tokens per step: {tokens_per_step:,}")
    log.info(f"Training for {cfg.training.max_steps:,} steps "
             f"= {cfg.training.max_steps * tokens_per_step / 1e9:.2f}B tokens")
    log.info(f"Schedule: {cfg.training.schedule}"
             + (f", cooldown from step {cooldown_start:,}" if cooldown_start is not None else ""))
    log_mixture_passes(cfg, tokens_per_step, cooldown_start)
    log.info("Starting training...")

    running_loss = 0.0
    t0 = time.perf_counter()
    t_start = t0
    optimizer.zero_grad()

    train_iter = iter(train_loader)

    while global_step < cfg.training.max_steps:

        # WSD cooldown start: keep a branch point, then switch to the cooldown mix
        if cooldown_start is not None and global_step == cooldown_start and global_step > start_step:
            if not (Path(cfg.logging.output_dir) / "checkpoints" / "pre_cooldown.pt").exists():
                checkpoint(tag="pre_cooldown")
            if phase == 0 and phase_at(global_step, cfg) == 1:
                log.info(f"Step {global_step}: cooldown starts, data mix -> {cfg.data.cooldown_mix}")
                del train_iter, train_loader
                phase = 1
                loader = make_loader(phase, 0)
                train_loader, _ = loader.get_dataloaders()
                train_iter = iter(train_loader)

        # LR update
        lr = cfg.training.learning_rate * lr_factor(global_step, cfg.training)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation (loss summed on the device: no host sync per micro-batch)
        step_loss = torch.zeros((), device=device)
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
            step_loss += loss.detach().float()

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
                max_row_norm = raw_model.max_weight_row_norm()   # before projection
            raw_model.project_parameters()

        global_step += 1
        running_loss += step_loss

        # Logging
        if global_step % cfg.training.log_every == 0:
            t1 = time.perf_counter()
            elapsed = t1 - t0
            tps = tokens_per_step * cfg.training.log_every / elapsed

            avg_loss = float(running_loss) / cfg.training.log_every
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
                "train/phase":      phase,
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
            val_metrics = evaluate_all(eval_periodic)
            extras = "".join(f" | {n} {val_metrics[f'val_{n}/loss']:.4f}" for n in extra_val)
            log.info(f"  Val loss: {val_metrics['val/loss']:.4f} | "
                     f"ppl: {val_metrics['val/perplexity']:.1f}{extras}")
            log_metrics(wandb_run, global_step, val_metrics)

        # Checkpoint
        if global_step % cfg.training.save_every == 0:
            checkpoint()
            for f in prune_checkpoints(cfg.logging.output_dir, cfg.training.keep_last,
                                       cfg.training.keep_steps):
                log.info(f"Checkpoint deleted (retention): {f.name}")

    # Final eval + checkpoint
    val_metrics = evaluate_all(eval_final)
    wall_hours = (time.perf_counter() - t_start) / 3600
    log.info(f"Final val loss: {val_metrics['val/loss']:.4f} | "
             f"ppl: {val_metrics['val/perplexity']:.1f} | {wall_hours:.2f}h")
    # val/loss -> final/loss, val_enc/loss -> final_enc/loss
    final = {"final" + k[len("val"):]: v for k, v in val_metrics.items()}
    final["final/wall_hours"] = wall_hours
    final["final/tokens"] = global_step * tokens_per_step
    log_metrics(wandb_run, global_step, {**val_metrics, **final})
    if wandb_run:
        for k, v in final.items():
            wandb_run.run.summary[k] = v
    checkpoint(tag="final")

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
    parser.add_argument("--schedule", choices=["cosine", "wsd"], default=None)
    parser.add_argument("--cooldown-start", type=int, default=None,
                        help="WSD: first cooldown step (branch a cooldown off a stable checkpoint)")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default=None)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="per-step micro-batch; gradient accumulation is rescaled to keep tokens/step")
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
    if args.schedule is not None:
        cfg.training.schedule = args.schedule
    if args.cooldown_start is not None:
        cfg.training.cooldown_start = args.cooldown_start
    if args.optimizer is not None:
        cfg.training.optimizer = args.optimizer
    if args.compile:
        cfg.training.compile = True
    if args.micro_batch is not None:
        total = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
        if total % args.micro_batch:
            parser.error(f"--micro-batch {args.micro_batch} does not divide {total} sequences per step")
        cfg.training.batch_size = args.micro_batch
        cfg.training.gradient_accumulation_steps = total // args.micro_batch

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

    resume = args.resume
    if resume == "auto":
        found = latest_checkpoint(cfg.logging.output_dir)
        log.info(f"--resume auto: {found if found else 'no checkpoint, starting from step 0'}")
        resume = str(found) if found else None

    train(cfg, resume=resume)


if __name__ == "__main__":
    main()