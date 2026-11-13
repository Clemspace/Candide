"""Checkpoint discovery and loading, driven by the configuration pickled in each checkpoint.

A checkpoint written by ramanujan/training/train_candide18c.py holds
{"model": state_dict, "optimizer": ..., "step": int, "config": FullConfig.__dict__, ...}.
The model is rebuilt from config["model"] through the trainer's own setup_model, as
scripts/memorization_probe.load_run does, so any architecture the trainer can build
(GPT+, anGPT, any shape) loads without being described here. Nothing is loaded with
strict=False: a key mismatch is an error, not a warning.

Discovery and file listing need no PyTorch; reading metadata and loading do.
"""
from __future__ import annotations

import importlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_STEP = re.compile(r"step_(\d+)")
_CONFIG_CLASSES = ("FullConfig", "ModelConfig", "TrainingConfig", "DataConfig", "LoggingConfig")


# ── Discovery (no torch) ─────────────────────────────────────────────────────

@dataclass
class CheckpointRef:
    id: str                 # stable key used by the API: "<run>/<file stem>"
    path: Path
    run: str
    name: str               # file stem: step_76000, pre_cooldown, final
    size_mb: float
    mtime: float

    @property
    def step_from_name(self) -> Optional[int]:
        m = _STEP.fullmatch(self.name)
        return int(m.group(1)) if m else None

    def sort_key(self):
        s = self.step_from_name
        rank = {"pre_cooldown": 1, "final": 2}.get(self.name, 1 if s is None else 0)
        return (self.run, rank if s is None else 0, s if s is not None else 0, self.name)


def discover(roots: Iterable[str | Path]) -> list[CheckpointRef]:
    """Every *.pt under a `checkpoints/` directory of the given roots (or the files given)."""
    found: dict[Path, CheckpointRef] = {}
    for root in roots:
        root = Path(root).expanduser()
        if root.is_file() and root.suffix == ".pt":
            paths, base = [root], root.parent.parent if root.parent.name == "checkpoints" else root.parent
        elif root.is_dir():
            paths, base = [p for p in root.rglob("*.pt") if p.parent.name == "checkpoints"], root
        else:
            continue
        for p in paths:
            p = p.resolve()
            run_dir = p.parent.parent if p.parent.name == "checkpoints" else p.parent
            try:
                run = str(run_dir.relative_to(base.resolve())) if run_dir != base.resolve() else run_dir.name
            except ValueError:
                run = run_dir.name
            run = run if run not in ("", ".") else run_dir.name
            st = p.stat()
            found[p] = CheckpointRef(f"{run}/{p.stem}", p, run, p.stem, st.st_size / 1e6, st.st_mtime)
    refs = sorted(found.values(), key=CheckpointRef.sort_key)
    seen: dict[str, int] = {}
    for r in refs:                                   # two roots holding the same run name
        n = seen.get(r.id, 0)
        seen[r.id] = n + 1
        if n:
            r.id = f"{r.id}~{n}"
    return refs


# ── Configuration ────────────────────────────────────────────────────────────

def trainer_module():
    """The training script as a module, with its config classes registered where the
    checkpoint pickled them (the run executed as __main__)."""
    import __main__
    T = importlib.import_module("ramanujan.training.train_candide18c")
    for name in _CONFIG_CLASSES:
        if not hasattr(__main__, name):
            setattr(__main__, name, getattr(T, name))
    return T


def _as_dict(section) -> dict:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    return dict(vars(section))


def _torch_load(path: Path):
    import torch
    trainer_module()            # before unpickling: the run pickled its config classes as __main__.*
    try:            # mmap: reading the config of a 2.7 GB checkpoint touches only the header
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _sections(ckpt: dict):
    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError("checkpoint has no 'config': not written by train_candide18c.py")
    if isinstance(cfg, dict):
        return cfg.get("model"), cfg.get("training"), cfg.get("data")
    return getattr(cfg, "model", None), getattr(cfg, "training", None), getattr(cfg, "data", None)


def _state(ckpt: dict) -> dict:
    state = ckpt.get("model")
    if state is None:
        raise ValueError("checkpoint has no 'model' state dict")
    return {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in state.items()}


def describe(ckpt: dict, ref: CheckpointRef) -> dict:
    """Everything the UI shows about a checkpoint, read from the checkpoint alone."""
    T = trainer_module()
    model_cfg, train_cfg, data_cfg = _sections(ckpt)
    m, t, d = _as_dict(model_cfg), _as_dict(train_cfg), _as_dict(data_cfg)
    state = _state(ckpt)
    step = int(ckpt.get("step", ref.step_from_name or 0))

    n_params = sum(v.numel() for v in state.values())
    if m.get("tie_word_embeddings") and "lm_head.weight" in state:
        n_params -= state["lm_head.weight"].numel()
    emb = state.get("token_embedding.embedding.weight")

    if isinstance(train_cfg, dict):
        tc = T.TrainingConfig()
        for k, v in train_cfg.items():
            setattr(tc, k, v)
        train_cfg = tc
    try:
        schedule = _schedule(T, train_cfg, t, d, step) if train_cfg is not None and t else {}
    except Exception as e:                      # metadata must not fail on an older config
        schedule = {"error": f"{type(e).__name__}: {e}"}

    return {
        "id": ref.id, "run": ref.run, "name": ref.name, "path": str(ref.path), "size_mb": round(ref.size_mb, 1),
        "step": step,
        "model": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in m.items()},
        "n_params": n_params,
        "vocab_size": int(emb.shape[0]) if emb is not None else m.get("vocab_size"),
        "train": schedule,
        "data": {k: d.get(k) for k in ("shards_dir", "tokenizer_path", "mix", "cooldown_mix", "sequence_length")
                 if k in d},
        "wandb_id": ckpt.get("wandb_id"),
        "has_optimizer": "optimizer" in ckpt,
    }


def _schedule(T, train_cfg, t: dict, d: dict, step: int) -> dict:
    """Phase and learning-rate factor at `step`, by the trainer's own rules."""
    cs = T.cooldown_start_step(train_cfg)
    warm = int(t.get("warmup_steps", 0))
    if step < warm:
        phase = "warmup"
    elif cs is not None and step >= cs:
        phase = "cooldown"
    else:
        phase = "stable" if cs is not None else "cosine"
    schedule = {
        "schedule": t.get("schedule"), "max_steps": t.get("max_steps"), "warmup_steps": warm,
        "cooldown_start": cs, "phase": phase,
        "lr_peak": t.get("learning_rate"), "lr_factor": round(T.lr_factor(step, train_cfg), 4),
        "optimizer": t.get("optimizer"), "tokens_per_step": _tokens_per_step(t, d),
    }
    if schedule["tokens_per_step"]:
        schedule["tokens_seen"] = step * schedule["tokens_per_step"]
    return schedule


def _tokens_per_step(t: dict, d: dict) -> Optional[int]:
    try:
        return int(t["batch_size"]) * int(t["gradient_accumulation_steps"]) * int(d.get("sequence_length", 1024))
    except (KeyError, TypeError, ValueError):
        return None


def build(ckpt: dict, device: str):
    """The model of a checkpoint, rebuilt by the trainer and loaded strictly."""
    T = trainer_module()
    model_cfg, _, _ = _sections(ckpt)
    if isinstance(model_cfg, dict):
        mc = T.ModelConfig()
        for k, v in model_cfg.items():
            setattr(mc, k, v)
        model_cfg = mc
    model = T.setup_model(model_cfg)
    model.load_state_dict(_state(ckpt))      # strict: a mismatch means the wrong builder
    return model.to(device).eval()


# ── Registry ─────────────────────────────────────────────────────────────────

@dataclass
class Loaded:
    model: object
    meta: dict
    ref: CheckpointRef


class Registry:
    """Discovered checkpoints, their metadata (cached by file mtime), and a small LRU of
    loaded models. One lock: a single GPU serves one request at a time."""

    def __init__(self, roots: Iterable[str | Path], device: str = "cpu", max_loaded: int = 3):
        self.roots = [Path(r) for r in roots]
        self.device = device
        self.max_loaded = max_loaded
        self.lock = threading.RLock()
        self._refs: dict[str, CheckpointRef] = {}
        self._meta: dict[str, tuple[float, dict]] = {}
        self._models: "OrderedDict[str, Loaded]" = OrderedDict()
        self.refresh()

    def refresh(self) -> list[CheckpointRef]:
        with self.lock:
            refs = discover(self.roots)
            self._refs = {r.id: r for r in refs}
            return refs

    def refs(self) -> list[CheckpointRef]:
        return list(self._refs.values())

    def ref(self, ckpt_id: str) -> CheckpointRef:
        if ckpt_id not in self._refs:
            self.refresh()
        if ckpt_id not in self._refs:
            raise KeyError(f"unknown checkpoint {ckpt_id!r}")
        return self._refs[ckpt_id]

    def meta(self, ckpt_id: str) -> dict:
        ref = self.ref(ckpt_id)
        with self.lock:
            cached = self._meta.get(ckpt_id)
            if cached and cached[0] == ref.mtime:
                return cached[1]
            ckpt = _torch_load(ref.path)
            meta = describe(ckpt, ref)
            del ckpt
            self._meta[ckpt_id] = (ref.mtime, meta)
            return meta

    def get(self, ckpt_id: str) -> Loaded:
        ref = self.ref(ckpt_id)
        with self.lock:
            hit = self._models.get(ckpt_id)
            if hit is not None and hit.ref.mtime == ref.mtime:
                self._models.move_to_end(ckpt_id)
                return hit
            while len(self._models) >= self.max_loaded:
                self._models.popitem(last=False)
                _empty_cache(self.device)
            ckpt = _torch_load(ref.path)
            meta = describe(ckpt, ref)
            model = build(ckpt, self.device)
            del ckpt
            meta["n_params_built"] = sum(p.numel() for p in model.parameters())
            self._meta[ckpt_id] = (ref.mtime, meta)
            loaded = Loaded(model, meta, ref)
            self._models[ckpt_id] = loaded
            return loaded

    def loaded_ids(self) -> list[str]:
        return list(self._models)


def _empty_cache(device: str):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()


def git_commit(repo: Path) -> Optional[str]:
    """Commit of the code serving the probe (recorded with every saved result)."""
    head = repo / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            target = repo / ".git" / ref[5:]
            if target.exists():
                return target.read_text().strip()[:7]
            packed = repo / ".git" / "packed-refs"
            for line in packed.read_text().splitlines():
                if line.endswith(ref[5:]):
                    return line.split()[0][:7]
            return None
        return ref[:7]
    except OSError:
        return None


def dirty(repo: Path) -> Optional[bool]:
    """True if tracked files differ from HEAD (None if git is unavailable)."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, timeout=5)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


__all__ = ["CheckpointRef", "discover", "describe", "build", "Registry", "trainer_module", "git_commit", "dirty"]
