#!/usr/bin/env python3
"""
Watch an encoder run and score each kept checkpoint on FreEM_LPM tagging.

Runs beside pretraining (same GPU, separate process). For each checkpoint in
--steps (plus pre_cooldown.pt and final.pt) that exists and has not been
scored, it runs scripts/pos_freem.py in a subprocess with the quick protocol
and writes <out-dir>/<checkpoint>.json. Baselines given with --baselines are
scored once with the same protocol, so the curve and the baselines are
comparable. It stops after final.pt is scored.

The quick protocol (3 epochs at 3e-5) is for the progression curve only; the
table of the paper uses pos_freem.py's default protocol and three seeds.

Usage (in tmux, next to training):
  PYTHONPATH=. python scripts/pos_watch.py \
      --run-dir runs/enc/candide-encoder-v2g-qk --freem ~/data/FreEMlpm \
      --baselines hf:pjox/dalembert,hf:almanach/camembert-base --wandb
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

DEFAULT_STEPS = "1000,2000,5000,10000,20000,30000,40000"


def candidates(ckpt_dir: Path, steps: List[int]) -> List[Path]:
    """Existing checkpoints to score (listed steps, pre_cooldown.pt, final.pt), oldest first."""
    names = [f"step_{s}.pt" for s in steps] + ["pre_cooldown.pt", "final.pt"]
    found = [ckpt_dir / n for n in names if (ckpt_dir / n).exists()]
    return sorted(found, key=lambda f: f.stat().st_mtime)


def is_stable(path: Path, min_age: float) -> bool:
    """A checkpoint is complete once its file has not changed for min_age seconds."""
    try:
        return time.time() - path.stat().st_mtime >= min_age
    except FileNotFoundError:
        return False


def result_name(spec: str) -> str:
    """File stem for a model spec: the checkpoint name, or the HF name with / -> __."""
    kind, _, name = spec.partition(":")
    if kind == "candide":
        return Path(name).stem
    return "baseline_" + re.sub(r"[^A-Za-z0-9_.-]", "_", name.replace("/", "__"))


def score(spec: str, out: Path, args) -> Optional[dict]:
    cmd = [sys.executable, str(Path(__file__).with_name("pos_freem.py")),
           "--model", spec, "--tokenizer", args.tokenizer,
           "--train", str(args.freem / "Models/Train_4/training_data/train/pos_full.tsv"),
           "--dev", str(args.freem / "Models/Train_4/training_data/dev/pos_full.tsv"),
           "--test-root", str(args.freem / "Data_outOfDomain"),
           "--out", str(out), "--seed", str(args.seed), "--lr", str(args.lr),
           "--epochs", str(args.epochs)]
    if args.max_train_sentences:
        cmd += ["--max-train-sentences", str(args.max_train_sentences)]
    if args.overlap:
        cmd += ["--overlap", str(args.overlap)]
    print(f"[watch] scoring {spec}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0 or not out.exists():
        print(f"[watch] FAILED ({proc.returncode}): {spec}", flush=True)
        return None
    return json.loads(out.read_text())


def headline(r: dict) -> dict:
    """Metrics logged per checkpoint: per-century accuracy (original spelling first)."""
    m = {f"pos/{k}": v["acc"] for k, v in r["test_centuries"].items()}
    orig = [v["acc"] for k, v in r["test_centuries"].items() if k.startswith("original/")]
    if orig:
        m["pos/original_mean"] = sum(orig) / len(orig)
    clean = (r.get("test_clean") or {}).get("centuries", {})
    m.update({f"pos_clean/{k}": v["acc"] for k, v in clean.items()})
    corig = [v["acc"] for k, v in clean.items() if k.startswith("original/")]
    if corig:
        m["pos_clean/original_mean"] = sum(corig) / len(corig)
    m["pos/dev_final"] = r["dev"][-1]["dev_acc"]
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--freem", required=True, type=Path, help="clone of FreEM-corpora/FreEMlpm (LFS pulled)")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/pos")
    p.add_argument("--steps", default=DEFAULT_STEPS)
    p.add_argument("--baselines", default="", help="comma-separated hf:<name> specs")
    p.add_argument("--tokenizer", default="/data/pretrain_candide/tokenizer_v2g/tokenizer.json")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-train-sentences", type=int, default=None)
    p.add_argument("--overlap", default=None, help="overlap_freem.py JSON, passed to pos_freem.py")
    p.add_argument("--poll", type=int, default=120, help="seconds between scans")
    p.add_argument("--min-age", type=int, default=60, help="seconds a checkpoint must be unchanged")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="candide-18c")
    p.add_argument("--wandb-name", default=None, help="default: <run-dir name>-pos")
    args = p.parse_args()

    ckpt_dir = args.run_dir / "checkpoints"
    out_dir = args.out_dir or args.run_dir / "pos"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in args.steps.split(",") if s]

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name or f"{args.run_dir.name}-pos",
                         group="encoder-pos", config=vars(args) | {"run_dir": str(args.run_dir)})
        run.define_metric("pretrain_step")
        run.define_metric("pos/*", step_metric="pretrain_step")

    for spec in [b for b in args.baselines.split(",") if b]:
        out = out_dir / f"{result_name(spec)}.json"
        r = json.loads(out.read_text()) if out.exists() else score(spec, out, args)
        if r and run:
            for k, v in headline(r).items():
                run.summary[f"baseline/{result_name(spec)}/{k}"] = v

    while True:
        done_final = False
        for ckpt in candidates(ckpt_dir, steps):
            out = out_dir / f"{ckpt.stem}.json"
            if out.exists():
                done_final |= ckpt.name == "final.pt"
                continue
            if not is_stable(ckpt, args.min_age):
                continue
            r = score(f"candide:{ckpt}", out, args)
            if r is None:
                continue
            if run:
                run.log({"pretrain_step": r["pretraining_step"], **headline(r)})
            done_final |= ckpt.name == "final.pt"
        if done_final:
            print("[watch] final.pt scored; done", flush=True)
            break
        time.sleep(args.poll)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
