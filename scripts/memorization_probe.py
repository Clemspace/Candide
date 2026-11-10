#!/usr/bin/env python3
"""
Memorization probe: how well do trained checkpoints reproduce training text,
as a function of how often that text is repeated in the corpus?

For each repetition bucket (a 50-token passage occurring 1, 2-9, 10-99 or
100+ times in the training shards), sample passages, give the model the 64
tokens that precede each one, and measure on the 50 target tokens:

  loss        mean cross-entropy (nats/token)
  acc         fraction of tokens where the model's top prediction is correct
  extracted   fraction of passages reproduced exactly by greedy decoding.
              Greedy decoding reproduces a passage iff the top prediction is
              correct at every position given the true prefix, so this is
              all(acc) per passage; no generation loop is needed.

"Extracted" is the verbatim-extraction test of Carlini et al. (2021, 2023)
restricted to training passages; it doubles as the pre-release memorization
check for a model trained on a closed corpus.

    PYTHONPATH=. python scripts/memorization_probe.py \\
        --shards /data/pretrain_candide/shards --per-bucket 300 \\
        --run gptplus=runs/pilot/<gptplus run>/checkpoints/final.pt \\
        --run angpt=runs/pilot/<angpt run>/checkpoints/final.pt

Repetition counts come from the same content-defined 50-gram sampling as
scripts/dup_audit.py (one CPU pass over the train shards, a few minutes).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dup_audit", _here / "dup_audit.py")
dup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dup)

BUCKETS = {"1": (1, 1), "2-9": (2, 9), "10-99": (10, 99), "100+": (100, 10**12)}


# ── Passage selection (numpy only) ────────────────────────────────────────────

def index_kgrams(shards, k: int, shift: int, block: int = 20_000_000):
    """Sampled k-gram hashes with their location: (hashes, shard index, position)."""
    hs, sh, pos = [], [], []
    for si, path in enumerate(shards):
        tokens = dup.load_shard(path)
        for start in range(0, max(1, len(tokens) - k + 1), block):
            chunk = np.asarray(tokens[start: start + block + k - 1])
            h = dup.kgram_hashes(chunk, k)
            keep = np.flatnonzero(dup.sample_mask(h, shift))
            hs.append(h[keep])
            pos.append((keep + start).astype(np.int64))
            sh.append(np.full(len(keep), si, dtype=np.int32))
    return np.concatenate(hs), np.concatenate(sh), np.concatenate(pos)


def select_passages(hashes, shard_idx, positions, per_bucket: int, min_pos: int, seed: int = 0):
    """
    For each bucket, up to `per_bucket` distinct k-grams with that corpus count,
    each located at its first occurrence with at least `min_pos` tokens before it.
    Returns {bucket: [(shard_idx, position, count), ...]}.
    """
    order = np.lexsort((positions, shard_idx, hashes))     # by hash, then location
    h, s, p = hashes[order], shard_idx[order], positions[order]
    uniq, first, counts = np.unique(h, return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    out = {}
    for name, (lo, hi) in BUCKETS.items():
        cand = np.flatnonzero((counts >= lo) & (counts <= hi))
        cand = cand[p[first[cand]] >= min_pos]
        pick = rng.choice(cand, size=min(per_bucket, len(cand)), replace=False) if len(cand) else []
        out[name] = [(int(s[first[i]]), int(p[first[i]]), int(counts[i])) for i in pick]
    return out


def passage_tokens(shards, sel, prefix: int, k: int) -> np.ndarray:
    """(n, prefix + k) token windows: `prefix` context tokens then the k-token passage."""
    mm = {}
    rows = []
    for si, pos, _ in sel:
        if si not in mm:
            mm[si] = dup.load_shard(shards[si])
        rows.append(np.asarray(mm[si][pos - prefix: pos + k], dtype=np.int64))
    return np.stack(rows) if rows else np.zeros((0, prefix + k), dtype=np.int64)


# ── Model side (torch) ────────────────────────────────────────────────────────

def load_run(path: str, device):
    """Rebuild a train_candide18c model from its checkpoint (config pickled with the run)."""
    import __main__
    import torch
    T = importlib.import_module("ramanujan.training.train_candide18c")
    for name in ("FullConfig", "ModelConfig", "TrainingConfig", "DataConfig", "LoggingConfig"):
        setattr(__main__, name, getattr(T, name))   # the run pickled them as __main__.*
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg["model"] if isinstance(cfg, dict) else cfg.model
    model = T.setup_model(model_cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def score(model, windows: np.ndarray, prefix: int, device, batch: int = 64) -> dict:
    import torch
    import torch.nn.functional as F
    losses, accs, exact = [], [], []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" \
        else torch.autocast("cpu", enabled=False)
    with torch.no_grad(), amp:
        for i in range(0, len(windows), batch):
            x = torch.from_numpy(windows[i: i + batch]).to(device)
            logits = model(x[:, :-1]).float()
            target = x[:, 1:]
            sl = slice(prefix - 1, None)                    # predictions of the passage tokens
            lg, tg = logits[:, sl], target[:, sl]
            nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1),
                                  reduction="none").view(tg.shape)
            hit = lg.argmax(-1) == tg
            losses.append(nll.mean(1).cpu())
            accs.append(hit.float().mean(1).cpu())
            exact.append(hit.all(1).float().cpu())
    if not losses:
        return {"n": 0}
    loss, acc, ex = (torch.cat(t) for t in (losses, accs, exact))
    return {"n": len(loss), "loss": loss.mean().item(), "acc": acc.mean().item(),
            "extracted": ex.mean().item()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True)
    p.add_argument("--run", action="append", required=True, help="name=path/to/checkpoint.pt")
    p.add_argument("--per-bucket", type=int, default=300)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--prefix", type=int, default=64)
    p.add_argument("--sample-shift", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="memorization_probe.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)   # fail-safe before long work

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shards = sorted((Path(args.shards) / "train").glob("shard_*.bin"))
    print("indexing 50-grams ...", flush=True)
    sel = select_passages(*index_kgrams(shards, args.k, args.sample_shift),
                          per_bucket=args.per_bucket, min_pos=args.prefix, seed=args.seed)
    windows = {b: passage_tokens(shards, s, args.prefix, args.k) for b, s in sel.items()}

    report = {"k": args.k, "prefix": args.prefix,
              "buckets": {b: len(s) for b, s in sel.items()}, "runs": {}}
    for spec in args.run:
        name, path = spec.split("=", 1)
        model = load_run(path, device)
        report["runs"][name] = {b: score(model, w, args.prefix, device) for b, w in windows.items()}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n{'run':<12}{'bucket':<8}{'n':>5}{'loss':>8}{'acc':>8}{'extracted':>11}")
    for name, res in report["runs"].items():
        for b, r in res.items():
            if r["n"]:
                print(f"{name:<12}{b:<8}{r['n']:>5}{r['loss']:>8.3f}{r['acc']:>8.3f}{r['extracted']:>11.3f}")
    print(f"\nreport: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
