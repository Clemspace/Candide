#!/usr/bin/env python3
"""
Duplication audit of pre-tokenized shards (CPU only, no GPU).

Measures how much of the training stream is repeated text, and how much of
the validation sample also occurs in training, using content-defined sampling
of token K-grams:

  1. every K-gram (K consecutive tokens, at every offset) gets a 64-bit
     polynomial hash;
  2. a K-gram is kept iff its mixed hash falls in a 1/2^s slice; the choice
     depends only on the K-gram's content, so a repeated K-gram is sampled
     at every position where it occurs, whatever its alignment;
  3. counting the kept hashes estimates, token-weighted, the fraction of the
     corpus covered by K-grams that occur more than once.

K = 50 follows Lee et al. (2022), "Deduplicating Training Data Makes Language
Models Better" (exact-substring threshold of 50 tokens).

    python scripts/dup_audit.py --shards /data/pretrain_candide/shards \\
        --tokenizer /data/pretrain_candide/tokenizer/tokenizer.json

Memory: about 1 GB per 20M-token block plus ~8 bytes per kept hash.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

MAGIC = 20240520
HEADER = 1024
BASE = np.uint64(1_000_003)
MIX = np.uint64(0x9E3779B97F4A7C15)


def load_shard(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER), dtype=np.int32)
    assert header[0] == MAGIC, f"bad magic in {path}"
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER, shape=(int(header[1]),))


def kgram_hashes(tokens: np.ndarray, k: int) -> np.ndarray:
    """Polynomial hash (mod 2^64) of every k-gram: h_i = sum_j t[i+j] * BASE^(k-1-j)."""
    n = len(tokens) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    t = tokens.astype(np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h *= BASE
            h += t[j: j + n]
    return h


def sample_mask(h: np.ndarray, shift: int) -> np.ndarray:
    """Content-defined 1/2^shift sample: keep hashes whose mixed top bits are zero."""
    with np.errstate(over="ignore"):
        return ((h * MIX) >> np.uint64(64 - shift)) == 0


def sampled_hashes(tokens: np.ndarray, k: int, shift: int, block: int = 20_000_000) -> np.ndarray:
    """Kept k-gram hashes of a token array, processed in overlapping blocks."""
    out = []
    for start in range(0, max(1, len(tokens) - k + 1), block):
        chunk = np.asarray(tokens[start: start + block + k - 1])
        h = kgram_hashes(chunk, k)
        out.append(h[sample_mask(h, shift)])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint64)


def duplicate_stats(hashes: np.ndarray) -> dict:
    """Token-weighted duplicate fraction and the count distribution of kept k-grams."""
    if len(hashes) == 0:
        return {"kept": 0, "dup_fraction": 0.0}
    uniq, counts = np.unique(hashes, return_counts=True)
    occ = counts[np.searchsorted(uniq, hashes)]        # occurrences of each kept position's k-gram
    return {
        "kept": int(len(hashes)),
        "distinct": int(len(uniq)),
        "dup_fraction": float((occ > 1).mean()),       # positions whose k-gram occurs elsewhere
        "excess_fraction": float(1 - len(uniq) / len(hashes)),   # removable by exact dedup
        "positions_by_count": {
            "1": float((occ == 1).mean()),
            "2-9": float(((occ >= 2) & (occ <= 9)).mean()),
            "10-99": float(((occ >= 10) & (occ <= 99)).mean()),
            "100+": float((occ >= 100).mean()),
        },
    }


def top_repeated(hashes: np.ndarray, n: int):
    uniq, counts = np.unique(hashes, return_counts=True)
    order = np.argsort(-counts)[:n]
    return uniq[order], counts[order]


def find_kgrams(shards, targets: np.ndarray, k: int, block: int = 20_000_000) -> dict:
    """One pass over the shards: tokens of the first occurrence of each target hash."""
    found, remaining = {}, set(int(x) for x in targets)
    for path in shards:
        tokens = load_shard(path)
        for start in range(0, max(1, len(tokens) - k + 1), block):
            chunk = np.asarray(tokens[start: start + block + k - 1])
            h = kgram_hashes(chunk, k)
            for i in np.flatnonzero(np.isin(h, targets)):
                key = int(h[i])
                if key in remaining:
                    found[key] = chunk[i: i + k]
                    remaining.discard(key)
            if not remaining:
                return found
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True, help="dir with train/ and val/")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--sample-shift", type=int, default=5, help="keep 1/2^s of k-grams (5 -> 1/32)")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--tokenizer", default=None, help="tokenizer.json, to print repeated text")
    p.add_argument("--out", default="dup_audit.json")
    args = p.parse_args()

    root = Path(args.shards)
    train = sorted((root / "train").glob("shard_*.bin"))
    val = sorted((root / "val").glob("shard_*.bin"))
    t0 = time.time()

    per_shard, train_hashes = [], []
    for s in train:
        h = sampled_hashes(load_shard(s), args.k, args.sample_shift)
        train_hashes.append(h)
        per_shard.append({"shard": s.name, **duplicate_stats(h)})
        print(f"{s.name}: within-shard dup {per_shard[-1]['dup_fraction']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    train_all = np.concatenate(train_hashes)
    train_stats = duplicate_stats(train_all)

    val_all = np.concatenate([sampled_hashes(load_shard(s), args.k, args.sample_shift) for s in val])
    val_stats = duplicate_stats(val_all)
    train_set = np.unique(train_all)
    val_in_train = float(np.isin(val_all, train_set).mean()) if len(val_all) else 0.0

    report = {"k": args.k, "sample": f"1/{2 ** args.sample_shift}",
              "train": train_stats, "val": val_stats,
              "val_kgrams_also_in_train": val_in_train, "train_shards": per_shard}

    hashes, counts = top_repeated(train_all, args.top)
    tok = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
    texts = find_kgrams(train, hashes, args.k) if tok is not None else {}
    top = []
    for h, c in zip(hashes, counts):
        entry = {"count_in_sample": int(c)}
        if tok is not None and int(h) in texts:
            entry["text"] = tok.decode(texts[int(h)].tolist())
        top.append(entry)
    report["top_repeated"] = top

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("train", "val", "val_kgrams_also_in_train")}, indent=2))
    print(f"\nfull report: {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
