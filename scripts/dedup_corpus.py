#!/usr/bin/env python3
"""
dedup_corpus.py — Document-level deduplication of the extracted ModERN corpus,
before tokenization, with a clean validation split.

Runs on the files that prepare_shards.py would select (same sub-corpus weights
and seed), and writes a manifest that prepare_shards.py --manifest consumes.

Method
------
Text of each file is cleaned (HTML entities), lowercased, whitespace-collapsed
and encoded as UTF-8 bytes. Every window of K bytes (default 200, about 50
tokens) gets a 64-bit polynomial hash; a window is kept iff its mixed hash
falls in a 1/2^s slice (content-defined sampling: a repeated window is kept
wherever it occurs). Files are ordered by sub-corpus priority (canon, press,
dicos, pamphlets, lettres, archive), then path.

  intra_dup    fraction of a file's sampled windows that repeat inside the same
               file (excess copies / total). >= --max-intra-dup: dropped as
               "internal_rep" (pages or blocks extracted several times).
  containment  fraction of a file's distinct sampled windows that occur in an
               earlier file. >= --max-containment: dropped as "near_dup" (the
               earlier file is the copy kept). One correction pass then
               recomputes containment against kept files only and restores any
               file that drops below the threshold (its only "source" was itself
               dropped).
  duplicate_of the earlier file sharing the most windows with it.
  shared       among kept files, fraction of a file's distinct windows that
               occur in any other kept file.

Validation split: per sub-corpus, files with shared <= --val-max-shared and
at least --val-min-chars characters are drawn at random until the sub-corpus
reaches val_fraction of its kept characters, so the validation mix matches the
training mix and validation text does not occur in training.

    python scripts/dedup_corpus.py --extracted-dir /data/pretrain_candide/extracted \\
        --out-dir /data/pretrain_candide/dedup_v2 --workers 16
"""

import argparse
import csv
import json
import sys
import time
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_shards as ps  # noqa: E402

BASE = 0x100000001B3                      # odd, so invertible mod 2^64
MASK64 = (1 << 64) - 1
MIX = np.uint64(0x9E3779B97F4A7C15)


def _inverse_mod_2_64(a: int) -> int:
    inv = a
    for _ in range(6):                     # Newton iteration doubles correct bits
        inv = (inv * (2 - a * inv)) & MASK64
    assert (a * inv) & MASK64 == 1
    return inv


BASE_INV = _inverse_mod_2_64(BASE)


# ── Hashing ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> bytes:
    text = unicodedata.normalize("NFC", ps.clean_text(text)).lower()
    return " ".join(text.split()).encode("utf-8")


def _powers(base: int, n: int) -> np.ndarray:
    """[base^0, base^1, ..., base^(n-1)] mod 2^64."""
    out = np.empty(n, dtype=np.uint64)
    if n == 0:
        return out
    out[0] = 1
    if n > 1:
        out[1:] = np.cumprod(np.full(n - 1, base, dtype=np.uint64))   # wraps mod 2^64
    return out


def window_hashes(data: bytes, k: int) -> np.ndarray:
    """
    h_i = sum_{j=0}^{k-1} c_{i+j} BASE^(k-1-j) mod 2^64 for every window, in O(n):
    with Q_m = sum_{j<m} c_j BASE^-j, h_i = (Q_{i+k} - Q_i) BASE^(i+k-1).
    """
    n = len(data)
    if n < k:
        return np.zeros(0, dtype=np.uint64)
    c = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    with np.errstate(over="ignore"):
        q = np.zeros(n + 1, dtype=np.uint64)
        np.cumsum(c * _powers(BASE_INV, n), out=q[1:])
        return (q[k:] - q[:-k]) * _powers(BASE, n)[k - 1:]


def sample_mask(h: np.ndarray, shift: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        return ((h * MIX) >> np.uint64(64 - shift)) == 0


def file_signature(args):
    """(index, n_chars, sampled hashes) for one file; runs in a worker."""
    i, path, k, shift = args
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return i, 0, np.zeros(0, dtype=np.uint64)
    return i, len(text), sampled_window_hashes(normalize(text), k, shift)


def sampled_window_hashes(data: bytes, k: int, shift: int, block: int = 1 << 24) -> np.ndarray:
    """Kept window hashes of `data`, in overlapping blocks (bounded memory for large files)."""
    out = []
    for start in range(0, max(1, len(data) - k + 1), block):
        h = window_hashes(data[start: start + block + k - 1], k)
        out.append(h[sample_mask(h, shift)])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint64)


# ── Decisions (vectorized over all files) ─────────────────────────────────────

def analyse(hashes: np.ndarray, docs: np.ndarray, n_docs: int,
            max_containment: float, max_intra_dup: float) -> dict:
    """
    hashes[i] was sampled in file docs[i] (files indexed in priority order).
    Returns per-file arrays: n_hashes, intra_dup, containment, duplicate_of,
    decision codes (0 keep, 1 near_dup, 2 internal_rep), shared.
    """
    total = np.bincount(docs, minlength=n_docs)

    # distinct (hash, doc) pairs, sorted by hash then doc
    order = np.lexsort((docs, hashes))
    h, d = hashes[order], docs[order]
    first_pair = np.ones(len(h), dtype=bool)
    first_pair[1:] = (h[1:] != h[:-1]) | (d[1:] != d[:-1])
    h, d = h[first_pair], d[first_pair]
    distinct = np.bincount(d, minlength=n_docs)
    intra_dup = np.where(total > 0, 1 - distinct / np.maximum(total, 1), 0.0)

    # owner = earliest file containing each hash (pairs are sorted by doc within hash)
    group_start = np.ones(len(h), dtype=bool)
    group_start[1:] = h[1:] != h[:-1]
    group_id = np.cumsum(group_start) - 1
    owner = d[group_start][group_id]
    contained = d != owner
    containment = np.where(distinct > 0,
                           np.bincount(d, weights=contained, minlength=n_docs) / np.maximum(distinct, 1),
                           0.0)

    # most frequent earlier owner among contained pairs
    duplicate_of = np.full(n_docs, -1, dtype=np.int64)
    if contained.any():
        key = d[contained].astype(np.int64) * n_docs + owner[contained]
        uk, cnt = np.unique(key, return_counts=True)
        doc_of, own_of = uk // n_docs, uk % n_docs
        best = np.lexsort((-cnt, doc_of))                 # per doc, highest count first
        first_of_doc = np.ones(len(best), dtype=bool)
        first_of_doc[1:] = doc_of[best][1:] != doc_of[best][:-1]
        duplicate_of[doc_of[best][first_of_doc]] = own_of[best][first_of_doc]

    decision = np.zeros(n_docs, dtype=np.int8)
    decision[containment >= max_containment] = 1
    decision[intra_dup >= max_intra_dup] = 2              # internal repetition wins the label

    # correction: containment against earlier KEPT files only
    kept_pair = decision[d] == 0
    big = np.iinfo(np.int64).max
    kept_owner = np.full(group_id.max() + 1 if len(group_id) else 0, big, dtype=np.int64)
    np.minimum.at(kept_owner, group_id[kept_pair], d[kept_pair].astype(np.int64))
    contained_kept = kept_owner[group_id] < d
    containment_kept = np.bincount(d, weights=contained_kept, minlength=n_docs) / np.maximum(distinct, 1)
    restore = (decision == 1) & (containment_kept < max_containment)
    decision[restore] = 0

    # shared: among kept files, windows found in any other kept file
    keep_pair = decision[d] == 0
    kh, kd = h[keep_pair], d[keep_pair]
    if len(kh):
        _, inv, df = np.unique(kh, return_inverse=True, return_counts=True)
        shared = np.bincount(kd, weights=df[inv] > 1, minlength=n_docs) / np.maximum(distinct, 1)
    else:
        shared = np.zeros(n_docs)

    return {"n_hashes": total, "intra_dup": intra_dup, "containment": containment,
            "duplicate_of": duplicate_of, "decision": decision, "shared": shared,
            "restored": restore}


def choose_val(subcorpora, n_chars, decision, shared, val_fraction, max_shared, min_chars, seed):
    """Per sub-corpus, eligible kept files at random until val_fraction of kept chars."""
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(subcorpora), dtype=bool)
    subcorpora = np.asarray(subcorpora)
    for name in dict.fromkeys(subcorpora):
        idx = np.flatnonzero((subcorpora == name) & (decision == 0))
        target = val_fraction * n_chars[idx].sum()
        eligible = idx[(shared[idx] <= max_shared) & (n_chars[idx] >= min_chars)]
        got = 0
        for i in rng.permutation(eligible):
            if got >= target:
                break
            if n_chars[i] > 2 * (target - got):
                continue                                  # no single file may overshoot 2x
            is_val[i] = True
            got += n_chars[i]
    return is_val


# ── Main ──────────────────────────────────────────────────────────────────────

DECISIONS = {0: "keep", 1: "near_dup", 2: "internal_rep", 3: "excluded"}


def read_exclude_list(path) -> set:
    """File stems to exclude (one per line; a path or a name, extension ignored)."""
    if not path:
        return set()
    with open(path, encoding="utf-8") as f:
        return {Path(line.strip()).stem for line in f if line.strip() and not line.startswith("#")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extracted-dir", default=str(ps.EXTRACTED_DIR))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42, help="same as prepare_shards.py (file selection)")
    p.add_argument("--sources", choices=["v1", "v2"], default="v1",
                   help="v1: the six v1 sub-corpora; v2: v1 + modern_merged (last in priority)")
    p.add_argument("--k", type=int, default=200, help="window length in bytes (~50 tokens)")
    p.add_argument("--sample-shift", type=int, default=6, help="keep 1/2^s windows")
    p.add_argument("--max-containment", type=float, default=0.8)
    p.add_argument("--max-intra-dup", type=float, default=0.3)
    p.add_argument("--val-fraction", type=float, default=0.005)
    p.add_argument("--val-max-shared", type=float, default=0.02)
    p.add_argument("--val-min-chars", type=int, default=2000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--exclude-list", default=None,
                   help="file stems to drop before deduplication (e.g. non-French files from the "
                        "inventory); recorded as 'excluded' and never used as a source")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = ps.select_files(Path(args.extracted_dir), args.seed, args.sources)
    n = len(files)
    exclude = read_exclude_list(args.exclude_list)
    excluded = np.array([Path(f).stem in exclude for _, f in files], dtype=bool)
    print(f"{n:,} files selected, {int(excluded.sum()):,} on the exclude list", flush=True)

    n_chars = np.zeros(n, dtype=np.int64)
    parts_h, parts_d = [], []
    jobs = [(i, str(f), args.k, args.sample_shift) for i, (_, f) in enumerate(files)]
    pool = Pool(args.workers) if args.workers > 1 else None
    results = pool.imap_unordered(file_signature, jobs, chunksize=64) if pool else map(file_signature, jobs)
    for done, (i, nc, h) in enumerate(results, 1):
        n_chars[i] = nc
        parts_h.append(h)
        parts_d.append(np.full(len(h), i, dtype=np.int64))
        if done % 5000 == 0:
            print(f"  hashed {done:,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
    if pool:
        pool.close()
        pool.join()
    hashes = np.concatenate(parts_h) if parts_h else np.zeros(0, dtype=np.uint64)
    docs = np.concatenate(parts_d) if parts_d else np.zeros(0, dtype=np.int64)
    del parts_h, parts_d
    if excluded.any():                       # excluded files are neither kept nor sources
        keep_pair = ~excluded[docs]
        hashes, docs = hashes[keep_pair], docs[keep_pair]
    print(f"{len(hashes):,} sampled windows ({time.time() - t0:.0f}s)", flush=True)

    r = analyse(hashes, docs, n, args.max_containment, args.max_intra_dup)
    r["decision"][excluded] = 3
    sub = [s for s, _ in files]
    is_val = choose_val(sub, n_chars, r["decision"], r["shared"], args.val_fraction,
                        args.val_max_shared, args.val_min_chars, args.seed)

    with open(out / "dedup_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["priority", "subcorpus", "path", "n_chars", "n_windows", "intra_dup",
                    "containment", "duplicate_of", "shared", "decision", "split"])
        for i, (s, path) in enumerate(files):
            dec = DECISIONS[int(r["decision"][i])]
            dup = r["duplicate_of"][i]
            w.writerow([i, s, str(path), int(n_chars[i]), int(r["n_hashes"][i]),
                        f"{r['intra_dup'][i]:.4f}", f"{r['containment'][i]:.4f}",
                        str(files[dup][1]) if dup >= 0 else "", f"{r['shared'][i]:.4f}", dec,
                        ("val" if is_val[i] else "train") if dec == "keep" else ""])

    summary = {"params": vars(args), "files": n, "seconds": round(time.time() - t0),
               "restored_by_correction": int(r["restored"].sum()), "subcorpora": {}}
    for name in dict.fromkeys(sub):
        idx = np.flatnonzero(np.asarray(sub) == name)
        chars = n_chars[idx]
        dec = r["decision"][idx]
        summary["subcorpora"][name] = {
            "files": int(len(idx)),
            "chars": int(chars.sum()),
            "kept_chars": int(chars[dec == 0].sum()),
            "near_dup_chars": int(chars[dec == 1].sum()),
            "internal_rep_chars": int(chars[dec == 2].sum()),
            "near_dup_files": int((dec == 1).sum()),
            "internal_rep_files": int((dec == 2).sum()),
            "excluded_chars": int(chars[dec == 3].sum()),
            "excluded_files": int((dec == 3).sum()),
            "val_files": int(is_val[idx].sum()),
            "val_chars": int(chars[is_val[idx]].sum()),
        }
    tot = {k: sum(v[k] for v in summary["subcorpora"].values())
           for k in ("chars", "kept_chars", "near_dup_chars", "internal_rep_chars",
                     "excluded_chars", "val_chars")}
    summary["total"] = tot
    (out / "dedup_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'subcorpus':<18}{'files':>8}{'chars (M)':>11}{'kept':>8}{'near-dup':>10}{'int-rep':>9}{'val files':>11}")
    for name, s in summary["subcorpora"].items():
        c = max(1, s["chars"])
        print(f"{name:<18}{s['files']:>8,}{s['chars']/1e6:>11.0f}{s['kept_chars']/c:>8.1%}"
              f"{s['near_dup_chars']/c:>10.1%}{s['internal_rep_chars']/c:>9.1%}{s['val_files']:>11,}")
    c = max(1, tot["chars"])
    print(f"{'total':<18}{n:>8,}{tot['chars']/1e6:>11.0f}{tot['kept_chars']/c:>8.1%}"
          f"{tot['near_dup_chars']/c:>10.1%}{tot['internal_rep_chars']/c:>9.1%}")
    print(f"\nmanifest: {out / 'dedup_manifest.csv'}\nsummary:  {out / 'dedup_summary.json'} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
