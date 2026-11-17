#!/usr/bin/env python3
"""
Summarize pos_freem.py result files into one table (summary.json) for the papers.

For each result file: model, pretraining step, protocol, and on original
spelling, 16th to 18th century:
  - mean     macro-average of the three century accuracies (D'AlemBERT's "Avg")
  - clean    the same on passages absent from the pretraining corpus
             (pos_freem.py --overlap)
  - read_minus_clean   pooled accuracy on the passages found in the corpus minus
             pooled accuracy on the others. Every model gets this difference,
             including models that never saw our corpus, so it measures how much
             easier those passages are, not memorization. Candide's excess over
             the baselines' difference is the part pretraining on them could
             explain.

Usage:
  python scripts/pos_summary.py results/2026-11-17/pos/*.json \
      --overlap results/2026-11-17/overlap_freem.json --out results/2026-11-17/pos/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pos_freem as pf  # noqa: E402

CENTURIES = ("16", "17", "18")


def mean_original(centuries: dict) -> float | None:
    vals = [centuries[f"original/{c}"]["acc"] for c in CENTURIES if f"original/{c}" in centuries]
    return sum(vals) / len(vals) if len(vals) == len(CENTURIES) else None


def read_minus_clean(test_files: dict, excluded: set) -> dict:
    """Pooled accuracy on read passages minus on clean ones (original spelling, 16th-18th c.)."""
    acc = {"read": [0, 0], "clean": [0, 0]}
    for f, v in test_files.items():
        if not any(f"/{c}_orig/" in f for c in CENTURIES):
            continue
        k = "read" if pf.passage_key(f) in excluded else "clean"
        acc[k][0] += v["correct"]
        acc[k][1] += v["words"]
    out = {k: {"acc": c / t if t else None, "words": t} for k, (c, t) in acc.items()}
    if out["read"]["acc"] is not None and out["clean"]["acc"] is not None:
        out["diff"] = out["read"]["acc"] - out["clean"]["acc"]
    return out


def summarize(paths, overlap: dict | None, max_overlap: float) -> dict:
    excluded = pf.contaminated(overlap, max_overlap) if overlap else set()
    rows = {}
    for p in paths:
        r = json.loads(Path(p).read_text())
        if "test_centuries" not in r:
            continue
        clean = (r.get("test_clean") or {}).get("centuries") or {}
        row = {
            "model": r["model"],
            "pretraining_step": r.get("pretraining_step", -1),
            "lr": r["protocol"]["lr"], "epochs": r["protocol"]["epochs"], "seed": r["protocol"]["seed"],
            "dev_final": r["dev"][-1]["dev_acc"],
            "centuries": {k: v["acc"] for k, v in r["test_centuries"].items()},
            "mean": mean_original(r["test_centuries"]),
            "clean_centuries": {k: v["acc"] for k, v in clean.items()},
            "clean": mean_original(clean) if clean else None,
        }
        if "test_files" in r and excluded:
            row["read_minus_clean"] = read_minus_clean(r["test_files"], excluded)
        rows[Path(p).stem] = row
    return {"max_overlap": max_overlap, "excluded_passages": len(excluded), "results": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--overlap", default=None)
    ap.add_argument("--max-overlap", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    overlap = json.loads(Path(a.overlap).read_text()) if a.overlap else None
    files = [f for f in a.files if Path(f).name != Path(a.out).name]
    s = summarize(files, overlap, a.max_overlap)
    Path(a.out).write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(f"{'result':40s} {'step':>6s} {'lr':>7s} {'mean':>6s} {'clean':>6s} {'read-clean':>10s}")
    for k, v in sorted(s["results"].items(), key=lambda kv: (kv[1]["model"].startswith("hf"), kv[1]["pretraining_step"])):
        d = v.get("read_minus_clean", {}).get("diff")
        print(f"{k:40s} {v['pretraining_step']:6d} {v['lr']:7.0e} {100 * v['mean']:6.2f} "
              f"{100 * v['clean'] if v['clean'] else float('nan'):6.2f} "
              f"{100 * d if d is not None else float('nan'):+10.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
