#!/usr/bin/env python3
"""
manifest_from_dir.py — a prepare_shards.py manifest for a directory of extracted
text files that is not part of the deduplicated corpus (e.g. the Encyclopédie,
built as its own shard set).

    python scripts/manifest_from_dir.py --dir /data/pretrain_candide/extracted_v1f/encyclopedie \\
        --subcorpus encyclopedie --val ENC_volume17 --out /data/pretrain_candide/enc_manifest.csv

Files whose stem is listed with --val go to the validation split; all others to train.
"""

import argparse
import csv
from pathlib import Path

FIELDS = ["priority", "subcorpus", "path", "n_chars", "n_windows", "intra_dup",
          "containment", "duplicate_of", "shared", "decision", "split"]


def build(directory: Path, subcorpus: str, val_stems=()):
    rows = []
    for i, f in enumerate(sorted(directory.glob("*.txt"))):
        rows.append({"priority": i, "subcorpus": subcorpus, "path": str(f),
                     "n_chars": len(f.read_text(encoding="utf-8", errors="replace")),
                     "n_windows": "", "intra_dup": "", "containment": "", "duplicate_of": "",
                     "shared": "", "decision": "keep",
                     "split": "val" if f.stem in set(val_stems) else "train"})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--subcorpus", required=True)
    p.add_argument("--val", nargs="*", default=[], help="file stems for the validation split")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = build(Path(args.dir), args.subcorpus, args.val)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_val = sum(r["split"] == "val" for r in rows)
    chars = sum(r["n_chars"] for r in rows)
    print(f"{len(rows)} files ({n_val} val), {chars/1e6:.0f}M chars -> {args.out}")


if __name__ == "__main__":
    main()
