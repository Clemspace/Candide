#!/usr/bin/env python3
"""Build a held-out test set (JSON lines) from the validation split of a dedup manifest.

The validation files of scripts/dedup_corpus.py are kept files that share almost nothing
with any other kept file, and they are never tokenized into the training shards. This
script gathers their text, with the metadata needed to break results down, into one file
that scripts/score_text.py --jsonl reads:

    {"id": "modern_canon/voltaire_candide.txt", "subcorpus": "modern_canon", "year": 1759,
     "decade": "1750s", "century": "18th", "genre": ..., "author": ..., "title": ...,
     "n_chars": 181234, "text": "..."}

Year, genre, author and title come from the inventory (scripts/inventory_corpus.py) when
it is given; a file is matched by sub-corpus and file name. Undated files get
decade "undated". The text is written as extracted: score_text.py applies the same
cleaning as training.

    python scripts/make_testset.py --manifest /data/pretrain_candide/dedup_v2g/dedup_manifest.csv \\
        --inventory /data/pretrain_candide/inventory_v2/inventory.csv \\
        --out /data/pretrain_candide/testsets/val_v2g.jsonl
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

CENTURIES = {15: "16th", 16: "17th", 17: "18th", 18: "19th"}


def load_inventory(path):
    """(sub-corpus, file stem) -> metadata; stems that occur twice in a sub-corpus are dropped."""
    index, seen = {}, Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["corpus"], Path(row["path"]).stem)
            seen[key] += 1
            index[key] = row
    return {k: v for k, v in index.items() if seen[k] == 1}


def period(year):
    """(decade, century) labels for a year, or ('undated', 'undated')."""
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "undated", "undated"
    return f"{y // 10 * 10}s", CENTURIES.get(y // 100, f"{y // 100 + 1}th")


def val_rows(manifest, split="val"):
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["decision"] == "keep" and row["split"] == split:
                yield row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="dedup_manifest.csv")
    ap.add_argument("--inventory", default=None, help="inventory.csv, for year, genre, author, title")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="keep at most this many characters per file, from the middle (default: whole file)")
    args = ap.parse_args(argv)

    inventory = load_inventory(args.inventory) if args.inventory else {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, chars, unmatched, missing = 0, Counter(), 0, 0
    with open(out, "w", encoding="utf-8") as fh:
        for row in val_rows(args.manifest, args.split):
            path = Path(row["path"])
            if not path.is_file():
                missing += 1
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if args.max_chars and len(text) > args.max_chars:
                start = (len(text) - args.max_chars) // 2
                nl = text.find("\n", start, start + args.max_chars // 10)
                start = nl + 1 if nl != -1 else start
                text = text[start:start + args.max_chars]
            meta = inventory.get((row["subcorpus"], path.stem))
            if inventory and meta is None:
                unmatched += 1
            meta = meta or {}
            decade, century = period(meta.get("year"))
            rec = {"id": f"{row['subcorpus']}/{path.name}", "subcorpus": row["subcorpus"],
                   "year": int(meta["year"]) if str(meta.get("year", "")).isdigit() else None,
                   "decade": decade, "century": century,
                   "genre": meta.get("genre") or None, "author": meta.get("author") or None,
                   "title": meta.get("title") or None, "n_chars": len(text), "text": text}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            chars[row["subcorpus"]] += len(text)
    print(f"{n} files, {sum(chars.values()) / 1e6:.1f}M characters -> {out}")
    for s, c in sorted(chars.items()):
        print(f"  {s:20s} {c / 1e6:8.1f}M")
    if missing:
        print(f"WARNING: {missing} file(s) listed in the manifest were not found", file=sys.stderr)
    if unmatched:
        print(f"note: {unmatched} file(s) have no single match in the inventory (undated)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
