#!/usr/bin/env python3
"""
prepare_shards.py — Pre-tokenize the ModERN corpus into binary shards for training.

Run on the machine that holds the extracted text, then rsync the shards to the
GPU server. Output: <out>/train/shard_*.bin and <out>/val/shard_*.bin, each a
1024-byte header (int32 magic 20240520, int32 token count) followed by uint16
tokens. Each file is wrapped in <|bos|> ... <|eos|>.

Two modes:

  legacy (no --manifest): the v1 pipeline, unchanged. Files are selected per
      sub-corpus with the weights in SUBCORPORA (weight < 1 = random file
      sample), shuffled, and the first 0.5% of files become validation.

  --manifest dedup_manifest.csv: the output of scripts/dedup_corpus.py. Train =
      files with decision "keep" and split "train"; val = split "val" (chosen
      by dedup_corpus.py: no text shared with any other kept file, sub-corpus
      mix matched to train). HTML entities are cleaned (clean_text) as in the
      dedup pass.

Usage:
    python scripts/prepare_shards.py                                   # legacy v1
    python scripts/prepare_shards.py --manifest dedup_manifest.csv \\
        --out-dir /data/pretrain_candide/shards_v2
    python scripts/prepare_shards.py --max-tokens 1e9                  # trial
"""

import argparse
import csv
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────

EXTRACTED_DIR = Path("/data/pretrain_candide/extracted")
SHARDS_DIR = Path("/data/pretrain_candide/shards")
TOKENIZER_PATH = Path("/data/pretrain_candide/tokenizer/tokenizer.json")

# Sub-corpora, in priority order (dedup keeps the earliest copy of a text), and
# their sampling weights: weight < 1 keeps that fraction of files at random.
SUBCORPORA = {
    "modern_canon": 1.0,      # Tier 1
    "modern_press": 1.0,      # Tier 1
    "modern_dicos": 1.0,      # Tier 1
    "modern_pamphlets": 0.8,  # Tier 2
    "modern_lettres": 0.8,    # Tier 2
    "modern_archive": 0.15,   # noisy, capped
}

# v2 = v1 + modern_merged (ECCO, Goldsmiths', Frantext, theatre, ... and renamed v1
# files), last in priority so that a text held twice keeps its v1 copy.
SUBCORPORA_V2 = {**SUBCORPORA, "modern_merged": 1.0}
SOURCES = {"v1": "SUBCORPORA", "v2": "SUBCORPORA_V2"}

SHARD_SIZE = int(1e8)    # 100M tokens per shard (~200 MB at uint16)
VAL_FRACTION = 0.005     # legacy mode: 0.5% of files held out
MAGIC = 20240520

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Text ──────────────────────────────────────────────────────────────────────

_NBSP = re.compile(r"&\s*nbsp\s*;", re.IGNORECASE)
_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}
_NAMED = re.compile(r"&(amp|lt|gt|quot|apos);", re.IGNORECASE)
_NUMERIC = re.compile(r"&#(\d{1,6});")


def clean_text(text: str) -> str:
    """Replace leftover HTML entities (&nbsp; -> space, &amp; -> &, &#233; -> é, ...)."""
    text = _NBSP.sub(" ", text)
    text = _NAMED.sub(lambda m: _ENTITIES[m.group(1).lower()], text)
    return _NUMERIC.sub(lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x110000 else m.group(0), text)


def read_text(path: Path, clean: bool = False) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return clean_text(text) if clean else text


# ── File selection ────────────────────────────────────────────────────────────

def select_files(extracted_dir: Path, seed: int, sources: str = "v1") -> List[Tuple[str, Path]]:
    """
    Sub-corpus file selection: (subcorpus, path) in priority order, weight < 1
    sub-corpora sampled at random. Seeds `random` exactly as v1 did, so with
    sources="v1" the selection is identical to v1 for the same seed; "v2" adds
    modern_merged after the v1 sub-corpora (same random draws for those).
    """
    subcorpora = globals()[SOURCES[sources]]
    random.seed(seed)
    selected = []
    for name, weight in subcorpora.items():
        subdir = extracted_dir / name
        if not subdir.exists():
            log.warning(f"Missing: {subdir}")
            continue
        files = sorted(subdir.glob("*.txt"))
        if weight < 1.0:
            n = max(1, int(len(files) * weight))
            files = random.sample(files, n)
        selected.extend((name, f) for f in files)
        log.info(f"  {name}: {len(files):,} files (weight={weight})")
    return selected


def legacy_split(selected: List[Tuple[str, Path]]) -> Tuple[List[Path], List[Path]]:
    """v1: shuffle all selected files (continuing the seeded stream), first 0.5% = val."""
    all_files = [f for _, f in selected]
    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * VAL_FRACTION))
    return all_files[n_val:], all_files[:n_val]


def manifest_split(manifest: Path, seed: int) -> Tuple[List[Path], List[Path]]:
    """Train/val files from a dedup_corpus.py manifest; train order shuffled with `seed`."""
    train, val = [], []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["decision"] != "keep":
                continue
            (val if row["split"] == "val" else train).append(Path(row["path"]))
    random.Random(seed).shuffle(train)
    return train, val


# ── Tokenization and shards ───────────────────────────────────────────────────

def tokenize_file(path: Path, tokenizer, clean: bool = False) -> np.ndarray:
    """Tokenize one file, wrapped in BOS/EOS; uint16 array (empty on failure)."""
    try:
        text = read_text(path, clean)
        if not text.strip():
            return np.array([], dtype=np.uint16)
        ids = tokenizer.encode(text).ids
        bos = tokenizer.token_to_id("<|bos|>")
        eos = tokenizer.token_to_id("<|eos|>")
        if bos is not None:
            ids = [bos] + ids
        if eos is not None:
            ids = ids + [eos]
        return np.array(ids, dtype=np.uint16)
    except Exception as e:
        log.debug(f"Skip {path.name}: {e}")
        return np.array([], dtype=np.uint16)


def write_shard(tokens: np.ndarray, path: Path):
    """Header [magic, n_tokens, 0...] (256 int32) then tokens."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = len(tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    log.info(f"  Wrote {len(tokens):,} tokens -> {path.name} ({path.stat().st_size / 1e6:.0f} MB)")


def write_split(files: List[Path], split_dir: Path, tokenizer, shard_size: int,
                max_tokens=None, clean: bool = False) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_idx, current, size, total = 0, [], 0, 0
    for i, fpath in enumerate(files):
        if (i + 1) % 5000 == 0:
            log.info(f"  [{i+1:,}/{len(files):,}] {total/1e6:.0f}M tokens")
        tokens = tokenize_file(fpath, tokenizer, clean)
        if len(tokens) == 0:
            continue
        current.append(tokens)
        size += len(tokens)
        total += len(tokens)
        if size >= shard_size:
            arr = np.concatenate(current)
            write_shard(arr[:shard_size], split_dir / f"shard_{shard_idx:05d}.bin")
            overflow = arr[shard_size:]
            current = [overflow] if len(overflow) > 0 else []
            size = len(overflow)
            shard_idx += 1
        if max_tokens and total >= max_tokens:
            log.info(f"  Reached max_tokens limit ({max_tokens/1e9:.2f}B)")
            break
    if current:
        arr = np.concatenate(current)
        if len(arr) > 1000:  # don't write tiny shards
            write_shard(arr, split_dir / f"shard_{shard_idx:05d}.bin")
    log.info(f"  {split_dir.name}: {total/1e6:.0f}M tokens -> {shard_idx + 1} shards")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=float, default=None,
                        help="Maximum total tokens per split (e.g. 1e9 for a trial run)")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extracted-dir", type=str, default=str(EXTRACTED_DIR))
    parser.add_argument("--tokenizer", type=str, default=str(TOKENIZER_PATH))
    parser.add_argument("--out-dir", type=str, default=str(SHARDS_DIR))
    parser.add_argument("--manifest", type=str, default=None,
                        help="dedup_corpus.py manifest: use its kept files and val split")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"],
                        help="which splits to write (e.g. --splits val for evaluation only)")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    out_dir = Path(args.out_dir)
    max_tokens = int(args.max_tokens) if args.max_tokens else None
    log.info("ModERN corpus — pre-tokenization + sharding")
    log.info(f"  Mode:      {'manifest ' + args.manifest if args.manifest else 'legacy v1'}")
    log.info(f"  Tokenizer: {args.tokenizer}")
    log.info(f"  Output:    {out_dir}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    log.info(f"  Vocab size: {tokenizer.get_vocab_size()}")

    if args.manifest:
        train_files, val_files = manifest_split(Path(args.manifest), args.seed)
        clean = True
    else:
        train_files, val_files = legacy_split(select_files(Path(args.extracted_dir), args.seed))
        clean = False
    log.info(f"  Train files: {len(train_files):,}   Val files: {len(val_files):,}")

    for split, files in [("train", train_files), ("val", val_files)]:
        if split not in args.splits:
            log.info(f"\nSkipping {split} split (--splits)")
            continue
        log.info(f"\nProcessing {split} split...")
        write_split(files, out_dir / split, tokenizer, args.shard_size, max_tokens, clean)

    log.info("\nDone. Transfer shards to the GPU server before training.")
    log.info(f"  rsync -avz --progress {out_dir}/ <gpu_server>:{out_dir}/")


if __name__ == "__main__":
    main()
