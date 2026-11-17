#!/usr/bin/env python3
"""
Overlap between the FreEM_LPM test passages and the pretraining corpus.

D'AlemBERT's corpus contains almost none of the out-of-domain test texts
(Gabay et al. 2022); ours may contain the plays and books they are taken from,
since ModERN's canon is eighteenth-century French. A model that has read a
test passage in pretraining is not tested on unseen text, so this is measured
before any tagging claim.

Method. The test files are tokenized (one token per line) and may come from
another edition than the corpus copy, so both sides are folded to letters and
digits only: entities decoded, accents and punctuation and spaces removed,
lower case, long s -> s, and the period variants u/v and i/j merged. Every
window of --k folded characters (default 60, about a dozen words) of each test
passage is hashed (all windows: the passages are short), and every window of
every kept training file is looked up among them. A test file's overlap is the
share of its windows found; the corpus files that contain them are listed.

Checks that the result is not an artefact: corpus paths that do not exist or
cannot be read are counted, and the run stops if there are more than
--max-unreadable of them (a manifest with stale paths would otherwise read as
"no overlap"); and --control N cuts N passages out of random corpus files,
tokenizes them like the test files, and scores them as the group _control,
which must come out near 100%.

Usage:
  PYTHONPATH=. python scripts/overlap_freem.py \
      --test-root ~/data/FreEMlpm/Data_outOfDomain \
      --manifest /data/pretrain_candide/dedup_v2g/dedup_manifest.csv \
      --extra /data/pretrain_candide/extracted_enc \
      --out results/2026-11-17/overlap_freem.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dedup_corpus as dc  # noqa: E402

VARIANTS = str.maketrans({"\u017f": "s", "u": "v", "j": "i", "\u00e6": "ae", "\u0153": "oe", "\u00df": "ss"})


def squeeze(text: str) -> bytes:
    """Letters and digits only: no accents, punctuation or spaces; long s, u/v, i/j folded."""
    t = unicodedata.normalize("NFKD", dc.ps.clean_text(text).lower())
    t = "".join(c for c in t if not unicodedata.combining(c)).translate(VARIANTS)
    return "".join(c for c in t if c.isalnum()).encode("utf-8")


def passage_text(path) -> str:
    """Tokens of a pie TSV (first column of the header's token column) joined by spaces."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = [h.strip().lower() for h in lines[0].split("\t")] if lines else []
    col = next((header.index(c) for c in ("token", "form", "word", "tok", "forme") if c in header), 0)
    return " ".join(l.split("\t")[col] for l in lines[1:] if l.strip() and len(l.split("\t")) > col)


def test_windows(paths: List[Path], k: int):
    """Sorted unique window hashes of all test files, and the hashes of each file."""
    per_file = {str(p): np.unique(dc.window_hashes(squeeze(passage_text(p)), k)) for p in paths}
    every = np.unique(np.concatenate(list(per_file.values()))) if per_file else np.zeros(0, np.uint64)
    return every, per_file


_TARGETS = np.zeros(0, np.uint64)
_K = 150


def _init(targets, k):
    global _TARGETS, _K
    _TARGETS, _K = targets, k


def member(h: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Boolean mask of h's entries present in the sorted array targets."""
    if len(targets) == 0:
        return np.zeros(len(h), dtype=bool)
    idx = np.minimum(np.searchsorted(targets, h), len(targets) - 1)
    return targets[idx] == h


def file_hits(path, k=None, targets=None):
    """Test-window hashes found in one corpus file (worker globals by default)."""
    k = _K if k is None else k
    targets = _TARGETS if targets is None else targets
    try:
        data = squeeze(Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return path, None, 0
    out = []
    for start in range(0, max(1, len(data) - k + 1), 1 << 24):
        h = dc.window_hashes(data[start: start + (1 << 24) + k - 1], k)
        out.append(h[member(h, targets)])
    return path, (np.unique(np.concatenate(out)) if out else np.zeros(0, np.uint64)), len(data)


def corpus_files(manifest: Path, extra: List[Path]) -> List[str]:
    """Kept training files of the manifest (split == train), plus every .txt under extra dirs."""
    files = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["decision"] == "keep" and row["split"] == "train":
                files.append(row["path"])
    for d in extra:
        files += [str(p) for p in sorted(Path(d).rglob("*.txt"))]
    return files


def control_passages(files: List[str], n: int, out_dir: Path, chars: int = 600, seed: int = 0) -> List[Path]:
    """Cut n passages from random corpus files and write them as one-token-per-line TSVs."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for path in rng.sample(files, min(len(files), 20 * n)):
        if len(made) == n:
            break
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) < 4 * chars:
            continue
        start = rng.randrange(len(text) // 4, len(text) - chars)
        toks = re.findall(r"\w+|[^\w\s]", text[start: start + chars])
        f = out_dir / f"control_{len(made):02d}.tsv"
        f.write_text("token\tPOS\n" + "".join(f"{t}\tX\n" for t in toks), encoding="utf-8")
        made.append(f)
    return made


def summarize(per_file: Dict[str, np.ndarray], found: Dict[str, set], root: Path) -> dict:
    """Overlap per test file and per test directory (group)."""
    files, groups = {}, defaultdict(lambda: [0, 0])
    all_found = set().union(*found.values()) if found else set()
    for path, hashes in per_file.items():
        hit = np.isin(hashes, np.fromiter(all_found, dtype=np.uint64, count=len(all_found)))
        sources = sorted(((src, int(np.isin(hashes, np.fromiter(s, np.uint64, len(s))).sum()))
                          for src, s in found.items() if s), key=lambda x: -x[1])
        sources = [(src, n) for src, n in sources if n][:5]
        rel = str(Path(path).relative_to(root))
        files[rel] = {"windows": int(len(hashes)), "found": int(hit.sum()),
                      "share": float(hit.mean()) if len(hashes) else 0.0, "top_sources": sources}
        g = str(Path(rel).parent)
        groups[g][0] += int(hit.sum())
        groups[g][1] += int(len(hashes))
    return {"files": files,
            "groups": {g: {"share": f / t if t else 0.0, "windows": t} for g, (f, t) in sorted(groups.items())}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-root", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--extra", nargs="*", default=[], type=Path, help="other training text dirs")
    p.add_argument("--k", type=int, default=60, help="window length in folded characters")
    p.add_argument("--control", type=int, default=10, help="positive-control passages from the corpus")
    p.add_argument("--max-unreadable", type=float, default=0.01, help="fraction of corpus files")
    p.add_argument("--only", default="original", help="substring the test path must contain ('' = all)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    t0 = time.time()
    tests = [f for f in sorted(a.test_root.rglob("*.tsv")) if a.only in str(f)]
    files = corpus_files(a.manifest, a.extra)
    missing = [f for f in files if not Path(f).exists()]
    if len(missing) > a.max_unreadable * len(files):
        sys.exit(f"{len(missing):,} of {len(files):,} corpus paths do not exist (first: {missing[0]}); "
                 f"fix the manifest paths first")
    tmp = Path(tempfile.mkdtemp(prefix="overlap_control_"))
    controls = control_passages(files, a.control, tmp / "_control")
    targets, per_file = test_windows(tests + controls, a.k)
    print(f"{len(tests)} test files, {len(targets):,} windows; {len(files):,} corpus files", flush=True)

    found, unreadable, chars = {}, 0, 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(targets, a.k)) as ex:
        for i, (path, hits, n) in enumerate(ex.map(file_hits, files, chunksize=64)):
            if hits is None:
                unreadable += 1
                continue
            chars += n
            if len(hits):
                found[path] = set(hits.tolist())
            if i % 10000 == 0:
                print(f"  {i:,}/{len(files):,} files, {len(found)} with hits", flush=True)

    if unreadable > a.max_unreadable * len(files):
        sys.exit(f"{unreadable:,} of {len(files):,} corpus files could not be read")
    per_test = {k: v for k, v in per_file.items() if not k.startswith(str(tmp))}
    per_ctrl = {k: v for k, v in per_file.items() if k.startswith(str(tmp))}
    result = {"params": {k: str(v) for k, v in vars(a).items()}, "corpus_files": len(files),
              "missing_paths": len(missing), "unreadable_files": unreadable,
              "folded_chars_scanned": chars, "seconds": round(time.time() - t0),
              **summarize(per_test, found, a.test_root),
              "control": summarize(per_ctrl, found, tmp)["groups"] if per_ctrl else {}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"read {len(files) - unreadable:,} files ({len(missing)} missing, {unreadable} unreadable), "
          f"{chars / 1e9:.2f}B folded characters")
    for g, v in {**result["groups"], **result["control"]}.items():
        print(f"{g:60s} {v['share'] * 100:6.2f}% of {v['windows']} windows")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
