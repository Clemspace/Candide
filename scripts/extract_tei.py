#!/usr/bin/env python3
"""
extract_tei.py — Extract plain text from the ModERN TEI/XML corpora (lxml).

Formerly ~/audit/extract_text.py on obtic1 (the v1 extraction of 2026-02-23).
Uses lxml with recover=True to handle malformed XML across all sub-corpora.
Output: one .txt per source document in <out-dir>/<subcorpus>/, one paragraph
(<p>) per block, blocks separated by a blank line.

Fixes of 2026-11-10 (v1 text was affected by all three):
  1. Tail text duplicated. Text following a skipped element (<pb/>, <note>,
     <ref>, <head>, <fw>, ...) inside a paragraph was emitted twice: once when
     the skipped element was visited, once by its tail entry.
  2. Nested paragraphs duplicated. A <p> inside another <p> was emitted as part
     of the outer paragraph and again on its own. Only outermost <p> are taken.
  3. Press files whose first 2,000 characters matched an earlier file were
     dropped as duplicates. Mostly right (content deduplication finds about
     half of modern_press to be near-copies), but unverifiable and without
     provenance. Off by default (--press-prefix-dedup restores it); duplicates
     are handled by content in scripts/dedup_corpus.py, which records each
     decision.
"""

import sys
import re
import json
import hashlib
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from lxml import etree

# ── Config ──────────────────────────────────────────────────────────────────

BASE_DIR = Path("/data/corpora/modern")
OUT_DIR = Path("/data/pretrain_candide/extracted")

SUBCORPORA = {
    "modern_big":       {"extensions": [".tei"], "recursive": True},
    "modern_canon":     {"extensions": [".xml"], "recursive": True},
    "modern_archive":   {"extensions": [".xml"], "recursive": True},
    "modern_press":     {"extensions": [".tei"], "recursive": True},
    "modern_lettres":   {"extensions": [".tei"], "recursive": True},
    "modern_pamphlets": {"extensions": [".xml"], "recursive": True},
    "modern_dicos":     {"extensions": [".xml"], "recursive": True},
    "modern_merged":    {"extensions": [".tei"], "recursive": True},   # v2
    # Diderot & d'Alembert, 17 text volumes (ARTFL). Kept out of the main corpus
    # selection; used as its own shard set (base mix and high-quality cooldown).
    # Encyclopediana is a different work and is not matched by the pattern.
    "encyclopedie":     {"extensions": [".xml"], "recursive": False, "pattern": "ENC_volume*.xml"},
}

SKIP_FILES = {"files2move.txt", "load_press_config.py", "nohup.out", "exclude.txt",
              "load_all_config.cpython-310.pyc"}

# Tags whose content we skip (tail text kept)
SKIP_TAGS = frozenset({"head", "note", "fw", "pb", "graphic", "milestone",
                        "figure", "figdesc", "bibl", "ref", "formula"})

# Header tags — skip entirely
HEADER_TAGS = frozenset({"teiheader", "filedesc", "profiledesc", "encodingdesc",
                          "revisiondesc", "metadata", "bookinfo", "citation",
                          "modernfiledesc"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Parsing ─────────────────────────────────────────────────────────────────


def fix_mojibake(text: str) -> str:
    """Repair double-encoded UTF-8 (Ã© → é etc.)"""
    try:
        # If text was decoded as latin-1 from utf-8 bytes, re-encode and decode
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text  # already clean, leave it


def parse_xml(filepath: Path, subcorpus: str):
    """Parse XML/TEI with maximum leniency."""
    try:
        raw = filepath.read_bytes()
    except Exception as e:
        log.debug(f"  Read error {filepath.name}: {e}")
        return None

    # Detect encoding
    enc = "utf-8"
    if subcorpus == "modern_dicos" and (b"ISO-8859-1" in raw[:300] or b"iso-8859-1" in raw[:300]):
        enc = "iso-8859-1"

    # Try 1: lxml recover mode
    parser = etree.XMLParser(recover=True, encoding=enc)
    try:
        root = etree.fromstring(raw, parser)
        if root is not None and len(root) > 0:
            # Check that we got more than just the header
            root_tag = local_tag(root)
            has_content = root_tag not in HEADER_TAGS and root_tag != "wrapper"
            if not has_content:
                # Root is just the header — check if body/text exists inside
                has_content = any(
                    local_tag(e) in ("body", "text")
                    for e in root.iter()
                    if isinstance(e.tag, str) and local_tag(e) != "textclass"
                )
            if has_content:
                return root
    except Exception:
        pass

    # Try 2: wrap fragments (pamphlets: sibling teiHeader + text, no root)
    try:
        text = raw.decode(enc, errors="replace")
        text = re.sub(r'<\?xml[^?]*\?>', '', text)
        wrapped = f"<wrapper>{text}</wrapper>"
        parser2 = etree.XMLParser(recover=True)
        root = etree.fromstring(wrapped.encode("utf-8"), parser2)
        if root is not None and len(root) > 0:
            return root
    except Exception:
        pass

    # Try 3: HTML parser (last resort)
    try:
        root = etree.HTML(raw)
        if root is not None:
            return root
    except Exception:
        pass

    return None


def local_tag(elem) -> str:
    """Tag name without namespace, lowercased."""
    tag = elem.tag if isinstance(elem.tag, str) else ""
    if "}" in tag:
        tag = tag.split("}")[-1]
    return tag.lower()


def iter_text(element) -> str:
    """Extract text iteratively (stack-based, no recursion limit)."""
    parts = []
    stack = [(element, True)]

    while stack:
        elem, visit = stack.pop()

        if not visit:
            if elem.tail and elem.tail.strip():
                parts.append(elem.tail)
            continue

        tag = local_tag(elem)

        # Skip headers and noise tags entirely. Their tail is emitted by the
        # (elem, False) entry pushed with them (fix 1: it was emitted here too).
        if tag in HEADER_TAGS or tag in SKIP_TAGS:
            continue

        if elem.text and elem.text.strip():
            parts.append(elem.text)

        # Push children in reverse (first child processed first)
        for child in reversed(list(elem)):
            stack.append((child, False))  # tail
            stack.append((child, True))   # content

    return " ".join(parts)


def find_body(root):
    """Find body element across schemas."""
    for elem in root.iter():
        tag = local_tag(elem)
        if tag == "body":
            return elem

    # Fallback: <text>
    for elem in root.iter():
        if local_tag(elem) == "text":
            raw_tag = elem.tag if isinstance(elem.tag, str) else ""
            local = raw_tag.split("}")[-1] if "}" in raw_tag else raw_tag
            if local.lower() == "text":
                return elem

    return None


def _inside_p(elem) -> bool:
    parent = elem.getparent()
    while parent is not None:
        if local_tag(parent) == "p":
            return True
        parent = parent.getparent()
    return False


def extract_paragraphs(body) -> list[str]:
    """Extract outermost <p> from body (fix 2: nested <p> are part of their parent)."""
    paragraphs = []
    for elem in body.iter():
        tag = local_tag(elem)
        if tag in HEADER_TAGS:
            continue
        if tag == "p" and not _inside_p(elem):
            text = iter_text(elem).strip()
            text = re.sub(r'\s+', ' ', text)
            if text and len(text) > 1 and not re.match(r'^\d+$', text):
                paragraphs.append(text)
    return paragraphs


def extract_file(filepath: Path, subcorpus: str) -> tuple[str, dict]:
    """Extract text from one file."""
    root = parse_xml(filepath, subcorpus)
    if root is None:
        return "", {}

    # Metadata
    meta = {}
    for elem in root.iter():
        tag = local_tag(elem)
        if tag in ("body",) or (tag == "text" and tag != "textclass"):
            break
        if tag == "title" and "title" not in meta:
            t = "".join(elem.itertext()).strip()
            if 2 < len(t) < 500:
                meta["title"] = t
        if tag == "author" and "author" not in meta:
            t = "".join(elem.itertext()).strip()
            if 2 < len(t) < 300:
                meta["author"] = t
        if tag == "date" and "date" not in meta:
            t = (elem.text or "").strip()
            if t and len(t) < 50:
                meta["date"] = t

    # Extract text
    body = find_body(root)
    if body is None:
        paragraphs = extract_paragraphs(root)
    else:
        paragraphs = extract_paragraphs(body)

    return "\n\n".join(paragraphs), meta


# ── File discovery ──────────────────────────────────────────────────────────

def find_files(src_dir: Path, extensions: list[str], recursive: bool, pattern: str = None) -> list[Path]:
    files = []
    if pattern:
        return sorted(f for f in src_dir.glob(pattern) if f.is_file() and f.stat().st_size > 100)
    for ext in extensions:
        if recursive:
            files.extend(src_dir.rglob(f"*{ext}"))
        else:
            files.extend(src_dir.glob(f"*{ext}"))
    return sorted(f for f in files
                  if f.name not in SKIP_FILES
                  and not f.name.endswith(".pyc")
                  and f.stat().st_size > 100)


# ── Processing ──────────────────────────────────────────────────────────────

def process_subcorpus(name: str, config: dict, dry_run: bool = False,
                      press_prefix_dedup: bool = False) -> dict:
    src_dir = BASE_DIR / name
    out_dir = OUT_DIR / name

    if not src_dir.exists():
        log.warning(f"  Missing: {src_dir}")
        return {"error": "missing"}

    files = find_files(src_dir, config["extensions"], config["recursive"], config.get("pattern"))
    log.info(f"  {len(files):,} files found")

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    stats["files_found"] = len(files)
    seen_hashes = set()

    for i, fpath in enumerate(files):
        if (i + 1) % 5000 == 0:
            log.info(f"    [{i+1:,}/{len(files):,}]")

        try:
            text, meta = extract_file(fpath, name)
        except Exception as e:
            log.debug(f"  Error {fpath.name}: {e}")
            stats["files_error"] += 1
            continue

        if not text or len(text) < 50:
            stats["files_empty"] += 1
            continue

        # Old press dedup on the first 2,000 characters (fix 3: off by default)
        if press_prefix_dedup and name == "modern_press":
            h = hashlib.md5(text[:2000].encode()).hexdigest()
            if h in seen_hashes:
                stats["files_deduped"] += 1
                continue
            seen_hashes.add(h)

        if name == "modern_dicos":
            text = fix_mojibake(text)

        stats["files_extracted"] += 1
        stats["total_chars"] += len(text)
        stats["total_words"] += len(text.split())

        if dry_run:
            continue

        # Write output
        out_name = fpath.stem + ".txt"
        out_path = out_dir / out_name
        if out_path.exists():
            out_name = f"{fpath.parent.name}_{fpath.stem}.txt"
            out_path = out_dir / out_name
            if out_path.exists():
                h = hashlib.md5(fpath.as_posix().encode()).hexdigest()[:8]
                out_name = f"{fpath.stem}_{h}.txt"
                out_path = out_dir / out_name

        out_path.write_text(text, encoding="utf-8")

    return dict(stats)


def main():

    global BASE_DIR, OUT_DIR

    parser = argparse.ArgumentParser(description="Extract text from ModERN corpus (v2, lxml)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--subcorpus", type=str)
    parser.add_argument("--base-dir", type=str, default=str(BASE_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--press-prefix-dedup", action="store_true",
                        help="v1 behaviour: drop press files whose first 2,000 chars repeat")
    args = parser.parse_args()

    BASE_DIR = Path(args.base_dir)
    OUT_DIR = Path(args.out_dir)

    log.info("ModERN corpus — text extraction (lxml, fixed 2026-11-10)")
    log.info(f"  Source: {BASE_DIR}")
    log.info(f"  Output: {OUT_DIR}")
    log.info(f"  Mode:   {'DRY RUN' if args.dry_run else 'EXTRACT'}")
    log.info("")

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = SUBCORPORA
    if args.subcorpus:
        if args.subcorpus not in SUBCORPORA:
            log.error(f"Unknown: {args.subcorpus}. Options: {list(SUBCORPORA.keys())}")
            sys.exit(1)
        targets = {args.subcorpus: SUBCORPORA[args.subcorpus]}

    all_stats = {}

    for name, config in targets.items():
        log.info(f"Processing {name}...")
        stats = process_subcorpus(name, config, dry_run=args.dry_run,
                                  press_prefix_dedup=args.press_prefix_dedup)
        all_stats[name] = stats

        w = stats.get("total_words", 0)
        f = stats.get("files_extracted", 0)
        e = stats.get("files_empty", 0)
        d = stats.get("files_deduped", 0)
        err = stats.get("files_error", 0)
        log.info(f"  → {f:,} extracted, {e:,} empty, {d:,} deduped, {err:,} errors")
        log.info(f"  → {w:,} words (~{w * 1.3 / 1e6:.0f}M tokens)")
        log.info("")

    # Summary
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)

    tw = tf = 0
    for name, stats in all_stats.items():
        w = stats.get("total_words", 0)
        f = stats.get("files_extracted", 0)
        tw += w; tf += f
        log.info(f"  {name:25s}  {f:>8,} files  {w:>15,} words  ~{w*1.3/1e9:.2f}B tok")

    log.info(f"  {'TOTAL':25s}  {tf:>8,} files  {tw:>15,} words  ~{tw*1.3/1e9:.2f}B tok")

    report = {
        "timestamp": datetime.now().isoformat(),
        "base_dir": str(BASE_DIR),
        "out_dir": str(OUT_DIR),
        "dry_run": args.dry_run,
        "subcorpora": all_stats,
        "total_words": tw,
        "total_files": tf,
        "total_tokens_est": tw * 1.3,
    }

    stem = f"extraction_stats_{args.subcorpus}" if args.subcorpus else "extraction_stats"
    rpath = OUT_DIR / f"{stem}.json" if not args.dry_run else Path(f"{stem}_dryrun.json")
    if not args.dry_run:
        rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info(f"\nReport: {rpath}")


if __name__ == "__main__":
    main()