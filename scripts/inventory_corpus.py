#!/usr/bin/env python3
"""
inventory_corpus.py — per-file inventory of the ModERN TEI corpora (data card).

For every .tei / .xml file in the given corpus directories, reads the ModERN
header (date, genre, author, title, BnF ark) and the body text, and records:

  corpus      directory under the corpora root (modern_canon, modern_merged, ...)
  source      file-name prefix after "modern_" (ecco, frantext18, goldsmith, ...)
  date, year  header <date> (modernFileDesc first, then sourceDesc); year = first 4 digits
  genre, author, title, ark
  chars, words  body text, tags stripped
  lang, lang_score  language guess from function-word frequencies (fr, en, la, it,
              de, es), and the share of the top language among recognized words

The language guess is deliberately simple and transparent: on running text,
function words identify the language reliably; on very short or OCR-damaged
texts lang_score is low, which flags them for inspection.

    python scripts/inventory_corpus.py --root /data/corpora/modern \\
        --corpora modern_canon modern_press modern_dicos modern_pamphlets \\
                  modern_lettres modern_archive modern_merged \\
        --out /data/pretrain_candide/inventory_v2 --workers 16
"""

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

FUNCTION_WORDS = {
    "fr": "le la les de des du et que qui dans est une un pour pas il elle ne se ce sur au aux par plus "
          "son sa ses leur nous vous on mais ou avec sont été être avoit étoit cette".split(),
    "en": "the and of to that is in it with was for his which be as by he not this but have had are "
          "from they were their all would".split(),
    "la": "et in est non ad cum quod sed ut qui quae enim autem nec atque esse sunt eius ab ex sibi "
          "etiam quam vel tamen".split(),
    "it": "il di che della non per una con del gli sono alla anche nel come questo più delle".split(),
    "de": "der die und das nicht ist ein eine zu den mit sich des auf für dem von auch wie".split(),
    "es": "el los las que del por con una para como pero sus está fue muy".split(),
}
# words shared by several languages carry no signal
_counts = Counter(w for ws in FUNCTION_WORDS.values() for w in set(ws))
LANG_OF = {w: lang for lang, ws in FUNCTION_WORDS.items() for w in ws if _counts[w] == 1}

_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_YEAR = re.compile(r"(1[0-9]{3})")


def _field(xml: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", xml, re.S)
    return " ".join(html.unescape(_TAG.sub(" ", m.group(1))).split()) if m else ""


def parse_header(xml: str) -> dict:
    head_end = xml.find("</teiHeader>")
    head = xml[:head_end] if head_end >= 0 else xml[:20000]
    mfd = re.search(r"<modernFileDesc>(.*?)</modernFileDesc>", head, re.S)
    mfd = mfd.group(1) if mfd else ""
    date = _field(mfd, "date") or _field(head, "date")
    y = _YEAR.search(date)
    return {
        "date": date,
        "year": int(y.group(1)) if y else "",
        "genre": _field(mfd, "genre"),
        "author": _field(mfd, "bnfName") or _field(mfd, "authorOriginal") or _field(head, "author"),
        "title": _field(mfd, "titleOriginal") or _field(head, "title"),
        "ark": _field(mfd, "arkBnf"),
    }


def body_text(xml: str) -> str:
    start = xml.find("</teiHeader>")
    body = xml[start + len("</teiHeader>"):] if start >= 0 else xml
    return " ".join(html.unescape(_TAG.sub(" ", body)).split())


def guess_language(text: str, max_words: int = 20000):
    counts = Counter()
    for i, m in enumerate(_WORD.finditer(text.lower())):
        if i >= max_words:
            break
        lang = LANG_OF.get(m.group(0))
        if lang:
            counts[lang] += 1
    total = sum(counts.values())
    if total == 0:
        return "", 0.0
    lang, n = counts.most_common(1)[0]
    return lang, n / total


def source_of(name: str) -> str:
    m = re.match(r"modern_([A-Za-z0-9]+)_", name)
    return m.group(1).lower() if m else ""


def inventory_file(args):
    corpus, path = args
    try:
        xml = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"corpus": corpus, "path": path, "error": str(e)}
    text = body_text(xml)
    lang, score = guess_language(text)
    return {"corpus": corpus, "path": path, "source": source_of(Path(path).name),
            **parse_header(xml), "chars": len(text), "words": len(_WORD.findall(text)),
            "lang": lang, "lang_score": round(score, 3), "error": ""}


FIELDS = ["corpus", "source", "path", "date", "year", "genre", "author", "title", "ark",
          "chars", "words", "lang", "lang_score", "error"]


def summarize(rows, period=(1680, 1815), min_source_files=20):
    """Per corpus and per (corpus, source): files, chars, language and period shares."""
    def agg(group):
        chars = sum(r["chars"] for r in group) or 1
        langs = Counter()
        for r in group:
            langs[r["lang"] or "?"] += r["chars"]
        years = [r["year"] for r in group if isinstance(r["year"], int)]
        in_period = sum(r["chars"] for r in group
                        if isinstance(r["year"], int) and period[0] <= r["year"] <= period[1])
        undated = sum(r["chars"] for r in group if not isinstance(r["year"], int))
        return {
            "files": len(group), "chars": chars,
            "lang_share": {k: round(v / chars, 4) for k, v in langs.most_common()},
            "in_period_share": round(in_period / chars, 4),
            "undated_share": round(undated / chars, 4),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
        }

    by_corpus, by_source = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        by_corpus[r["corpus"]].append(r)
        by_source[(r["corpus"], r["source"])].append(r)
    out = {"period": list(period), "corpora": {c: agg(g) for c, g in by_corpus.items()}, "sources": {}}
    for (c, s), g in sorted(by_source.items(), key=lambda kv: -sum(r["chars"] for r in kv[1])):
        key = f"{c}/{s or '(title-named)'}" if len(g) >= min_source_files else f"{c}/(other small)"
        out["sources"].setdefault(key, []).extend(g)
    out["sources"] = {k: agg(g) for k, g in out["sources"].items()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/data/corpora/modern")
    p.add_argument("--corpora", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--period", nargs=2, type=int, default=[1680, 1815])
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, str(f)) for c in args.corpora
            for f in sorted((Path(args.root) / c).rglob("*"))
            if f.is_file() and f.suffix in (".tei", ".xml")]
    print(f"{len(jobs):,} files", flush=True)
    rows = []
    pool = Pool(args.workers) if args.workers > 1 else None
    results = pool.imap_unordered(inventory_file, jobs, chunksize=32) if pool else map(inventory_file, jobs)
    for i, r in enumerate(results, 1):
        rows.append(r)
        if i % 10000 == 0:
            print(f"  {i:,}", flush=True)
    if pool:
        pool.close()
        pool.join()

    rows.sort(key=lambda r: (r["corpus"], r["path"]))
    with open(out / "inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, tuple(args.period))
    summary["errors"] = sum(1 for r in rows if r.get("error"))
    (out / "inventory_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    def line(name, s):
        top = ", ".join(f"{k} {v:.0%}" for k, v in list(s["lang_share"].items())[:3])
        yr = f"{s['year_min']}-{s['year_max']}" if s["year_min"] else "-"
        print(f"{name:<38}{s['files']:>8,}{s['chars']/1e6:>10.0f}  {top:<26}"
              f"{s['in_period_share']:>7.0%}{s['undated_share']:>9.0%}  {yr}")
    hdr = f"{'':<38}{'files':>8}{'Mchars':>10}  {'language (by chars)':<26}{'period':>7}{'undated':>9}  years"
    print("\n" + hdr)
    for c, s in summary["corpora"].items():
        line(c, s)
    print("\n" + hdr)
    for k, s in summary["sources"].items():
        line(k, s)
    print(f"\ninventory: {out / 'inventory.csv'}\nsummary:   {out / 'inventory_summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
