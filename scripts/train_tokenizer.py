#!/usr/bin/env python3
"""
train_tokenizer.py — train and evaluate lossless BPE tokenizers for the ModERN corpus.

Why. The v1 tokenizer splits on whitespace and punctuation before BPE and
decodes with a WordPiece decoder; it cannot reproduce text: "L'Encyclopédie"
decodes to "L ' Encyclopédie", "dit-on ?" to "dit - on?", and pieces of one
word come back separated by spaces. A model trained on it never sees word
boundaries inside split words and cannot print clean text.

The new tokenizer is byte-level BPE (as GPT-2 / Qwen): every string encodes and
decodes exactly, and no character is ever unknown. Before BPE, text is split by
a regex adapted to French: elided articles and pronouns stay attached to their
apostrophe (" l'", " qu'", " lorsqu'"), a word keeps its leading space, digits
are split one by one, punctuation is grouped. Text is NFC-normalized only (no
lowercasing, no accent stripping, ſ and old spellings kept).

Noisy text. Training text is drawn from the dedup manifest's training files,
with a set share coming from the noisy sub-corpora (default modern_archive,
OCR): enough for systematic OCR patterns to get tokens, not enough for noise to
take a large part of the vocabulary. `eval` compares candidates on the
validation files: characters per token on clean and noisy text, exact
round-trip, and the share of the vocabulary used mostly by noisy text.

    # three candidates
    for s in 0.0 0.1 0.3; do python scripts/train_tokenizer.py train \\
        --manifest /data/pretrain_candide/dedup_v2f/dedup_manifest.csv --noisy-share $s \\
        --old-tokenizer /data/pretrain_candide/tokenizer/tokenizer.json \\
        --out /data/pretrain_candide/tokenizer_v2/noisy$s/tokenizer.json; done
    python scripts/train_tokenizer.py eval --manifest .../dedup_manifest.csv \\
        --tokenizer v1=/data/pretrain_candide/tokenizer/tokenizer.json \\
        --tokenizer n0=.../noisy0.0/tokenizer.json --tokenizer n10=... --tokenizer n30=...
"""

import argparse
import csv
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_shards as ps  # noqa: E402

# Elisions first (kept with their apostrophe and leading space), then letter runs
# with one optional leading non-letter (usually the space), single digits,
# punctuation runs, newlines, other whitespace.
SPLIT_PATTERN = (
    r" ?(?i:lorsqu|puisqu|jusqu|quelqu|presqu|quoiqu|qu|[cdjlmnst])['’]"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
DEFAULT_SPECIALS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "[MASK]"]


def build_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def special_tokens(old_tokenizer: str = None) -> list:
    """Specials of the old tokenizer, in id order (so their ids are kept), plus [MASK]."""
    specials = []
    if old_tokenizer:
        from tokenizers import Tokenizer
        old = Tokenizer.from_file(old_tokenizer)
        specials = [t.content for _, t in sorted(old.get_added_tokens_decoder().items())]
    for s in DEFAULT_SPECIALS:
        if s not in specials:
            specials.append(s)
    return specials


def read_manifest(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["decision"] == "keep"]


def sample_files(rows, split: str, noisy_subcorpora, noisy_share: float, sample_chars: float, seed: int):
    """Files of `split`, drawn at random until the clean and noisy character budgets are met."""
    rng = random.Random(seed)
    rows = [r for r in rows if r["split"] == split]
    noisy = [r for r in rows if r["subcorpus"] in noisy_subcorpora]
    clean = [r for r in rows if r["subcorpus"] not in noisy_subcorpora]
    picked = []
    for group, budget in ((clean, (1 - noisy_share) * sample_chars), (noisy, noisy_share * sample_chars)):
        rng.shuffle(group)
        got = 0
        for r in group:
            if got >= budget:
                break
            picked.append(r)
            got += int(r["n_chars"])
    rng.shuffle(picked)
    return picked


def budgets_not_met(clean_chars: int, noisy_chars: int, noisy_share: float, sample_chars: float) -> list:
    """Names of the character budgets ('clean', 'noisy') the sampled files fall short of."""
    return [name for name, got, want in
            (("clean", clean_chars, (1 - noisy_share) * sample_chars),
             ("noisy", noisy_chars, noisy_share * sample_chars)) if got < want]


def iter_texts(rows, chunk_chars: int = 1_000_000):
    """Cleaned file texts (same cleaning as prepare_shards), in chunks."""
    for r in rows:
        text = ps.read_text(Path(r["path"]), clean=True)
        for i in range(0, len(text), chunk_chars):
            yield text[i: i + chunk_chars]


def train(args):
    from tokenizers import trainers, pre_tokenizers
    rows = read_manifest(args.manifest)
    picked = sample_files(rows, "train", set(args.noisy_subcorpora), args.noisy_share,
                          args.sample_chars, args.seed)
    noisy_chars = sum(int(r["n_chars"]) for r in picked if r["subcorpus"] in args.noisy_subcorpora)
    total = sum(int(r["n_chars"]) for r in picked)
    print(f"{len(picked):,} files, {total/1e6:.0f}M chars, noisy share {noisy_chars/max(1, total):.3f}",
          flush=True)
    # A budget larger than the available text is filled with all of it, and the
    # share then follows what exists, not the target: say so.
    short = budgets_not_met(total - noisy_chars, noisy_chars, args.noisy_share, args.sample_chars)
    if short:
        print(f"WARNING: {' and '.join(short)} budget not met (not enough training text); "
              f"the noisy share is {noisy_chars/max(1, total):.3f}, target {args.noisy_share}", flush=True)
    specials = special_tokens(args.old_tokenizer)
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=args.min_frequency,
                                  special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=True)
    tok.train_from_iterator(iter_texts(picked), trainer=trainer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    meta = {"manifest": args.manifest, "noisy_subcorpora": args.noisy_subcorpora,
            "noisy_share_target": args.noisy_share, "noisy_share_actual": noisy_chars / max(1, total),
            "train_chars": total, "files": len(picked), "budgets_not_met": short,
            "vocab_size": tok.get_vocab_size(),
            "specials": {s: tok.token_to_id(s) for s in specials}, "seed": args.seed,
            "split_pattern": SPLIT_PATTERN}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta["specials"]), f"vocab {tok.get_vocab_size()}", f"-> {out}")


def evaluate(tokenizers: dict, rows, noisy_subcorpora, max_files: int, seed: int) -> dict:
    """Chars/token per sub-corpus, round-trip exactness, vocabulary share used mostly by noisy text."""
    from tokenizers import Tokenizer
    rng = random.Random(seed)
    val = defaultdict(list)
    for r in rows:
        if r["split"] == "val":
            val[r["subcorpus"]].append(r)
    for s in val:
        rng.shuffle(val[s])
        val[s] = val[s][:max_files]
    texts = {s: [unicodedata.normalize("NFC", ps.read_text(Path(r["path"]), clean=True)) for r in rs]
             for s, rs in val.items()}

    report = {}
    for name, path in tokenizers.items():
        tok = Tokenizer.from_file(path)
        res = {"per_subcorpus": {}, "roundtrip_exact": 0, "files": 0}
        counts = {"clean": Counter(), "noisy": Counter()}
        clean_chars = clean_toks = noisy_chars = noisy_toks = 0
        for s, ts in texts.items():
            chars = toks = 0
            kind = "noisy" if s in noisy_subcorpora else "clean"
            for t in ts:
                enc = tok.encode(t)
                chars += len(t)
                toks += len(enc.ids)
                counts[kind].update(enc.ids)
                res["roundtrip_exact"] += int(tok.decode(enc.ids) == t)
                res["files"] += 1
            res["per_subcorpus"][s] = round(chars / max(1, toks), 3)
            if kind == "noisy":
                noisy_chars, noisy_toks = noisy_chars + chars, noisy_toks + toks
            else:
                clean_chars, clean_toks = clean_chars + chars, clean_toks + toks
        res["chars_per_token_clean"] = round(clean_chars / max(1, clean_toks), 3)
        res["chars_per_token_noisy"] = round(noisy_chars / max(1, noisy_toks), 3)
        used = set(counts["clean"]) | set(counts["noisy"])
        noisy_mostly = [i for i in used if counts["noisy"][i] >= 5 and
                        counts["noisy"][i] >= 0.9 * (counts["noisy"][i] + counts["clean"][i])]
        res["vocab_used"] = len(used)
        res["vocab_share_mostly_noisy"] = round(len(noisy_mostly) / max(1, len(used)), 4)
        res["roundtrip_exact_rate"] = round(res["roundtrip_exact"] / max(1, res["files"]), 4)
        report[name] = res
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--manifest", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--vocab-size", type=int, default=32132, help="including special tokens")
    t.add_argument("--noisy-share", type=float, default=0.1)
    t.add_argument("--noisy-subcorpora", nargs="+", default=["modern_archive"])
    t.add_argument("--sample-chars", type=float, default=2e9)
    t.add_argument("--min-frequency", type=int, default=2)
    t.add_argument("--old-tokenizer", default=None, help="keep its special tokens and their ids")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--manifest", required=True)
    e.add_argument("--tokenizer", action="append", required=True, help="name=path")
    e.add_argument("--noisy-subcorpora", nargs="+", default=["modern_archive"])
    e.add_argument("--max-files", type=int, default=100, help="per sub-corpus")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    args = p.parse_args()

    if args.cmd == "train":
        train(args)
        return
    toks = dict(s.split("=", 1) for s in args.tokenizer)
    report = evaluate(toks, read_manifest(args.manifest), set(args.noisy_subcorpora),
                      args.max_files, args.seed)
    print(f"{'tokenizer':<12}{'chars/tok clean':>16}{'noisy':>8}{'exact':>8}{'noisy-vocab':>13}")
    for name, r in report.items():
        print(f"{name:<12}{r['chars_per_token_clean']:>16}{r['chars_per_token_noisy']:>8}"
              f"{r['roundtrip_exact_rate']:>8.1%}{r['vocab_share_mostly_noisy']:>13.1%}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
