#!/usr/bin/env python3
"""Score text in bits per byte, with Candide or with any Hugging Face causal model.

Bits per byte is the number of bits a language model needs, on average, to encode one
byte of a text. Lower is better. Unlike loss per token or perplexity it does not depend
on the tokenizer, so models with different vocabularies can be compared on the same text:

    bits per byte = (sum over tokens of -log2 p(token | previous tokens)) / bytes of the text

What is measured, exactly
  * The text is prepared the same way for every model: HTML entities cleaned as in
    training (prepare_shards.clean_text), then Unicode NFC. The byte count is that of the
    prepared text in UTF-8. Use --raw to skip the cleaning.
  * Every token of the text is scored exactly once. A text longer than the context is
    read through overlapping windows; a token is only ever scored with at least
    (max_len - stride) tokens of context before it, when the text has that many.
  * The text starts after a beginning-of-text token, which is context and is never
    scored. A model without one cannot score its first token: that is counted and
    reported (n_tokens - n_scored), not hidden.
  * All models should be run with the same --max-len so that none sees more context.

Inputs (one of)
    --jsonl FILE         one JSON object per line: {"id": ..., "text": ..., <metadata>}
    --files PATH ...     text files, directories (searched for *.txt), or @list.txt with one
                         path per line; the metadata field "dir" is the parent directory name
    --pairs FILE         JSON lines {"id": ..., "a": ..., "b": ...}: which of two versions of
                         a passage the model finds more likely (spelling variants, a curated
                         transcription against its OCR, a true statement against a false one)
    --from-docs FILE     a .docs.jsonl from an earlier run: summarize again by another field
                         (--group-by decade, then --group-by subcorpus) without rescoring

Outputs
    <out>.docs.jsonl     one line per document (or pair)
    <out>.summary.json   totals per group (--group-by), with a bootstrap 95% interval over
                         documents, and every setting needed to reproduce the run

Examples
    # Candide on held-out text, by sub-corpus
    PYTHONPATH=. python scripts/score_text.py --model candide \\
        --checkpoint runs/full/candide-v2g-224m/checkpoints/final.pt \\
        --tokenizer /data/pretrain_candide/tokenizer/tokenizer.json \\
        --files @val_files.txt --group-by dir --out results/2026-11-13/bpb_candide_val

    # the same text with a model from the Hugging Face hub
    PYTHONPATH=. python scripts/score_text.py --model hf:gpt2 \\
        --files @val_files.txt --group-by dir --out results/2026-11-13/bpb_gpt2_val

    # and with a file compressor, as a reference point anyone can relate to
    python scripts/score_text.py --model compress:xz \\
        --files @val_files.txt --group-by dir --out results/2026-11-13/bpb_xz_val
"""
import argparse
import importlib.util
import json
import math
import random
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

LN2 = math.log(2.0)
_HERE = Path(__file__).resolve().parent
_SIBLINGS: dict = {}


def _sibling(name: str):
    """Import another script of this directory as a module (same idiom as rescore_checkpoints.py)."""
    if name not in _SIBLINGS:
        spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SIBLINGS[name] = mod
    return _SIBLINGS[name]


# ── Text ────────────────────────────────────────────────────────────────────

def prepare(text: str, clean: bool = True) -> str:
    """The text as every model will see it: entities cleaned as in training, then NFC."""
    if clean:
        text = _sibling("prepare_shards").clean_text(text)
    return unicodedata.normalize("NFC", text)


def excerpt(text: str, n_chars: int | None) -> str:
    """At most n_chars characters from the middle of the text, starting on a line break if
    one is near. The middle avoids title pages and tables at either end of a book."""
    if not n_chars or len(text) <= n_chars:
        return text
    start = (len(text) - n_chars) // 2
    nl = text.find("\n", start, start + n_chars // 10)
    if nl != -1:
        start = nl + 1
    return text[start:start + n_chars]


# ── Windows ─────────────────────────────────────────────────────────────────

def windows(n: int, max_len: int, stride: int) -> Iterator[tuple[int, int, int]]:
    """Cover positions 1..n-1 of a sequence of length n (position 0 is context only).

    Yields (begin, end, first_target): the model is fed positions begin..end-1 and the
    tokens at first_target..end-1 are scored. Every position from 1 to n-1 is a target in
    exactly one window, and each window has at most max_len positions.
    """
    if not 1 <= stride <= max_len - 1:
        raise ValueError(f"stride must be between 1 and max_len - 1, got {stride} for max_len {max_len}")
    done = 1
    while done < n:
        end = min(done + stride, n)
        begin = max(0, end - max_len)
        yield begin, end, done
        done = end


# ── Models ──────────────────────────────────────────────────────────────────
# A backend has: name, bos_id (or None), max_len, encode(text) -> list[int],
# and logits(ids[B, T], attention_mask[B, T]) -> float tensor [B, T, vocabulary].

class CandideBackend:
    """A checkpoint written by ramanujan/training/train_candide18c.py, with its tokenizer."""

    def __init__(self, checkpoint: str, tokenizer: str, device: str):
        from tokenizers import Tokenizer
        self.model = _sibling("memorization_probe").load_run(checkpoint, device)
        self.tok = Tokenizer.from_file(tokenizer)
        vocab_model = int(self.model.config.vocab_size)
        if self.tok.get_vocab_size() != vocab_model:
            raise ValueError(
                f"tokenizer has {self.tok.get_vocab_size()} entries, the checkpoint expects {vocab_model}: "
                "wrong tokenizer for this checkpoint (v1 and v2g differ)")
        self.bos_id = self.tok.token_to_id("<|bos|>")
        self.max_len = int(self.model.config.max_seq_len)
        self.name = f"candide:{checkpoint}"
        self.tokenizer_name = str(tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def logits(self, ids, attention_mask):
        return self.model(ids)      # right-padded batches need no mask in a causal model


class HFBackend:
    """Any causal language model of the Hugging Face hub (or a local directory)."""

    def __init__(self, name: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        self.bos_id = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        cfg = self.model.config
        self.max_len = int(getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None) or 1024)
        self.name = f"hf:{name}"
        self.tokenizer_name = name

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def logits(self, ids, attention_mask):
        return self.model(input_ids=ids, attention_mask=attention_mask).logits


class CompressorBackend:
    """A general-purpose compressor as a reference point: 'compress:xz', 'compress:bz2', 'compress:gzip'.

    Not a language model. A compressor also turns text into bits, so its bits per byte sit
    on the same scale: 8 is no compression at all, and a language model should be well
    below what a compressor reaches. Each text is compressed on its own.
    """
    bos_id, max_len, tokenizer_name = None, 10 ** 9, "none"

    def __init__(self, algo: str):
        import bz2, gzip, lzma
        table = {"xz": lambda b: lzma.compress(b, preset=9), "bz2": lambda b: bz2.compress(b, 9),
                 "gzip": lambda b: gzip.compress(b, 9)}
        if algo not in table:
            raise SystemExit(f"unknown compressor {algo!r}: use one of {sorted(table)}")
        self.compress, self.name = table[algo], f"compress:{algo}"

    def score(self, text: str) -> tuple[float, int, int]:
        """(nats, tokens, tokens scored) for a prepared text."""
        return len(self.compress(text.encode("utf-8"))) * 8 * LN2, 0, 0


def load_backend(spec: str, checkpoint: str | None, tokenizer: str | None, device: str):
    if spec.startswith("compress:"):
        return CompressorBackend(spec[len("compress:"):])
    if spec == "candide":
        if not (checkpoint and tokenizer):
            raise SystemExit("--model candide needs --checkpoint and --tokenizer")
        return CandideBackend(checkpoint, tokenizer, device)
    if spec.startswith("hf:"):
        return HFBackend(spec[3:], device)
    raise SystemExit(f"--model must be 'candide', 'hf:<name>' or 'compress:<xz|bz2|gzip>', got {spec!r}")


# ── Scoring ─────────────────────────────────────────────────────────────────

def score_ids(backend, ids: Sequence[int], max_len: int, stride: int,
              batch_size: int = 8, device: str = "cpu") -> tuple[float, int]:
    """Total negative log-likelihood (nats) of a token sequence, and the number of tokens scored."""
    import torch
    import torch.nn.functional as F

    if len(ids) == 0:
        return 0.0, 0
    seq = ([backend.bos_id] if backend.bos_id is not None else []) + list(ids)
    spans = list(windows(len(seq), max_len, stride))
    total, count = 0.0, 0
    if str(device).startswith("cuda"):
        precision = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        import contextlib
        precision = contextlib.nullcontext
    for i in range(0, len(spans), batch_size):
        chunk = spans[i:i + batch_size]
        width = max(end - begin for begin, end, _ in chunk)
        x = torch.zeros(len(chunk), width, dtype=torch.long)
        attention = torch.zeros(len(chunk), width, dtype=torch.long)
        is_target = torch.zeros(len(chunk), width, dtype=torch.bool)
        for row, (begin, end, first_target) in enumerate(chunk):
            x[row, :end - begin] = torch.tensor(seq[begin:end], dtype=torch.long)
            attention[row, :end - begin] = 1
            is_target[row, first_target - begin:end - begin] = True
        x, attention, is_target = x.to(device), attention.to(device), is_target.to(device)
        with torch.no_grad(), precision():
            logits = backend.logits(x, attention)
        logits = logits[:, :-1].float()                     # position j predicts token j + 1
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                              reduction="none").reshape(x.shape[0], -1)
        keep = is_target[:, 1:]
        total += float(nll[keep].double().sum())
        count += int(keep.sum())
    return total, count


@dataclass
class Score:
    id: str
    n_bytes: int
    n_tokens: int
    n_scored: int
    nll_nats: float
    meta: dict = field(default_factory=dict)

    @property
    def bits_per_byte(self) -> float:
        return self.nll_nats / LN2 / self.n_bytes if self.n_bytes else float("nan")

    @property
    def bits_per_token(self) -> float:
        return self.nll_nats / LN2 / self.n_scored if self.n_scored else float("nan")

    def as_dict(self) -> dict:
        return {"id": self.id, "n_bytes": self.n_bytes, "n_tokens": self.n_tokens,
                "n_scored": self.n_scored, "nll_nats": round(self.nll_nats, 4),
                "bits_per_byte": _round(self.bits_per_byte, 5),
                "bits_per_token": _round(self.bits_per_token, 5), **self.meta}


def score_text(backend, text: str, doc_id: str = "", meta: dict | None = None, *,
               max_len: int = 1024, stride: int = 512, batch_size: int = 8, device: str = "cpu",
               clean: bool = True, excerpt_chars: int | None = None) -> Score:
    """Prepare one text, score it, and return its totals."""
    text = excerpt(prepare(text, clean=clean), excerpt_chars)
    n_bytes = len(text.encode("utf-8"))
    if hasattr(backend, "score"):                       # a compressor: no tokens, no windows
        nll, n_tokens, scored = backend.score(text)
        return Score(doc_id, n_bytes, n_tokens, scored, nll, dict(meta or {}))
    ids = backend.encode(text)
    nll, scored = score_ids(backend, ids, min(max_len, backend.max_len), stride, batch_size, device)
    return Score(doc_id, n_bytes, len(ids), scored, nll, dict(meta or {}))


# ── Aggregation ─────────────────────────────────────────────────────────────

def _round(x: float, digits: int):
    return None if x != x else round(x, digits)      # NaN is not valid JSON


def _ratio(scores: Sequence[Score]) -> float:
    b = sum(s.n_bytes for s in scores)
    return sum(s.nll_nats for s in scores) / LN2 / b if b else float("nan")


def summarize(scores: Sequence[Score], group_by: str | None = None,
              n_boot: int = 1000, seed: int = 0) -> dict:
    """Bits per byte over all documents and per group: total bits over total bytes (long
    documents weigh more), with a 95% interval from resampling documents."""
    groups: dict[str, list[Score]] = {"all": list(scores)}
    if group_by:
        for s in scores:
            groups.setdefault(str(s.meta.get(group_by, "unknown")), []).append(s)
    rng = random.Random(seed)
    out = {}
    for name, members in groups.items():
        entry = {
            "documents": len(members),
            "bytes": sum(s.n_bytes for s in members),
            "tokens": sum(s.n_tokens for s in members),
            "tokens_not_scored": sum(s.n_tokens - s.n_scored for s in members),
            "bits_per_byte": round(_ratio(members), 5),
            "bits_per_token": _round(sum(s.nll_nats for s in members) / LN2 / scored, 5) if (scored := sum(s.n_scored for s in members)) else None,
            "bytes_per_token": round(sum(s.n_bytes for s in members) / n_tok, 4) if (n_tok := sum(s.n_tokens for s in members)) else None,
        }
        if len(members) >= 2 and n_boot:
            draws = sorted(_ratio([members[rng.randrange(len(members))] for _ in members])
                           for _ in range(n_boot))
            entry["ci95"] = [round(draws[int(0.025 * n_boot)], 5), round(draws[min(n_boot - 1, int(0.975 * n_boot))], 5)]
        out[name] = entry
    return out


# ── Inputs ──────────────────────────────────────────────────────────────────

def iter_files(args_files: Sequence[str]) -> Iterator[Path]:
    for a in args_files:
        if a.startswith("@"):
            for line in Path(a[1:]).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield Path(line.strip())
        elif Path(a).is_dir():
            yield from sorted(Path(a).rglob("*.txt"))
        else:
            yield Path(a)


def iter_documents(args) -> Iterator[tuple[str, str, dict]]:
    if args.jsonl:
        with open(args.jsonl, encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.pop(args.text_field)
                yield str(row.get(args.id_field, n)), text, {k: v for k, v in row.items() if k != args.id_field}
    else:
        for p in iter_files(args.files):
            yield str(p), p.read_text(encoding="utf-8", errors="replace"), {"dir": p.parent.name}


# ── Command line ────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="'candide', 'hf:<hub name or local directory>', or 'compress:xz|bz2|gzip' as a reference point")
    ap.add_argument("--checkpoint", help="Candide checkpoint (.pt)")
    ap.add_argument("--tokenizer", help="Candide tokenizer.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl")
    src.add_argument("--files", nargs="+")
    src.add_argument("--pairs")
    src.add_argument("--from-docs", help="a .docs.jsonl written earlier: summarize it again by another "
                                         "--group-by, without a model and without rescoring")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--group-by", default=None, help="metadata field to break the summary down by")
    ap.add_argument("--max-len", type=int, default=1024, help="context in tokens; use the same for every model")
    ap.add_argument("--stride", type=int, default=512, help="new tokens per window; the rest is context")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--excerpt-chars", type=int, default=None, help="score only this many characters from the middle of each text")
    ap.add_argument("--raw", action="store_true", help="do not clean HTML entities (NFC is always applied)")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args(argv)

    if args.from_docs:
        keys = {"id", "n_bytes", "n_tokens", "n_scored", "nll_nats", "bits_per_byte", "bits_per_token"}
        scores = []
        for line in open(args.from_docs, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                scores.append(Score(str(r["id"]), r["n_bytes"], r["n_tokens"], r["n_scored"], r["nll_nats"],
                                    {k: v for k, v in r.items() if k not in keys}))
        summary = {"from_docs": args.from_docs, "group_by": args.group_by,
                   "groups": summarize(scores, args.group_by, args.bootstrap)}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
        print(json.dumps(summary["groups"], indent=1, ensure_ascii=False))
        print(f"wrote {args.out}.summary.json")
        return 0
    if not args.model:
        ap.error("--model is required unless --from-docs is given")

    if args.model.startswith("compress:"):
        device, torch_version = "cpu", None             # no model, no PyTorch needed
    else:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_version = torch.__version__
    backend = load_backend(args.model, args.checkpoint, args.tokenizer, device)
    max_len = min(args.max_len, backend.max_len)
    if max_len < args.max_len:
        print(f"note: {backend.name} accepts {backend.max_len} tokens; scoring with max_len {max_len}", file=sys.stderr)
    stride = min(args.stride, max_len - 1)
    kw = dict(max_len=max_len, stride=stride, batch_size=args.batch_size, device=device,
              clean=not args.raw, excerpt_chars=args.excerpt_chars)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {"model": backend.name, "tokenizer": backend.tokenizer_name, "bos_token": backend.bos_id is not None,
                "max_len": max_len, "stride": stride, "clean_entities": not args.raw,
                "excerpt_chars": args.excerpt_chars, "device": device, "torch": torch_version,
                "input": args.jsonl or args.pairs or args.files, "group_by": args.group_by}

    with open(f"{out}.docs.jsonl", "w", encoding="utf-8") as fh:
        if args.pairs:
            rows = []
            for n, line in enumerate(open(args.pairs, encoding="utf-8")):
                if not line.strip():
                    continue
                row = json.loads(line)
                a = score_text(backend, row["a"], **kw)
                b = score_text(backend, row["b"], **kw)
                res = {"id": row.get("id", n), "nll_a": round(a.nll_nats, 4), "nll_b": round(b.nll_nats, 4),
                       "bits_per_byte_a": round(a.bits_per_byte, 5), "bits_per_byte_b": round(b.bits_per_byte, 5),
                       "prefers_a_total": a.nll_nats < b.nll_nats,
                       "prefers_a_per_byte": a.bits_per_byte < b.bits_per_byte,
                       **{k: v for k, v in row.items() if k not in ("a", "b", "id")}}
                rows.append(res)
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            n_pairs = max(1, len(rows))
            summary = {"pairs": len(rows),
                       "prefers_a_total": round(sum(r["prefers_a_total"] for r in rows) / n_pairs, 4),
                       "prefers_a_per_byte": round(sum(r["prefers_a_per_byte"] for r in rows) / n_pairs, 4)}
        else:
            scores = []
            for doc_id, text, meta in iter_documents(args):
                s = score_text(backend, text, doc_id, meta, **kw)
                if s.n_bytes == 0:
                    continue
                scores.append(s)
                fh.write(json.dumps(s.as_dict(), ensure_ascii=False) + "\n")
            summary = {"groups": summarize(scores, args.group_by, args.bootstrap)}

    Path(f"{out}.summary.json").write_text(
        json.dumps({"settings": settings, **summary}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {out}.docs.jsonl and {out}.summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
