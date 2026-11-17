#!/usr/bin/env python3
"""
Part-of-speech tagging on FreEM_LPM: fine-tune one encoder, score it per century.

The same harness serves the Candide encoder (a training checkpoint) and the
Hugging Face baselines (D'AlemBERT, CamemBERT), so every model gets the same
data, word pooling, head, schedule and metric.

Protocol (default = D'AlemBERT's, Gabay et al. 2022, Sect. 5): full fine-tuning,
learning rate 5e-6 for 10 epochs, linear warmup over 10% of the steps then
linear decay, word vector = mean of its subword vectors, a linear layer over
tags. Metric: word accuracy. The paper names no metric; its numbers are read
as accuracy. Our harness is plain PyTorch, not flair, and the head is a single
linear layer: reproducing D'AlemBERT's published scores with it is the check
that the harness is sound.

Data (FreEM-corpora/FreEMlpm, files under git LFS):
  --train / --dev   Models/Train_4/training_data/{train,dev}/pos_full.tsv
  --test-root       Data_outOfDomain: every directory holding .tsv files is
                    one test group, named by its path (e.g.
                    Data_outOfDomain_original/theatre_original/18_orig).

Model spec:
  candide:/path/to/step_N.pt   Candide encoder checkpoint (+ --tokenizer)
  hf:pjox/dalembert            any Hugging Face encoder

Usage:
  PYTHONPATH=. python scripts/pos_freem.py --model hf:pjox/dalembert \
      --train $F/Models/Train_4/training_data/train/pos_full.tsv \
      --dev   $F/Models/Train_4/training_data/dev/pos_full.tsv \
      --test-root $F/Data_outOfDomain --seed 1 --out results/pos/dalembert_s1.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TOKEN_COLUMNS = ("token", "form", "word", "tok", "forme")
SENT_END = {".", "!", "?", "…", "...", ";", ":"}
DEFAULT_TOKENIZER = "/data/pretrain_candide/tokenizer_v2g/tokenizer.json"


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class Sentence:
    words: List[str]
    tags: List[str]


def read_tsv(path, max_words: int = 64, split_on_punct: bool = True) -> Tuple[List[Sentence], dict]:
    """
    Read a pie-style TSV (header row with a POS column; one token per line).

    Sentences end at a blank line, after a sentence-final punctuation token, or
    after max_words words. The split only sets the context each word is seen
    in: every word of the file is scored once whatever the split.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        return [], {"words": 0, "skipped_lines": 0}
    header = [h.strip().lower() for h in lines[0].split("\t")]
    if "pos" not in header:
        raise ValueError(f"{path}: no POS column in header {lines[0]!r}")
    pos_col = header.index("pos")
    tok_col = next((header.index(c) for c in TOKEN_COLUMNS if c in header), 0)

    sents, words, tags, skipped = [], [], [], 0

    def flush():
        if words:
            sents.append(Sentence(list(words), list(tags)))
            words.clear()
            tags.clear()

    for line in lines[1:]:
        if not line.strip():
            flush()
            continue
        cols = line.split("\t")
        if len(cols) <= max(pos_col, tok_col) or not cols[tok_col].strip() or not cols[pos_col].strip():
            skipped += 1
            continue
        words.append(cols[tok_col].strip())
        tags.append(cols[pos_col].strip())
        if (split_on_punct and words[-1] in SENT_END) or len(words) >= max_words:
            flush()
    flush()
    return sents, {"words": sum(len(s.words) for s in sents), "skipped_lines": skipped}


def test_groups(root) -> Dict[str, List[Path]]:
    """Every directory under root that holds .tsv files is one group."""
    root = Path(root)
    groups = defaultdict(list)
    for f in sorted(root.rglob("*.tsv")):
        groups[str(f.parent.relative_to(root))].append(f)
    return dict(groups)


def century_summary(group_acc: Dict[str, Tuple[int, int]]) -> Dict[str, dict]:
    """
    Pool groups by (spelling, century) over genres, as D'AlemBERT's "Both" rows:
    micro-average of words. Group names look like
    Data_outOfDomain_original/theatre_original/18_orig.
    """
    pooled = defaultdict(lambda: [0, 0])
    for name, (correct, total) in group_acc.items():
        spelling = "original" if "original" in name else "normalised" if "normalised" in name else "other"
        m = re.search(r"(?:^|/)(\d{2})(?:_orig)?$", name)
        key = f"{spelling}/{m.group(1) if m else name}"
        pooled[key][0] += correct
        pooled[key][1] += total
    return {k: {"acc": c / t if t else float("nan"), "words": t} for k, (c, t) in sorted(pooled.items())}


def passage_key(path) -> str:
    """Century directory and file stem without _orig: one key per passage, both spellings."""
    p = Path(path)
    century = p.parent.name.replace("_orig", "")
    genre = "theatre" if "theatre" in str(p) else "varia" if "varia" in str(p) else ""
    return f"{genre}/{century}/{p.stem.replace('_orig', '')}"


def contaminated(overlap: dict, max_share: float) -> set:
    """Passage keys whose share of windows found in the pretraining corpus exceeds max_share."""
    return {passage_key(f) for f, v in overlap["files"].items() if v["share"] > max_share}


def clean_groups(per_file: Dict[str, Tuple[int, int]], excluded: set) -> Dict[str, Tuple[int, int]]:
    """Group totals over test files whose passage is not excluded."""
    out = defaultdict(lambda: [0, 0])
    for f, (c, t) in per_file.items():
        if passage_key(f) not in excluded:
            g = str(Path(f).parent)
            out[g][0] += c
            out[g][1] += t
    return {g: tuple(v) for g, v in out.items()}


class TagSet:
    def __init__(self, train: Sequence[Sentence]):
        counts = Counter(t for s in train for t in s.tags)
        self.tags = sorted(counts)
        self.index = {t: i for i, t in enumerate(self.tags)}

    def __len__(self):
        return len(self.tags)

    def ids(self, tags: Sequence[str]) -> List[int]:
        """Tag ids; a tag never seen in training is -1 (scored as an error)."""
        return [self.index.get(t, -1) for t in tags]


# ── Word to subword alignment ─────────────────────────────────────────────────

def align_words(word_pieces: Sequence[Sequence[int]], bos: Optional[int], eos: Optional[int]
                ) -> Tuple[List[int], List[int]]:
    """
    Concatenate per-word subword ids between bos and eos.

    Returns (ids, word_index) where word_index[j] is the word subword j belongs
    to, -1 for special tokens.
    """
    ids, widx = [], []
    if bos is not None:
        ids.append(bos)
        widx.append(-1)
    for w, pieces in enumerate(word_pieces):
        if not pieces:
            raise ValueError(f"word {w} encodes to no subword")
        ids.extend(pieces)
        widx.extend([w] * len(pieces))
    if eos is not None:
        ids.append(eos)
        widx.append(-1)
    return ids, widx


def group_by_word(ids: Sequence[int], word_ids: Sequence[Optional[int]], n_words: int,
                  unk: int) -> List[List[int]]:
    """Subword ids grouped per word from a fast tokenizer's word_ids(); unk for a word with none."""
    out = [[] for _ in range(n_words)]
    for i, w in zip(ids, word_ids):
        if w is not None:
            out[w].append(i)
    return [p or [unk] for p in out]


def split_to_fit(words: List[str], tags: List[str], n_subwords, max_len: int) -> List[Tuple[list, list]]:
    """Halve a sentence until each part has at most max_len subwords (specials included)."""
    if n_subwords(words) <= max_len or len(words) == 1:
        return [(words, tags)]
    h = len(words) // 2
    return (split_to_fit(words[:h], tags[:h], n_subwords, max_len)
            + split_to_fit(words[h:], tags[h:], n_subwords, max_len))


# ── Backbones ─────────────────────────────────────────────────────────────────

class CandideBackbone:
    """A Candide encoder checkpoint and the v2g tokenizer."""

    def __init__(self, ckpt_path: str, tokenizer_path: str):
        import torch
        from tokenizers import Tokenizer
        from ramanujan.models.architectures.encoder import CandideEncoder
        from ramanujan.models.architectures.encoder import CandideEncoderConfig

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = CandideEncoder(CandideEncoderConfig.from_dict(ckpt["model_config"]))
        self.model.load_state_dict(ckpt["model"])
        self.step = int(ckpt.get("step", -1))
        del ckpt
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.bos, self.eos = self.tok.token_to_id("<|bos|>"), self.tok.token_to_id("<|eos|>")
        self.pad = self.tok.token_to_id("<pad>")
        if self.pad is None:
            self.pad = 1
        self.max_len = self.model.config.max_seq_len
        self.hidden_size = self.model.config.d_model

    def pieces(self, words: Sequence[str]) -> List[List[int]]:
        # inside running text every word carries its leading space, as in pretraining
        texts = [(" " if i else "") + w for i, w in enumerate(words)]
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def encode(self, words):
        return align_words(self.pieces(words), self.bos, self.eos)

    def hidden(self, ids, attn):
        import torch
        none = torch.zeros_like(ids, dtype=torch.bool)       # skip the MLM head
        _, _, h = self.model(ids, attention_mask=attn, return_hidden_states=True,
                             masked_positions=none)
        return h


class HFBackbone:
    """Any Hugging Face encoder (D'AlemBERT, CamemBERT, ...)."""

    def __init__(self, name: str):
        from transformers import AutoModel, AutoTokenizer

        try:
            self.tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        except TypeError:
            self.tok = AutoTokenizer.from_pretrained(name)
        # safetensors first: transformers refuses .bin weights with torch < 2.6 (CVE-2025-32434)
        try:
            self.model = AutoModel.from_pretrained(name, use_safetensors=True)
        except (OSError, EnvironmentError):
            self.model = AutoModel.from_pretrained(name)
        self.step = -1
        self.bos, self.eos = self.tok.cls_token_id, self.tok.sep_token_id
        self.pad = self.tok.pad_token_id
        self.max_len = min(int(self.tok.model_max_length), 512)
        self.hidden_size = self.model.config.hidden_size

    def pieces(self, words):
        enc = self.tok(list(words), add_special_tokens=False, is_split_into_words=True)
        return group_by_word(enc["input_ids"], enc.word_ids(), len(words), self.tok.unk_token_id)

    def encode(self, words):
        return align_words(self.pieces(words), self.bos, self.eos)

    def hidden(self, ids, attn):
        return self.model(input_ids=ids, attention_mask=attn).last_hidden_state


def load_backbone(spec: str, tokenizer_path: str):
    kind, _, name = spec.partition(":")
    if kind == "candide":
        return CandideBackbone(name, tokenizer_path)
    if kind == "hf":
        return HFBackbone(name)
    raise ValueError(f"model spec {spec!r}: expected candide:<checkpoint> or hf:<name>")


# ── Batching and pooling ──────────────────────────────────────────────────────

@dataclass
class Example:
    ids: List[int]
    word_index: List[int]
    labels: List[int]
    group: str = ""
    file: str = ""


def make_examples(sents: Sequence[Sentence], backbone, tagset: TagSet, group: str = "",
                  file: str = "") -> List[Example]:
    out = []
    for s in sents:
        for words, tags in split_to_fit(s.words, s.tags, lambda ws: len(backbone.encode(ws)[0]),
                                        backbone.max_len):
            ids, widx = backbone.encode(words)
            out.append(Example(ids, widx, tagset.ids(tags), group, file))
    return out


def collate(batch: Sequence[Example], pad: int):
    import torch
    T = max(len(e.ids) for e in batch)
    W = max(len(e.labels) for e in batch)
    ids = torch.full((len(batch), T), pad, dtype=torch.long)
    attn = torch.zeros((len(batch), T), dtype=torch.long)
    widx = torch.full((len(batch), T), -1, dtype=torch.long)
    labels = torch.full((len(batch), W), -100, dtype=torch.long)
    wmask = torch.zeros((len(batch), W), dtype=torch.bool)
    for b, e in enumerate(batch):
        n = len(e.ids)
        ids[b, :n] = torch.tensor(e.ids)
        attn[b, :n] = 1
        widx[b, :n] = torch.tensor(e.word_index)
        labels[b, :len(e.labels)] = torch.tensor(e.labels)
        wmask[b, :len(e.labels)] = True
    return ids, attn, widx, labels, wmask


def pool_words(hidden, widx, n_words: int, mode: str = "mean"):
    """(B, T, H) subword states -> (B, n_words, H) word vectors (mean or first subword)."""
    import torch
    B, T, H = hidden.shape
    valid = widx >= 0
    gid = (torch.arange(B, device=hidden.device).unsqueeze(1) * n_words + widx.clamp(min=0))[valid]
    h = hidden[valid].float()
    if mode == "first":
        first = torch.ones_like(widx, dtype=torch.bool)
        first[:, 1:] = widx[:, 1:] != widx[:, :-1]
        keep = first[valid]
        gid, h = gid[keep], h[keep]
    sums = torch.zeros(B * n_words, H, device=hidden.device).index_add_(0, gid, h)
    counts = torch.zeros(B * n_words, device=hidden.device).index_add_(
        0, gid, torch.ones_like(gid, dtype=torch.float))
    return (sums / counts.clamp(min=1).unsqueeze(1)).view(B, n_words, H)


# ── Train and evaluate ────────────────────────────────────────────────────────

def batches(examples, size, shuffle, rng):
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        yield [examples[j] for j in order[i:i + size]]


def evaluate(backbone, head, examples, args, device, autocast):
    import torch
    backbone.model.eval()
    per_group, per_file = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    with torch.no_grad(), autocast():
        for batch in batches(examples, args.eval_batch_size, False, None):
            ids, attn, widx, labels, wmask = (t.to(device) for t in collate(batch, backbone.pad))
            words = pool_words(backbone.hidden(ids, attn), widx, labels.shape[1], args.pooling)
            pred = head(words).argmax(-1)
            correct = ((pred == labels) & wmask).sum(1).tolist()
            for e, c in zip(batch, correct):
                for d, k in ((per_group, e.group), (per_file, e.file)):
                    d[k][0] += int(c)
                    d[k][1] += len(e.labels)
    backbone.model.train()
    return {g: tuple(v) for g, v in per_group.items()}, {f: tuple(v) for f, v in per_file.items()}


def run(args) -> dict:
    import numpy as np
    import torch

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cpu", enabled=False)

    train_s, train_info = read_tsv(args.train, args.max_words)
    dev_s, dev_info = read_tsv(args.dev, args.max_words)
    if args.max_train_sentences:
        train_s = random.Random(args.seed).sample(train_s, min(args.max_train_sentences, len(train_s)))
    tagset = TagSet(train_s)

    backbone = load_backbone(args.model, args.tokenizer)
    backbone.model.to(device).train()
    head = torch.nn.Sequential(torch.nn.Dropout(args.dropout),
                               torch.nn.Linear(backbone.hidden_size, len(tagset))).to(device)

    train_ex = make_examples(train_s, backbone, tagset, "train")
    dev_ex = make_examples(dev_s, backbone, tagset, "dev")
    test_ex, test_info = [], {}
    for group, files in test_groups(args.test_root).items():
        n_words, unseen = 0, 0
        for f in files:
            s, info = read_tsv(f, args.max_words)
            n_words += info["words"]
            unseen += sum(t not in tagset.index for x in s for t in x.tags)
            test_ex += make_examples(s, backbone, tagset, group, str(f.relative_to(args.test_root)))
        test_info[group] = {"files": len(files), "words": n_words, "unseen_tag_words": unseen}

    head_lr = args.head_lr or args.lr
    opt = torch.optim.AdamW([
        {"params": backbone.model.parameters(), "lr": args.lr},
        {"params": head.parameters(), "lr": head_lr},
    ], weight_decay=args.weight_decay)
    total = args.epochs * math.ceil(len(train_ex) / args.batch_size)
    warm = int(args.warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    rng = random.Random(args.seed)

    history = []
    for epoch in range(1, args.epochs + 1):
        running, n = 0.0, 0
        for batch in batches(train_ex, args.batch_size, True, rng):
            ids, attn, widx, labels, _ = (t.to(device) for t in collate(batch, backbone.pad))
            labels = labels.masked_fill(labels < 0, -100)
            with autocast():
                words = pool_words(backbone.hidden(ids, attn), widx, labels.shape[1], args.pooling)
                logits = head(words).float()
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.model.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n += 1
        c, t = evaluate(backbone, head, dev_ex, args, device, autocast)[0]["dev"]
        history.append({"epoch": epoch, "train_loss": running / max(1, n), "dev_acc": c / t})
        print(f"epoch {epoch}: train loss {running / max(1, n):.4f} | dev acc {c / t:.4f}", flush=True)

    group_acc, file_acc = evaluate(backbone, head, test_ex, args, device, autocast)
    clean = {}
    if args.overlap:
        excluded = contaminated(json.loads(Path(args.overlap).read_text()), args.max_overlap)
        clean = {"overlap_file": str(args.overlap), "max_overlap": args.max_overlap,
                 "excluded_passages": sorted(excluded),
                 "centuries": century_summary(clean_groups(file_acc, excluded))}
    result = {
        "model": args.model,
        "pretraining_step": backbone.step,
        "protocol": {k: getattr(args, k) for k in (
            "lr", "head_lr", "epochs", "batch_size", "warmup_frac", "weight_decay", "dropout",
            "pooling", "max_words", "max_train_sentences", "seed")},
        "data": {"train": str(args.train), "dev": str(args.dev), "test_root": str(args.test_root),
                 "train_sentences": len(train_s), "train_words": sum(len(s.words) for s in train_s),
                 "dev_words": dev_info["words"], "skipped_lines": train_info["skipped_lines"],
                 "tags": len(tagset)},
        "params_m": round(sum(p.numel() for p in backbone.model.parameters()) / 1e6, 1),
        "dev": history,
        "test_groups": {g: {"acc": c / t if t else float("nan"), **test_info[g]}
                        for g, (c, t) in sorted(group_acc.items())},
        "test_centuries": century_summary(group_acc),
        "test_clean": clean,
        "test_files": {f: {"correct": c, "words": t} for f, (c, t) in sorted(file_acc.items())},
        "seconds": round(time.time() - t0),
        "torch": torch.__version__,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="candide:<checkpoint.pt> or hf:<name>")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Candide models only")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--head-lr", type=float, default=None, help="default: --lr")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16, help="sentences")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["mean", "first"], default="mean")
    p.add_argument("--max-words", type=int, default=64, help="longest sentence chunk, in words")
    p.add_argument("--max-train-sentences", type=int, default=None, help="subsample (quick protocol)")
    p.add_argument("--overlap", default=None, help="overlap_freem.py JSON: also score without read passages")
    p.add_argument("--max-overlap", type=float, default=0.05, help="share above which a passage is excluded")
    return p


def main():
    args = build_parser().parse_args()
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    clean = result["test_clean"].get("centuries", {})
    for k, v in result["test_centuries"].items():
        c = clean.get(k)
        extra = f"   clean {c['acc'] * 100:6.2f} ({c['words']} words)" if c else ""
        print(f"{k:16s} {v['acc'] * 100:6.2f}  ({v['words']} words){extra}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
