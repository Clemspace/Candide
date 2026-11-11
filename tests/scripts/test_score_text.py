"""Bits-per-byte scorer: windowing, exactly-once scoring, byte counts, aggregation."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "score_text", Path(__file__).resolve().parents[2] / "scripts" / "score_text.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


class TableBackend:
    """A model whose next-token distribution depends only on the previous token.

    Its total loss on a sequence does not depend on how the sequence is cut into windows,
    so any correct windowing, batching and padding must return the same number.
    """
    name = "table"
    tokenizer_name = "bytes"

    def __init__(self, vocab=50, bos=True, max_len=64, seed=0):
        self.table = torch.randn(vocab, vocab, generator=torch.Generator().manual_seed(seed))
        self.vocab, self.max_len = vocab, max_len
        self.bos_id = 0 if bos else None

    def encode(self, text):
        return [1 + (b % (self.vocab - 1)) for b in text.encode("utf-8")]

    def logits(self, ids, attention_mask):
        return self.table[ids]

    def reference(self, ids):
        """Loss computed directly, one token after the other."""
        seq = ([self.bos_id] if self.bos_id is not None else []) + list(ids)
        logp = torch.log_softmax(self.table.double(), dim=-1)
        return -sum(float(logp[a, b]) for a, b in zip(seq[:-1], seq[1:]))


@pytest.mark.parametrize("n", [1, 2, 5, 64, 65, 200, 1000])
@pytest.mark.parametrize("max_len,stride", [(8, 1), (8, 7), (64, 32), (64, 63), (1024, 512)])
def test_windows_score_every_position_once_within_the_context(n, max_len, stride):
    seen = []
    for begin, end, first in S.windows(n, max_len, stride):
        assert 0 <= begin < first < end <= n          # every target has a token before it
        assert end - begin <= max_len
        assert first - begin >= min(first, max_len - stride)   # guaranteed context
        seen += list(range(first, end))
    assert seen == list(range(1, n))


def test_windows_reject_a_stride_that_leaves_no_context():
    with pytest.raises(ValueError):
        list(S.windows(100, 16, 16))


@pytest.mark.parametrize("max_len,stride,batch", [(64, 32, 1), (64, 32, 8), (64, 63, 3), (16, 1, 5), (16, 15, 2)])
def test_total_loss_does_not_depend_on_windows_or_batches(max_len, stride, batch):
    backend = TableBackend()
    ids = backend.encode("Il faut cultiver notre jardin. " * 12)          # several windows
    nll, scored = S.score_ids(backend, ids, max_len, stride, batch)
    assert scored == len(ids)
    assert nll == pytest.approx(backend.reference(ids), rel=1e-5)


def test_short_text_and_empty_text():
    backend = TableBackend()
    assert S.score_ids(backend, [], 64, 32) == (0.0, 0)
    nll, scored = S.score_ids(backend, [7], 64, 32)
    assert scored == 1 and nll == pytest.approx(backend.reference([7]), rel=1e-5)


def test_model_without_bos_cannot_score_its_first_token():
    backend = TableBackend(bos=False)
    ids = backend.encode("Candide, ou l'Optimisme")
    nll, scored = S.score_ids(backend, ids, 64, 32)
    assert scored == len(ids) - 1
    assert nll == pytest.approx(backend.reference(ids), rel=1e-5)


def test_bits_per_byte_uses_bytes_of_the_nfc_text():
    backend = TableBackend()
    composed = S.score_text(backend, "été", "c", clean=False, max_len=64, stride=32)
    decomposed = S.score_text(backend, "été", "d", clean=False, max_len=64, stride=32)
    assert composed.n_bytes == decomposed.n_bytes == 5            # é t é = 2 + 1 + 2 bytes
    assert composed.nll_nats == pytest.approx(decomposed.nll_nats)
    assert composed.bits_per_byte == pytest.approx(composed.nll_nats / math.log(2) / 5)


def test_uniform_model_costs_log2_vocabulary_per_token():
    backend = TableBackend(vocab=32)
    backend.table = torch.zeros(32, 32)
    s = S.score_text(backend, "abcdefgh" * 30, clean=False, max_len=64, stride=32)
    assert s.bits_per_token == pytest.approx(5.0, rel=1e-6)        # log2(32)
    assert s.bits_per_byte == pytest.approx(5.0, rel=1e-6)         # one token per byte here


def test_compressor_reference_point():
    backend = S.load_backend("compress:xz", None, None, "cpu")
    repetitive = S.score_text(backend, "Tout est bien. " * 400, clean=False)
    import random
    rng = random.Random(0)
    varied = S.score_text(backend, "".join(chr(33 + rng.randrange(90)) for _ in range(6000)), clean=False)
    assert 0 < repetitive.bits_per_byte < 1 < varied.bits_per_byte < 9
    assert repetitive.n_tokens == 0 and repetitive.as_dict()["bits_per_token"] is None
    assert S.summarize([repetitive, varied])["all"]["bits_per_token"] is None


def test_cleaning_matches_training_and_changes_the_byte_count():
    raw = "Tom &amp; Jerry"
    cleaned = S.prepare(raw, clean=True)
    assert cleaned == S._sibling("prepare_shards").clean_text(raw)
    assert S.prepare(raw, clean=False) == raw


def test_excerpt_takes_the_middle():
    text = "".join(f"ligne {i}\n" for i in range(1000))
    cut = S.excerpt(text, 200)
    assert len(cut) <= 200 and cut.startswith("ligne ")
    assert "ligne 0\n" not in cut and "ligne 999\n" not in cut
    assert S.excerpt("court", 200) == "court" and S.excerpt(text, None) == text


def test_summary_is_total_bits_over_total_bytes_per_group():
    scores = [
        S.Score("a", n_bytes=100, n_tokens=30, n_scored=30, nll_nats=100 * math.log(2), meta={"dir": "x"}),
        S.Score("b", n_bytes=300, n_tokens=90, n_scored=90, nll_nats=900 * math.log(2), meta={"dir": "x"}),
        S.Score("c", n_bytes=100, n_tokens=30, n_scored=29, nll_nats=200 * math.log(2), meta={"dir": "y"}),
    ]
    out = S.summarize(scores, "dir", n_boot=200)
    assert out["x"]["bits_per_byte"] == pytest.approx(1000 / 400)   # not the mean of 1 and 3
    assert out["y"]["bits_per_byte"] == pytest.approx(2.0)
    assert out["all"]["bits_per_byte"] == pytest.approx(1200 / 500)
    assert out["all"]["tokens_not_scored"] == 1
    lo, hi = out["x"]["ci95"]
    assert lo <= out["x"]["bits_per_byte"] <= hi
    assert "ci95" not in out["y"]                                   # one document: no interval


def test_command_line_on_files_and_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "load_backend", lambda *a, **k: TableBackend())
    for d, name, text in [("canon", "a.txt", "Tout est bien. " * 20), ("press", "b.txt", "Nouvelles de Paris. " * 20)]:
        (tmp_path / d).mkdir(exist_ok=True)
        (tmp_path / d / name).write_text(text, encoding="utf-8")
    out = tmp_path / "res" / "bpb"
    assert S.main(["--model", "candide", "--files", str(tmp_path / "canon"), str(tmp_path / "press" / "b.txt"),
                   "--group-by", "dir", "--raw", "--max-len", "64", "--stride", "32", "--device", "cpu",
                   "--out", str(out)]) == 0
    import json
    summary = json.loads(Path(f"{out}.summary.json").read_text())
    assert set(summary["groups"]) == {"all", "canon", "press"}
    assert summary["settings"]["max_len"] == 64
    assert len(Path(f"{out}.docs.jsonl").read_text().splitlines()) == 2

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{"id": 1, "a": "il estoit", "b": "il \\u00e9tait", "kind": "spelling"}\n', encoding="utf-8")
    assert S.main(["--model", "candide", "--pairs", str(pairs), "--raw", "--max-len", "64", "--stride", "32",
                   "--device", "cpu", "--out", str(tmp_path / "res" / "pairs")]) == 0
    row = json.loads(Path(f"{tmp_path / 'res' / 'pairs'}.docs.jsonl").read_text())
    assert row["kind"] == "spelling" and isinstance(row["prefers_a_total"], bool)


def test_summarize_again_from_saved_documents_without_a_model(tmp_path):
    import json
    docs = tmp_path / "x.docs.jsonl"
    rows = [{"id": "a", "n_bytes": 100, "n_tokens": 30, "n_scored": 30, "nll_nats": 100 * math.log(2),
             "bits_per_byte": 1.0, "bits_per_token": 3.3, "subcorpus": "canon", "decade": "1750s"},
            {"id": "b", "n_bytes": 100, "n_tokens": 30, "n_scored": 30, "nll_nats": 300 * math.log(2),
             "bits_per_byte": 3.0, "bits_per_token": 10.0, "subcorpus": "canon", "decade": "1760s"}]
    docs.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert S.main(["--from-docs", str(docs), "--group-by", "decade", "--out", str(tmp_path / "by_decade")]) == 0
    summary = json.loads((tmp_path / "by_decade.summary.json").read_text())
    assert summary["group_by"] == "decade"
    assert summary["groups"]["1750s"]["bits_per_byte"] == pytest.approx(1.0)
    assert summary["groups"]["1760s"]["bits_per_byte"] == pytest.approx(3.0)
    assert summary["groups"]["all"]["bits_per_byte"] == pytest.approx(2.0)
