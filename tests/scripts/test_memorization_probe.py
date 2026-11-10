"""Memorization probe: passage selection by corpus repetition count."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "memorization_probe", Path(__file__).resolve().parents[2] / "scripts" / "memorization_probe.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def _write_shard(path, tokens):
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1] = M.dup.MAGIC, len(tokens)
    path.write_bytes(header.tobytes() + tokens.astype(np.uint16).tobytes())


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.default_rng(0)
    repeated = rng.integers(10, 30000, 200)               # a passage repeated 12 times
    parts = []
    for _ in range(12):
        parts += [rng.integers(10, 30000, 300), repeated]
    parts.append(rng.integers(10, 30000, 3000))
    tokens = np.concatenate(parts)
    _write_shard(tmp_path / "shard_00000.bin", tokens[: len(tokens) // 2])
    _write_shard(tmp_path / "shard_00001.bin", tokens[len(tokens) // 2:])
    return sorted(tmp_path.glob("shard_*.bin")), repeated


def test_index_locations_match_hashes(corpus):
    shards, _ = corpus
    h, s, p = M.index_kgrams(shards, k=20, shift=2)
    for i in range(0, len(h), max(1, len(h) // 50)):
        toks = np.asarray(M.dup.load_shard(shards[s[i]])[p[i]: p[i] + 20])
        assert M.dup.kgram_hashes(toks, 20)[0] == h[i]


def test_buckets_get_the_right_counts(corpus):
    shards, repeated = corpus
    sel = M.select_passages(*M.index_kgrams(shards, k=20, shift=2), per_bucket=50, min_pos=32)
    assert sel["1"] and sel["10-99"]
    assert all(c == 1 for _, _, c in sel["1"])
    assert all(10 <= c <= 99 for _, _, c in sel["10-99"])
    assert not sel["100+"]
    # every 10-99 passage is a piece of the repeated block
    w = M.passage_tokens(shards, sel["10-99"], prefix=32, k=20)
    block = repeated.tolist()
    for row in w:
        target = row[32:].tolist()
        assert any(block[j: j + 20] == target for j in range(len(block) - 19))


def test_passages_have_full_prefix(corpus):
    shards, _ = corpus
    sel = M.select_passages(*M.index_kgrams(shards, k=20, shift=2), per_bucket=20, min_pos=32)
    for b, items in sel.items():
        w = M.passage_tokens(shards, items, prefix=32, k=20)
        assert w.shape == (len(items), 52)
        assert all(pos >= 32 for _, pos, _ in items)
