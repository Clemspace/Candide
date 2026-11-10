"""Duplication audit: hashing, content-defined sampling, statistics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "dup_audit", Path(__file__).resolve().parents[2] / "scripts" / "dup_audit.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def test_hash_matches_direct_computation():
    t = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.uint16)
    h = D.kgram_hashes(t, 3)
    B = 1_000_003
    direct = [(int(t[i]) * B * B + int(t[i + 1]) * B + int(t[i + 2])) % 2 ** 64 for i in range(6)]
    assert [int(x) for x in h] == direct


def test_equal_kgrams_equal_hashes_at_any_offset():
    rng = np.random.default_rng(0)
    seg = rng.integers(0, 30000, 40, dtype=np.uint16)
    t = np.concatenate([rng.integers(0, 30000, 7, dtype=np.uint16), seg,
                        rng.integers(0, 30000, 13, dtype=np.uint16), seg])
    h = D.kgram_hashes(t, 10)
    a, b = 7, 7 + 40 + 13
    assert np.array_equal(h[a: a + 31], h[b: b + 31])


def test_sampling_depends_only_on_content():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 30000, 5000, dtype=np.uint16)
    h = D.kgram_hashes(t, 8)
    m = D.sample_mask(h, 3)
    assert 0.08 < m.mean() < 0.17                      # ~1/8
    assert np.array_equal(D.sample_mask(h.copy(), 3), m)


def test_blocks_give_same_result_as_whole():
    rng = np.random.default_rng(2)
    t = rng.integers(0, 30000, 10_000, dtype=np.uint16)
    whole = D.sampled_hashes(t, 16, 2, block=10**9)
    blocked = D.sampled_hashes(t, 16, 2, block=777)
    assert np.array_equal(np.sort(whole), np.sort(blocked))


def test_duplicate_fraction_of_a_doubled_corpus():
    rng = np.random.default_rng(3)
    doc = rng.integers(0, 30000, 20_000, dtype=np.uint16)
    unique = rng.integers(0, 30000, 20_000, dtype=np.uint16)
    corpus = np.concatenate([doc, unique, doc])          # doc appears twice: ~2/3 duplicated
    s = D.duplicate_stats(D.sampled_hashes(corpus, 32, 2))
    assert 0.6 < s["dup_fraction"] < 0.72
    assert 0.27 < s["excess_fraction"] < 0.39            # one copy of doc is removable (~1/3)
    fresh = D.duplicate_stats(D.sampled_hashes(unique, 32, 2))
    assert fresh["dup_fraction"] < 0.01


def test_shard_roundtrip(tmp_path):
    tokens = np.arange(1000, dtype=np.uint16)
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1] = D.MAGIC, len(tokens)
    path = tmp_path / "shard_00000.bin"
    path.write_bytes(header.tobytes() + tokens.tobytes())
    assert np.array_equal(np.asarray(D.load_shard(path)), tokens)
    h = D.kgram_hashes(tokens, 5)
    found = D.find_kgrams([path], h[[100, 400]], 5)
    assert np.array_equal(found[int(h[100])], tokens[100:105])
    assert np.array_equal(found[int(h[400])], tokens[400:405])
