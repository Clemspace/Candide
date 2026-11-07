"""Tests for the shard loader: windowing, global order, fixed validation sample."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from ramanujan.data.loaders.shard_loader import (
    HEADER_SIZE,
    MAGIC,
    FixedWindowDataset,
    ShardedDataset,
    ShardLoader,
    global_order,
    windows_per_shard,
)

SEQ = 8


def _write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_SIZE // 4, dtype=np.int32)
    header[0] = MAGIC
    header[1] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def _make_dir(root: Path, n_shards: int = 4, n_tokens: int = 81, base: int = 1000) -> Path:
    """Shard s holds tokens base*(s+1) + 0..n_tokens-1, so a token identifies its shard."""
    root.mkdir(parents=True, exist_ok=True)
    for s in range(n_shards):
        _write_shard(root / f"shard_{s:04d}.bin", base * (s + 1) + np.arange(n_tokens))
    return root


def _take(dataset, n):
    it = iter(dataset)
    return [next(it) for _ in range(n)]


def _shard_of(item) -> int:
    return int(item["input_ids"][0]) // 1000 - 1


@pytest.fixture
def train_dir(tmp_path):
    return _make_dir(tmp_path / "train")


@pytest.fixture
def data_root(tmp_path):
    _make_dir(tmp_path / "train", n_shards=4)
    _make_dir(tmp_path / "val", n_shards=3, base=1000)
    return tmp_path


# ============================================================================
# WINDOWS
# ============================================================================

def test_windows_per_shard():
    assert windows_per_shard(81, 8) == 10
    assert windows_per_shard(80, 8) == 9     # needs seq_len + 1 tokens per window
    assert windows_per_shard(5, 8) == 0


@pytest.mark.parametrize("order", ["global", "sequential"])
def test_labels_are_next_tokens(train_dir, order):
    for item in _take(ShardedDataset(train_dir, seq_len=SEQ, order=order, seed=1), 20):
        assert item["input_ids"].shape == (SEQ,)
        assert torch.equal(item["labels"][:-1], item["input_ids"][1:])
        assert item["labels"][-1] == item["input_ids"][-1] + 1   # consecutive tokens in fixtures


def test_global_and_sequential_cover_the_same_windows(train_dir):
    """The fix changes the order only, not which windows exist."""
    n = 4 * windows_per_shard(81, SEQ)
    key = lambda it: int(it["input_ids"][0])
    g = sorted(key(x) for x in _take(ShardedDataset(train_dir, SEQ, order="global", seed=3), n))
    s = sorted(key(x) for x in _take(ShardedDataset(train_dir, SEQ, order="sequential", seed=3), n))
    assert g == s


# ============================================================================
# GLOBAL ORDER
# ============================================================================

def test_global_order_each_window_once_per_epoch_across_workers():
    total, k = 1001, 3
    parts = [global_order(total, seed=7, epoch=0, worker_id=w, n_workers=k) for w in range(k)]
    allv = np.concatenate(parts)
    assert len(allv) == total
    assert len(np.unique(allv)) == total


def test_global_order_deterministic_and_seed_dependent():
    a = global_order(500, seed=1, epoch=0)
    assert np.array_equal(a, global_order(500, seed=1, epoch=0))
    assert not np.array_equal(a, global_order(500, seed=2, epoch=0))
    assert not np.array_equal(a, global_order(500, seed=1, epoch=1))


def test_global_epoch_visits_every_window_once(train_dir):
    n = 4 * windows_per_shard(81, SEQ)
    starts = [int(x["input_ids"][0]) for x in _take(ShardedDataset(train_dir, SEQ, seed=5), n)]
    assert len(set(starts)) == n


def test_global_batches_mix_shards(train_dir):
    """A batch of 8 is not 8 consecutive windows of one shard (the old behaviour)."""
    items = _take(ShardedDataset(train_dir, SEQ, order="global", seed=1), 8)
    assert len({_shard_of(x) for x in items}) >= 2

    legacy = _take(ShardedDataset(train_dir, SEQ, order="sequential", seed=1), 8)
    assert len({_shard_of(x) for x in legacy}) == 1


def test_global_dataloader_two_workers_no_duplicates(train_dir):
    n = 4 * windows_per_shard(81, SEQ)
    ds = ShardedDataset(train_dir, SEQ, order="global", seed=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2)
    starts, it = [], iter(loader)
    while len(starts) < n:
        starts.extend(next(it)["input_ids"][:, 0].tolist())
    assert len(set(starts[:n])) == n


def test_invalid_order(train_dir):
    with pytest.raises(ValueError):
        ShardedDataset(train_dir, SEQ, order="random")


# ============================================================================
# FIXED VALIDATION SAMPLE
# ============================================================================

def test_fixed_val_sample_size_and_spread(data_root):
    ds = FixedWindowDataset(data_root / "val", SEQ, n_windows=15, sample_seed=0)
    assert len(ds) == 15
    assert len({_shard_of(ds[i]) for i in range(len(ds))}) >= 2


def test_fixed_val_sample_independent_of_train_seed(data_root):
    a = ShardLoader(str(data_root), seq_len=SEQ, batch_size=4, num_workers=0, pin_memory=False,
                    seed=1, val_windows=12)
    b = ShardLoader(str(data_root), seq_len=SEQ, batch_size=4, num_workers=0, pin_memory=False,
                    seed=99, val_windows=12)
    _, va = a.get_dataloaders()
    _, vb = b.get_dataloaders()
    for x, y in zip(va, vb):
        assert torch.equal(x["input_ids"], y["input_ids"])


def test_fixed_val_is_finite_and_repeatable(data_root):
    loader = ShardLoader(str(data_root), seq_len=SEQ, batch_size=4, num_workers=0,
                         pin_memory=False, seed=1, val_windows=10)
    _, val = loader.get_dataloaders()
    first = [b["input_ids"] for b in val]
    second = [b["input_ids"] for b in val]
    assert sum(len(b) for b in first) == 10
    assert all(torch.equal(x, y) for x, y in zip(first, second))


def test_fixed_val_caps_at_available(data_root):
    total = 3 * windows_per_shard(81, SEQ)
    ds = FixedWindowDataset(data_root / "val", SEQ, n_windows=10_000, sample_seed=0)
    assert len(ds) == total


def test_legacy_val_stream_still_available(data_root):
    loader = ShardLoader(str(data_root), seq_len=SEQ, batch_size=4, num_workers=0,
                         pin_memory=False, seed=1, val_windows=None)
    _, val = loader.get_dataloaders()
    batch = next(iter(val))
    assert batch["input_ids"].shape == (4, SEQ)
