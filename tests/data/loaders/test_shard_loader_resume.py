"""Shard loader: exact resume (any number of workers) and source mixtures."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ramanujan.data.loaders.shard_loader import (
    HEADER_SIZE,
    MAGIC,
    MixtureDataset,
    ShardedDataset,
    ShardLoader,
    items_already_read,
    worker_role,
)

SEQ = 8


def _write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_SIZE // 4, dtype=np.int32)
    header[0] = MAGIC
    header[1] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def _make_dir(root: Path, n_shards: int = 3, n_tokens: int = 81, base: int = 1000) -> Path:
    """Shard s holds base*(s+1) + 0..n_tokens-1: a token tells its source and shard."""
    root.mkdir(parents=True, exist_ok=True)
    for s in range(n_shards):
        _write_shard(root / f"shard_{s:04d}.bin", base * (s + 1) + np.arange(n_tokens))
    return root


def _firsts(items):
    return [int(it["input_ids"][0]) for it in items]


def _batches(dataset, n, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    it = iter(loader)
    out = [next(it)["input_ids"][:, 0].tolist() for _ in range(n)]
    del it
    return out


def test_items_already_read_round_robin():
    # 5 batches, 2 workers: worker 0 made batches 0, 2, 4; worker 1 made 1, 3
    assert items_already_read(5, 4, 0, 2) == 12
    assert items_already_read(5, 4, 1, 2) == 8
    assert items_already_read(1, 4, 1, 2) == 0
    assert items_already_read(7, 3, 0, 0) == 21          # no workers = one stream
    assert items_already_read(0, 3, 0, 1) == 0


def test_worker_role_rotates_with_start_batch():
    assert [worker_role(w, 2, 9) for w in (0, 1)] == [1, 0]     # batch 9 came from worker 1
    assert [worker_role(w, 3, 4) for w in (0, 1, 2)] == [1, 2, 0]
    assert worker_role(0, 0, 7) == 0 and worker_role(0, 1, 7) == 0


def test_sharded_resume_in_process(tmp_path):
    d = _make_dir(tmp_path / "a")
    full = ShardedDataset(d, SEQ, seed=3, batch_size=2)
    it = iter(full)
    ref = _firsts([next(it) for _ in range(60)])          # > 1 epoch (30 windows)
    resumed = ShardedDataset(d, SEQ, seed=3, start_batch=17, batch_size=2)
    it = iter(resumed)
    assert _firsts([next(it) for _ in range(26)]) == ref[34:60]


@pytest.mark.parametrize("workers", [0, 2, 3])
@pytest.mark.parametrize("start", [8, 9, 10])       # every residue mod 2 and mod 3
def test_sharded_resume_through_dataloader(tmp_path, workers, start):
    d = _make_dir(tmp_path / "a")
    ref = _batches(ShardedDataset(d, SEQ, seed=3, batch_size=2), start + 15, 2, workers)
    got = _batches(ShardedDataset(d, SEQ, seed=3, start_batch=start, batch_size=2), 15, 2, workers)
    assert got == ref[start:]


def test_mixture_single_source_is_sharded_order(tmp_path):
    d = _make_dir(tmp_path / "a")
    a = iter(ShardedDataset(d, SEQ, seed=5))
    b = iter(MixtureDataset({"main": d}, {"main": 1.0}, SEQ, seed=5))
    assert _firsts([next(a) for _ in range(70)]) == _firsts([next(b) for _ in range(70)])


def test_mixture_weights_and_full_cycles(tmp_path):
    main = _make_dir(tmp_path / "main", n_shards=3, base=1000)
    enc = _make_dir(tmp_path / "enc", n_shards=1, base=30000)       # 10 windows
    ds = MixtureDataset({"main": main, "enc": enc}, {"main": 0.8, "enc": 0.2}, SEQ, seed=0)
    it = iter(ds)
    firsts = _firsts([next(it) for _ in range(5000)])
    from_enc = [f for f in firsts if f >= 30000]
    assert abs(len(from_enc) / 5000 - 0.2) < 0.03
    # a source is cycled in full before any window repeats
    assert len(set(from_enc[:10])) == 10
    assert ds.windows == {"main": 30, "enc": 10}


def test_mixture_zero_weight_source_unused(tmp_path):
    main = _make_dir(tmp_path / "main")
    enc = _make_dir(tmp_path / "enc", base=30000)
    it = iter(MixtureDataset({"main": main, "enc": enc}, {"main": 1.0, "enc": 0.0}, SEQ))
    assert all(f < 30000 for f in _firsts([next(it) for _ in range(200)]))


def test_mixture_phases_draw_different_orders(tmp_path):
    main = _make_dir(tmp_path / "main")
    a = iter(MixtureDataset({"main": main}, {"main": 1.0}, SEQ, seed=1, phase=0))
    b = iter(MixtureDataset({"main": main}, {"main": 1.0}, SEQ, seed=1, phase=1))
    assert _firsts([next(a) for _ in range(30)]) != _firsts([next(b) for _ in range(30)])


@pytest.mark.parametrize("workers", [0, 2, 3])
@pytest.mark.parametrize("start", [3, 1500, 1501])   # inside the first choice block, past it; odd and even
def test_mixture_resume_through_dataloader(tmp_path, workers, start):
    main = _make_dir(tmp_path / "main")
    enc = _make_dir(tmp_path / "enc", n_shards=1, base=30000)
    src, w = {"main": main, "enc": enc}, {"main": 0.7, "enc": 0.3}
    n = start + 12
    ref = _batches(MixtureDataset(src, w, SEQ, seed=2, batch_size=4), n, 4, workers)
    got = _batches(MixtureDataset(src, w, SEQ, seed=2, start_batch=start, batch_size=4), 12, 4, workers)
    assert got == ref[start:n]


def test_mixture_rejects_unknown_or_empty(tmp_path):
    main = _make_dir(tmp_path / "main")
    with pytest.raises(ValueError, match="unknown"):
        MixtureDataset({"main": main}, {"main": 1.0, "enc": 0.5}, SEQ)
    with pytest.raises(ValueError, match="positive"):
        MixtureDataset({"main": main}, {"main": 0.0}, SEQ)


def test_shard_loader_mixture_mode(tmp_path):
    root = tmp_path / "v2"
    _make_dir(root / "train")
    _make_dir(root / "val")
    enc = tmp_path / "enc"
    _make_dir(enc / "train", n_shards=1, base=30000)
    loader = ShardLoader(str(root), seq_len=SEQ, batch_size=2, num_workers=0, pin_memory=False,
                         val_windows=5, train_sources={"main": str(root), "enc": str(enc)},
                         train_mix={"main": 0.5, "enc": 0.5}, phase=1, start_batch=4)
    assert isinstance(loader.train_dataset, MixtureDataset)
    assert loader.train_dataset.phase == 1 and loader.train_dataset.start_batch == 4
    train, val = loader.get_dataloaders()
    assert next(iter(train))["input_ids"].shape == (2, SEQ)
    assert len(list(val)) == 3                               # 5 windows, batches of 2


def test_iterating_loaders_leaves_global_rng_alone(tmp_path):
    root = tmp_path / "v2"
    _make_dir(root / "train")
    _make_dir(root / "val")
    train, val = ShardLoader(str(root), seq_len=SEQ, batch_size=2, num_workers=0,
                             pin_memory=False, val_windows=4).get_dataloaders()
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    next(iter(train))
    list(val)
    assert torch.equal(torch.rand(3), expected)
