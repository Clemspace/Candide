"""
ramanujan/data/loaders/shard_loader.py

DataLoader for pre-tokenized binary shards produced by prepare_shards.py.
Reads shards with numpy memmap: zero copies, minimal RAM, works across
NFS/mounted volumes.

Windows
-------
Each shard of n tokens is cut into (n - 1) // seq_len windows of seq_len + 1
tokens: window i covers tokens[i*L : i*L + L + 1], giving input_ids = w[:-1]
and labels = w[1:] (next-token targets, shifted once, here).

Train order
-----------
- "global" (default): every epoch, all windows of all train shards are put in
  one random permutation (seeded by seed + epoch); worker k of K takes every
  K-th window of it. Each window is seen exactly once per epoch and a batch
  mixes windows from across the corpus.
- "sequential": legacy behaviour (shards shuffled as whole files, windows read
  in order within a shard). A batch is then seq_len * batch_size contiguous
  tokens of one document. Kept only to reproduce runs made before 2026-11-07.

Validation
----------
A fixed random sample of val_windows windows drawn across all val shards,
with its own seed (val_seed) so that every run, arm and training seed is
evaluated on exactly the same windows. Finite and map-style: an evaluation
pass reads the whole sample. val_windows=None keeps the legacy behaviour
(sequential stream from the first val shard).
"""

import logging
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

log = logging.getLogger(__name__)

MAGIC = 20240520
HEADER_SIZE = 256 * 4  # 256 int32s = 1024 bytes

TRAIN_ORDERS = ("global", "sequential")


def read_shard_header(path: Path) -> dict:
    """Read shard metadata without loading tokens."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_SIZE), dtype=np.int32)
    assert header[0] == MAGIC, f"Bad magic in {path}: {header[0]}"
    return {"n_tokens": int(header[1])}


def load_shard(path: Path) -> np.ndarray:
    """Load shard tokens as a uint16 memmap."""
    n = read_shard_header(path)["n_tokens"]
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_SIZE, shape=(n,))


def windows_per_shard(n_tokens: int, seq_len: int) -> int:
    """Number of (seq_len + 1)-token windows in a shard of n_tokens."""
    return max(0, (n_tokens - 1) // seq_len)


def _make_item(tokens: np.ndarray, window: int, seq_len: int) -> Dict[str, torch.Tensor]:
    start = window * seq_len
    chunk = torch.from_numpy(tokens[start: start + seq_len + 1].astype(np.int64))
    return {"input_ids": chunk[:-1], "labels": chunk[1:]}


class _WindowIndex:
    """Maps a global window id to (shard index, window within shard)."""

    def __init__(self, shards: List[Path], seq_len: int):
        self.shards = shards
        self.seq_len = seq_len
        counts = np.array(
            [windows_per_shard(read_shard_header(s)["n_tokens"], seq_len) for s in shards],
            dtype=np.int64,
        )
        self.counts = counts
        self.starts = np.concatenate([[0], np.cumsum(counts)])  # len = n_shards + 1
        self.total = int(self.starts[-1])
        self._memmaps: Dict[int, np.ndarray] = {}

    def __getstate__(self):
        # Never pickle open memmaps into worker processes (spawn would copy them)
        state = self.__dict__.copy()
        state["_memmaps"] = {}
        return state

    def locate(self, gid: int) -> Tuple[int, int]:
        shard = int(np.searchsorted(self.starts, gid, side="right") - 1)
        return shard, int(gid - self.starts[shard])

    def tokens(self, shard: int) -> np.ndarray:
        # Opened lazily so each DataLoader worker holds its own memmaps
        if shard not in self._memmaps:
            self._memmaps[shard] = load_shard(self.shards[shard])
        return self._memmaps[shard]

    def item(self, gid: int) -> Dict[str, torch.Tensor]:
        shard, window = self.locate(gid)
        return _make_item(self.tokens(shard), window, self.seq_len)


def global_order(total: int, seed: int, epoch: int, worker_id: int = 0, n_workers: int = 1) -> np.ndarray:
    """Window ids worker `worker_id` reads in `epoch`: its stride of one global permutation."""
    perm = np.random.default_rng((seed, epoch)).permutation(total)
    return perm[worker_id::n_workers]


class ShardedDataset(IterableDataset):
    """
    Infinite iterable over the windows of a set of shards.

    Args:
        shard_dir: directory containing shard_*.bin
        seq_len: sequence length (windows are seq_len + 1 tokens)
        order: "global" (default) or "sequential" (legacy)
        shuffle_shards: legacy mode only; shuffle whole shards each epoch
        seed: data-order seed (shared by both arms of an experiment)
    """

    def __init__(
        self,
        shard_dir: Path,
        seq_len: int = 1024,
        order: str = "global",
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        if order not in TRAIN_ORDERS:
            raise ValueError(f"Unknown order: {order}. Available: {TRAIN_ORDERS}")
        self.shard_dir = Path(shard_dir)
        self.seq_len = seq_len
        self.order = order
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        self.shards = sorted(self.shard_dir.glob("shard_*.bin"))
        assert len(self.shards) > 0, f"No shards found in {shard_dir}"

        self.index = _WindowIndex(self.shards, seq_len)
        total_tokens = sum(read_shard_header(s)["n_tokens"] for s in self.shards)
        log.info(f"ShardedDataset: {len(self.shards)} shards, {total_tokens/1e9:.3f}B tokens, "
                 f"{self.index.total:,} windows, order={order} ({shard_dir})")

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        w_id, n_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        if self.order == "global":
            yield from self._iter_global(w_id, n_workers)
        else:
            yield from self._iter_sequential(w_id, n_workers)

    def _iter_global(self, w_id: int, n_workers: int) -> Iterator[dict]:
        epoch = 0
        while True:
            for gid in global_order(self.index.total, self.seed, epoch, w_id, n_workers):
                yield self.index.item(int(gid))
            epoch += 1
            log.debug(f"ShardedDataset: worker {w_id} completed epoch {epoch}")

    def _iter_sequential(self, w_id: int, n_workers: int) -> Iterator[dict]:
        """Legacy order (pre 2026-11-07): whole shards, windows in order."""
        shard_ids = [i for i in range(len(self.shards)) if i % n_workers == w_id]
        rng = random.Random(self.seed)
        while True:
            if self.shuffle_shards:
                rng.shuffle(shard_ids)
            for s in shard_ids:
                tokens = self.index.tokens(s)
                for w in range(int(self.index.counts[s])):
                    yield _make_item(tokens, w, self.seq_len)


class FixedWindowDataset(Dataset):
    """
    A fixed random sample of windows across a set of shards (map-style).

    The sample depends only on (shards, seq_len, n_windows, sample_seed), never
    on the training seed, so every run is evaluated on the same windows.
    Windows are returned in a fixed order.
    """

    def __init__(self, shard_dir: Path, seq_len: int, n_windows: int, sample_seed: int = 0):
        shards = sorted(Path(shard_dir).glob("shard_*.bin"))
        assert len(shards) > 0, f"No shards found in {shard_dir}"
        self.index = _WindowIndex(shards, seq_len)
        n = min(int(n_windows), self.index.total)
        if n < n_windows:
            log.warning(f"FixedWindowDataset: only {n} windows available (asked {n_windows})")
        rng = np.random.default_rng(sample_seed)
        self.gids = np.sort(rng.choice(self.index.total, size=n, replace=False))
        log.info(f"FixedWindowDataset: {n:,} windows ({n * seq_len / 1e6:.2f}M tokens) "
                 f"sampled from {len(shards)} shards, sample_seed={sample_seed}")

    def __len__(self) -> int:
        return len(self.gids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.index.item(int(self.gids[i]))


class ShardLoader:
    """
    Factory for train/val DataLoaders backed by pre-tokenized shards.

    Args:
        data_dir: root dir containing train/ and val/ shard subdirs
        seq_len: sequence length
        batch_size: batch size
        num_workers: DataLoader workers (2-4 recommended for NFS; 0 disables multiprocessing)
        prefetch_factor: batches to prefetch per worker
        pin_memory: True if training on GPU
        seed: train data-order seed
        train_order: "global" (default) or "sequential" (legacy, pre 2026-11-07)
        val_windows: size of the fixed validation sample (None = legacy stream)
        val_seed: seed of the validation sample (keep fixed across all runs)
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 1024,
        batch_size: int = 8,
        num_workers: int = 2,
        prefetch_factor: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        train_order: str = "global",
        val_windows: Optional[int] = 1600,
        val_seed: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.seed = seed

        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        assert train_dir.exists(), f"Missing train shards: {train_dir}"
        assert val_dir.exists(), f"Missing val shards: {val_dir}"

        self.train_dataset = ShardedDataset(train_dir, seq_len=seq_len, order=train_order,
                                            shuffle_shards=True, seed=seed)
        if val_windows is None:
            self.val_dataset = ShardedDataset(val_dir, seq_len=seq_len, order="sequential",
                                              shuffle_shards=False, seed=seed)
        else:
            self.val_dataset = FixedWindowDataset(val_dir, seq_len, val_windows, val_seed)

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        def common(n_workers):
            return dict(
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                num_workers=n_workers,
                **({"prefetch_factor": self.prefetch_factor, "persistent_workers": True}
                   if n_workers > 0 else {}),
            )

        train_loader = DataLoader(self.train_dataset, **common(self.num_workers))
        val_loader = DataLoader(self.val_dataset, shuffle=False,
                                **common(max(0, self.num_workers // 2)))
        return train_loader, val_loader
