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

Resuming
--------
A DataLoader over an IterableDataset takes batches from its workers in turn
(batch j from worker j % K). Given start_batch = batches already consumed,
worker w of the resumed run takes the role of worker (w + start_batch) % K of
the original run (the one whose batch comes next in that position) and skips
exactly the windows that worker produced before, so a resumed run reads the
same windows, in the same order, as an uninterrupted one.

Mixtures
--------
MixtureDataset draws each window from one of several shard sets ("sources")
with fixed probabilities. Each source keeps its own global order (a fresh
permutation per epoch of that source), so a small source sampled often is
cycled through in full before any window repeats. With a single source of
weight 1 at phase 0 the order is exactly ShardedDataset's global order.

Validation
----------
A fixed random sample of val_windows windows drawn across all val shards,
with its own seed (val_seed) so that every run, arm and training seed is
evaluated on exactly the same windows. Finite and map-style: an evaluation
pass reads the whole sample. val_windows=None keeps the legacy behaviour
(sequential stream from the first val shard).
"""

import itertools
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


def worker_role(worker_id: int, n_workers: int, start_batch: int = 0) -> int:
    """
    Which worker of the original run this worker stands in for after a resume.

    The resumed DataLoader asks worker 0 first, but batch `start_batch` of the
    original run came from worker start_batch % K; rotating the roles keeps
    batch j of the resumed run = batch start_batch + j of the original.
    """
    n_workers = max(1, n_workers)
    return (worker_id + start_batch) % n_workers


def items_already_read(start_batch: int, batch_size: int, worker_id: int = 0, n_workers: int = 1) -> int:
    """
    Windows worker `worker_id` produced for the first `start_batch` batches of a
    DataLoader with `n_workers` workers (0 workers counts as 1): batch j comes
    from worker j % n_workers.
    """
    n_workers = max(1, n_workers)
    if start_batch <= worker_id:
        return 0
    return ((start_batch - worker_id + n_workers - 1) // n_workers) * batch_size


class ShardedDataset(IterableDataset):
    """
    Infinite iterable over the windows of a set of shards.

    Args:
        shard_dir: directory containing shard_*.bin
        seq_len: sequence length (windows are seq_len + 1 tokens)
        order: "global" (default) or "sequential" (legacy)
        shuffle_shards: legacy mode only; shuffle whole shards each epoch
        seed: data-order seed (shared by both arms of an experiment)
        start_batch, batch_size: resume after start_batch DataLoader batches
            of batch_size windows (see "Resuming" above)
    """

    def __init__(
        self,
        shard_dir: Path,
        seq_len: int = 1024,
        order: str = "global",
        shuffle_shards: bool = True,
        seed: int = 42,
        start_batch: int = 0,
        batch_size: int = 1,
    ):
        super().__init__()
        if order not in TRAIN_ORDERS:
            raise ValueError(f"Unknown order: {order}. Available: {TRAIN_ORDERS}")
        self.shard_dir = Path(shard_dir)
        self.seq_len = seq_len
        self.order = order
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.start_batch = start_batch
        self.batch_size = batch_size

        self.shards = sorted(self.shard_dir.glob("shard_*.bin"))
        assert len(self.shards) > 0, f"No shards found in {shard_dir}"

        self.index = _WindowIndex(self.shards, seq_len)
        total_tokens = sum(read_shard_header(s)["n_tokens"] for s in self.shards)
        log.info(f"ShardedDataset: {len(self.shards)} shards, {total_tokens/1e9:.3f}B tokens, "
                 f"{self.index.total:,} windows, order={order} ({shard_dir})")

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        w_id, n_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        w_id = worker_role(w_id, n_workers, self.start_batch)
        if self.order == "global":
            yield from self._iter_global(w_id, n_workers)
        else:
            yield from self._iter_sequential(w_id, n_workers)

    def _iter_global(self, w_id: int, n_workers: int) -> Iterator[dict]:
        skip = items_already_read(self.start_batch, self.batch_size, w_id, n_workers)
        epoch = 0
        while True:
            order = global_order(self.index.total, self.seed, epoch, w_id, n_workers)
            if len(order) == 0:
                raise ValueError(f"{self.shard_dir}: fewer windows than DataLoader workers")
            if skip >= len(order):          # whole epoch already read before the resume
                skip -= len(order)
                epoch += 1
                continue
            for gid in order[skip:]:
                yield self.index.item(int(gid))
            skip = 0
            epoch += 1
            log.debug(f"ShardedDataset: worker {w_id} completed epoch {epoch}")

    def _iter_sequential(self, w_id: int, n_workers: int) -> Iterator[dict]:
        """Legacy order (pre 2026-11-07): whole shards, windows in order."""
        skip = items_already_read(self.start_batch, self.batch_size, w_id, n_workers)
        yield from itertools.islice(self._iter_sequential_all(w_id, n_workers), skip, None)

    def _iter_sequential_all(self, w_id: int, n_workers: int) -> Iterator[dict]:
        shard_ids = [i for i in range(len(self.shards)) if i % n_workers == w_id]
        rng = random.Random(self.seed)
        while True:
            if self.shuffle_shards:
                rng.shuffle(shard_ids)
            for s in shard_ids:
                tokens = self.index.tokens(s)
                for w in range(int(self.index.counts[s])):
                    yield _make_item(tokens, w, self.seq_len)


class _SourceStream:
    """Window ids of one source for one worker: its stride of a fresh permutation per epoch."""

    def __init__(self, total: int, seed: int, worker_id: int, n_workers: int, name: str = ""):
        self.total, self.seed, self.w, self.k, self.name = total, seed, worker_id, n_workers, name
        self.epoch, self.pos = 0, 0
        self.order = self._order()

    def _order(self) -> np.ndarray:
        order = global_order(self.total, self.seed, self.epoch, self.w, self.k)
        if len(order) == 0:
            raise ValueError(f"source {self.name!r}: fewer windows than DataLoader workers")
        return order

    def _next_epoch(self):
        self.epoch += 1
        self.pos = 0
        self.order = self._order()

    def advance(self, n: int):
        while n > 0:
            left = len(self.order) - self.pos
            if n < left:
                self.pos += n
                return
            n -= left
            self._next_epoch()

    def next(self) -> int:
        if self.pos >= len(self.order):
            self._next_epoch()
        gid = int(self.order[self.pos])
        self.pos += 1
        return gid


class MixtureDataset(IterableDataset):
    """
    Infinite iterable drawing each window from one of several shard sets.

    Args:
        sources: name -> directory of shard_*.bin
        weights: name -> sampling weight (normalized; sources with weight 0 are unused)
        seq_len: sequence length
        seed: data-order seed
        phase: training phase (0 = stable, 1 = cooldown); each phase draws its
            own source choices and its own permutations, so the cooldown does
            not replay the first windows of the stable phase
        start_batch, batch_size: resume after start_batch DataLoader batches

    Source choices are drawn per worker in fixed blocks from a generator seeded
    by (seed, phase, worker); skipping only advances counters, so a resume is
    exact without reading any skipped window.
    """

    BLOCK = 4096

    def __init__(self, sources: Dict[str, Path], weights: Dict[str, float], seq_len: int = 1024,
                 seed: int = 42, phase: int = 0, start_batch: int = 0, batch_size: int = 1):
        super().__init__()
        unknown = [n for n in weights if n not in sources]
        if unknown:
            raise ValueError(f"mixture weights name unknown sources {unknown}; known: {list(sources)}")
        if any(w < 0 for w in weights.values()):
            raise ValueError(f"negative mixture weight: {weights}")
        self.names = [n for n, w in weights.items() if w > 0]
        if not self.names:
            raise ValueError("mixture has no source with a positive weight")
        p = np.array([weights[n] for n in self.names], dtype=np.float64)
        self.p = p / p.sum()
        self.seq_len, self.seed, self.phase = seq_len, seed, phase
        self.start_batch, self.batch_size = start_batch, batch_size
        self.indexes = []
        for n in self.names:
            shards = sorted(Path(sources[n]).glob("shard_*.bin"))
            assert len(shards) > 0, f"No shards found for source {n!r} in {sources[n]}"
            self.indexes.append(_WindowIndex(shards, seq_len))
        self.windows = {n: ix.total for n, ix in zip(self.names, self.indexes)}
        log.info("MixtureDataset phase %d: %s", phase, ", ".join(
            f"{n} p={pi:.3f} ({self.windows[n]:,} windows)" for n, pi in zip(self.names, self.p)))

    def stream_seed(self, i: int) -> int:
        # source 0 of phase 0 uses the plain seed: one source = ShardedDataset's order
        return self.seed + 1_000_003 * i + 7_919 * self.phase

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        w_id, n_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        w_id = worker_role(w_id, n_workers, self.start_batch)
        streams = [_SourceStream(ix.total, self.stream_seed(i), w_id, n_workers, n)
                   for i, (n, ix) in enumerate(zip(self.names, self.indexes))]
        rng = np.random.default_rng((self.seed, self.phase, w_id, 1))
        n_src = len(self.names)

        def new_block():
            return rng.choice(n_src, size=self.BLOCK, p=self.p)

        block, i = new_block(), 0
        skip = items_already_read(self.start_batch, self.batch_size, w_id, n_workers)
        while skip > 0:
            take = min(skip, self.BLOCK - i)
            for s, c in enumerate(np.bincount(block[i: i + take], minlength=n_src)):
                streams[s].advance(int(c))
            i += take
            skip -= take
            if i == self.BLOCK:
                block, i = new_block(), 0
        while True:
            while i < self.BLOCK:
                s = int(block[i])
                i += 1
                yield self.indexes[s].item(streams[s].next())
            block, i = new_block(), 0


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
        train_sources: name -> root dir with train/ shards (mixture mode)
        train_mix: name -> weight; None = data_dir/train alone (ShardedDataset)
        phase: training phase passed to the mixture (0 stable, 1 cooldown)
        start_batch: train batches already consumed (resume)
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
        train_sources: Optional[Dict[str, str]] = None,
        train_mix: Optional[Dict[str, float]] = None,
        phase: int = 0,
        start_batch: int = 0,
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

        if train_mix:
            if train_order != "global":
                raise ValueError("mixtures use the global order only")
            sources = {n: Path(d) / "train" for n, d in (train_sources or {}).items()}
            self.train_dataset = MixtureDataset(sources, train_mix, seq_len=seq_len, seed=seed,
                                                phase=phase, start_batch=start_batch,
                                                batch_size=batch_size)
        else:
            self.train_dataset = ShardedDataset(train_dir, seq_len=seq_len, order=train_order,
                                                shuffle_shards=True, seed=seed,
                                                start_batch=start_batch, batch_size=batch_size)
        if val_windows is None:
            self.val_dataset = ShardedDataset(val_dir, seq_len=seq_len, order="sequential",
                                              shuffle_shards=False, seed=seed)
        else:
            self.val_dataset = FixedWindowDataset(val_dir, seq_len, val_windows, val_seed)

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        # Each loader has its own torch generator: creating an iterator then
        # never draws from the global RNG, so dropout draws stay identical
        # between an uninterrupted run and a resumed one.
        def common(n_workers):
            return dict(
                generator=torch.Generator().manual_seed(self.seed),
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
