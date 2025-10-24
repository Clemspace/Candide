"""Tests for data base classes."""

import pytest
import torch
from typing import Dict, Any, List

from ramanujan.data.base import (
    BaseDataset,
    StreamingDataset,
    TokenizedDataset,
    DatasetConfig,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

class DummyDataset(BaseDataset):
    """Dummy dataset for testing."""
    
    def __init__(self, size=100, **kwargs):
        super().__init__(**kwargs)
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        item = {'data': torch.tensor([idx, idx + 1, idx + 2])}
        return self._apply_transform(item)


class DummyStreamingDataset(StreamingDataset):
    """Dummy streaming dataset for testing."""
    
    def __getitem__(self, idx):
        return {'data': torch.tensor([idx % 10])}


class DummyTokenizedDataset(TokenizedDataset):
    """Dummy tokenized dataset for testing."""
    
    def __init__(self, sequences, **kwargs):
        super().__init__(**kwargs)
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq, dtype=torch.long)
        return {'input_ids': input_ids, 'labels': input_ids.clone()}


# ============================================================================
# TEST BASE DATASET
# ============================================================================

def test_base_dataset_creation():
    """Test creating base dataset."""
    dataset = DummyDataset(size=50)
    
    assert len(dataset) == 50
    assert dataset.transform is None
    assert dataset.cache_dir is None


def test_base_dataset_getitem():
    """Test getting items from dataset."""
    dataset = DummyDataset(size=10)
    
    item = dataset[0]
    
    assert 'data' in item
    assert isinstance(item['data'], torch.Tensor)
    assert torch.equal(item['data'], torch.tensor([0, 1, 2]))


def test_base_dataset_with_transform():
    """Test dataset with transform."""
    def transform(item):
        item['data'] = item['data'] * 2
        return item
    
    dataset = DummyDataset(size=10, transform=transform)
    item = dataset[5]
    
    # Should be [5, 6, 7] * 2 = [10, 12, 14]
    assert torch.equal(item['data'], torch.tensor([10, 12, 14]))


def test_base_dataset_collate():
    """Test default collate function."""
    dataset = DummyDataset(size=10)
    
    batch = [dataset[0], dataset[1], dataset[2]]
    collated = dataset.collate_fn(batch)
    
    assert 'data' in collated
    assert collated['data'].shape == (3, 3)


def test_base_dataset_metadata():
    """Test metadata tracking."""
    dataset = DummyDataset(size=42)
    
    metadata = dataset.get_metadata()
    
    assert metadata['size'] == 42
    assert metadata['type'] == 'DummyDataset'


# ============================================================================
# TEST STREAMING DATASET
# ============================================================================

def test_streaming_dataset_creation():
    """Test creating streaming dataset."""
    dataset = DummyStreamingDataset(estimated_size=1_000_000)
    
    assert len(dataset) == 1_000_000
    assert dataset.metadata['streaming'] is True


def test_streaming_dataset_getitem():
    """Test getting items from streaming dataset."""
    dataset = DummyStreamingDataset()
    
    item = dataset[0]
    assert 'data' in item
    
    item = dataset[100]
    assert 'data' in item


# ============================================================================
# TEST TOKENIZED DATASET
# ============================================================================

def test_tokenized_dataset_creation():
    """Test creating tokenized dataset."""
    sequences = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    dataset = DummyTokenizedDataset(
        sequences,
        max_length=10,
        pad_token_id=0
    )
    
    assert len(dataset) == 3
    assert dataset.max_length == 10
    assert dataset.pad_token_id == 0


def test_tokenized_dataset_padding():
    """Test sequence padding."""
    sequences = [[1, 2, 3]]
    dataset = DummyTokenizedDataset(
        sequences,
        max_length=10,
        pad_token_id=0
    )
    
    padded = dataset._pad_sequence([1, 2, 3])
    
    assert len(padded) == 10
    assert padded[:3] == [1, 2, 3]
    assert padded[3:] == [0] * 7


def test_tokenized_dataset_truncation():
    """Test sequence truncation."""
    sequences = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    dataset = DummyTokenizedDataset(
        sequences,
        max_length=5,
        pad_token_id=0
    )
    
    truncated = dataset._truncate_sequence([1, 2, 3, 4, 5, 6, 7, 8])
    
    assert len(truncated) == 5
    assert truncated == [1, 2, 3, 4, 5]


def test_tokenized_dataset_chunking():
    """Test sequence chunking with sliding window."""
    sequences = [[]]
    dataset = DummyTokenizedDataset(
        sequences,
        max_length=5,
        stride=3,
        pad_token_id=0
    )
    
    long_seq = list(range(20))  # [0, 1, 2, ..., 19]
    chunks = dataset._chunk_sequence(long_seq, min_chunk_size=3)
    
    assert len(chunks) > 1
    assert chunks[0] == [0, 1, 2, 3, 4]
    assert chunks[1] == [3, 4, 5, 6, 7]  # Overlap due to stride


def test_tokenized_dataset_chunking_min_size():
    """Test chunking respects minimum chunk size."""
    sequences = [[]]
    dataset = DummyTokenizedDataset(
        sequences,
        max_length=10,
        stride=10,
        pad_token_id=0
    )
    
    short_seq = [1, 2, 3]
    chunks = dataset._chunk_sequence(short_seq, min_chunk_size=5)
    
    # Should not create chunks smaller than min_chunk_size
    assert len(chunks) == 0


# ============================================================================
# TEST DATASET CONFIG
# ============================================================================

def test_dataset_config_creation():
    """Test creating dataset config."""
    config = DatasetConfig(
        name='wikitext',
        split='train',
        max_length=1024,
        batch_size=32
    )
    
    assert config.name == 'wikitext'
    assert config.split == 'train'
    assert config.max_length == 1024
    assert config.batch_size == 32


def test_dataset_config_to_dict():
    """Test converting config to dict."""
    config = DatasetConfig(
        name='wikitext',
        batch_size=16,
        custom_arg='value'
    )
    
    config_dict = config.to_dict()
    
    assert config_dict['name'] == 'wikitext'
    assert config_dict['batch_size'] == 16
    assert config_dict['custom_arg'] == 'value'


def test_dataset_config_from_dict():
    """Test creating config from dict."""
    config_dict = {
        'name': 'fineweb',
        'split': 'train',
        'max_length': 2048,
        'batch_size': 64,
        'streaming': True
    }
    
    config = DatasetConfig.from_dict(config_dict)
    
    assert config.name == 'fineweb'
    assert config.max_length == 2048
    assert config.streaming is True


def test_dataset_config_with_extra_kwargs():
    """Test config with extra kwargs."""
    config = DatasetConfig(
        name='custom',
        subset='sample-10BT',
        buffer_size=100
    )
    
    assert config.extra['subset'] == 'sample-10BT'
    assert config.extra['buffer_size'] == 100


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_empty_batch_collate():
    """Test collating empty batch."""
    dataset = DummyDataset(size=10)
    collated = dataset.collate_fn([])
    
    assert collated == {}


def test_tokenized_dataset_with_bos_eos():
    """Test tokenized dataset with special tokens."""
    sequences = [[1, 2, 3]]
    dataset = DummyTokenizedDataset(
        sequences,
        max_length=10,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2
    )
    
    assert dataset.bos_token_id == 1
    assert dataset.eos_token_id == 2


def test_tokenized_dataset_metadata():
    """Test tokenized dataset metadata."""
    sequences = [[1, 2, 3]]
    dataset = DummyTokenizedDataset(
        sequences,
        max_length=512,
        pad_token_id=0,
        stride=256
    )
    
    metadata = dataset.get_metadata()
    
    assert metadata['max_length'] == 512
    assert metadata['pad_token_id'] == 0
    assert metadata['stride'] == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])