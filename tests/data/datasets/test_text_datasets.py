"""Tests for text datasets."""

import pytest
import torch
from typing import List

from ramanujan.data.datasets.text import (
    TextDataset,
    StreamingTextDataset,
    MemoryMappedTextDataset,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    
    def encode(self, text: str) -> List[int]:
        """Simple encoding: char codes mod vocab_size."""
        return [ord(c) % self.vocab_size for c in text]


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer fixture."""
    return MockTokenizer()


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating!",
    ]


# ============================================================================
# TEST TEXT DATASET
# ============================================================================

def test_text_dataset_creation(mock_tokenizer, sample_texts):
    """Test creating text dataset."""
    dataset = TextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=20,
        min_chunk_size=1,  # Allow small chunks for testing
        verbose=False
    )
    
    assert len(dataset) > 0
    assert dataset.max_length == 20


def test_text_dataset_getitem(mock_tokenizer, sample_texts):
    """Test getting item from text dataset."""
    dataset = TextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=20,
        min_chunk_size=1,
        verbose=False
    )
    
    item = dataset[0]
    
    assert 'input_ids' in item
    assert 'labels' in item
    assert isinstance(item['input_ids'], torch.Tensor)
    assert item['input_ids'].shape == (20,)


def test_text_dataset_chunking(mock_tokenizer):
    """Test text dataset creates multiple chunks."""
    # Long text that will be chunked
    long_text = "a" * 100  # 100 characters
    
    dataset = TextDataset(
        texts=[long_text],
        tokenizer=mock_tokenizer,
        max_length=10,
        stride=5,
        min_chunk_size=5,
        verbose=False
    )
    
    # Should create multiple chunks due to sliding window
    assert len(dataset) > 1


def test_text_dataset_min_chunk_size(mock_tokenizer):
    """Test dataset respects minimum chunk size."""
    short_text = "abc"  # Only 3 tokens
    
    dataset = TextDataset(
        texts=[short_text],
        tokenizer=mock_tokenizer,
        max_length=10,
        min_chunk_size=5,
        verbose=False
    )
    
    # Should not create chunks smaller than min_chunk_size
    assert len(dataset) == 0


def test_text_dataset_padding(mock_tokenizer):
    """Test dataset pads sequences correctly."""
    short_text = "hi"
    
    dataset = TextDataset(
        texts=[short_text],
        tokenizer=mock_tokenizer,
        max_length=10,
        min_chunk_size=1,
        verbose=False
    )
    
    item = dataset[0]
    
    # Should be padded to max_length
    assert item['input_ids'].shape == (10,)


def test_text_dataset_labels_match_input(mock_tokenizer, sample_texts):
    """Test labels match input_ids for causal LM."""
    dataset = TextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=20,
        min_chunk_size=1,
        verbose=False
    )
    
    item = dataset[0]
    
    # For causal LM, labels should equal input_ids
    assert torch.equal(item['input_ids'], item['labels'])


def test_text_dataset_metadata(mock_tokenizer, sample_texts):
    """Test dataset metadata."""
    dataset = TextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=20,
        min_chunk_size=1,
        verbose=False
    )
    
    metadata = dataset.get_metadata()
    
    assert 'num_texts' in metadata
    assert 'num_sequences' in metadata
    assert metadata['num_texts'] == 3


# ============================================================================
# TEST STREAMING TEXT DATASET (without HF datasets)
# ============================================================================

def test_streaming_dataset_requires_datasets():
    """Test streaming dataset requires datasets library."""
    # This would fail without datasets library
    # Just checking the class exists
    assert StreamingTextDataset is not None


# ============================================================================
# TEST MEMORY MAPPED DATASET (requires numpy)
# ============================================================================

def test_memory_mapped_dataset_requires_cache_file():
    """Test memory mapped dataset requires cache file."""
    with pytest.raises(ValueError, match="cache_file must be provided"):
        MemoryMappedTextDataset(cache_file=None)


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_text_dataset_empty_texts(mock_tokenizer):
    """Test dataset with empty texts list."""
    dataset = TextDataset(
        texts=[],
        tokenizer=mock_tokenizer,
        max_length=10,
        verbose=False
    )
    
    assert len(dataset) == 0


def test_text_dataset_single_text(mock_tokenizer):
    """Test dataset with single text."""
    dataset = TextDataset(
        texts=["Hello world"],
        tokenizer=mock_tokenizer,
        max_length=20,
        min_chunk_size=1,
        verbose=False
    )
    
    assert len(dataset) >= 1


def test_text_dataset_stride_equals_max_length(mock_tokenizer):
    """Test dataset when stride equals max_length (no overlap)."""
    long_text = "a" * 50
    
    dataset = TextDataset(
        texts=[long_text],
        tokenizer=mock_tokenizer,
        max_length=10,
        stride=10,  # No overlap
        min_chunk_size=1,
        verbose=False
    )
    
    # Should create non-overlapping chunks
    assert len(dataset) >= 1


def test_text_dataset_with_transform(mock_tokenizer, sample_texts):
    """Test dataset with custom transform."""
    def transform(item):
        # Add 1 to all input_ids
        item['input_ids'] = item['input_ids'] + 1
        item['labels'] = item['labels'] + 1
        return item
    
    dataset = TextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=20,
        min_chunk_size=1,
        transform=transform,
        verbose=False
    )
    
    item = dataset[0]
    
    # Transform should have been applied
    assert item['input_ids'].min() >= 1  # All values increased


if __name__ == "__main__":
    pytest.main([__file__, "-v"])