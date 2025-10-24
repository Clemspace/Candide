"""Tests for collation functions."""

import pytest
import torch

from ramanujan.data.loaders.collation import (
    default_collate,
    padded_collate,
    text_collate,
    causal_lm_collate,
    sequence_classification_collate,
    get_collate_fn,
)


# ============================================================================
# TEST DEFAULT COLLATE
# ============================================================================

def test_default_collate():
    """Test default collate function."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5, 6])},
    ]
    
    collated = default_collate(batch)
    
    assert 'input_ids' in collated
    assert collated['input_ids'].shape == (2, 3)


def test_default_collate_empty():
    """Test default collate with empty batch."""
    collated = default_collate([])
    assert collated == {}


def test_default_collate_multiple_keys():
    """Test default collate with multiple keys."""
    batch = [
        {'input_ids': torch.tensor([1, 2]), 'labels': torch.tensor([3, 4])},
        {'input_ids': torch.tensor([5, 6]), 'labels': torch.tensor([7, 8])},
    ]
    
    collated = default_collate(batch)
    
    assert 'input_ids' in collated
    assert 'labels' in collated
    assert collated['input_ids'].shape == (2, 2)


# ============================================================================
# TEST PADDED COLLATE
# ============================================================================

def test_padded_collate():
    """Test padded collate function."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5])},
    ]
    
    collated = padded_collate(batch, pad_token_id=0)
    
    assert collated['input_ids'].shape == (2, 3)
    assert torch.equal(collated['input_ids'][0], torch.tensor([1, 2, 3]))
    assert torch.equal(collated['input_ids'][1], torch.tensor([4, 5, 0]))


def test_padded_collate_left_padding():
    """Test padded collate with left padding."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5])},
    ]
    
    collated = padded_collate(batch, pad_token_id=0, padding_side='left')
    
    assert torch.equal(collated['input_ids'][1], torch.tensor([0, 4, 5]))


def test_padded_collate_custom_keys():
    """Test padded collate with custom keys to pad."""
    batch = [
        {
            'input_ids': torch.tensor([1, 2, 3]),
            'labels': torch.tensor([4, 5]),
            'other': torch.tensor([1])
        },
        {
            'input_ids': torch.tensor([6, 7]),
            'labels': torch.tensor([8, 9, 10]),
            'other': torch.tensor([2])
        }
    ]
    
    collated = padded_collate(
        batch,
        pad_token_id=0,
        keys_to_pad=['input_ids', 'labels']
    )
    
    # input_ids and labels should be padded
    assert collated['input_ids'].shape == (2, 3)
    assert collated['labels'].shape == (2, 3)
    # other should be stacked without padding
    assert collated['other'].shape == (2, 1)


# ============================================================================
# TEST TEXT COLLATE
# ============================================================================

def test_text_collate():
    """Test text collate function."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4])},
        {'input_ids': torch.tensor([5, 6]), 'labels': torch.tensor([6, 7])},
    ]
    
    collated = text_collate(batch, pad_token_id=0)
    
    assert collated['input_ids'].shape == (2, 3)
    assert collated['labels'].shape == (2, 3)
    assert collated['attention_mask'].shape == (2, 3)


def test_text_collate_attention_mask():
    """Test text collate creates correct attention mask."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5])},
    ]
    
    collated = text_collate(batch, pad_token_id=0)
    
    expected_mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0]
    ])
    
    assert torch.equal(collated['attention_mask'], expected_mask)


def test_text_collate_label_padding():
    """Test text collate pads labels with ignore_label_id."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4])},
        {'input_ids': torch.tensor([5, 6]), 'labels': torch.tensor([6, 7])},
    ]
    
    collated = text_collate(batch, pad_token_id=0, ignore_label_id=-100)
    
    assert collated['labels'][1, 2] == -100  # Padded position


def test_text_collate_without_labels():
    """Test text collate without labels."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5])},
    ]
    
    collated = text_collate(batch)
    
    assert 'input_ids' in collated
    assert 'attention_mask' in collated
    assert 'labels' not in collated


# ============================================================================
# TEST CAUSAL LM COLLATE
# ============================================================================

def test_causal_lm_collate():
    """Test causal LM collate (same as text collate)."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4])},
        {'input_ids': torch.tensor([5, 6]), 'labels': torch.tensor([6, 7])},
    ]
    
    collated = causal_lm_collate(batch)
    
    assert 'input_ids' in collated
    assert 'labels' in collated
    assert 'attention_mask' in collated


# ============================================================================
# TEST SEQUENCE CLASSIFICATION COLLATE
# ============================================================================

def test_sequence_classification_collate():
    """Test sequence classification collate."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor(0)},
        {'input_ids': torch.tensor([4, 5]), 'labels': torch.tensor(1)},
    ]
    
    collated = sequence_classification_collate(batch)
    
    assert collated['input_ids'].shape == (2, 3)
    assert collated['labels'].shape == (2,)
    assert torch.equal(collated['labels'], torch.tensor([0, 1]))


def test_sequence_classification_collate_attention_mask():
    """Test sequence classification creates attention mask."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor(0)},
        {'input_ids': torch.tensor([4, 5]), 'labels': torch.tensor(1)},
    ]
    
    collated = sequence_classification_collate(batch)
    
    expected_mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0]
    ])
    
    assert torch.equal(collated['attention_mask'], expected_mask)


# ============================================================================
# TEST GET COLLATE FN
# ============================================================================

def test_get_collate_fn_default():
    """Test getting default collate function."""
    collate_fn = get_collate_fn('default')
    
    batch = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5, 6])},
    ]
    
    collated = collate_fn(batch)
    assert collated['input_ids'].shape == (2, 3)


def test_get_collate_fn_text():
    """Test getting text collate function."""
    collate_fn = get_collate_fn('text', pad_token_id=0)
    
    batch = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5])},
    ]
    
    collated = collate_fn(batch)
    assert 'attention_mask' in collated


def test_get_collate_fn_padded():
    """Test getting padded collate function."""
    collate_fn = get_collate_fn('padded', pad_token_id=0)
    
    batch = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5])},
    ]
    
    collated = collate_fn(batch)
    assert collated['input_ids'].shape == (2, 3)


def test_get_collate_fn_invalid():
    """Test getting invalid collate function raises error."""
    with pytest.raises(ValueError, match="Unknown collate_type"):
        get_collate_fn('invalid_type')


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_collate_single_item_batch():
    """Test collating batch with single item."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4])}
    ]
    
    collated = text_collate(batch)
    
    assert collated['input_ids'].shape == (1, 3)
    assert collated['labels'].shape == (1, 3)


def test_collate_large_padding():
    """Test collating with large padding difference."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])},
        {'input_ids': torch.tensor([11, 12])},
    ]
    
    collated = text_collate(batch, pad_token_id=0)
    
    assert collated['input_ids'].shape == (2, 10)
    assert collated['attention_mask'][1].sum() == 2  # Only 2 real tokens


def test_collate_all_same_length():
    """Test collating when all sequences same length."""
    batch = [
        {'input_ids': torch.tensor([1, 2, 3])},
        {'input_ids': torch.tensor([4, 5, 6])},
        {'input_ids': torch.tensor([7, 8, 9])},
    ]
    
    collated = text_collate(batch)
    
    # No padding needed
    assert collated['input_ids'].shape == (3, 3)
    assert collated['attention_mask'].sum() == 9  # All ones


if __name__ == "__main__":
    pytest.main([__file__, "-v"])