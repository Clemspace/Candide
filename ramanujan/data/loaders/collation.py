"""Collation functions for batching data."""

import torch
from torch import Tensor
from typing import List, Dict, Any, Optional, Tuple


def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
    """
    Default collate function - simple stacking.
    
    Args:
        batch: List of dictionaries
    
    Returns:
        Dictionary with stacked tensors
    
    Example:
        >>> batch = [
        ...     {'input_ids': torch.tensor([1, 2, 3])},
        ...     {'input_ids': torch.tensor([4, 5, 6])}
        ... ]
        >>> collated = default_collate(batch)
        >>> collated['input_ids'].shape
        torch.Size([2, 3])
    """
    if not batch:
        return {}
    
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        if all(isinstance(v, Tensor) for v in values):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    
    return collated


def padded_collate(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
    padding_side: str = 'right',
    keys_to_pad: Optional[List[str]] = None
) -> Dict[str, Tensor]:
    """
    Collate with padding to max length in batch.
    
    Args:
        batch: List of dictionaries
        pad_token_id: Token ID to use for padding
        padding_side: 'right' or 'left'
        keys_to_pad: Keys to pad (default: ['input_ids', 'labels'])
    
    Returns:
        Dictionary with padded tensors
    
    Example:
        >>> batch = [
        ...     {'input_ids': torch.tensor([1, 2, 3])},
        ...     {'input_ids': torch.tensor([4, 5, 6, 7, 8])}
        ... ]
        >>> collated = padded_collate(batch)
        >>> collated['input_ids'].shape
        torch.Size([2, 5])
    """
    if not batch:
        return {}
    
    keys_to_pad = keys_to_pad or ['input_ids', 'labels']
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # Check if all are tensors
        if not all(isinstance(v, Tensor) for v in values):
            collated[key] = values
            continue
        
        # Pad if in keys_to_pad
        if key in keys_to_pad:
            # Find max length
            max_len = max(v.size(0) for v in values)
            
            # Pad each tensor
            padded = []
            for v in values:
                if v.size(0) < max_len:
                    pad_size = max_len - v.size(0)
                    
                    if padding_side == 'right':
                        padding = torch.full((pad_size,), pad_token_id, dtype=v.dtype)
                        v = torch.cat([v, padding], dim=0)
                    else:  # left
                        padding = torch.full((pad_size,), pad_token_id, dtype=v.dtype)
                        v = torch.cat([padding, v], dim=0)
                
                padded.append(v)
            
            collated[key] = torch.stack(padded)
        else:
            # Stack without padding
            collated[key] = torch.stack(values)
    
    return collated


def text_collate(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
    ignore_label_id: int = -100
) -> Dict[str, Tensor]:
    """
    Collate function for text data.
    
    Pads input_ids and creates attention_mask.
    Sets padding positions in labels to ignore_label_id.
    
    Args:
        batch: List of dictionaries with 'input_ids' and optionally 'labels'
        pad_token_id: Token ID to use for padding
        ignore_label_id: Label ID for padding positions
    
    Returns:
        Dictionary with:
        - input_ids: Padded input IDs (batch, max_len)
        - attention_mask: Attention mask (batch, max_len)
        - labels: Padded labels if present (batch, max_len)
    
    Example:
        >>> batch = [
        ...     {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4])},
        ...     {'input_ids': torch.tensor([5, 6]), 'labels': torch.tensor([6, 7])}
        ... ]
        >>> collated = text_collate(batch)
        >>> collated['input_ids'].shape
        torch.Size([2, 3])
        >>> collated['attention_mask']
        tensor([[1, 1, 1],
                [1, 1, 0]])
    """
    if not batch:
        return {}
    
    # Extract input_ids
    input_ids_list = [item['input_ids'] for item in batch]
    max_len = max(ids.size(0) for ids in input_ids_list)
    batch_size = len(input_ids_list)
    
    # Pad input_ids and create attention_mask
    padded_input_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=input_ids_list[0].dtype
    )
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i, ids in enumerate(input_ids_list):
        seq_len = ids.size(0)
        padded_input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1
    
    result = {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask
    }
    
    # Handle labels if present
    if 'labels' in batch[0]:
        labels_list = [item['labels'] for item in batch]
        padded_labels = torch.full(
            (batch_size, max_len),
            ignore_label_id,
            dtype=labels_list[0].dtype
        )
        
        for i, labels in enumerate(labels_list):
            seq_len = labels.size(0)
            padded_labels[i, :seq_len] = labels
        
        result['labels'] = padded_labels
    
    return result


def causal_lm_collate(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
    ignore_label_id: int = -100
) -> Dict[str, Tensor]:
    """
    Collate function for causal language modeling.
    
    Same as text_collate but optimized for causal LM:
    - Labels are input_ids shifted by 1
    - Efficient padding
    
    Args:
        batch: List of dictionaries with 'input_ids'
        pad_token_id: Token ID to use for padding
        ignore_label_id: Label ID for padding positions
    
    Returns:
        Dictionary with input_ids, attention_mask, labels
    """
    return text_collate(batch, pad_token_id, ignore_label_id)


def sequence_classification_collate(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0
) -> Dict[str, Tensor]:
    """
    Collate function for sequence classification.
    
    Args:
        batch: List of dictionaries with 'input_ids' and 'labels'
        pad_token_id: Token ID to use for padding
    
    Returns:
        Dictionary with padded inputs and label tensor
    
    Example:
        >>> batch = [
        ...     {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor(0)},
        ...     {'input_ids': torch.tensor([4, 5]), 'labels': torch.tensor(1)}
        ... ]
        >>> collated = sequence_classification_collate(batch)
        >>> collated['labels']
        tensor([0, 1])
    """
    # Pad input_ids
    input_ids_list = [item['input_ids'] for item in batch]
    max_len = max(ids.size(0) for ids in input_ids_list)
    batch_size = len(input_ids_list)
    
    padded_input_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=input_ids_list[0].dtype
    )
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i, ids in enumerate(input_ids_list):
        seq_len = ids.size(0)
        padded_input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1
    
    result = {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask
    }
    
    # Stack labels (should be scalars)
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        result['labels'] = labels
    
    return result


def get_collate_fn(
    collate_type: str = 'default',
    pad_token_id: int = 0,
    ignore_label_id: int = -100,
    **kwargs
) -> callable:
    """
    Get collate function by name.
    
    Args:
        collate_type: Type of collate ('default', 'padded', 'text', 'causal_lm', 'seq_class')
        pad_token_id: Token ID for padding
        ignore_label_id: Label ID for ignoring in loss
        **kwargs: Additional arguments
    
    Returns:
        Collate function
    
    Example:
        >>> collate_fn = get_collate_fn('text', pad_token_id=0)
        >>> # Use with DataLoader
        >>> loader = DataLoader(dataset, collate_fn=collate_fn)
    """
    collate_type = collate_type.lower()
    
    if collate_type == 'default':
        return default_collate
    
    elif collate_type == 'padded':
        return lambda batch: padded_collate(
            batch,
            pad_token_id=pad_token_id,
            **kwargs
        )
    
    elif collate_type == 'text' or collate_type == 'causal_lm':
        return lambda batch: text_collate(
            batch,
            pad_token_id=pad_token_id,
            ignore_label_id=ignore_label_id
        )
    
    elif collate_type == 'seq_class' or collate_type == 'classification':
        return lambda batch: sequence_classification_collate(
            batch,
            pad_token_id=pad_token_id
        )
    
    else:
        raise ValueError(
            f"Unknown collate_type: {collate_type}. "
            f"Available: default, padded, text, causal_lm, seq_class"
        )