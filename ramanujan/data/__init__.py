"""Data loading and processing for Candide framework."""

# Base classes
from .base import (
    DatasetProtocol,
    BaseDataset,
    StreamingDataset,
    TokenizedDataset,
    DatasetConfig,
)

# Text datasets
from .datasets.text import (
    TextDataset,
    StreamingTextDataset,
    MemoryMappedTextDataset,
)

# Collation functions
from .loaders.collation import (
    default_collate,
    padded_collate,
    text_collate,
    causal_lm_collate,
    sequence_classification_collate,
    get_collate_fn,
)

# DataLoader factories
from .loaders.factory import (
    WikiTextLoader,
    FineWebLoader,
    create_dataloader,
    create_dataloader_from_config,
)


__all__ = [
    # Base classes
    'DatasetProtocol',
    'BaseDataset',
    'StreamingDataset',
    'TokenizedDataset',
    'DatasetConfig',
    
    # Text datasets
    'TextDataset',
    'StreamingTextDataset',
    'MemoryMappedTextDataset',
    
    # Collation
    'default_collate',
    'padded_collate',
    'text_collate',
    'causal_lm_collate',
    'sequence_classification_collate',
    'get_collate_fn',
    
    # Loaders
    'WikiTextLoader',
    'FineWebLoader',
    'create_dataloader',
    'create_dataloader_from_config',
]