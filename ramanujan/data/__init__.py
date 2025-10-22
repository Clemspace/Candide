"""
Data: Dataset loaders and preprocessing utilities.

This module provides data loading and preprocessing for various datasets
including WikiText, FineWeb, and custom datasets.

Components:
- Dataset loaders (WikiText, FineWeb)
- Tokenizer wrappers
- Curriculum learning samplers
- Data utilities

Example:
    >>> from ramanujan.data import WikiTextLoader
    >>> 
    >>> loader = WikiTextLoader(
    ...     dataset_name="wikitext-2",
    ...     vocab_size=31980,
    ...     sequence_length=128
    ... )
    >>> train_loader, eval_loader = loader.get_dataloaders(batch_size=8)
"""

from .datasets import (
    WikiTextLoader,
    FineWebLoader
)

from .tokenizer import (
    VocabConstrainedTokenizer,
    MistralTokenizerWrapper,
    get_tokenizer
)

from .curriculum import (
    ConfidenceBasedSampler,
    CurriculumDataLoader
)

__all__ = [
    # Datasets
    'WikiTextLoader',
    'FineWebLoader',
    
    # Tokenizer
    'VocabConstrainedTokenizer',
    'MistralTokenizerWrapper',
    'get_tokenizer',
    
    # Curriculum
    'ConfidenceBasedSampler',
    'CurriculumDataLoader',
]