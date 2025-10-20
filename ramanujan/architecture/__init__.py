"""
Architecture: Neural network components with Ramanujan sparsity.

This module provides transformer architecture components that can optionally
use Ramanujan graph sparsity for efficiency.

Components:
- Attention mechanisms (GQA, sliding window, sparse)
- Feedforward networks (SwiGLU, MoE)
- Normalization layers (RMSNorm, QKNorm)
- Transformer blocks
- Complete models

Example:
    >>> from ramanujan.architecture import create_model
    >>> from ramanujan.utils import ModelConfig
    >>> 
    >>> config = ModelConfig(dim=890, num_layers=6, num_heads=10)
    >>> model = create_model(config)
"""

# Import components
from .attention import (
    StandardGQA,
    SlidingWindowGQA,
    SparseRamanujanGQA,
    AttentionFactory,
    RotaryPositionalEmbedding,
    apply_rotary_emb
)

from .feedforward import (
    SwiGLU,
    SparseRamanujanSwiGLU,
    FeedForwardFactory
)

from .normalization import (
    RMSNorm,
    QKNorm
)

from .blocks import (
    TransformerBlock,
    EnhancedPretrainingBlock
)

from .model import (
    create_model,
    EnhancedPretrainingModel,
    StandardModel
)

__all__ = [
    # Attention
    'StandardGQA',
    'SlidingWindowGQA',
    'SparseRamanujanGQA',
    'AttentionFactory',
    'RotaryPositionalEmbedding',
    'apply_rotary_emb',
    
    # Feedforward
    'SwiGLU',
    'SparseRamanujanSwiGLU',
    'FeedForwardFactory',
    
    # Normalization
    'RMSNorm',
    'QKNorm',
    
    # Blocks
    'TransformerBlock',
    'EnhancedPretrainingBlock',
    
    # Models
    'create_model',
    'EnhancedPretrainingModel',
    'StandardModel',
]
