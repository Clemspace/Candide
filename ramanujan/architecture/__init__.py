"""
Ramanujan architecture components.
"""

# Attention
from .attention import (
    ImprovedGQA,
    ImprovedSlidingWindowGQA,
    StandardGQA,
    SlidingWindowGQA,
)

# Feedforward - use actual names
from .feedforward import (
    SwiGLU,
    SparseRamanujanSwiGLU,
    StandardFFN,
    FeedForwardFactory,
    FeedForwardConfig,
)

# Embeddings (optional)
try:
    from .embeddings import RotaryEmbedding, create_embeddings
except ImportError:
    RotaryEmbedding = create_embeddings = None

# Blocks (optional)
try:
    from .blocks import TransformerBlock, create_transformer_block
except ImportError:
    TransformerBlock = create_transformer_block = None

# Models (optional)
try:
    from .models import EnhancedPretrainingModel, create_model
except ImportError:
    EnhancedPretrainingModel = create_model = None

__all__ = [
    # Attention
    'ImprovedGQA',
    'ImprovedSlidingWindowGQA',
    'StandardGQA',
    'SlidingWindowGQA',
    
    # Feedforward
    'SwiGLU',
    'SparseRamanujanSwiGLU',
    'StandardFFN',
    'FeedForwardFactory',
    'FeedForwardConfig',
    
    # Embeddings
    'RotaryEmbedding',
    'create_embeddings',
    
    # Blocks
    'TransformerBlock',
    'create_transformer_block',
    
    # Models
    'EnhancedPretrainingModel',
    'create_model',
]
