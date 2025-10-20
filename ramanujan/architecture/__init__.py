"""
Ramanujan architecture components.

Modular design:
- attention: Multi-head attention variants (GQA, MQA, etc.)
- feedforward: FFN layers with sparsity
- embeddings: Token and position embeddings
- blocks: Transformer blocks
- models: Complete model definitions
"""

from .attention import (
    ImprovedGQA,
    ImprovedSlidingWindowGQA,

)

from .feedforward import (
    SwiGLU,
    SparseRamanujanSwiGLU,
    StandardFFN,
    FeedForwardFactory,
    FeedForwardConfig,
)

from .embeddings import (
    RotaryEmbedding,
    SinusoidalEmbedding,
    LearnedPositionEmbedding,
    create_position_embedding
)

from .blocks import (
    EnhancedPretrainingBlock,
    PostNormTransformerBlock,
    TransformerBlock,
    BlockFactory,
    BlockConfig,
)

from .model import (
    EnhancedPretrainingModel,
    create_model,
    StandardModel,
    BaselineModel,
    ModelConfig,
    count_parameters,
    estimate_model_memory,
    compare_models,
    save_checkpoint,
    save_model,
    load_model,
)

__all__ = [
    # Attention
    'ImprovedGQA',
    'ImprovedSlidingWindowGQA',

    
    # Feedforward
    'SwiGLU',
    'SparseRamanujanSwiGLU',
    'StandardFFN',
    'FeedForwardFactory',
    'FeedForwardConfig',
    
    # Embeddings
    'RotaryEmbedding',
    'SinusoidalEmbedding',
    'LearnedPositionEmbedding',
    'create_position_embedding',
    
    # Blocks
    'TransformerBlock',
    'BlockFactory',
    'BlockConfig',
    'EnhancedPretrainingBlock',
    'PostNormTransformerBlock',
    'create_transformer_block',
    
    # Models
    'EnhancedPretrainingModel',
    'create_model', 'StandardModel', 'BaselineModel', 'ModelConfig', 'create_model_from_dict', 'count_parameters', 'estimate_model_memory', 'compare_models', 'save_checkpoint', 'save_model', 'load_model',
]
