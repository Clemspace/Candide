
"""
Complete transformer models for Ramanujan Transformer.

This module provides complete transformer architectures built from blocks:
- StandardModel: Standard transformer for general tasks
- EnhancedPretrainingModel: Optimized for efficient pretraining
- BaselineModel: Simple baseline for ablation studies

All models support:
- Causal language modeling
- Optional Ramanujan sparsity
- Flexible configuration
- Easy model creation via factory

Example:
    >>> from ramanujan.architecture import create_model
    >>> from ramanujan.utils import ModelConfig
    >>> 
    >>> # Standard model
    >>> config = ModelConfig(
    ...     dim=512,
    ...     num_layers=6,
    ...     num_heads=8,
    ...     num_kv_heads=4,
    ...     vocab_size=32000
    ... )
    >>> model = create_model(config)
    >>> 
    >>> # Enhanced pretraining model
    >>> config_enhanced = ModelConfig(
    ...     dim=890,
    ...     num_layers=6,
    ...     num_heads=10,
    ...     num_kv_heads=5,
    ...     vocab_size=31980,
    ...     model_type='enhanced',
    ...     use_sliding_window=True
    ... )
    >>> model_enhanced = create_model(config_enhanced)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math


from .blocks import (
    EnhancedPretrainingBlock,
    PostNormTransformerBlock,
    TransformerBlock,
    BlockFactory,
    BlockConfig
)
from .normalization import RMSNorm, NormalizationFactory


class SafeEmbedding(nn.Embedding):
    """Embedding with automatic input clamping for vocab-truncated models."""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Clamp to valid range
        input = torch.clamp(input, 0, self.num_embeddings - 1)
        return super().forward(input)


# ============================================================================
# BASE MODEL CLASS
# ============================================================================

class BaseTransformerLM(nn.Module):
    """
    Base class for transformer language models.
    
    Provides common functionality:
    - Token embeddings
    - Position embeddings (optional)
    - Output projection to vocabulary
    - Parameter initialization
    
    Subclasses should implement the forward method.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        max_seq_len: int = 2048,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        #self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.token_embedding = SafeEmbedding(vocab_size, dim, padding_idx=pad_token_id)

        
        # Output projection (tied with input embeddings by default)
        self.output_projection = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters using scaled initialization."""
        # Token embeddings: normal init with std=0.02
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()
        
        # Output projection: normal init with std=0.02
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
    
    def tie_weights(self):
        """Tie input and output embeddings (weight sharing)."""
        self.output_projection.weight = self.token_embedding.weight
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters in model.
        
        Args:
            non_embedding: If True, exclude embedding parameters
        
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        
        return n_params
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'type': self.__class__.__name__,
            'vocab_size': self.vocab_size,
            'dim': self.dim,
            'max_seq_len': self.max_seq_len,
            'total_params': self.get_num_params(non_embedding=False),
            'non_embedding_params': self.get_num_params(non_embedding=True)
        }


# ============================================================================
# STANDARD MODEL
# ============================================================================

class StandardModel(BaseTransformerLM):
    """
    Standard transformer language model.
    
    Architecture:
        - Token embeddings
        - N transformer blocks (pre-norm)
        - Final layer norm
        - Output projection to vocabulary
    
    Features:
    - Grouped Query Attention
    - SwiGLU feedforward
    - RMSNorm for efficiency
    - Optional Ramanujan sparsity
    - Causal masking
    
    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        hidden_dim: FFN hidden dimension (default: None, computed as 8*dim/3)
        dropout: Dropout probability (default: 0.0)
        max_seq_len: Maximum sequence length (default: 2048)
        pad_token_id: Padding token ID (default: 0)
        tie_embeddings: Whether to tie input/output embeddings (default: True)
        foundation: Optional RamanujanFoundation for sparsity
        attention_sparsity: Target sparsity for attention (default: 0.0)
        ffn_sparsity: Target sparsity for FFN (default: 0.0)
        ffn_type: Type of FFN ('swiglu' or 'standard', default: 'swiglu')
        norm_type: Type of normalization ('rms' or 'layer', default: 'rms')
    
    Example:
        >>> model = StandardModel(
        ...     vocab_size=32000,
        ...     dim=512,
        ...     num_layers=6,
        ...     num_heads=8,
        ...     num_kv_heads=4,
        ...     dropout=0.1
        ... )
        >>> 
        >>> # Forward pass
        >>> input_ids = torch.randint(0, 32000, (2, 128))
        >>> logits = model(input_ids)
        >>> print(logits.shape)  # [2, 128, 32000]
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        pad_token_id: int = 0,
        tie_embeddings: bool = True,
        foundation: Optional['RamanujanFoundation'] = None,
        attention_sparsity: float = 0.0,
        ffn_sparsity: float = 0.0,
        ffn_type: str = 'swiglu',
        norm_type: str = 'rms'
    ):
        super().__init__(vocab_size, dim, max_seq_len, pad_token_id)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        # Compute hidden_dim if not provided (SwiGLU uses ~8/3 * dim)
        if hidden_dim is None:
            if ffn_type == 'swiglu':
                hidden_dim = int(8 * dim / 3)
                # Round to nearest multiple of 256 for efficiency
                hidden_dim = ((hidden_dim + 255) // 256) * 256
            else:
                hidden_dim = 4 * dim
        
        self.hidden_dim = hidden_dim
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                attention_dropout=dropout,
                ffn_type=ffn_type,
                norm_type=norm_type,
                foundation=foundation,
                attention_sparsity=attention_sparsity,
                ffn_sparsity=ffn_sparsity
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = NormalizationFactory.create(norm_type, dim)
        
        # Tie embeddings if requested
        if tie_embeddings:
            self.tie_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            return_hidden_states: If True, return (logits, hidden_states)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            or (logits, hidden_states) if return_hidden_states=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        if return_hidden_states:
            return logits, hidden_states
        return logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()
        info.update({
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'hidden_dim': self.hidden_dim,
            'blocks': [block.get_info() for block in self.blocks]
        })
        return info


# ============================================================================
# ENHANCED PRETRAINING MODEL
# ============================================================================

class EnhancedPretrainingModel(BaseTransformerLM):
    """
    Enhanced transformer optimized for efficient pretraining.
    
    Includes all efficiency improvements:
    - Sliding window attention for long sequences
    - Ramanujan graph sparsity
    - SwiGLU feedforward
    - RMSNorm
    - GQA for memory efficiency
    
    This model is designed for pretraining on long sequences with
    limited compute budget.
    
    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        hidden_dim: FFN hidden dimension (default: None, computed)
        dropout: Dropout probability (default: 0.0)
        max_seq_len: Maximum sequence length (default: 2048)
        pad_token_id: Padding token ID (default: 0)
        tie_embeddings: Whether to tie input/output embeddings (default: True)
        foundation: RamanujanFoundation for sparsity (recommended)
        attention_sparsity: Target sparsity for attention (default: 0.82)
        ffn_sparsity: Target sparsity for FFN (default: 0.88)
        use_sliding_window: Enable sliding window attention (default: True)
        window_size: Sliding window size (default: 512)
        num_global_tokens: Number of global attention tokens (default: 64)
    
    Example:
        >>> from ramanujan.foundation import RamanujanFoundation
        >>> foundation = RamanujanFoundation(max_prime=1000)
        >>> 
        >>> model = EnhancedPretrainingModel(
        ...     vocab_size=31980,
        ...     dim=890,
        ...     num_layers=6,
        ...     num_heads=10,
        ...     num_kv_heads=5,
        ...     foundation=foundation,
        ...     attention_sparsity=0.82,
        ...     ffn_sparsity=0.88,
        ...     use_sliding_window=True,
        ...     window_size=512
        ... )
        >>> 
        >>> # Can handle long sequences efficiently
        >>> input_ids = torch.randint(0, 31980, (2, 2048))
        >>> logits = model(input_ids)
        >>> print(logits.shape)  # [2, 2048, 31980]
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        pad_token_id: int = 0,
        tie_embeddings: bool = True,
        foundation: Optional['RamanujanFoundation'] = None,
        attention_sparsity: float = 0.82,
        ffn_sparsity: float = 0.88,
        use_sliding_window: bool = True,
        window_size: int = 512,
        num_global_tokens: int = 64
    ):
        super().__init__(vocab_size, dim, max_seq_len, pad_token_id)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_sliding_window = use_sliding_window
        
        # Compute hidden_dim if not provided
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)
            hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.hidden_dim = hidden_dim
        
        # Enhanced transformer blocks
        self.blocks = nn.ModuleList([
            EnhancedPretrainingBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                attention_dropout=dropout,
                foundation=foundation,
                attention_sparsity=attention_sparsity,
                ffn_sparsity=ffn_sparsity,
                use_sliding_window=use_sliding_window,
                window_size=window_size,
                num_global_tokens=num_global_tokens
            )
            for _ in range(num_layers)
        ])
        
        # Final RMSNorm
        self.final_norm = RMSNorm(dim)
        
        # Tie embeddings if requested
        if tie_embeddings:
            self.tie_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            return_hidden_states: If True, return (logits, hidden_states)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            or (logits, hidden_states) if return_hidden_states=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        if return_hidden_states:
            return logits, hidden_states
        return logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()
        info.update({
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'hidden_dim': self.hidden_dim,
            'use_sliding_window': self.use_sliding_window,
            'blocks': [block.get_info() for block in self.blocks]
        })
        return info


# ============================================================================
# BASELINE MODEL (for ablation)
# ============================================================================

class BaselineModel(BaseTransformerLM):
    """
    Simple baseline transformer for ablation studies.
    
    Minimal implementation with:
    - Standard multi-head attention
    - ReLU feedforward
    - LayerNorm
    - Post-normalization
    
    Use this as a baseline to compare against enhanced versions.
    
    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        hidden_dim: FFN hidden dimension (default: 4*dim)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 2048)
        pad_token_id: Padding token ID (default: 0)
    
    Example:
        >>> # Simple baseline
        >>> baseline = BaselineModel(
        ...     vocab_size=32000,
        ...     dim=512,
        ...     num_layers=6,
        ...     num_heads=8
        ... )
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pad_token_id: int = 0
    ):
        super().__init__(vocab_size, dim, max_seq_len, pad_token_id)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        self.hidden_dim = hidden_dim
        
        # Post-norm transformer blocks
        self.blocks = nn.ModuleList([
            PostNormTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,  # No GQA in baseline
                hidden_dim=hidden_dim,
                dropout=dropout,
                ffn_type='standard',  # Standard FFN with ReLU
                norm_type='layer'  # LayerNorm instead of RMSNorm
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(dim)
        
        # Tie embeddings
        self.tie_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> torch.Tensor:
        """Forward pass."""
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        if return_hidden_states:
            return logits, hidden_states
        return logits


# ============================================================================
# MODEL FACTORY
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for transformer models."""
    # Architecture
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_dim: Optional[int] = None
    
    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0
    max_seq_len: int = 2048
    pad_token_id: int = 0
    tie_embeddings: bool = True

    
    # Model type
    model_type: str = 'standard'  # 'standard', 'enhanced', 'baseline'
    
    # Block config
    ffn_type: str = 'swiglu'
    norm_type: str = 'rms'
    
    # Ramanujan sparsity
    foundation: Optional['RamanujanFoundation'] = None
    attention_sparsity: float = 0.0
    ffn_sparsity: float = 0.0
    
    # Sliding window (for enhanced model)
    use_sliding_window: bool = False
    window_size: int = 512
    num_global_tokens: int = 64


def create_model(config: ModelConfig) -> nn.Module:
    """
    Create transformer model based on config.
    
    Main entry point for model creation. Automatically selects
    the appropriate model class based on config.model_type.
    
    Args:
        config: ModelConfig instance
    
    Returns:
        Transformer model
    
    Example:
        >>> from ramanujan.architecture import create_model, ModelConfig
        >>> 
        >>> config = ModelConfig(
        ...     vocab_size=32000,
        ...     dim=512,
        ...     num_layers=6,
        ...     num_heads=8,
        ...     num_kv_heads=4,
        ...     model_type='standard'
        ... )
        >>> model = create_model(config)
    """
    model_type = config.model_type.lower()
    
    if model_type in ['standard', 'default']:
        return StandardModel(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            pad_token_id=config.pad_token_id,
            tie_embeddings=config.tie_embeddings,
            foundation=config.foundation,
            attention_sparsity=config.attention_sparsity,
            ffn_sparsity=config.ffn_sparsity,
            ffn_type=config.ffn_type,
            norm_type=config.norm_type
        )
    elif model_type in ['enhanced', 'pretraining']:
        return EnhancedPretrainingModel(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            pad_token_id=config.pad_token_id,
            tie_embeddings=config.tie_embeddings,
            foundation=config.foundation,
            attention_sparsity=config.attention_sparsity,
            ffn_sparsity=config.ffn_sparsity,
            use_sliding_window=config.use_sliding_window,
            window_size=config.window_size,
            num_global_tokens=config.num_global_tokens
        )
    elif model_type == 'baseline':
        return BaselineModel(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            pad_token_id=config.pad_token_id
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from: 'standard', 'enhanced', 'baseline'"
        )


class ModelFactory:
    """Factory for creating transformer models."""
    
    @staticmethod
    def create(config: ModelConfig) -> nn.Module:
        """Create model based on config."""
        # This is your existing create_model() logic
        model_type = config.model_type.lower()
        
        if model_type in ['standard', 'default']:
            return StandardModel(...)
        elif model_type in ['enhanced', 'pretraining']:
            return EnhancedPretrainingModel(...)
        elif model_type == 'baseline':
            return BaselineModel(...)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    @staticmethod
    def create_from_dict(config_dict: dict) -> nn.Module:
        """Create from dictionary."""
        config = ModelConfig(**config_dict)
        return ModelFactory.create(config)

# Keep old function for backward compatibility
def create_model(config: ModelConfig) -> nn.Module:
    """Legacy function - use ModelFactory.create() instead."""
    return ModelFactory.create(config)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, int]:
    """
    Count parameters in model by category.
    
    Args:
        model: Model to analyze
        trainable_only: If True, only count trainable parameters
    
    Returns:
        Dictionary with parameter counts by category
    
    Example:
        >>> model = create_model(config)
        >>> params = count_parameters(model)
        >>> print(f"Total: {params['total']:,}")
        >>> print(f"Embeddings: {params['embeddings']:,}")
        >>> print(f"Blocks: {params['blocks']:,}")
    """
    def count_params(module):
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in module.parameters())
    
    counts = {
        'total': count_params(model),
        'embeddings': count_params(model.token_embedding),
        'output_projection': count_params(model.output_projection),
        'blocks': count_params(model.blocks),
        'final_norm': count_params(model.final_norm)
    }
    
    # Per-block breakdown
    counts['per_block'] = [count_params(block) for block in model.blocks]
    
    return counts


def estimate_model_memory(
    config: ModelConfig,
    batch_size: int = 1,
    seq_len: int = 512,
    dtype: str = 'float32'
) -> Dict[str, float]:
    """
    Estimate memory usage for model.
    
    Args:
        config: Model configuration
        batch_size: Batch size
        seq_len: Sequence length
        dtype: Data type ('float32', 'float16', 'bfloat16')
    
    Returns:
        Dictionary with memory estimates in MB
    
    Example:
        >>> config = ModelConfig(dim=512, num_layers=6, ...)
        >>> mem = estimate_model_memory(config, batch_size=8, seq_len=512)
        >>> print(f"Model params: {mem['model_params_mb']:.1f} MB")
        >>> print(f"Activations: {mem['activations_mb']:.1f} MB")
        >>> print(f"Total: {mem['total_mb']:.1f} MB")
    """
    # Bytes per element
    bytes_per_element = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2
    }[dtype]
    
    # Model parameters
    # Rough estimate based on architecture
    embedding_params = config.vocab_size * config.dim
    
    # Per-block parameters (attention + FFN + norms)
    head_dim = config.dim // config.num_heads
    attn_params_per_block = (
        config.dim * (config.num_heads * head_dim) +  # Q
        config.dim * (config.num_kv_heads * head_dim) +  # K
        config.dim * (config.num_kv_heads * head_dim) +  # V
        (config.num_heads * head_dim) * config.dim  # O
    )
    
    hidden_dim = config.hidden_dim or int(8 * config.dim / 3)
    if config.ffn_type == 'swiglu':
        ffn_params_per_block = 3 * config.dim * hidden_dim
    else:
        ffn_params_per_block = 2 * config.dim * hidden_dim
    
    norm_params_per_block = 2 * config.dim
    block_params = attn_params_per_block + ffn_params_per_block + norm_params_per_block
    
    total_params = embedding_params + (config.num_layers * block_params) + config.dim
    
    # Apply sparsity if applicable
    if config.attention_sparsity > 0:
        attn_params_per_block = int(attn_params_per_block * (1 - config.attention_sparsity))
    if config.ffn_sparsity > 0:
        ffn_params_per_block = int(ffn_params_per_block * (1 - config.ffn_sparsity))
        block_params = attn_params_per_block + ffn_params_per_block + norm_params_per_block
        total_params = embedding_params + (config.num_layers * block_params) + config.dim
    
    model_params_mb = (total_params * bytes_per_element) / (1024 ** 2)
    
    # Activation memory (rough estimate)
    # Hidden states: batch * seq * dim per layer
    hidden_states_mb = (batch_size * seq_len * config.dim * bytes_per_element * config.num_layers) / (1024 ** 2)
    
    # Attention scores: batch * num_heads * seq * seq per layer
    if config.use_sliding_window:
        # Approximate with window size
        effective_seq = min(config.window_size, seq_len)
        attn_scores_mb = (batch_size * config.num_heads * seq_len * effective_seq * bytes_per_element * config.num_layers) / (1024 ** 2)
    else:
        attn_scores_mb = (batch_size * config.num_heads * seq_len * seq_len * bytes_per_element * config.num_layers) / (1024 ** 2)
    
    activations_mb = hidden_states_mb + attn_scores_mb
    
    # KV cache (for inference)
    kv_cache_mb = (2 * batch_size * config.num_kv_heads * seq_len * (config.dim // config.num_heads) * bytes_per_element * config.num_layers) / (1024 ** 2)
    
    total_mb = model_params_mb + activations_mb
    total_with_kv_mb = total_mb + kv_cache_mb
    
    return {
        'model_params_mb': model_params_mb,
        'hidden_states_mb': hidden_states_mb,
        'attention_scores_mb': attn_scores_mb,
        'activations_mb': activations_mb,
        'kv_cache_mb': kv_cache_mb,
        'total_mb': total_mb,
        'total_with_kv_mb': total_with_kv_mb
    }


def compare_models(configs: Dict[str, ModelConfig]) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple model configurations.
    
    Args:
        configs: Dictionary mapping names to ModelConfig instances
    
    Returns:
        Dictionary with comparison results
    
    Example:
        >>> configs = {
        ...     'small': ModelConfig(dim=512, num_layers=6, ...),
        ...     'optimal': ModelConfig(dim=890, num_layers=6, ...)
        ... }
        >>> comparison = compare_models(configs)
        >>> for name, info in comparison.items():
        ...     print(f"{name}: {info['params']:,} params")
    """
    results = {}
    
    for name, config in configs.items():
        # Create model
        model = create_model(config)
        
        # Get parameter counts
        params = count_parameters(model)
        
        # Get memory estimates
        memory = estimate_model_memory(config)
        
        results[name] = {
            'config': config,
            'total_params': params['total'],
            'trainable_params': count_parameters(model, trainable_only=True)['total'],
            'embedding_params': params['embeddings'],
            'block_params': params['blocks'],
            'params_per_block': params['per_block'][0] if params['per_block'] else 0,
            'model_memory_mb': memory['model_params_mb'],
            'total_memory_mb': memory['total_mb']
        }
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optional optimizer to save
        checkpoint_path: Path to save checkpoint
        step: Training step
        epoch: Training epoch
        loss: Current loss
        **kwargs: Additional metadata to save
    
    Example:
        >>> save_checkpoint(
        ...     model,
        ...     optimizer,
        ...     'checkpoints/model_step_10000.pt',
        ...     step=10000,
        ...     loss=2.5
        ... )
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'step': step,
        'epoch': epoch,
        'loss': loss
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add model config if available
    if hasattr(model, 'get_model_info'):
        checkpoint['model_info'] = model.get_model_info()
    
    # Add any additional metadata
    checkpoint.update(kwargs)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")


def save_model(
    model: nn.Module,
    save_path: str,
    save_format: str = 'pytorch'
):
    """
    Save model weights only (no optimizer state).
    
    Args:
        model: Model to save
        save_path: Path to save model
        save_format: Format to save ('pytorch', 'safetensors')
    
    Example:
        >>> save_model(model, 'models/my_model.pt')
    """
    if save_format == 'pytorch':
        # Create directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save state dict
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
    
    elif save_format == 'safetensors':
        try:
            from safetensors.torch import save_file
            
            # Create directory
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save with safetensors
            save_file(model.state_dict(), save_path)
            print(f"Model saved to: {save_path} (safetensors)")
        
        except ImportError:
            print("safetensors not installed. Install with: pip install safetensors")
            print("Falling back to PyTorch format...")
            save_model(model, save_path, save_format='pytorch')
    
    else:
        raise ValueError(f"Unknown save_format: {save_format}")


def load_model(
    model: nn.Module,
    model_path: str,
    strict: bool = True,
    map_location: Optional[str] = None
) -> nn.Module:
    """
    Load model weights.
    
    Args:
        model: Model to load weights into
        model_path: Path to model file
        strict: Whether to strictly enforce matching keys
        map_location: Device to map tensors to
    
    Returns:
        Model with loaded weights
    
    Example:
        >>> model = load_model(model, 'models/my_model.pt', map_location='cuda')
    """
    print(f"Loading model from: {model_path}")
    
    # Detect format
    if model_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        except ImportError:
            raise ImportError("safetensors not installed. Install with: pip install safetensors")
    else:
        state_dict = torch.load(model_path, map_location=map_location)
    
    # Load state dict
    model.load_state_dict(state_dict, strict=strict)
    print("Model loaded successfully!")
    
    return model


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing model.py module")
    print("="*70)
    
    # Test StandardModel
    print("\n1. Testing StandardModel...")
    model = StandardModel(
        vocab_size=1000,
        dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.1
    )
    
    input_ids = torch.randint(0, 1000, (2, 64))
    logits = model(input_ids)
    
    assert logits.shape == (2, 64, 1000), "Shape mismatch!"
    
    print(f"   Input: {input_ids.shape}")
    print(f"   Output: {logits.shape}")
    print(f"   Parameters: {model.get_num_params():,}")
    print(f"   ✅ StandardModel working!")
    
    # Test with return_hidden_states
    logits, hidden_states = model(input_ids, return_hidden_states=True)
    assert len(hidden_states) == 2, "Wrong number of hidden states!"
    print(f"   Hidden states: {len(hidden_states)} layers")
    print(f"   ✅ Hidden states working!")
    
    # Test EnhancedPretrainingModel
    print("\n2. Testing EnhancedPretrainingModel...")
    enhanced = EnhancedPretrainingModel(
        vocab_size=1000,
        dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        use_sliding_window=True,
        window_size=512
    )
    
    logits_enhanced = enhanced(input_ids)
    
    assert logits_enhanced.shape == (2, 64, 1000), "Shape mismatch!"
    
    print(f"   Input: {input_ids.shape}")
    print(f"   Output: {logits_enhanced.shape}")
    print(f"   Parameters: {enhanced.get_num_params():,}")
    print(f"   ✅ EnhancedPretrainingModel working!")
    
    # Test with longer sequence
    input_ids_long = torch.randint(0, 1000, (1, 512))
    logits_long = enhanced(input_ids_long)
    print(f"   Long sequence: {input_ids_long.shape} -> {logits_long.shape}")
    print(f"   ✅ Long sequence working!")
    
    # Test BaselineModel
    print("\n3. Testing BaselineModel...")
    baseline = BaselineModel(
        vocab_size=1000,
        dim=256,
        num_layers=2,
        num_heads=4
    )
    
    logits_baseline = baseline(input_ids)
    
    assert logits_baseline.shape == (2, 64, 1000), "Shape mismatch!"
    
    print(f"   Input: {input_ids.shape}")
    print(f"   Output: {logits_baseline.shape}")
    print(f"   Parameters: {baseline.get_num_params():,}")
    print(f"   ✅ BaselineModel working!")
    
    # Test model factory
    print("\n4. Testing create_model factory...")
    config = ModelConfig(
        vocab_size=1000,
        dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        model_type='standard'
    )
    model_factory = create_model(config)
    logits_factory = model_factory(input_ids)
    
    print(f"   Factory created: {type(model_factory).__name__}")
    print(f"   Output: {logits_factory.shape}")
    print(f"   ✅ Model factory working!")
    
    # Test with enhanced config
    config_enhanced = ModelConfig(
        vocab_size=1000,
        dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        model_type='enhanced',
        use_sliding_window=True
    )
    model_enhanced_factory = create_model(config_enhanced)
    print(f"   Enhanced factory created: {type(model_enhanced_factory).__name__}")
    print(f"   ✅ Enhanced factory working!")
    
    # Test parameter counting
    print("\n5. Testing parameter counting...")
    params = count_parameters(model)
    
    print(f"   Total params: {params['total']:,}")
    print(f"   Embedding params: {params['embeddings']:,}")
    print(f"   Block params: {params['blocks']:,}")
    print(f"   Params per block: {params['per_block'][0]:,}")
    print(f"   ✅ Parameter counting working!")
    
    # Test memory estimation
    print("\n6. Testing memory estimation...")
    mem = estimate_model_memory(config, batch_size=8, seq_len=512)
    
    print(f"   Model params: {mem['model_params_mb']:.1f} MB")
    print(f"   Activations: {mem['activations_mb']:.1f} MB")
    print(f"   KV cache: {mem['kv_cache_mb']:.1f} MB")
    print(f"   Total: {mem['total_mb']:.1f} MB")
    print(f"   Total with KV: {mem['total_with_kv_mb']:.1f} MB")
    print(f"   ✅ Memory estimation working!")
    
    # Test model comparison
    print("\n7. Testing model comparison...")
    configs = {
        'small': ModelConfig(
            vocab_size=1000, dim=256, num_layers=2,
            num_heads=4, num_kv_heads=2, model_type='standard'
        ),
        'medium': ModelConfig(
            vocab_size=1000, dim=512, num_layers=4,
            num_heads=8, num_kv_heads=4, model_type='standard'
        )
    }
    comparison = compare_models(configs)
    
    print("\n   Model Comparison:")
    for name, info in comparison.items():
        print(f"   {name:8s}: {info['total_params']:8,} params, "
              f"{info['model_memory_mb']:6.1f} MB")
    print(f"   ✅ Model comparison working!")
    
    # Test gradient flow
    print("\n8. Testing gradient flow...")
    model_grad = StandardModel(
        vocab_size=1000,
        dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2
    )
    input_ids_grad = torch.randint(0, 1000, (2, 64))
    logits_grad = model_grad(input_ids_grad)
    
    # Create fake loss
    targets = torch.randint(0, 1000, (2, 64))
    loss = F.cross_entropy(
        logits_grad.view(-1, 1000),
        targets.view(-1)
    )
    loss.backward()
    
    # Check that gradients exist
    has_grads = any(p.grad is not None for p in model_grad.parameters())
    assert has_grads, "No gradients computed!"
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model_grad.parameters(), 1.0)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradient norm: {grad_norm:.4f}")
    print(f"   ✅ Gradient flow working!")
    
    # Test get_model_info
    print("\n9. Testing get_model_info...")
    info = model.get_model_info()
    
    print(f"   Model type: {info['type']}")
    print(f"   Dimension: {info['dim']}")
    print(f"   Num layers: {info['num_layers']}")
    print(f"   Total params: {info['total_params']:,}")
    print(f"   ✅ get_model_info working!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.architecture import create_model, ModelConfig")
    print("  from ramanujan.architecture import StandardModel, EnhancedPretrainingModel")
    print("="*70)