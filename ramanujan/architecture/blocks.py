"""
Transformer blocks for Ramanujan Transformer.

This module provides transformer block architectures that combine
attention and feedforward networks with optional Ramanujan sparsity.

Blocks:
- TransformerBlock: Standard pre-norm transformer block
- EnhancedPretrainingBlock: Block with all improvements (sliding window, etc.)
- PostNormTransformerBlock: Post-norm variant (for ablation)

All blocks follow the pre-normalization pattern for better gradient flow:
    x = x + Attention(Norm(x))
    x = x + FFN(Norm(x))

Example:
    >>> from ramanujan.architecture import TransformerBlock
    >>> from ramanujan.foundation import RamanujanFoundation
    >>> 
    >>> # Standard block
    >>> block = TransformerBlock(
    ...     dim=512,
    ...     num_heads=8,
    ...     num_kv_heads=4,
    ...     hidden_dim=2048
    ... )
    >>> 
    >>> # Enhanced block with all improvements
    >>> foundation = RamanujanFoundation(max_prime=1000)
    >>> enhanced_block = EnhancedPretrainingBlock(
    ...     dim=512,
    ...     num_heads=8,
    ...     num_kv_heads=4,
    ...     hidden_dim=2048,
    ...     foundation=foundation,
    ...     attention_sparsity=0.82,
    ...     ffn_sparsity=0.88,
    ...     use_sliding_window=True,
    ...     window_size=512
    ... )
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

from .attention_factory import (
    AttentionFactory, 
    AttentionConfig,
)
from .attention import (
    ImprovedSlidingWindowGQA
)

from .feedforward import (
    FeedForwardFactory,
    FeedForwardConfig,
    SwiGLU,
    StandardFFN
)
from .normalization import (
    RMSNorm,
    NormalizationFactory
)


# ============================================================================
# STANDARD TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Standard pre-normalization transformer block.
    
    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    
    Pre-normalization (as opposed to post-norm) provides:
    - Better gradient flow
    - More stable training
    - Can train deeper models
    
    Features:
    - Grouped Query Attention with RoPE
    - SwiGLU feedforward (default) or standard FFN
    - RMSNorm for efficiency
    - Optional Ramanujan sparsity in all projections
    - Optional dropout
    
    Args:
        dim: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        hidden_dim: FFN hidden dimension (typically ~2.67*dim for SwiGLU)
        dropout: Dropout probability (default: 0.0)
        attention_dropout: Attention-specific dropout (default: 0.0)
        ffn_type: Type of FFN ('swiglu' or 'standard', default: 'swiglu')
        norm_type: Type of normalization ('rms' or 'layer', default: 'rms')
        foundation: Optional RamanujanFoundation for sparse layers
        attention_sparsity: Target sparsity for attention projections
        ffn_sparsity: Target sparsity for FFN layers
    
    Example:
        >>> block = TransformerBlock(
        ...     dim=512,
        ...     num_heads=8,
        ...     num_kv_heads=4,
        ...     hidden_dim=2048,
        ...     dropout=0.1
        ... )
        >>> x = torch.randn(2, 128, 512)
        >>> out = block(x)
        >>> print(out.shape)  # [2, 128, 512]
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        ffn_type: str = 'swiglu',
        norm_type: str = 'rms',
        foundation: Optional['RamanujanFoundation'] = None,
        attention_sparsity: float = 0.0,
        ffn_sparsity: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_dim = hidden_dim
        
        # Pre-normalization layers
        self.attn_norm = NormalizationFactory.create(norm_type, dim)
        self.ffn_norm = NormalizationFactory.create(norm_type, dim)
        
        # Attention
        attn_config = AttentionConfig(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=attention_dropout,
            foundation=foundation,
            attention_sparsity=attention_sparsity,
            use_sliding_window=False  # Standard block doesn't use sliding window
        )
        self.attention = AttentionFactory.create(attn_config)
        
        # Feedforward
        ffn_config = FeedForwardConfig(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ffn_type=ffn_type,
            foundation=foundation,
            ffn_sparsity=ffn_sparsity
        )
        self.ffn = FeedForwardFactory.create(ffn_config)
        
        # Residual dropout (applied after attention and FFN)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            mask: Optional attention mask [batch, seq_len] or [batch, seq_len, seq_len]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Self-attention with residual
        attn_out = self.attention(self.attn_norm(x), mask=mask)
        x = x + self.dropout(attn_out)
        
        # Feedforward with residual
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)
        
        return x
    
    def get_info(self) -> dict:
        """Get information about this block."""
        from .attention import get_attention_info
        from .feedforward import get_ffn_info
        
        info = {
            'type': 'TransformerBlock',
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'hidden_dim': self.hidden_dim,
            'attention': get_attention_info(self.attention),
            'ffn': get_ffn_info(self.ffn)
        }
        
        return info


# ============================================================================
# ENHANCED PRETRAINING BLOCK
# ============================================================================

class EnhancedPretrainingBlock(nn.Module):
    """
    Enhanced transformer block with all improvements.
    
    This block includes all the enhancements for efficient pretraining:
    - Sliding window attention for O(n*window) complexity
    - Ramanujan graph sparsity in all projections
    - SwiGLU feedforward
    - RMSNorm for efficiency
    - QK-Normalization in attention
    
    Architecture:
        x = x + SlidingWindowGQA(RMSNorm(x))
        x = x + SparseRamanujanSwiGLU(RMSNorm(x))
    
    This is the recommended block for pretraining on long sequences
    with limited compute.
    
    Args:
        dim: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        hidden_dim: FFN hidden dimension
        dropout: Dropout probability (default: 0.0)
        attention_dropout: Attention-specific dropout (default: 0.0)
        foundation: RamanujanFoundation for sparse layers (required for sparsity)
        attention_sparsity: Target sparsity for attention (default: 0.82)
        ffn_sparsity: Target sparsity for FFN (default: 0.88)
        use_sliding_window: Enable sliding window attention (default: True)
        window_size: Sliding window size (default: 512)
        num_global_tokens: Number of global attention tokens (default: 64)
    
    Example:
        >>> from ramanujan.foundation import RamanujanFoundation
        >>> foundation = RamanujanFoundation(max_prime=1000)
        >>> 
        >>> block = EnhancedPretrainingBlock(
        ...     dim=890,
        ...     num_heads=10,
        ...     num_kv_heads=5,
        ...     hidden_dim=2370,
        ...     foundation=foundation,
        ...     attention_sparsity=0.82,
        ...     ffn_sparsity=0.88,
        ...     use_sliding_window=True,
        ...     window_size=512
        ... )
        >>> 
        >>> # Can handle long sequences efficiently
        >>> x = torch.randn(2, 2048, 890)
        >>> out = block(x)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        foundation: Optional['RamanujanFoundation'] = None,
        attention_sparsity: float = 0.82,
        ffn_sparsity: float = 0.88,
        use_sliding_window: bool = True,
        window_size: int = 512,
        num_global_tokens: int = 64
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_dim = hidden_dim
        self.use_sliding_window = use_sliding_window
        
        # Pre-normalization layers (RMSNorm for efficiency)
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
        # Attention with sliding window
        attn_config = AttentionConfig(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=attention_dropout,
            foundation=foundation,
            attention_sparsity=attention_sparsity,
            use_sliding_window=use_sliding_window,
            window_size=window_size,
            num_global_tokens=num_global_tokens
        )
        self.attention = AttentionFactory.create(attn_config)
        
        # Feedforward (SwiGLU with Ramanujan sparsity)
        ffn_config = FeedForwardConfig(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ffn_type='swiglu',  # Always use SwiGLU for enhanced block
            foundation=foundation,
            ffn_sparsity=ffn_sparsity
        )
        self.ffn = FeedForwardFactory.create(ffn_config)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            mask: Optional attention mask [batch, seq_len] or [batch, seq_len, seq_len]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Self-attention with residual
        attn_out = self.attention(self.attn_norm(x), mask=mask)
        x = x + self.dropout(attn_out)
        
        # Feedforward with residual
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)
        
        return x
    
    def get_info(self) -> dict:
        """Get information about this block."""
        from .attention import get_attention_info
        from .feedforward import get_ffn_info
        
        info = {
            'type': 'EnhancedPretrainingBlock',
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'hidden_dim': self.hidden_dim,
            'use_sliding_window': self.use_sliding_window,
            'attention': get_attention_info(self.attention),
            'ffn': get_ffn_info(self.ffn)
        }
        
        return info


# ============================================================================
# POST-NORM TRANSFORMER BLOCK (for ablation)
# ============================================================================

class PostNormTransformerBlock(nn.Module):
    """
    Post-normalization transformer block.
    
    Architecture:
        x = Norm(x + Attention(x))
        x = Norm(x + FFN(x))
    
    This is the original transformer normalization pattern.
    Generally harder to train than pre-norm but useful for ablation studies.
    
    Args:
        dim: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        hidden_dim: FFN hidden dimension
        dropout: Dropout probability (default: 0.0)
        attention_dropout: Attention-specific dropout (default: 0.0)
        ffn_type: Type of FFN ('swiglu' or 'standard', default: 'swiglu')
        norm_type: Type of normalization ('rms' or 'layer', default: 'layer')
        foundation: Optional RamanujanFoundation for sparse layers
        attention_sparsity: Target sparsity for attention projections
        ffn_sparsity: Target sparsity for FFN layers
    
    Example:
        >>> # For ablation study comparing pre-norm vs post-norm
        >>> block_post = PostNormTransformerBlock(
        ...     dim=512,
        ...     num_heads=8,
        ...     num_kv_heads=4,
        ...     hidden_dim=2048
        ... )
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        ffn_type: str = 'swiglu',
        norm_type: str = 'layer',  # Original transformer uses LayerNorm
        foundation: Optional['RamanujanFoundation'] = None,
        attention_sparsity: float = 0.0,
        ffn_sparsity: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_dim = hidden_dim
        
        # Post-normalization layers
        self.attn_norm = NormalizationFactory.create(norm_type, dim)
        self.ffn_norm = NormalizationFactory.create(norm_type, dim)
        
        # Attention
        attn_config = AttentionConfig(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=attention_dropout,
            foundation=foundation,
            attention_sparsity=attention_sparsity,
            use_sliding_window=False
        )
        self.attention = AttentionFactory.create(attn_config)
        
        # Feedforward
        ffn_config = FeedForwardConfig(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ffn_type=ffn_type,
            foundation=foundation,
            ffn_sparsity=ffn_sparsity
        )
        self.ffn = FeedForwardFactory.create(ffn_config)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with post-normalization.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            mask: Optional attention mask [batch, seq_len] or [batch, seq_len, seq_len]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Self-attention with post-norm
        attn_out = self.attention(x, mask=mask)
        x = self.attn_norm(x + self.dropout(attn_out))
        
        # Feedforward with post-norm
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        
        return x
    
    def get_info(self) -> dict:
        """Get information about this block."""
        from .attention import get_attention_info
        from .feedforward import get_ffn_info
        
        info = {
            'type': 'PostNormTransformerBlock',
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'hidden_dim': self.hidden_dim,
            'attention': get_attention_info(self.attention),
            'ffn': get_ffn_info(self.ffn)
        }
        
        return info


# ============================================================================
# BLOCK FACTORY
# ============================================================================

@dataclass
class BlockConfig:
    """Configuration for transformer block."""
    dim: int
    num_heads: int
    num_kv_heads: int
    hidden_dim: int
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Block type
    block_type: str = 'standard'  # 'standard', 'enhanced', 'postnorm'
    
    # FFN config
    ffn_type: str = 'swiglu'  # 'swiglu', 'standard'
    norm_type: str = 'rms'  # 'rms', 'layer'
    
    # Ramanujan sparsity
    foundation: Optional['RamanujanFoundation'] = None
    attention_sparsity: float = 0.0
    ffn_sparsity: float = 0.0
    
    # Sliding window (for enhanced block)
    use_sliding_window: bool = False
    window_size: int = 512
    num_global_tokens: int = 64


class BlockFactory:
    """
    Factory for creating transformer blocks.
    
    Example:
        >>> from ramanujan.architecture import BlockFactory, BlockConfig
        >>> 
        >>> # Standard block
        >>> config = BlockConfig(
        ...     dim=512,
        ...     num_heads=8,
        ...     num_kv_heads=4,
        ...     hidden_dim=2048,
        ...     block_type='standard'
        ... )
        >>> block = BlockFactory.create(config)
        >>> 
        >>> # Enhanced block
        >>> config_enhanced = BlockConfig(
        ...     dim=890,
        ...     num_heads=10,
        ...     num_kv_heads=5,
        ...     hidden_dim=2370,
        ...     block_type='enhanced',
        ...     use_sliding_window=True
        ... )
        >>> block_enhanced = BlockFactory.create(config_enhanced)
    """
    
    @staticmethod
    def create(config: BlockConfig) -> nn.Module:
        """
        Create transformer block based on config.
        
        Args:
            config: BlockConfig instance
        
        Returns:
            Appropriate block module
        """
        block_type = config.block_type.lower()
        
        if block_type in ['standard', 'default']:
            return TransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                ffn_type=config.ffn_type,
                norm_type=config.norm_type,
                foundation=config.foundation,
                attention_sparsity=config.attention_sparsity,
                ffn_sparsity=config.ffn_sparsity
            )
        elif block_type in ['enhanced', 'pretraining']:
            return EnhancedPretrainingBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                foundation=config.foundation,
                attention_sparsity=config.attention_sparsity,
                ffn_sparsity=config.ffn_sparsity,
                use_sliding_window=config.use_sliding_window,
                window_size=config.window_size,
                num_global_tokens=config.num_global_tokens
            )
        elif block_type in ['postnorm', 'post_norm']:
            return PostNormTransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                ffn_type=config.ffn_type,
                norm_type=config.norm_type,
                foundation=config.foundation,
                attention_sparsity=config.attention_sparsity,
                ffn_sparsity=config.ffn_sparsity
            )
        else:
            raise ValueError(
                f"Unknown block_type: {block_type}. "
                f"Choose from: 'standard', 'enhanced', 'postnorm'"
            )
    
    @staticmethod
    def create_from_dict(config_dict: dict) -> nn.Module:
        """
        Create block from dictionary config.
        
        Args:
            config_dict: Dictionary with block parameters
        
        Returns:
            Block module
        """
        config = BlockConfig(**config_dict)
        return BlockFactory.create(config)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_block_info(module: nn.Module) -> dict:
    """
    Get information about a block module.
    
    Args:
        module: Block module instance
    
    Returns:
        Dictionary with block info
    """
    if hasattr(module, 'get_info'):
        return module.get_info()
    
    # Fallback
    return {
        'type': module.__class__.__name__
    }


def estimate_block_parameters(
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    hidden_dim: int,
    attention_sparsity: float = 0.0,
    ffn_sparsity: float = 0.0,
    ffn_type: str = 'swiglu'
) -> dict:
    """
    Estimate parameter count for a transformer block.
    
    Args:
        dim: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        hidden_dim: FFN hidden dimension
        attention_sparsity: Sparsity for attention projections
        ffn_sparsity: Sparsity for FFN layers
        ffn_type: Type of FFN ('swiglu' or 'standard')
    
    Returns:
        Dictionary with parameter counts
    """
    head_dim = dim // num_heads
    
    # Attention parameters
    # Q: dim -> num_heads * head_dim
    # K, V: dim -> num_kv_heads * head_dim
    # O: num_heads * head_dim -> dim
    attn_params = (
        dim * (num_heads * head_dim) +  # Q
        dim * (num_kv_heads * head_dim) +  # K
        dim * (num_kv_heads * head_dim) +  # V
        (num_heads * head_dim) * dim  # O
    )
    attn_params_sparse = int(attn_params * (1 - attention_sparsity))
    
    # FFN parameters
    if ffn_type == 'swiglu':
        # 3 projections: gate, value, output
        ffn_params = 3 * dim * hidden_dim
    else:
        # 2 projections: fc1, fc2
        ffn_params = 2 * dim * hidden_dim
    ffn_params_sparse = int(ffn_params * (1 - ffn_sparsity))
    
    # Normalization parameters (2 RMSNorm layers)
    norm_params = 2 * dim
    
    # Total
    total_dense = attn_params + ffn_params + norm_params
    total_sparse = attn_params_sparse + ffn_params_sparse + norm_params
    
    return {
        'attention_params': attn_params,
        'attention_params_sparse': attn_params_sparse,
        'ffn_params': ffn_params,
        'ffn_params_sparse': ffn_params_sparse,
        'norm_params': norm_params,
        'total_dense': total_dense,
        'total_sparse': total_sparse,
        'total_savings': total_dense - total_sparse,
        'savings_percentage': ((total_dense - total_sparse) / total_dense) * 100
    }


def compare_block_types(
    dim: int = 512,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    hidden_dim: int = 2048,
    batch_size: int = 2,
    seq_len: int = 128
) -> dict:
    """
    Compare different block types.
    
    Args:
        dim: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        hidden_dim: FFN hidden dimension
        batch_size: Batch size for test
        seq_len: Sequence length for test
    
    Returns:
        Dictionary with comparison results
    """
    x = torch.randn(batch_size, seq_len, dim)
    
    results = {}
    
    # Standard block
    standard = TransformerBlock(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_dim=hidden_dim
    )
    standard_params = sum(p.numel() for p in standard.parameters())
    with torch.no_grad():
        out_standard = standard(x)
    
    results['standard'] = {
        'params': standard_params,
        'output_shape': out_standard.shape,
        'output_mean': out_standard.mean().item(),
        'output_std': out_standard.std().item()
    }
    
    # Post-norm block
    postnorm = PostNormTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_dim=hidden_dim
    )
    postnorm_params = sum(p.numel() for p in postnorm.parameters())
    with torch.no_grad():
        out_postnorm = postnorm(x)
    
    results['postnorm'] = {
        'params': postnorm_params,
        'output_shape': out_postnorm.shape,
        'output_mean': out_postnorm.mean().item(),
        'output_std': out_postnorm.std().item()
    }
    
    return results


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing blocks.py module")
    print("="*70)
    
    # Test TransformerBlock
    print("\n1. Testing TransformerBlock...")
    block = TransformerBlock(
        dim=512,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=2048,
        dropout=0.1
    )
    x = torch.randn(2, 128, 512)
    out = block(x)
    
    assert out.shape == x.shape, "Shape mismatch!"
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in block.parameters()):,}")
    
    info = block.get_info()
    print(f"   Type: {info['type']}")
    print(f"   ✅ TransformerBlock working!")
    
    # Test EnhancedPretrainingBlock
    print("\n2. Testing EnhancedPretrainingBlock...")
    enhanced = EnhancedPretrainingBlock(
        dim=512,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=2048,
        use_sliding_window=True,
        window_size=512
    )
    out_enhanced = enhanced(x)
    
    assert out_enhanced.shape == x.shape, "Shape mismatch!"
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {out_enhanced.shape}")
    print(f"   Parameters: {sum(p.numel() for p in enhanced.parameters()):,}")
    print(f"   ✅ EnhancedPretrainingBlock working!")
    
    # Test with longer sequence
    x_long = torch.randn(1, 1024, 512)
    out_long = enhanced(x_long)
    print(f"   Long sequence: {x_long.shape} -> {out_long.shape}")
    print(f"   ✅ Long sequence working!")
    
    # Test PostNormTransformerBlock
    print("\n3. Testing PostNormTransformerBlock...")
    postnorm = PostNormTransformerBlock(
        dim=512,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=2048
    )
    out_postnorm = postnorm(x)
    
    assert out_postnorm.shape == x.shape, "Shape mismatch!"
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {out_postnorm.shape}")
    print(f"   Parameters: {sum(p.numel() for p in postnorm.parameters()):,}")
    print(f"   ✅ PostNormTransformerBlock working!")
    
    # Test BlockFactory
    print("\n4. Testing BlockFactory...")
    config = BlockConfig(
        dim=512,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=2048,
        block_type='standard',
        dropout=0.1
    )
    block_factory = BlockFactory.create(config)
    out_factory = block_factory(x)
    
    print(f"   Factory created: {type(block_factory).__name__}")
    print(f"   Output: {out_factory.shape}")
    print(f"   ✅ BlockFactory working!")
    
    # Test with enhanced config
    config_enhanced = BlockConfig(
        dim=512,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=2048,
        block_type='enhanced',
        use_sliding_window=True,
        window_size=512
    )
    block_enhanced_factory = BlockFactory.create(config_enhanced)
    out_enhanced_factory = block_enhanced_factory(x)
    
    print(f"   Enhanced factory created: {type(block_enhanced_factory).__name__}")
    print(f"   Output: {out_enhanced_factory.shape}")
    print(f"   ✅ Enhanced factory working!")
    
    # Test parameter estimation
    print("\n5. Testing parameter estimation...")
    params = estimate_block_parameters(
        dim=512,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=2048,
        attention_sparsity=0.82,
        ffn_sparsity=0.88,
        ffn_type='swiglu'
    )
    
    print(f"   Attention params (dense): {params['attention_params']:,}")
    print(f"   Attention params (sparse): {params['attention_params_sparse']:,}")
    print(f"   FFN params (dense): {params['ffn_params']:,}")
    print(f"   FFN params (sparse): {params['ffn_params_sparse']:,}")
    print(f"   Total savings: {params['total_savings']:,} ({params['savings_percentage']:.1f}%)")
    print(f"   ✅ Parameter estimation working!")
    
    # Test comparison
    print("\n6. Testing block comparison...")
    comparison = compare_block_types(dim=512, num_heads=8, num_kv_heads=4, hidden_dim=2048)
    
    print("\n   Block Type Comparison:")
    for block_type, info in comparison.items():
        print(f"   {block_type:10s}: {info['params']:8,} params, "
              f"mean={info['output_mean']:7.4f}, std={info['output_std']:7.4f}")
    print(f"   ✅ Comparison working!")
    
    # Test gradient flow
    print("\n7. Testing gradient flow...")
    block_grad = TransformerBlock(
        dim=512,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=2048
    )
    x_grad = torch.randn(2, 128, 512, requires_grad=True)
    out_grad = block_grad(x_grad)
    loss = out_grad.sum()
    loss.backward()
    
    assert x_grad.grad is not None, "No gradient computed!"
    grad_norm = x_grad.grad.norm().item()
    print(f"   Input gradient norm: {grad_norm:.4f}")
    print(f"   ✅ Gradient flow working!")
    
    # Test with attention mask
    print("\n8. Testing with attention mask...")
    mask = torch.ones(2, 128, dtype=torch.bool)
    mask[:, 64:] = False  # Mask out second half
    out_masked = block(x, mask=mask)
    
    assert out_masked.shape == x.shape, "Shape mismatch with mask!"
    print(f"   Output with mask: {out_masked.shape}")
    print(f"   ✅ Attention masking working!")
    
    # Test get_info for all block types
    print("\n9. Testing get_info for all blocks...")
    info_standard = block.get_info()
    info_enhanced = enhanced.get_info()
    info_postnorm = postnorm.get_info()
    
    print(f"   Standard block info: {info_standard['type']}")
    print(f"   Enhanced block info: {info_enhanced['type']}")
    print(f"   PostNorm block info: {info_postnorm['type']}")
    print(f"   ✅ get_info working for all blocks!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.architecture.blocks import TransformerBlock")
    print("  from ramanujan.architecture.blocks import EnhancedPretrainingBlock")
    print("  from ramanujan.architecture.blocks import BlockFactory")
    print("="*70)