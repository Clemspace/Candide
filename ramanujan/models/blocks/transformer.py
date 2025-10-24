"""Transformer blocks - composed components."""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from ..components.base import BaseTransformerBlock
from ..components import (
    BaseAttention,
    BaseFeedForward,
    BaseNormalization,
    get_attention,
    get_feedforward,
    get_normalization,
)


class TransformerBlock(BaseTransformerBlock):
    """
    Transformer block with self-attention and feedforward.
    
    Modern pre-norm architecture (LLaMA-style):
        x = x + attention(norm(x))
        x = x + ffn(norm(x))
    
    vs. Classic post-norm (original Transformer):
        x = norm(x + attention(x))
        x = norm(x + ffn(x))
    
    Pre-norm is more stable and commonly used in modern LLMs.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feedforward dimension (default: 4 * d_model)
        dropout: Dropout probability
        norm_first: Use pre-norm (True) or post-norm (False)
        norm_type: Type of normalization ('rms', 'layer')
        attention_type: Type of attention ('mha', 'gqa', 'mqa')
        ffn_type: Type of feedforward ('swiglu', 'gelu')
        n_kv_heads: Number of KV heads for GQA (optional)
        rope: Optional RoPE module
        bias: Whether to use bias in projections
        **kwargs: Additional arguments
    
    Example:
        >>> from ramanujan.models.components import RotaryEmbedding
        >>> rope = RotaryEmbedding(d_model=64, max_seq_len=2048)
        >>> block = TransformerBlock(
        ...     d_model=768,
        ...     n_heads=12,
        ...     norm_type='rms',
        ...     attention_type='gqa',
        ...     ffn_type='swiglu',
        ...     rope=rope
        ... )
        >>> x = torch.randn(2, 10, 768)
        >>> out, cache = block(x, use_cache=False)
        >>> out.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        norm_first: bool = True,
        norm_type: str = 'rms',
        attention_type: str = 'mha',
        ffn_type: str = 'swiglu',
        n_kv_heads: Optional[int] = None,
        rope: Optional[nn.Module] = None,
        bias: bool = False,
        **kwargs
    ):
        """Initialize transformer block."""
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            norm_first=norm_first,
            **kwargs
        )
        
        # Attention
        self.attention = get_attention(
            attention_type=attention_type,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            bias=bias,
            rope=rope,
        )
        
        # Feedforward
        self.ffn = get_feedforward(
            ffn_type=ffn_type,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
        )
        
        # Normalization layers
        self.norm1 = get_normalization(norm_type, d_model)
        self.norm2 = get_normalization(norm_type, d_model)
        
        # Dropout (optional)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len) or (batch, 1, seq_len, seq_len)
            position_ids: Position indices (batch, seq_len)
            past_key_value: Cached (key, value) from previous step
            use_cache: Whether to return key-value cache
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (output, past_key_value)
        """
        # Self-attention with residual
        if self.norm_first:
            # Pre-norm: x = x + attention(norm(x))
            attn_input = self.norm1(x)
            attn_output, past_key_value = self.attention(
                attn_input,
                mask=mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if self.dropout is not None:
                attn_output = self.dropout(attn_output)
            x = x + attn_output
            
            # Feedforward with residual: x = x + ffn(norm(x))
            ffn_input = self.norm2(x)
            ffn_output = self.ffn(ffn_input)
            if self.dropout is not None:
                ffn_output = self.dropout(ffn_output)
            x = x + ffn_output
        else:
            # Post-norm: x = norm(x + attention(x))
            attn_output, past_key_value = self.attention(
                x,
                mask=mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if self.dropout is not None:
                attn_output = self.dropout(attn_output)
            x = self.norm1(x + attn_output)
            
            # Feedforward: x = norm(x + ffn(x))
            ffn_output = self.ffn(x)
            if self.dropout is not None:
                ffn_output = self.dropout(ffn_output)
            x = self.norm2(x + ffn_output)
        
        return x, past_key_value
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f'd_model={self.d_model}, n_heads={self.n_heads}, '
            f'd_ff={self.d_ff}, dropout={self.dropout}, '
            f'norm_first={self.norm_first}'
        )


def create_transformer_block(
    d_model: int,
    n_heads: int,
    preset: Optional[str] = None,
    **kwargs
) -> TransformerBlock:
    """
    Factory function for creating transformer blocks with presets.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        preset: Preset configuration ('llama', 'gpt', 'bert')
        **kwargs: Override preset values
    
    Returns:
        TransformerBlock
    
    Presets:
        - 'llama': Pre-norm, RMSNorm, GQA, SwiGLU, no bias
        - 'gpt': Pre-norm, LayerNorm, MHA, GELU, bias
        - 'bert': Post-norm, LayerNorm, MHA, GELU, bias
    
    Example:
        >>> # LLaMA-style block
        >>> block = create_transformer_block(
        ...     d_model=4096,
        ...     n_heads=32,
        ...     preset='llama',
        ...     n_kv_heads=8  # Override for GQA
        ... )
    """
    presets = {
        'llama': {
            'norm_first': True,
            'norm_type': 'rms',
            'attention_type': 'gqa',
            'ffn_type': 'swiglu',
            'bias': False,
            'n_kv_heads': max(1, n_heads // 4),  # Default GQA
        },
        'gpt': {
            'norm_first': True,
            'norm_type': 'layer',
            'attention_type': 'mha',
            'ffn_type': 'gelu',
            'bias': True,
        },
        'bert': {
            'norm_first': False,
            'norm_type': 'layer',
            'attention_type': 'mha',
            'ffn_type': 'gelu',
            'bias': True,
        },
    }
    
    # Get preset config
    if preset is not None:
        if preset.lower() not in presets:
            raise ValueError(
                f"Unknown preset: {preset}. "
                f"Available: {list(presets.keys())}"
            )
        config = presets[preset.lower()].copy()
        config.update(kwargs)  # Override with kwargs
    else:
        config = kwargs
    
    return TransformerBlock(d_model=d_model, n_heads=n_heads, **config)