"""
Factory for creating attention mechanisms.
"""

from typing import Optional
from dataclasses import dataclass
import torch.nn as nn

from .attention import (
    ImprovedGQA,
    ImprovedSlidingWindowGQA,
)
from ..foundation import RamanujanFoundation


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    
    # Core parameters
    dim: int
    num_heads: int
    num_kv_heads: int
    max_seq_len: int = 2048
    
    # Dropout
    dropout: float = 0.0
    
    # Sparsity
    foundation: Optional[RamanujanFoundation] = None
    attention_sparsity: float = 0.0
    
    # Sliding window
    use_sliding_window: bool = False
    window_size: int = 256
    num_global_tokens: int = 0
    
    # Improvements
    use_qk_norm: bool = False
    use_improved: bool = False
    
    # Attention type (for manual override)
    attention_type: str = 'auto'  # 'auto', 'standard', 'improved', 'sliding', 'improved_sliding'


class AttentionFactory:
    """
    Factory for creating attention mechanisms.
    
    Automatically selects the best attention type based on config,
    or allows manual specification.
    
    Example:
        >>> config = AttentionConfig(
        ...     dim=512,
        ...     num_heads=8,
        ...     num_kv_heads=4,
        ...     use_sliding_window=True,
        ...     window_size=256
        ... )
        >>> attention = AttentionFactory.create(config)
    """
    
    @staticmethod
    def create(config: AttentionConfig) -> nn.Module:
        """
        Create attention mechanism based on config.
        
        Args:
            config: AttentionConfig instance
            
        Returns:
            Attention module
        """
        # Manual type specification
        if config.attention_type != 'auto':
            return AttentionFactory._create_by_type(config)
        
        # Auto-select based on config
        if config.use_sliding_window:
            if config.use_improved:
                return ImprovedSlidingWindowGQA(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    dropout=config.dropout
                )
            else:
                return ImprovedSlidingWindowGQA( #TODO: Change to StandardSlidingWindowGQA
                    dim=config.dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    dropout=config.dropout
                )
        else:
            if config.use_improved:
                return ImprovedGQA(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    max_seq_len=config.max_seq_len,
                    foundation=config.foundation,
                    attention_sparsity=config.attention_sparsity,
                    dropout=config.dropout
                )
            else:
                return StandardGQA(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    max_seq_len=config.max_seq_len,
                    foundation=config.foundation,
                    attention_sparsity=config.attention_sparsity,
                    dropout=config.dropout
                )
    
    @staticmethod
    def _create_by_type(config: AttentionConfig) -> nn.Module:
        """Create attention by explicit type."""
        
        type_map = {
            'standard': StandardGQA,
            'improved': ImprovedGQA,
            'sliding': SlidingWindowGQA,
            'improved_sliding': ImprovedSlidingWindowGQA,
        }
        
        attn_class = type_map.get(config.attention_type)
        if attn_class is None:
            raise ValueError(
                f"Unknown attention_type: {config.attention_type}. "
                f"Choose from: {list(type_map.keys())}"
            )
        
        # Build kwargs based on class
        kwargs = {
            'dim': config.dim,
            'num_heads': config.num_heads,
            'num_kv_heads': config.num_kv_heads,
            'max_seq_len': config.max_seq_len,
            'dropout': config.dropout,
        }
        
        # Add window_size for sliding window variants
        if 'sliding' in config.attention_type:
            kwargs['window_size'] = config.window_size
        
        # Add foundation and sparsity for non-sliding variants
        if 'sliding' not in config.attention_type:
            kwargs['foundation'] = config.foundation
            kwargs['attention_sparsity'] = config.attention_sparsity
        
        return attn_class(**kwargs)
    
    @staticmethod
    def create_from_dict(config_dict: dict) -> nn.Module:
        """Create attention from dictionary config."""
        config = AttentionConfig(**config_dict)
        return AttentionFactory.create(config)


def get_attention_info(module: nn.Module) -> dict:
    """Get information about an attention module."""
    return {
        'type': module.__class__.__name__,
        'dim': getattr(module, 'dim', None),
        'num_heads': getattr(module, 'num_heads', None),
        'num_kv_heads': getattr(module, 'num_kv_heads', None),
    }


__all__ = [
    'AttentionConfig',
    'AttentionFactory',
    'get_attention_info',
]
