"""Attention mechanisms for transformer models."""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .base import BaseAttention


class MultiHeadAttention(BaseAttention):
    """
    Multi-Head Attention (MHA).
    
    Standard attention mechanism from "Attention is All You Need".
    Each head attends to all positions independently.
    
    Used in: BERT, GPT-2, original Transformer.
    
    Example:
        >>> attn = MultiHeadAttention(d_model=768, n_heads=12)
        >>> x = torch.randn(2, 10, 768)
        >>> out, cache = attn(x, use_cache=False)
        >>> out.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        pruning_mask: Optional[Tensor] = None,
        rope: Optional[nn.Module] = None,
        **kwargs
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
            pruning_mask: Optional mask for pruning
            rope: Optional RoPE module for position encoding
            **kwargs: Additional arguments
        """
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            pruning_mask=pruning_mask,
            **kwargs
        )
        
        self.rope = rope
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.q_proj.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.k_proj.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.v_proj.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.out_proj.weight, mean=0, std=self.d_model ** -0.5)
        
        if self.use_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
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
        Forward pass through multi-head attention.
        
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
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if self.rope is not None:
            q, k = self.rope(q, k, position_ids=position_ids)
        
        # Handle past key-value cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # Concat on seq_len dim
            v = torch.cat([past_v, v], dim=2)
        
        # Cache for next iteration
        past_key_value = (k, v) if use_cache else None
        
        # Compute attention scores
        # (batch, n_heads, seq_len, head_dim) @ (batch, n_heads, head_dim, kv_seq_len)
        # = (batch, n_heads, seq_len, kv_seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if mask is not None:
            # Expand mask if needed: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Dropout on attention weights
        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # (batch, n_heads, seq_len, kv_seq_len) @ (batch, n_heads, kv_seq_len, head_dim)
        # = (batch, n_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Apply pruning mask if set
        if self.pruning_mask.numel() > 1:
            output = output * self.pruning_mask
        
        # Output projection
        output = self.out_proj(output)
        
        # Residual dropout
        if self.resid_dropout is not None:
            output = self.resid_dropout(output)
        
        return output, past_key_value


class GroupedQueryAttention(BaseAttention):
    """
    Grouped Query Attention (GQA).
    
    Memory-efficient attention variant where keys and values are shared
    across multiple query heads. Reduces KV cache size significantly.
    
    GQA with n_kv_heads=1 is equivalent to Multi-Query Attention (MQA).
    GQA with n_kv_heads=n_heads is equivalent to standard MHA.
    
    Used in: LLaMA 2, Mistral, and other modern efficient LLMs.
    
    Reference: https://arxiv.org/abs/2305.13245
    
    Example:
        >>> # 32 query heads, 8 KV heads (4x reduction)
        >>> attn = GroupedQueryAttention(d_model=2048, n_heads=32, n_kv_heads=8)
        >>> x = torch.randn(2, 10, 2048)
        >>> out, cache = attn(x, use_cache=False)
        >>> out.shape
        torch.Size([2, 10, 2048])
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        pruning_mask: Optional[Tensor] = None,
        rope: Optional[nn.Module] = None,
        **kwargs
    ):
        """
        Initialize grouped query attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            n_kv_heads: Number of key-value heads (default: n_heads // 4 or 1)
            dropout: Dropout probability
            bias: Whether to use bias (typically False for LLaMA-style)
            pruning_mask: Optional mask for pruning
            rope: Optional RoPE module
            **kwargs: Additional arguments
        """
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            pruning_mask=pruning_mask,
            **kwargs
        )
        
        # Default: use fewer KV heads for efficiency
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 4)
        
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # How many Q heads per KV head
        self.kv_head_dim = self.head_dim
        
        self.rope = rope
        
        # Q projection (all heads)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # K, V projections (fewer heads)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.kv_head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.kv_head_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.q_proj.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.k_proj.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.v_proj.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.out_proj.weight, mean=0, std=self.d_model ** -0.5)
        
        if self.use_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def _repeat_kv(self, x: Tensor) -> Tensor:
        """
        Repeat K/V heads to match number of Q heads.
        
        Args:
            x: (batch, n_kv_heads, seq_len, head_dim)
        
        Returns:
            (batch, n_heads, seq_len, head_dim)
        """
        batch_size, n_kv_heads, seq_len, head_dim = x.shape
        
        if self.n_groups == 1:
            return x
        
        # Repeat each KV head n_groups times
        x = x[:, :, None, :, :].expand(
            batch_size, n_kv_heads, self.n_groups, seq_len, head_dim
        )
        return x.reshape(batch_size, n_kv_heads * self.n_groups, seq_len, head_dim)
    
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
        Forward pass through grouped query attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask
            position_ids: Position indices (batch, seq_len)
            past_key_value: Cached (key, value)
            use_cache: Whether to return cache
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (output, past_key_value)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape queries (all heads)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Reshape keys and values (fewer heads)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.kv_head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if self.rope is not None:
            q, k = self.rope(q, k, position_ids=position_ids)
        
        # Handle past key-value cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Cache for next iteration
        past_key_value = (k, v) if use_cache else None
        
        # Repeat K/V heads to match Q heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Dropout on attention weights
        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Apply pruning mask if set
        if self.pruning_mask.numel() > 1:
            output = output * self.pruning_mask
        
        # Output projection
        output = self.out_proj(output)
        
        # Residual dropout
        if self.resid_dropout is not None:
            output = self.resid_dropout(output)
        
        return output, past_key_value


def get_attention(
    attention_type: str,
    d_model: int,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
    dropout: float = 0.0,
    bias: bool = True,
    rope: Optional[nn.Module] = None,
    **kwargs
) -> BaseAttention:
    """
    Factory function for attention mechanisms.
    
    Args:
        attention_type: Type of attention ('mha', 'gqa', 'mqa')
        d_model: Model dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of KV heads (for GQA/MQA)
        dropout: Dropout probability
        bias: Whether to use bias
        rope: Optional RoPE module
        **kwargs: Additional arguments
    
    Returns:
        Attention module
    
    Example:
        >>> attn = get_attention('gqa', d_model=2048, n_heads=32, n_kv_heads=8)
        >>> isinstance(attn, GroupedQueryAttention)
        True
    """
    attention_type = attention_type.lower()
    
    if attention_type == 'mha' or attention_type == 'multi_head':
        return MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            rope=rope,
            **kwargs
        )
    
    elif attention_type == 'gqa' or attention_type == 'grouped_query':
        return GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            bias=bias,
            rope=rope,
            **kwargs
        )
    
    elif attention_type == 'mqa' or attention_type == 'multi_query':
        # MQA is GQA with n_kv_heads=1
        return GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=1,
            dropout=dropout,
            bias=bias,
            rope=rope,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown attention_type: {attention_type}. "
            f"Available: 'mha', 'gqa', 'mqa'"
        )