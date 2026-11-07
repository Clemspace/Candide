"""Attention mechanisms for transformer models."""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .base import BaseAttention
from .normalization import RMSNorm


def _sdpa_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
    """
    Convert a Candide attention mask (nonzero / True = attend) to the boolean
    mask expected by F.scaled_dot_product_attention (True = attend),
    broadcastable to (batch, heads, q_len, kv_len).
    """
    if mask is None:
        return None
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dtype != torch.bool:
        mask = mask != 0
    return mask


def _attend(module: nn.Module, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
    """
    softmax(q k^T * scale + mask) v via PyTorch's fused SDPA kernel.

    Numerically equivalent to the explicit path it replaces (scores,
    masked_fill(-inf), softmax, dropout, matmul); same kernel as ANGPTAttention,
    so the two arms of the normalization pilot compare like for like.
    """
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=_sdpa_mask(mask),
        dropout_p=module.dropout if module.training else 0.0,
        scale=module.scale,
    )


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
        qk_norm: bool = False,
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
            qk_norm: RMSNorm q and k over head_dim before RoPE (QK-norm)
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

        # QK-norm (Henry et al., 2020): RMSNorm with learnable scale over head_dim.
        # With the 1/sqrt(head_dim) softmax scale this gives logits of
        # sqrt(head_dim) * cos(q, k) at init.
        self.qk_norm = qk_norm
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        
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
        
        # QK-norm (before RoPE; RoPE is a rotation and preserves the norm)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

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
        
        # Attention (fused SDPA kernel; dropout on attention weights inside)
        output = _attend(self, q, k, v, mask)
        
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
        qk_norm: bool = False,
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
            qk_norm: RMSNorm q and k over head_dim before RoPE (QK-norm)
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

        # QK-norm (Henry et al., 2020): RMSNorm with learnable scale over head_dim.
        # With the 1/sqrt(head_dim) softmax scale this gives logits of
        # sqrt(head_dim) * cos(q, k) at init.
        self.qk_norm = qk_norm
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        
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
        
        # QK-norm (before RoPE; RoPE is a rotation and preserves the norm)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

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
        
        # Attention (fused SDPA kernel; dropout on attention weights inside)
        output = _attend(self, q, k, v, mask)
        
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