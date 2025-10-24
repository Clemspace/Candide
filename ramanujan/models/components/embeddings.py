"""Embedding layers for transformer models."""

import torch
from torch import nn, Tensor
import math
from typing import Optional, Tuple

from .base import BaseEmbedding


class TokenEmbedding(BaseEmbedding):
    """
    Token embedding layer.
    
    Maps token IDs to dense vectors.
    
    Example:
        >>> embed = TokenEmbedding(vocab_size=50000, d_model=768)
        >>> tokens = torch.randint(0, 50000, (2, 10))
        >>> out = embed(tokens)
        >>> out.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        scale_grad_by_freq: bool = False
    ):
        """
        Initialize token embedding.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            padding_idx: Padding token ID (embeddings zeroed)
            max_norm: If given, renormalize embeddings to this value
            scale_grad_by_freq: Scale gradients by token frequency
        """
        super().__init__(vocab_size=vocab_size, d_model=d_model)
        
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.scale_grad_by_freq = scale_grad_by_freq
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
            max_norm=max_norm,
            scale_grad_by_freq=scale_grad_by_freq
        )
        
        self.reset_parameters()
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Embed token IDs.
        
        Args:
            input_ids: Token IDs (..., seq_len)
        
        Returns:
            Embeddings (..., seq_len, d_model)
        """
        return self.embedding(input_ids)
    
    def reset_parameters(self) -> None:
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)


def _precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Precompute complex frequencies for RoPE.
    
    Uses complex number formulation for efficiency:
    e^(i*theta) = cos(theta) + i*sin(theta)
    
    This avoids separate sin/cos computations and uses
    a single complex multiplication for rotation.
    
    Args:
        dim: Embedding dimension (must be even)
        end: Maximum sequence length
        theta: Base for frequency computation (default: 10000.0)
        device: Device for tensor
    
    Returns:
        Complex tensor [end, dim//2] for rotations
    """
    # Compute inverse frequencies: 1 / (theta^(2i/dim))
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    
    # Position indices
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # Outer product: [end, dim//2]
    freqs = torch.outer(t, freqs).float()
    
    # Convert to complex: e^(i*theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def _reshape_for_broadcast(
    freqs_cis: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Reshape frequency tensor to broadcast correctly.
    
    Handles both:
    - freqs_cis: [seq_len, dim//2] for sequential positions
    - freqs_cis: [batch, seq_len, dim//2] for custom position_ids
    
    Args:
        freqs_cis: [seq_len, dim//2] or [batch, seq_len, dim//2]
        x: [..., seq_len, dim//2]
    
    Returns:
        Reshaped freqs_cis that broadcasts with x
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    
    # Check if freqs_cis has batch dimension
    if freqs_cis.ndim == 3:
        # freqs_cis: [batch, seq_len, dim//2]
        # x: [batch, n_heads, seq_len, dim//2]
        # Need to add head dimension
        assert freqs_cis.shape[0] == x.shape[0], f"Batch size mismatch: freqs {freqs_cis.shape[0]} vs x {x.shape[0]}"
        assert freqs_cis.shape[1] == x.shape[-2], f"Seq len mismatch: freqs {freqs_cis.shape[1]} vs x {x.shape[-2]}"
        assert freqs_cis.shape[2] == x.shape[-1], f"Dim mismatch: freqs {freqs_cis.shape[2]} vs x {x.shape[-1]}"
        # Add dimension for heads: [batch, seq_len, dim] -> [batch, 1, seq_len, dim]
        return freqs_cis.unsqueeze(1)
    else:
        # freqs_cis: [seq_len, dim//2]
        # x: [batch, n_heads, seq_len, dim//2]
        assert freqs_cis.shape == (x.shape[-2], x.shape[-1]), \
            f"Shape mismatch: freqs_cis {freqs_cis.shape} vs x {x.shape}"
        
        # Create shape: [1, 1, seq_len, dim//2]
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


class RotaryEmbedding(BaseEmbedding):
    """
    Rotary Position Embedding (RoPE) using complex multiplication.
    
    More efficient than separate sin/cos formulation.
    Uses complex number representation: e^(i*theta) = cos(theta) + i*sin(theta)
    
    Used in: LLaMA, Mistral, and other modern LLMs.
    
    Reference: https://arxiv.org/abs/2104.09864
    
    Example:
        >>> rope = RotaryEmbedding(d_model=64, max_seq_len=2048)
        >>> q = torch.randn(2, 8, 10, 64)  # (batch, heads, seq, head_dim)
        >>> k = torch.randn(2, 8, 10, 64)
        >>> q_rot, k_rot = rope(q, k, start_pos=0)
        >>> q_rot.shape
        torch.Size([2, 8, 10, 64])
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize RoPE.
        
        Args:
            d_model: Dimension per head (must be even!)
            max_seq_len: Maximum sequence length
            theta: Base for frequency computation (default: 10000.0)
            device: Device for computation
        """
        super().__init__(d_model=d_model, max_seq_len=max_seq_len)
        
        assert d_model % 2 == 0, "Dimension must be even for RoPE"
        
        self.theta = theta
        self.device = device
        
        # Precompute and register as buffer (not a parameter)
        freqs_cis = _precompute_freqs_cis(d_model, max_seq_len, theta, device)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    
    def forward(
        self,
        xq: Tensor,
        xk: Tensor,
        position_ids: Optional[Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            xq: Query tensor [..., seq_len, dim]
            xk: Key tensor [..., seq_len, dim]
            position_ids: Position indices (batch, seq_len) - if provided, overrides start_pos
            start_pos: Starting position (for KV caching)
        
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = xq.shape[-2]
        
        # Get frequencies for current sequence
        if position_ids is not None:
            # Use specific positions
            freqs_cis = self.freqs_cis[position_ids]  # (batch, seq_len, dim//2)
        else:
            # Use sequential positions
            freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        
        # Reshape to pairs: [..., seq_len, dim] -> [..., seq_len, dim//2, 2]
        # Then view as complex: [..., seq_len, dim//2]
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], -1, 2)
        )
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], -1, 2)
        )
        
        # Reshape freqs for broadcasting
        freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
        
        # Complex multiplication = rotation
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
        
        # Return with original dtype
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    def extend_seq_len(self, new_max_len: int):
        """
        Extend cached frequencies for longer sequences.
        
        Useful for handling sequences longer than max_seq_len.
        """
        if new_max_len > self.max_seq_len:
            freqs_cis = _precompute_freqs_cis(
                self.d_model,
                new_max_len,
                self.theta,
                device=self.freqs_cis.device
            )
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
            self.max_seq_len = new_max_len
    
    def extra_repr(self) -> str:
        """String representation."""
        return f'd_model={self.d_model}, max_seq_len={self.max_seq_len}, theta={self.theta}'


class LearnedPositionalEmbedding(BaseEmbedding):
    """
    Rotary Position Embedding (RoPE).
    
    Applies rotation to query and key based on position.
    More efficient and effective than learned positional embeddings.
    
    Used in: LLaMA, GPT-NeoX, PaLM, and other modern LLMs.
    
    Reference: https://arxiv.org/abs/2104.09864
    
    Example:
        >>> rope = RotaryEmbedding(d_model=128, max_seq_len=2048)
        >>> q = torch.randn(2, 8, 10, 16)  # (batch, heads, seq, head_dim)
        >>> k = torch.randn(2, 8, 10, 16)
        >>> position_ids = torch.arange(10).unsqueeze(0)
        >>> q_rot, k_rot = rope(q, k, position_ids)
        >>> q_rot.shape
        torch.Size([2, 8, 10, 16])
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize RoPE.
        
        Args:
            d_model: Dimension per head (not total model dimension!)
            max_seq_len: Maximum sequence length
            base: Base for exponential (10000 in original paper)
            device: Device for computation
        """
        super().__init__(d_model=d_model, max_seq_len=max_seq_len)
        
        self.base = base
        self.device = device
        
        # Precompute frequency tensor
        # inv_freq = 1 / (base ^ (2i / d_model)) for i in [0, d_model/2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Compute cos and sin values for sequence length.
        
        Args:
            seq_len: Sequence length
            device: Device for computation
        
        Returns:
            Tuple of (cos, sin) tensors
        """
        # Check cache
        if (
            self._cos_cached is not None
            and seq_len <= self._seq_len_cached
            and self._cos_cached.device == device
        ):
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]
        
        # Compute position indices
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies: outer product of positions and inv_freq
        # freqs shape: (seq_len, d_model//2)
        freqs = torch.outer(t, self.inv_freq.to(device))
        
        # Concatenate to get full dimension
        # emb shape: (seq_len, d_model)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Compute cos and sin
        cos = emb.cos()
        sin = emb.sin()
        
        # Cache
        self._cos_cached = cos
        self._sin_cached = sin
        self._seq_len_cached = seq_len
        
        return cos, sin
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        Rotate half the hidden dims.
        
        For RoPE, we split x into two halves and rotate:
        [x1, x2] -> [-x2, x1]
        
        Args:
            x: Input tensor (..., d_model)
        
        Returns:
            Rotated tensor (..., d_model)
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to query and key.
        
        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim)
            k: Key tensor (batch, n_heads, seq_len, head_dim)
            position_ids: Position indices (batch, seq_len)
                         If None, assumes positions are [0, 1, ..., seq_len-1]
        
        Returns:
            Tuple of (q_rotated, k_rotated)
        """
        batch_size, n_heads, seq_len, head_dim = q.shape
        
        # Get cos/sin values
        cos, sin = self._compute_cos_sin(seq_len, q.device)
        
        # Handle position_ids
        if position_ids is not None:
            # Gather cos/sin for specific positions
            # position_ids: (batch, seq_len)
            cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
            sin = sin[position_ids].unsqueeze(1)
        else:
            # Use sequential positions
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
            sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        # q_embed = q * cos + rotate_half(q) * sin
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def extra_repr(self) -> str:
        """String representation."""
        return f'd_model={self.d_model}, max_seq_len={self.max_seq_len}, base={self.base}'


class LearnedPositionalEmbedding(BaseEmbedding):
    """
    Learned positional embeddings.
    
    Standard learned position embeddings like in BERT/GPT-2.
    Less efficient than RoPE but sometimes useful.
    
    Example:
        >>> pos_embed = LearnedPositionalEmbedding(max_seq_len=512, d_model=768)
        >>> position_ids = torch.arange(10).unsqueeze(0)
        >>> out = pos_embed(position_ids)
        >>> out.shape
        torch.Size([1, 10, 768])
    """
    
    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        padding_idx: Optional[int] = None
    ):
        """
        Initialize learned positional embedding.
        
        Args:
            max_seq_len: Maximum sequence length
            d_model: Model dimension
            padding_idx: Padding position (if any)
        """
        super().__init__(max_seq_len=max_seq_len, d_model=d_model)
        
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        
        self.reset_parameters()
    
    def forward(self, position_ids: Tensor) -> Tensor:
        """
        Embed position IDs.
        
        Args:
            position_ids: Position indices (..., seq_len)
        
        Returns:
            Position embeddings (..., seq_len, d_model)
        """
        return self.embedding(position_ids)
    
    def reset_parameters(self) -> None:
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)