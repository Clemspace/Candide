"""
Position Embeddings for Transformers.

Provides various positional encoding schemes:
- RoPE (Rotary Position Embedding) - Complex formulation
- Learned positional embeddings
- Sinusoidal embeddings (classic Transformer)

The complex RoPE formulation is more efficient than sin/cos
and is used in modern LLMs (LLaMA, Mistral, etc.).
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


# ============================================================================
# ROTARY POSITION EMBEDDINGS (RoPE)
# ============================================================================

def precompute_freqs_cis(
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
        
    Example:
        >>> freqs = precompute_freqs_cis(dim=64, end=2048)
        >>> print(freqs.shape)  # [2048, 32]
        >>> print(freqs.dtype)  # torch.complex64
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


def reshape_for_broadcast(
    freqs_cis: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Reshape frequency tensor to broadcast correctly.
    
    Handles arbitrary tensor shapes by adding dimensions
    as needed for broadcasting.
    
    Args:
        freqs_cis: [seq_len, dim//2]
        x: [..., seq_len, dim//2]
    
    Returns:
        Reshaped freqs_cis that broadcasts with x
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1]), \
        f"Shape mismatch: freqs_cis {freqs_cis.shape} vs x {x.shape}"
    
    # Create shape: [1, ..., 1, seq_len, dim//2]
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings using complex multiplication.
    
    This is the core RoPE operation. Works by:
    1. Reshape input to pairs of values (real, imag)
    2. Convert to complex numbers
    3. Multiply by rotation (complex multiplication = rotation)
    4. Convert back to real representation
    
    More efficient than separate sin/cos rotations.
    
    Args:
        xq: Query tensor [..., seq_len, dim]
        xk: Key tensor [..., seq_len, dim]
        freqs_cis: Precomputed frequencies [seq_len, dim//2]
    
    Returns:
        Tuple of (rotated_queries, rotated_keys)
        
    Example:
        >>> freqs = precompute_freqs_cis(64, 128)
        >>> q = torch.randn(2, 8, 128, 64)  # [B, H, L, D]
        >>> k = torch.randn(2, 8, 128, 64)
        >>> q_rot, k_rot = apply_rotary_emb(q, k, freqs)
        >>> print(q_rot.shape)  # [2, 8, 128, 64]
    """
    # Reshape to pairs: [..., seq_len, dim] -> [..., seq_len, dim//2, 2]
    # Then view as complex: [..., seq_len, dim//2]
    xq_ = torch.view_as_complex(
        xq.float().reshape(*xq.shape[:-1], -1, 2)
    )
    xk_ = torch.view_as_complex(
        xk.float().reshape(*xk.shape[:-1], -1, 2)
    )
    
    # Reshape freqs for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Complex multiplication = rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    
    # Return with original dtype
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.
    
    Precomputes and caches rotation frequencies for efficiency.
    Can be shared across multiple attention layers.
    
    Args:
        dim: Embedding dimension (head_dim, not model dim)
        max_seq_len: Maximum sequence length to cache
        theta: Base for frequency computation (default: 10000.0)
        
    Example:
        >>> rope = RotaryEmbedding(dim=64, max_seq_len=2048)
        >>> q = torch.randn(2, 8, 128, 64)
        >>> k = torch.randn(2, 8, 128, 64)
        >>> q_rot, k_rot = rope(q, k, start_pos=0)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0
    ):
        super().__init__()
        
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute and register as buffer (not a parameter)
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    
    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            xq: Query tensor [..., seq_len, dim]
            xk: Key tensor [..., seq_len, dim]
            start_pos: Starting position (for KV caching)
            
        Returns:
            (rotated_q, rotated_k)
        """
        seq_len = xq.shape[-2]
        
        # Get frequencies for current sequence
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        
        return apply_rotary_emb(xq, xk, freqs_cis)
    
    def extend_seq_len(self, new_max_len: int):
        """
        Extend cached frequencies for longer sequences.
        
        Useful for handling sequences longer than max_seq_len.
        """
        if new_max_len > self.max_seq_len:
            freqs_cis = precompute_freqs_cis(
                self.dim,
                new_max_len,
                self.theta,
                device=self.freqs_cis.device
            )
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
            self.max_seq_len = new_max_len


# ============================================================================
# OTHER POSITION EMBEDDINGS (for comparison)
# ============================================================================

class SinusoidalEmbedding(nn.Module):
    """
    Classic sinusoidal position embedding from "Attention is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    
    Args:
        dim: Embedding dimension
        max_seq_len: Maximum sequence length
        
    Example:
        >>> pe = SinusoidalEmbedding(dim=512, max_seq_len=2048)
        >>> positions = torch.arange(128)
        >>> emb = pe(positions)
    """
    
    def __init__(self, dim: int, max_seq_len: int = 5000):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute embeddings
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get position embeddings.
        
        Args:
            positions: Position indices [seq_len] or [batch, seq_len]
            
        Returns:
            Position embeddings [..., seq_len, dim]
        """
        return self.pe[positions]


class LearnedPositionEmbedding(nn.Module):
    """
    Learned position embeddings.
    
    Simple embedding table for positions.
    Used in BERT and other models.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Embedding dimension
        
    Example:
        >>> pe = LearnedPositionEmbedding(max_seq_len=512, dim=768)
        >>> positions = torch.arange(128)
        >>> emb = pe(positions)
    """
    
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Get position embeddings."""
        return self.embedding(positions)


# ============================================================================
# FACTORY
# ============================================================================


@dataclass
class EmbeddingConfig:
    """Configuration for position embeddings."""
    embedding_type: str  # 'rope', 'sinusoidal', 'learned'
    dim: int
    max_seq_len: int = 2048
    theta: float = 10000.0  # For RoPE only

class EmbeddingFactory:
    """Factory for creating position embeddings."""
    
    @staticmethod
    def create(config: EmbeddingConfig) -> nn.Module:
        """Create position embedding based on config."""
        embedding_type = config.embedding_type.lower()
        
        if embedding_type == 'rope':
            return RotaryEmbedding(config.dim, config.max_seq_len, config.theta)
        elif embedding_type == 'sinusoidal':
            return SinusoidalEmbedding(config.dim, config.max_seq_len)
        elif embedding_type == 'learned':
            return LearnedPositionEmbedding(config.max_seq_len, config.dim)
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
    
    @staticmethod
    def create_from_dict(config_dict: dict) -> nn.Module:
        """Create from dictionary."""
        config = EmbeddingConfig(**config_dict)
        return EmbeddingFactory.create(config)

# Keep the old function for backward compatibility
def create_position_embedding(embedding_type: str, dim: int, 
                              max_seq_len: int = 2048, theta: float = 10000.0) -> nn.Module:
    """Legacy function - use EmbeddingFactory.create() instead."""
    config = EmbeddingConfig(embedding_type=embedding_type, dim=dim, 
                            max_seq_len=max_seq_len, theta=theta)
    return EmbeddingFactory.create(config)
