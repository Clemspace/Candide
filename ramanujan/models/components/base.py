"""Base interfaces and protocols for model components."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Protocol
import torch
from torch import nn, Tensor


class NormalizationProtocol(Protocol):
    """Protocol for normalization layers."""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply normalization."""
        ...
    
    def reset_parameters(self) -> None:
        """Reset parameters."""
        ...


class AttentionProtocol(Protocol):
    """Protocol for attention mechanisms."""
    
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
        Apply attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len)
            position_ids: Position indices (batch, seq_len)
            past_key_value: Cached key-value pairs
            use_cache: Whether to return key-value cache
        
        Returns:
            Tuple of (output, past_key_value)
        """
        ...


class FeedForwardProtocol(Protocol):
    """Protocol for feedforward networks."""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply feedforward transformation."""
        ...


class BaseNormalization(nn.Module, ABC):
    """
    Abstract base class for normalization layers.
    
    Provides consistent interface for different normalization types:
    - RMSNorm (default for transformers)
    - LayerNorm (standard)
    - Future: Other variants
    """
    
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        **kwargs
    ):
        """
        Initialize normalization layer.
        
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
            **kwargs: Additional arguments for subclasses
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply normalization."""
        pass
    
    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset layer parameters."""
        pass


class BaseAttention(nn.Module, ABC):
    """
    Abstract base class for attention mechanisms.
    
    All attention mechanisms inherit from this to ensure consistent interface.
    Supports:
    - Multi-head attention (standard)
    - Grouped query attention (memory efficient)
    - Future: Sparse attention, local attention, etc.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        pruning_mask: Optional[Tensor] = None,
        **kwargs
    ):
        """
        Initialize attention mechanism.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
            pruning_mask: Optional mask for pruning (future Ramanujan)
            **kwargs: Additional arguments for subclasses
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_bias = bias
        
        # Pruning support (for future Ramanujan sparsification)
        self.register_buffer(
            'pruning_mask',
            pruning_mask if pruning_mask is not None else torch.ones(1),
            persistent=False
        )
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
    
    @abstractmethod
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
        Apply attention mechanism.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len) or (batch, 1, seq_len, seq_len)
            position_ids: Position indices (batch, seq_len)
            past_key_value: Cached (key, value) from previous step
            use_cache: Whether to return key-value cache for generation
            **kwargs: Additional arguments
        
        Returns:
            Tuple of:
                - output: Attention output (batch, seq_len, d_model)
                - past_key_value: Updated cache (key, value) if use_cache=True
        """
        pass
    
    def _apply_pruning_mask(self, weight: Tensor) -> Tensor:
        """
        Apply pruning mask to weights (for future Ramanujan).
        
        Args:
            weight: Weight tensor
        
        Returns:
            Masked weight tensor
        """
        if self.pruning_mask.numel() > 1:
            return weight * self.pruning_mask
        return weight
    
    def set_pruning_mask(self, mask: Tensor) -> None:
        """Set pruning mask for structured sparsity."""
        self.register_buffer('pruning_mask', mask, persistent=False)


class BaseFeedForward(nn.Module, ABC):
    """
    Abstract base class for feedforward networks.
    
    All FFN variants inherit from this:
    - SwiGLU (default, used in LLaMA)
    - GELU (standard)
    - Future: Sparse FFN, MoE, etc.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        pruning_mask: Optional[Tensor] = None,
        **kwargs
    ):
        """
        Initialize feedforward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feedforward dimension (default: 4 * d_model)
            dropout: Dropout probability
            bias: Whether to use bias
            pruning_mask: Optional mask for pruning
            **kwargs: Additional arguments for subclasses
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.dropout = dropout
        self.use_bias = bias
        
        # Pruning support
        self.register_buffer(
            'pruning_mask',
            pruning_mask if pruning_mask is not None else torch.ones(1),
            persistent=False
        )
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feedforward transformation.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        pass
    
    def _apply_pruning_mask(self, weight: Tensor) -> Tensor:
        """Apply pruning mask to weights."""
        if self.pruning_mask.numel() > 1:
            return weight * self.pruning_mask
        return weight
    
    def set_pruning_mask(self, mask: Tensor) -> None:
        """Set pruning mask for structured sparsity."""
        self.register_buffer('pruning_mask', mask, persistent=False)


class BaseEmbedding(nn.Module, ABC):
    """
    Abstract base class for embeddings.
    
    Supports:
    - Token embeddings
    - Position embeddings (learned, sinusoidal, RoPE)
    - Future: Relative position, ALiBi, etc.
    """
    
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: int = 768,
        max_seq_len: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Vocabulary size (for token embeddings)
            d_model: Model dimension
            max_seq_len: Maximum sequence length (for position embeddings)
            **kwargs: Additional arguments for subclasses
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
    
    @abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply embedding.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
        
        Returns:
            Embedded tensor
        """
        pass


class BaseTransformerBlock(nn.Module, ABC):
    """
    Abstract base class for transformer blocks.
    
    A transformer block consists of:
    1. Self-attention
    2. Feedforward network
    3. Normalization layers
    4. Residual connections
    
    Variants:
    - Pre-norm (default, used in modern LLMs)
    - Post-norm (original transformer)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        norm_first: bool = True,
        **kwargs
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feedforward dimension
            dropout: Dropout probability
            norm_first: Use pre-norm (True) or post-norm (False)
            **kwargs: Additional arguments for subclasses
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff or 4 * d_model
        self.dropout = dropout
        self.norm_first = norm_first
    
    @abstractmethod
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
            mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key-value pairs
            use_cache: Whether to return cache
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (output, past_key_value)
        """
        pass