"""Feedforward networks for transformer models."""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

from .base import BaseFeedForward


class SwiGLUFeedForward(BaseFeedForward):
    """
    SwiGLU Feedforward Network.
    
    Uses Swish activation with Gated Linear Units.
    More effective than standard GELU for language models.
    
    Used in: LLaMA, PaLM, and other modern LLMs.
    
    Reference: https://arxiv.org/abs/2002.05202
    
    Architecture:
        x -> Linear(d_model, d_ff) -> Swish -> * -> Linear(d_ff, d_model)
              Linear(d_model, d_ff) ------->
    
    Example:
        >>> ffn = SwiGLUFeedForward(d_model=768, d_ff=2048)
        >>> x = torch.randn(2, 10, 768)
        >>> out = ffn(x)
        >>> out.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs
    ):
        """
        Initialize SwiGLU feedforward.
        
        Args:
            d_model: Model dimension
            d_ff: Feedforward dimension (default: 8/3 * d_model for SwiGLU)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            **kwargs: Additional arguments
        """
        # SwiGLU typically uses d_ff = 8/3 * d_model to match GELU params
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            # Round to nearest multiple of 256 for efficiency
            d_ff = ((d_ff + 255) // 256) * 256
        
        super().__init__(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias, **kwargs)
        
        # Gate and up projections
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # Gate
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)  # Up
        
        # Down projection
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        self.reset_parameters()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU feedforward.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # SwiGLU: SiLU(x @ W1) * (x @ W2) @ W3
        gate = F.silu(self.w1(x))  # Swish/SiLU activation
        up = self.w2(x)
        
        # Element-wise multiplication (gating)
        hidden = gate * up
        
        # Apply pruning mask if set
        if self.pruning_mask.numel() > 1:
            hidden = hidden * self.pruning_mask
        
        # Down projection
        output = self.w3(hidden)
        
        if self.dropout_layer is not None:
            output = self.dropout_layer(output)
        
        return output
    
    def reset_parameters(self) -> None:
        """Initialize weights."""
        # Use scaled initialization
        nn.init.normal_(self.w1.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.w2.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.w3.weight, mean=0, std=self.d_ff ** -0.5)
        
        if self.use_bias:
            nn.init.zeros_(self.w1.bias)
            nn.init.zeros_(self.w2.bias)
            nn.init.zeros_(self.w3.bias)


class GELUFeedForward(BaseFeedForward):
    """
    Standard GELU Feedforward Network.
    
    Traditional transformer feedforward with GELU activation.
    
    Used in: BERT, GPT-2, GPT-3, and original Transformer.
    
    Architecture:
        x -> Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
    
    Example:
        >>> ffn = GELUFeedForward(d_model=768, d_ff=3072)
        >>> x = torch.randn(2, 10, 768)
        >>> out = ffn(x)
        >>> out.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = 'gelu',
        **kwargs
    ):
        """
        Initialize GELU feedforward.
        
        Args:
            d_model: Model dimension
            d_ff: Feedforward dimension (default: 4 * d_model)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            activation: Activation function ('gelu', 'relu', 'gelu_new')
            **kwargs: Additional arguments
        """
        super().__init__(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias, **kwargs)
        
        self.fc1 = nn.Linear(d_model, self.d_ff, bias=bias)
        self.fc2 = nn.Linear(self.d_ff, d_model, bias=bias)
        
        # Activation function
        self.activation_name = activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'gelu_new':
            self.activation = nn.GELU(approximate='tanh')
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        self.reset_parameters()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply GELU feedforward.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Up projection + activation
        hidden = self.activation(self.fc1(x))
        
        # Apply pruning mask if set
        if self.pruning_mask.numel() > 1:
            hidden = hidden * self.pruning_mask
        
        # Down projection
        output = self.fc2(hidden)
        
        if self.dropout_layer is not None:
            output = self.dropout_layer(output)
        
        return output
    
    def reset_parameters(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.fc1.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.fc2.weight, mean=0, std=self.d_ff ** -0.5)
        
        if self.use_bias:
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)


def get_feedforward(
    ffn_type: str,
    d_model: int,
    d_ff: Optional[int] = None,
    dropout: float = 0.0,
    bias: bool = True,
    **kwargs
) -> BaseFeedForward:
    """
    Factory function for feedforward networks.
    
    Args:
        ffn_type: Type of feedforward ('swiglu', 'gelu', 'relu')
        d_model: Model dimension
        d_ff: Feedforward dimension
        dropout: Dropout probability
        bias: Whether to use bias
        **kwargs: Additional arguments
    
    Returns:
        Feedforward network
    
    Example:
        >>> ffn = get_feedforward('swiglu', d_model=768)
        >>> isinstance(ffn, SwiGLUFeedForward)
        True
    """
    ffn_type = ffn_type.lower()
    
    if ffn_type == 'swiglu' or ffn_type == 'swish':
        return SwiGLUFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            **kwargs
        )
    
    elif ffn_type == 'gelu':
        return GELUFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            activation='gelu',
            **kwargs
        )
    
    elif ffn_type == 'gelu_new':
        return GELUFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            activation='gelu_new',
            **kwargs
        )
    
    elif ffn_type == 'relu':
        return GELUFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            activation='relu',
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown ffn_type: {ffn_type}. "
            f"Available: 'swiglu', 'gelu', 'gelu_new', 'relu'"
        )