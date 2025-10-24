"""Normalization layers for transformer models."""

import torch
from torch import nn, Tensor
from typing import Optional

from .base import BaseNormalization


class RMSNorm(BaseNormalization):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is simpler and more efficient than LayerNorm:
    - No mean centering
    - No learned bias
    - Faster computation
    
    Used in: LLaMA, Mistral, and other modern LLMs.
    
    Reference: https://arxiv.org/abs/1910.07467
    
    Example:
        >>> norm = RMSNorm(d_model=768)
        >>> x = torch.randn(2, 10, 768)
        >>> out = norm(x)
        >>> out.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        """
        Initialize RMSNorm.
        
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
            elementwise_affine: Whether to learn scale parameter
        """
        super().__init__(d_model=d_model, eps=eps)
        
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMSNorm.
        
        Args:
            x: Input tensor (..., d_model)
        
        Returns:
            Normalized tensor (..., d_model)
        """
        # Compute RMS: sqrt(mean(x^2))
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        # Apply learned scale
        if self.elementwise_affine:
            x = x * self.weight
        
        return x
    
    def reset_parameters(self) -> None:
        """Reset parameters to default values."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
    
    def extra_repr(self) -> str:
        """String representation."""
        return f'd_model={self.d_model}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class LayerNorm(BaseNormalization):
    """
    Standard Layer Normalization.
    
    Standard LayerNorm with mean centering and learned scale/bias.
    More expensive than RMSNorm but sometimes more stable.
    
    Used in: Original Transformer, BERT, GPT-2.
    
    Reference: https://arxiv.org/abs/1607.06450
    
    Example:
        >>> norm = LayerNorm(d_model=768)
        >>> x = torch.randn(2, 10, 768)
        >>> out = norm(x)
        >>> out.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True
    ):
        """
        Initialize LayerNorm.
        
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
            elementwise_affine: Whether to learn scale/bias
            bias: Whether to use bias (only if elementwise_affine=True)
        """
        super().__init__(d_model=d_model, eps=eps)
        
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            if bias:
                self.bias = nn.Parameter(torch.zeros(d_model))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply LayerNorm.
        
        Args:
            x: Input tensor (..., d_model)
        
        Returns:
            Normalized tensor (..., d_model)
        """
        # Use PyTorch's efficient implementation
        return torch.nn.functional.layer_norm(
            x,
            normalized_shape=(self.d_model,),
            weight=self.weight if self.elementwise_affine else None,
            bias=self.bias if self.elementwise_affine and self.use_bias else None,
            eps=self.eps
        )
    
    def reset_parameters(self) -> None:
        """Reset parameters to default values."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.use_bias:
                nn.init.zeros_(self.bias)
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f'd_model={self.d_model}, eps={self.eps}, '
            f'elementwise_affine={self.elementwise_affine}, bias={self.use_bias}'
        )


def get_normalization(
    norm_type: str,
    d_model: int,
    eps: float = 1e-6,
    **kwargs
) -> BaseNormalization:
    """
    Factory function for normalization layers.
    
    Args:
        norm_type: Type of normalization ('rms', 'layer')
        d_model: Model dimension
        eps: Small constant for numerical stability
        **kwargs: Additional arguments
    
    Returns:
        Normalization layer
    
    Example:
        >>> norm = get_normalization('rms', d_model=768)
        >>> isinstance(norm, RMSNorm)
        True
    """
    norm_type = norm_type.lower()
    
    if norm_type == 'rms' or norm_type == 'rmsnorm':
        return RMSNorm(d_model=d_model, eps=eps, **kwargs)
    
    elif norm_type == 'layer' or norm_type == 'layernorm':
        return LayerNorm(d_model=d_model, eps=eps, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown norm_type: {norm_type}. "
            f"Available: 'rms', 'layer'"
        )