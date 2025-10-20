"""
Normalization layers for Ramanujan Transformer.

This module provides various normalization techniques used throughout
the architecture:
- RMSNorm: Root Mean Square normalization (more efficient than LayerNorm)
- QKNorm: Query-Key normalization for attention stability

These normalizations improve training stability and performance while
being more computationally efficient than standard LayerNorm.

Example:
    >>> from ramanujan.architecture.normalization import RMSNorm, QKNorm
    >>> 
    >>> # RMS normalization for hidden states
    >>> rms_norm = RMSNorm(dim=512)
    >>> x = torch.randn(2, 128, 512)
    >>> x_normalized = rms_norm(x)
    >>> 
    >>> # QK normalization for attention
    >>> qk_norm = QKNorm(dim=64)
    >>> q = torch.randn(2, 8, 128, 64)
    >>> q_normalized = qk_norm(q)
"""

import torch
import torch.nn as nn
from typing import Optional


# ============================================================================
# RMS NORMALIZATION
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is a simplified version of LayerNorm that:
    - Only normalizes by RMS (no mean centering)
    - Uses only a scale parameter (no bias)
    - Is more efficient computationally
    - Often performs as well or better than LayerNorm
    
    From "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    
    Math:
        RMS(x) = sqrt(mean(x²) + eps)
        output = (x / RMS(x)) * scale
    
    Args:
        dim: Dimension to normalize
        eps: Small constant for numerical stability (default: 1e-6)
        elementwise_affine: Whether to learn scale parameter (default: True)
    
    Example:
        >>> norm = RMSNorm(dim=512)
        >>> x = torch.randn(2, 128, 512)
        >>> x_norm = norm(x)
        >>> print(x_norm.shape)  # [2, 128, 512]
        >>> 
        >>> # Verify normalization
        >>> rms = x_norm.pow(2).mean(-1, keepdim=True).sqrt()
        >>> print(rms.mean())  # Should be close to 1.0
    """
    
    def __init__(
        self, 
        dim: int, 
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            # Learnable scale parameter (initialized to 1)
            self.scale = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('scale', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor [..., dim]
        
        Returns:
            Normalized tensor with same shape as input
        """
        # Compute RMS: sqrt(mean(x²) + eps)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        
        # Normalize
        x_norm = x / rms
        
        # Apply learnable scale if enabled
        if self.elementwise_affine:
            x_norm = x_norm * self.scale
        
        return x_norm
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


# ============================================================================
# QK NORMALIZATION
# ============================================================================

class QKNorm(nn.Module):
    """
    Query-Key Normalization for attention stability.
    
    QKNorm normalizes queries and keys before computing attention scores.
    This helps:
    - Prevent attention entropy collapse
    - Improve training stability
    - Enable better gradient flow
    - Allow higher learning rates
    
    From "NormFormer: Improved Transformer Pretraining with Extra Normalization"
    
    Unlike standard LayerNorm, QKNorm uses RMS normalization which is:
    - Faster to compute
    - Simpler (no bias term)
    - Often performs better in attention
    
    Args:
        dim: Dimension to normalize (typically head_dim)
        eps: Small constant for numerical stability (default: 1e-6)
        elementwise_affine: Whether to learn scale parameter (default: True)
    
    Example:
        >>> qk_norm = QKNorm(dim=64)
        >>> 
        >>> # Normalize queries and keys in attention
        >>> q = torch.randn(2, 8, 128, 64)  # [batch, heads, seq, head_dim]
        >>> k = torch.randn(2, 8, 128, 64)
        >>> 
        >>> q_norm = qk_norm(q)
        >>> k_norm = qk_norm(k)
        >>> 
        >>> # Compute attention with normalized q, k
        >>> scores = torch.matmul(q_norm, k_norm.transpose(-2, -1))
    """
    
    def __init__(
        self, 
        dim: int, 
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            # Learnable scale parameter (initialized to 1)
            self.scale = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('scale', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply QK normalization.
        
        Args:
            x: Input tensor [..., dim] - typically queries or keys
        
        Returns:
            Normalized tensor with same shape as input
        """
        # Compute RMS normalization
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        
        # Normalize
        x_norm = x / rms
        
        # Apply learnable scale if enabled
        if self.elementwise_affine:
            x_norm = x_norm * self.scale
        
        return x_norm.type_as(x)
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


# ============================================================================
# LAYER NORM (for comparison/compatibility)
# ============================================================================

class LayerNorm(nn.LayerNorm):
    """
    Standard LayerNorm with slight modifications for consistency.
    
    This is just nn.LayerNorm but with the same interface as RMSNorm
    for easy swapping. Useful for ablation studies.
    
    Args:
        dim: Dimension to normalize
        eps: Small constant for numerical stability (default: 1e-6)
        elementwise_affine: Whether to learn affine parameters (default: True)
    
    Example:
        >>> # Can swap RMSNorm and LayerNorm easily
        >>> norm = LayerNorm(dim=512)  # or RMSNorm(dim=512)
        >>> x_norm = norm(x)
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__(
            normalized_shape=dim,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
        self.dim = dim


# ============================================================================
# NORMALIZATION FACTORY
# ============================================================================

class NormalizationFactory:
    """
    Factory for creating normalization layers.
    
    Makes it easy to switch between different normalization types
    for experiments and ablation studies.
    
    Example:
        >>> # Create RMSNorm
        >>> norm = NormalizationFactory.create('rms', dim=512)
        >>> 
        >>> # Create LayerNorm for comparison
        >>> norm_ln = NormalizationFactory.create('layer', dim=512)
        >>> 
        >>> # Create QKNorm for attention
        >>> qk_norm = NormalizationFactory.create('qk', dim=64)
    """
    
    @staticmethod
    def create(
        norm_type: str,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ) -> nn.Module:
        """
        Create normalization layer.
        
        Args:
            norm_type: Type of normalization ('rms', 'layer', 'qk')
            dim: Dimension to normalize
            eps: Small constant for numerical stability
            elementwise_affine: Whether to learn affine parameters
        
        Returns:
            Normalization module
        
        Raises:
            ValueError: If norm_type is not recognized
        """
        norm_type = norm_type.lower()
        
        if norm_type in ['rms', 'rmsnorm']:
            return RMSNorm(dim, eps, elementwise_affine)
        elif norm_type in ['layer', 'layernorm', 'ln']:
            return LayerNorm(dim, eps, elementwise_affine)
        elif norm_type in ['qk', 'qknorm']:
            return QKNorm(dim, eps, elementwise_affine)
        else:
            raise ValueError(
                f"Unknown norm_type: {norm_type}. "
                f"Choose from: 'rms', 'layer', 'qk'"
            )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_normalizations(
    x: torch.Tensor,
    dim: int = -1
) -> dict:
    """
    Compare different normalization methods on the same input.
    
    Useful for understanding the differences between normalization types.
    
    Args:
        x: Input tensor
        dim: Dimension to normalize (default: -1, last dimension)
    
    Returns:
        Dictionary with statistics for each normalization type
    
    Example:
        >>> x = torch.randn(2, 128, 512)
        >>> stats = compare_normalizations(x)
        >>> for norm_type, s in stats.items():
        ...     print(f"{norm_type}: mean={s['mean']:.4f}, std={s['std']:.4f}")
    """
    if dim != -1:
        # Reshape to put target dim at end
        perm = list(range(x.ndim))
        perm[-1], perm[dim] = perm[dim], perm[-1]
        x = x.permute(perm)
    
    feature_dim = x.shape[-1]
    
    results = {}
    
    # Original
    results['original'] = {
        'mean': x.mean(dim=-1).mean().item(),
        'std': x.std(dim=-1).mean().item(),
        'max': x.max().item(),
        'min': x.min().item()
    }
    
    # RMSNorm
    rms_norm = RMSNorm(feature_dim, elementwise_affine=False)
    x_rms = rms_norm(x)
    results['rms'] = {
        'mean': x_rms.mean(dim=-1).mean().item(),
        'std': x_rms.std(dim=-1).mean().item(),
        'max': x_rms.max().item(),
        'min': x_rms.min().item()
    }
    
    # LayerNorm
    layer_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
    x_ln = layer_norm(x)
    results['layer'] = {
        'mean': x_ln.mean(dim=-1).mean().item(),
        'std': x_ln.std(dim=-1).mean().item(),
        'max': x_ln.max().item(),
        'min': x_ln.min().item()
    }
    
    return results


def get_norm_info(module: nn.Module) -> dict:
    """
    Get information about a normalization module.
    
    Args:
        module: Normalization module
    
    Returns:
        Dictionary with module information
    
    Example:
        >>> norm = RMSNorm(dim=512)
        >>> info = get_norm_info(norm)
        >>> print(info['type'])  # 'RMSNorm'
        >>> print(info['dim'])   # 512
    """
    info = {
        'type': module.__class__.__name__,
    }
    
    if hasattr(module, 'dim'):
        info['dim'] = module.dim
    elif hasattr(module, 'normalized_shape'):
        info['dim'] = module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
    
    if hasattr(module, 'eps'):
        info['eps'] = module.eps
    
    if hasattr(module, 'elementwise_affine'):
        info['elementwise_affine'] = module.elementwise_affine
    
    if hasattr(module, 'scale') and module.scale is not None:
        info['has_learnable_scale'] = True
        info['scale_mean'] = module.scale.mean().item()
        info['scale_std'] = module.scale.std().item()
    else:
        info['has_learnable_scale'] = False
    
    return info


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing normalization.py module")
    print("="*70)
    
    # Test RMSNorm
    print("\n1. Testing RMSNorm...")
    rms_norm = RMSNorm(dim=512)
    x = torch.randn(2, 128, 512)
    x_norm = rms_norm(x)
    
    # Check shape
    assert x_norm.shape == x.shape, "Shape mismatch!"
    
    # Check that RMS is approximately 1
    rms = x_norm.pow(2).mean(-1).sqrt().mean()
    print(f"   Input: {x.shape}")
    print(f"   Output: {x_norm.shape}")
    print(f"   RMS after normalization: {rms:.4f} (should be ~1.0)")
    print(f"   ✅ RMSNorm working!")
    
    # Test info
    info = get_norm_info(rms_norm)
    print(f"   Info: {info}")
    
    # Test QKNorm
    print("\n2. Testing QKNorm...")
    qk_norm = QKNorm(dim=64)
    q = torch.randn(2, 8, 128, 64)  # [batch, heads, seq, head_dim]
    q_norm = qk_norm(q)
    
    assert q_norm.shape == q.shape, "Shape mismatch!"
    
    rms_q = q_norm.pow(2).mean(-1).sqrt().mean()
    print(f"   Input: {q.shape}")
    print(f"   Output: {q_norm.shape}")
    print(f"   RMS after normalization: {rms_q:.4f} (should be ~1.0)")
    print(f"   ✅ QKNorm working!")
    
    # Test LayerNorm compatibility
    print("\n3. Testing LayerNorm compatibility...")
    ln = LayerNorm(dim=512)
    x_ln = ln(x)
    
    assert x_ln.shape == x.shape, "Shape mismatch!"
    
    # LayerNorm should have mean ~0 and std ~1
    mean_ln = x_ln.mean(-1).mean()
    std_ln = x_ln.std(-1).mean()
    print(f"   Input: {x.shape}")
    print(f"   Output: {x_ln.shape}")
    print(f"   Mean after normalization: {mean_ln:.4f} (should be ~0.0)")
    print(f"   Std after normalization: {std_ln:.4f} (should be ~1.0)")
    print(f"   ✅ LayerNorm working!")
    
    # Test NormalizationFactory
    print("\n4. Testing NormalizationFactory...")
    norm_rms = NormalizationFactory.create('rms', dim=512)
    norm_ln = NormalizationFactory.create('layer', dim=512)
    norm_qk = NormalizationFactory.create('qk', dim=64)
    
    print(f"   Created RMS: {type(norm_rms).__name__}")
    print(f"   Created Layer: {type(norm_ln).__name__}")
    print(f"   Created QK: {type(norm_qk).__name__}")
    print(f"   ✅ NormalizationFactory working!")
    
    # Test comparison utility
    print("\n5. Testing normalization comparison...")
    x_test = torch.randn(2, 128, 512)
    stats = compare_normalizations(x_test)
    
    print("\n   Statistics comparison:")
    for norm_type, s in stats.items():
        print(f"   {norm_type:8s}: mean={s['mean']:7.4f}, std={s['std']:7.4f}")
    print(f"   ✅ Comparison utility working!")
    
    # Test without affine
    print("\n6. Testing without learnable parameters...")
    rms_no_affine = RMSNorm(dim=512, elementwise_affine=False)
    x_no_affine = rms_no_affine(x)
    
    # Check that scale parameter doesn't exist
    assert rms_no_affine.scale is None, "Scale should be None!"
    print(f"   Created RMSNorm without scale parameter")
    print(f"   Output shape: {x_no_affine.shape}")
    print(f"   ✅ Non-affine mode working!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.architecture.normalization import RMSNorm, QKNorm")
    print("  from ramanujan.architecture.normalization import NormalizationFactory")
    print("="*70)