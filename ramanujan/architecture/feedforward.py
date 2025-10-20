"""
Feedforward networks for Ramanujan Transformer.

This module provides various feedforward network architectures:
- SwiGLU: Gated Linear Unit with Swish activation
- SparseRamanujanSwiGLU: SwiGLU with Ramanujan graph sparsity
- Standard FFN: Classic two-layer feedforward (for comparison)

SwiGLU has been shown to outperform standard ReLU FFNs in transformers
while maintaining similar computational cost.

Example:
    >>> from ramanujan.architecture.feedforward import FeedForwardFactory
    >>> from ramanujan.foundation import RamanujanFoundation
    >>> 
    >>> # Standard SwiGLU
    >>> ffn = FeedForwardFactory.create(
    ...     dim=512,
    ...     hidden_dim=2048,
    ...     ffn_type='swiglu'
    ... )
    >>> 
    >>> # With Ramanujan sparsity
    >>> foundation = RamanujanFoundation(max_prime=1000)
    >>> ffn_sparse = FeedForwardFactory.create(
    ...     dim=512,
    ...     hidden_dim=2048,
    ...     ffn_type='swiglu',
    ...     foundation=foundation,
    ...     ffn_sparsity=0.88
    ... )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU), also known as Swish.
    
    SiLU(x) = x * sigmoid(x)
    
    More smooth than ReLU and has been shown to improve performance
    in deep networks, especially transformers.
    
    Note: PyTorch has nn.SiLU, but we define it here for clarity.
    
    Example:
        >>> silu = SiLU()
        >>> x = torch.randn(2, 512)
        >>> y = silu(x)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU activation."""
        return x * torch.sigmoid(x)


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU).
    
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
    
    Commonly used in BERT and other transformer models.
    
    Example:
        >>> gelu = GELU()
        >>> x = torch.randn(2, 512)
        >>> y = gelu(x)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation."""
        return F.gelu(x)


# ============================================================================
# SWIGLU FEEDFORWARD
# ============================================================================

class SwiGLU(nn.Module):
    """
    Gated Linear Unit with SiLU (Swish) activation.
    
    SwiGLU is a variant of GLU that uses SiLU activation instead of sigmoid.
    It has been shown to outperform standard FFN in transformers (PaLM paper).
    
    Architecture:
        gate = SiLU(W1(x))
        value = W2(x)
        output = W3(gate * value)
    
    The gating mechanism allows the network to control information flow,
    similar to LSTMs but without recurrence.
    
    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (typically 4*dim for standard, or ~2.67*dim for SwiGLU)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in linear layers (default: False)
        foundation: Optional RamanujanFoundation for sparse layers
        ffn_sparsity: Target sparsity for FFN layers (if foundation provided)
    
    Example:
        >>> # Standard SwiGLU with 4x hidden dim
        >>> ffn = SwiGLU(dim=512, hidden_dim=2048)
        >>> x = torch.randn(2, 128, 512)
        >>> out = ffn(x)
        >>> print(out.shape)  # [2, 128, 512]
        >>> 
        >>> # With Ramanujan sparsity
        >>> from ramanujan.foundation import RamanujanFoundation
        >>> foundation = RamanujanFoundation(max_prime=1000)
        >>> ffn_sparse = SwiGLU(
        ...     dim=512,
        ...     hidden_dim=2048,
        ...     foundation=foundation,
        ...     ffn_sparsity=0.88
        ... )
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        foundation: Optional['RamanujanFoundation'] = None,
        ffn_sparsity: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        
        # Determine if we should use sparse layers
        use_sparse = foundation is not None and ffn_sparsity > 0.05
        
        if use_sparse:
            # Sparse Ramanujan layers
            self.gate_proj = foundation.create_layer(
                dim, hidden_dim,
                target_sparsity=ffn_sparsity,
                bias=bias,
                force_method="lps"
            )
            self.value_proj = foundation.create_layer(
                dim, hidden_dim,
                target_sparsity=ffn_sparsity,
                bias=bias,
                force_method="lps"
            )
            self.output_proj = foundation.create_layer(
                hidden_dim, dim,
                target_sparsity=ffn_sparsity,
                bias=bias,
                force_method="lps"
            )
        else:
            # Standard dense layers
            self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
            self.value_proj = nn.Linear(dim, hidden_dim, bias=bias)
            self.output_proj = nn.Linear(hidden_dim, dim, bias=bias)
        
        # Activation
        self.activation = SiLU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Gate pathway: SiLU(W1(x))
        gate = self.activation(self.gate_proj(x))
        
        # Value pathway: W2(x)
        value = self.value_proj(x)
        
        # Element-wise gating
        hidden = gate * value
        
        # Project back to original dimension
        output = self.output_proj(hidden)
        
        # Dropout
        output = self.dropout(output)
        
        return output
    
    def get_info(self) -> dict:
        """Get information about this FFN module."""
        info = {
            'type': 'SwiGLU',
            'dim': self.dim,
            'hidden_dim': self.hidden_dim,
            'expansion_ratio': self.hidden_dim / self.dim,
            'dropout': self.dropout_prob
        }
        
        # Check if using sparse layers
        if hasattr(self.gate_proj, 'mask'):
            gate_sparsity = 1.0 - (self.gate_proj.mask.sum() / self.gate_proj.mask.numel())
            info['ffn_sparsity'] = gate_sparsity.item()
            info['sparse'] = True
        else:
            info['sparse'] = False
        
        return info


# ============================================================================
# STANDARD FEEDFORWARD (for comparison)
# ============================================================================

class StandardFFN(nn.Module):
    """
    Standard two-layer feedforward network with ReLU or GELU.
    
    This is the classic FFN used in original Transformer paper.
    Useful for ablation studies and comparison with SwiGLU.
    
    Architecture:
        output = W2(activation(W1(x)))
    
    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (typically 4*dim)
        dropout: Dropout probability (default: 0.0)
        activation: Activation function ('relu' or 'gelu', default: 'relu')
        bias: Whether to use bias in linear layers (default: False)
        foundation: Optional RamanujanFoundation for sparse layers
        ffn_sparsity: Target sparsity for FFN layers (if foundation provided)
    
    Example:
        >>> # Standard FFN with ReLU
        >>> ffn = StandardFFN(dim=512, hidden_dim=2048, activation='relu')
        >>> 
        >>> # With GELU activation
        >>> ffn_gelu = StandardFFN(dim=512, hidden_dim=2048, activation='gelu')
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = 'relu',
        bias: bool = False,
        foundation: Optional['RamanujanFoundation'] = None,
        ffn_sparsity: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        self.activation_name = activation
        
        # Determine if we should use sparse layers
        use_sparse = foundation is not None and ffn_sparsity > 0.05
        
        if use_sparse:
            # Sparse Ramanujan layers
            self.fc1 = foundation.create_layer(
                dim, hidden_dim,
                target_sparsity=ffn_sparsity,
                bias=bias,
                force_method="lps"
            )
            self.fc2 = foundation.create_layer(
                hidden_dim, dim,
                target_sparsity=ffn_sparsity,
                bias=bias,
                force_method="lps"
            )
        else:
            # Standard dense layers
            self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        
        # Activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}. Choose 'relu' or 'gelu'")
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # First layer with activation
        hidden = self.activation(self.fc1(x))
        
        # Dropout after activation
        hidden = self.dropout(hidden)
        
        # Second layer
        output = self.fc2(hidden)
        
        # Dropout after output
        output = self.dropout(output)
        
        return output
    
    def get_info(self) -> dict:
        """Get information about this FFN module."""
        info = {
            'type': 'StandardFFN',
            'dim': self.dim,
            'hidden_dim': self.hidden_dim,
            'expansion_ratio': self.hidden_dim / self.dim,
            'activation': self.activation_name,
            'dropout': self.dropout_prob
        }
        
        # Check if using sparse layers
        if hasattr(self.fc1, 'mask'):
            sparsity = 1.0 - (self.fc1.mask.sum() / self.fc1.mask.numel())
            info['ffn_sparsity'] = sparsity.item()
            info['sparse'] = True
        else:
            info['sparse'] = False
        
        return info


# ============================================================================
# SPARSE RAMANUJAN SWIGLU (Convenience Alias)
# ============================================================================

class SparseRamanujanSwiGLU(SwiGLU):
    """
    Convenience class for SwiGLU with Ramanujan sparsity.
    
    This is just SwiGLU with foundation and sparsity explicitly set.
    Use this when you want to emphasize that sparsity is being used.
    
    Args:
        foundation: RamanujanFoundation (required)
        ffn_sparsity: Target sparsity (required, >0.05)
        **kwargs: Passed to SwiGLU
    
    Example:
        >>> from ramanujan.foundation import RamanujanFoundation
        >>> foundation = RamanujanFoundation(max_prime=1000)
        >>> ffn = SparseRamanujanSwiGLU(
        ...     dim=512,
        ...     hidden_dim=2048,
        ...     foundation=foundation,
        ...     ffn_sparsity=0.88
        ... )
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        foundation: 'RamanujanFoundation',
        ffn_sparsity: float,
        dropout: float = 0.0,
        bias: bool = False
    ):
        assert foundation is not None, "SparseRamanujanSwiGLU requires foundation"
        assert ffn_sparsity > 0.05, "SparseRamanujanSwiGLU requires ffn_sparsity > 0.05"
        
        super().__init__(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
            foundation=foundation,
            ffn_sparsity=ffn_sparsity
        )


# ============================================================================
# FEEDFORWARD FACTORY
# ============================================================================

@dataclass
class FeedForwardConfig:
    """Configuration for feedforward network."""
    dim: int
    hidden_dim: int
    dropout: float = 0.0
    bias: bool = False
    
    # FFN type
    ffn_type: str = 'swiglu'  # 'swiglu', 'standard'
    activation: str = 'relu'  # For standard FFN: 'relu', 'gelu'
    
    # Ramanujan sparsity
    foundation: Optional['RamanujanFoundation'] = None
    ffn_sparsity: float = 0.0


class FeedForwardFactory:
    """
    Factory for creating feedforward networks.
    
    Automatically selects the appropriate FFN class based on configuration.
    
    Example:
        >>> from ramanujan.architecture.feedforward import FeedForwardFactory, FeedForwardConfig
        >>> 
        >>> # SwiGLU
        >>> config = FeedForwardConfig(
        ...     dim=512,
        ...     hidden_dim=2048,
        ...     ffn_type='swiglu'
        ... )
        >>> ffn = FeedForwardFactory.create(config)
        >>> 
        >>> # Standard FFN with GELU
        >>> config_standard = FeedForwardConfig(
        ...     dim=512,
        ...     hidden_dim=2048,
        ...     ffn_type='standard',
        ...     activation='gelu'
        ... )
        >>> ffn_standard = FeedForwardFactory.create(config_standard)
    """
    
    @staticmethod
    def create(config: FeedForwardConfig) -> nn.Module:
        """
        Create feedforward network based on config.
        
        Args:
            config: FeedForwardConfig instance
        
        Returns:
            Appropriate feedforward module
        """
        ffn_type = config.ffn_type.lower()
        
        if ffn_type == 'swiglu':
            return SwiGLU(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                bias=config.bias,
                foundation=config.foundation,
                ffn_sparsity=config.ffn_sparsity
            )
        elif ffn_type in ['standard', 'ffn']:
            return StandardFFN(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                activation=config.activation,
                bias=config.bias,
                foundation=config.foundation,
                ffn_sparsity=config.ffn_sparsity
            )
        else:
            raise ValueError(
                f"Unknown ffn_type: {ffn_type}. "
                f"Choose from: 'swiglu', 'standard'"
            )
    
    @staticmethod
    def create_from_dict(config_dict: dict) -> nn.Module:
        """
        Create feedforward from dictionary config.
        
        Useful for loading from YAML/JSON configs.
        
        Args:
            config_dict: Dictionary with FFN parameters
        
        Returns:
            Feedforward module
        
        Example:
            >>> config = {
            ...     'dim': 512,
            ...     'hidden_dim': 2048,
            ...     'ffn_type': 'swiglu',
            ...     'dropout': 0.1
            ... }
            >>> ffn = FeedForwardFactory.create_from_dict(config)
        """
        config = FeedForwardConfig(**config_dict)
        return FeedForwardFactory.create(config)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_ffn_info(module: nn.Module) -> dict:
    """
    Get information about a feedforward module.
    
    Args:
        module: Feedforward module instance
    
    Returns:
        Dictionary with FFN info
    
    Example:
        >>> ffn = SwiGLU(dim=512, hidden_dim=2048)
        >>> info = get_ffn_info(ffn)
        >>> print(info['type'])  # 'SwiGLU'
        >>> print(info['expansion_ratio'])  # 4.0
    """
    if hasattr(module, 'get_info'):
        return module.get_info()
    
    # Fallback for modules without get_info method
    info = {
        'type': module.__class__.__name__,
    }
    
    if hasattr(module, 'dim'):
        info['dim'] = module.dim
    if hasattr(module, 'hidden_dim'):
        info['hidden_dim'] = module.hidden_dim
        if hasattr(module, 'dim'):
            info['expansion_ratio'] = module.hidden_dim / module.dim
    
    return info


def estimate_ffn_parameters(
    dim: int,
    hidden_dim: int,
    ffn_type: str = 'swiglu',
    sparsity: float = 0.0
) -> dict:
    """
    Estimate parameter count for feedforward network.
    
    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension
        ffn_type: Type of FFN ('swiglu' or 'standard')
        sparsity: Target sparsity (0.0 = dense)
    
    Returns:
        Dictionary with parameter counts
    
    Example:
        >>> params = estimate_ffn_parameters(
        ...     dim=512,
        ...     hidden_dim=2048,
        ...     ffn_type='swiglu',
        ...     sparsity=0.88
        ... )
        >>> print(f"Dense params: {params['dense_params']:,}")
        >>> print(f"Sparse params: {params['sparse_params']:,}")
        >>> print(f"Savings: {params['savings_percentage']:.1f}%")
    """
    if ffn_type.lower() == 'swiglu':
        # SwiGLU has 3 projections: gate, value, output
        # gate_proj: dim -> hidden_dim
        # value_proj: dim -> hidden_dim
        # output_proj: hidden_dim -> dim
        dense_params = 2 * (dim * hidden_dim) + (hidden_dim * dim)
        dense_params = 3 * dim * hidden_dim  # Simplified
    else:
        # Standard FFN has 2 projections
        # fc1: dim -> hidden_dim
        # fc2: hidden_dim -> dim
        dense_params = (dim * hidden_dim) + (hidden_dim * dim)
        dense_params = 2 * dim * hidden_dim  # Simplified
    
    # Apply sparsity
    sparse_params = int(dense_params * (1 - sparsity))
    
    # Calculate savings
    saved_params = dense_params - sparse_params
    savings_percentage = (saved_params / dense_params) * 100
    
    return {
        'dense_params': dense_params,
        'sparse_params': sparse_params,
        'saved_params': saved_params,
        'savings_percentage': savings_percentage,
        'sparsity': sparsity
    }


def compare_ffn_types(
    dim: int = 512,
    hidden_dim: int = 2048,
    batch_size: int = 2,
    seq_len: int = 128
) -> dict:
    """
    Compare different FFN types on the same input.
    
    Useful for understanding differences and performance characteristics.
    
    Args:
        dim: Model dimension
        hidden_dim: Hidden dimension
        batch_size: Batch size for test
        seq_len: Sequence length for test
    
    Returns:
        Dictionary with comparison results
    
    Example:
        >>> results = compare_ffn_types(dim=512, hidden_dim=2048)
        >>> for ffn_type, info in results.items():
        ...     print(f"{ffn_type}: {info['params']:,} params")
    """
    x = torch.randn(batch_size, seq_len, dim)
    
    results = {}
    
    # SwiGLU
    swiglu = SwiGLU(dim=dim, hidden_dim=hidden_dim)
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    with torch.no_grad():
        out_swiglu = swiglu(x)
    
    results['swiglu'] = {
        'params': swiglu_params,
        'output_shape': out_swiglu.shape,
        'output_mean': out_swiglu.mean().item(),
        'output_std': out_swiglu.std().item()
    }
    
    # Standard FFN with ReLU
    standard_relu = StandardFFN(dim=dim, hidden_dim=hidden_dim, activation='relu')
    standard_relu_params = sum(p.numel() for p in standard_relu.parameters())
    with torch.no_grad():
        out_relu = standard_relu(x)
    
    results['standard_relu'] = {
        'params': standard_relu_params,
        'output_shape': out_relu.shape,
        'output_mean': out_relu.mean().item(),
        'output_std': out_relu.std().item()
    }
    
    # Standard FFN with GELU
    standard_gelu = StandardFFN(dim=dim, hidden_dim=hidden_dim, activation='gelu')
    standard_gelu_params = sum(p.numel() for p in standard_gelu.parameters())
    with torch.no_grad():
        out_gelu = standard_gelu(x)
    
    results['standard_gelu'] = {
        'params': standard_gelu_params,
        'output_shape': out_gelu.shape,
        'output_mean': out_gelu.mean().item(),
        'output_std': out_gelu.std().item()
    }
    
    return results


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing feedforward.py module")
    print("="*70)
    
    # Test SwiGLU
    print("\n1. Testing SwiGLU...")
    swiglu = SwiGLU(dim=512, hidden_dim=2048, dropout=0.1)
    x = torch.randn(2, 128, 512)
    out = swiglu(x)
    
    assert out.shape == x.shape, "Shape mismatch!"
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in swiglu.parameters()):,}")
    
    info = swiglu.get_info()
    print(f"   Info: {info}")
    print(f"   ✅ SwiGLU working!")
    
    # Test StandardFFN
    print("\n2. Testing StandardFFN...")
    ffn_relu = StandardFFN(dim=512, hidden_dim=2048, activation='relu')
    ffn_gelu = StandardFFN(dim=512, hidden_dim=2048, activation='gelu')
    
    out_relu = ffn_relu(x)
    out_gelu = ffn_gelu(x)
    
    assert out_relu.shape == x.shape, "ReLU shape mismatch!"
    assert out_gelu.shape == x.shape, "GELU shape mismatch!"
    
    print(f"   ReLU output: {out_relu.shape}")
    print(f"   GELU output: {out_gelu.shape}")
    print(f"   ReLU params: {sum(p.numel() for p in ffn_relu.parameters()):,}")
    print(f"   GELU params: {sum(p.numel() for p in ffn_gelu.parameters()):,}")
    print(f"   ✅ StandardFFN working!")
    
    # Test FeedForwardFactory
    print("\n3. Testing FeedForwardFactory...")
    config_swiglu = FeedForwardConfig(
        dim=512,
        hidden_dim=2048,
        ffn_type='swiglu',
        dropout=0.1
    )
    ffn_factory = FeedForwardFactory.create(config_swiglu)
    out_factory = ffn_factory(x)
    
    print(f"   Factory created: {type(ffn_factory).__name__}")
    print(f"   Output: {out_factory.shape}")
    print(f"   ✅ FeedForwardFactory working!")
    
    # Test parameter estimation
    print("\n4. Testing parameter estimation...")
    params = estimate_ffn_parameters(
        dim=512,
        hidden_dim=2048,
        ffn_type='swiglu',
        sparsity=0.88
    )
    
    print(f"   Dense params: {params['dense_params']:,}")
    print(f"   Sparse params: {params['sparse_params']:,}")
    print(f"   Savings: {params['savings_percentage']:.1f}%")
    print(f"   ✅ Parameter estimation working!")
    
    # Test comparison
    print("\n5. Testing FFN comparison...")
    comparison = compare_ffn_types(dim=512, hidden_dim=2048)
    
    print("\n   FFN Type Comparison:")
    for ffn_type, info in comparison.items():
        print(f"   {ffn_type:15s}: {info['params']:8,} params, "
              f"mean={info['output_mean']:7.4f}, std={info['output_std']:7.4f}")
    print(f"   ✅ Comparison working!")
    
    # Test gradient flow
    print("\n6. Testing gradient flow...")
    swiglu_grad = SwiGLU(dim=512, hidden_dim=2048)
    x_grad = torch.randn(2, 128, 512, requires_grad=True)
    out_grad = swiglu_grad(x_grad)
    loss = out_grad.sum()
    loss.backward()
    
    assert x_grad.grad is not None, "No gradient computed!"
    grad_norm = x_grad.grad.norm().item()
    print(f"   Input gradient norm: {grad_norm:.4f}")
    print(f"   ✅ Gradient flow working!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.architecture.feedforward import SwiGLU, StandardFFN")
    print("  from ramanujan.architecture.feedforward import FeedForwardFactory")
    print("="*70)

