"""
Ramanujan Foundation: Graph-theoretic sparsity for neural networks.

This module provides the mathematical foundation for creating structured
sparse neural network layers based on Ramanujan graphs.

Components:
- RamanujanMath: Mathematical computations (primes, Legendre symbols, etc.)
- RamanujanGraphBuilder: LPS and bi-regular graph construction
- RamanujanLinearLayer: Sparse linear layers with Ramanujan structure
- RamanujanFoundation: Main interface for creating sparse layers

Example:
    >>> foundation = RamanujanFoundation(max_prime=1000)
    >>> layer = foundation.create_layer(
    ...     in_features=512,
    ...     out_features=512,
    ...     target_sparsity=0.85,
    ...     force_method="lps"
    ... )
    >>> print(f"Actual sparsity: {layer.get_info()['actual_sparsity']:.2%}")
"""

from .math_core import RamanujanMath
from .graph_builder import RamanujanGraphBuilder
from .sparse_layers import RamanujanLinearLayer, RamanujanFoundation

__all__ = [
    'RamanujanMath',
    'RamanujanGraphBuilder',
    'RamanujanLinearLayer',
    'RamanujanFoundation',
]