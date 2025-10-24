"""
Optimizers for Candide framework.

This module provides various optimization algorithms with a unified interface:
- AdamW: Adam with decoupled weight decay
- Muon: Momentum with Newton-Schulz orthogonalization
- AdEMAMix: Adaptive EMA mixing
- SGD: Stochastic Gradient Descent with momentum
- Lion: Memory-efficient optimizer

Example:
    >>> from ramanujan.training.optimizers import create_optimizer
    >>> 
    >>> # Simple usage
    >>> optimizer = create_optimizer('adamw', model.parameters(), lr=1e-3)
    >>> 
    >>> # With parameter groups
    >>> optimizer = create_optimizer_with_param_groups(
    ...     'muon',
    ...     model,
    ...     base_lr=0.02,
    ...     rules={'bias': {'weight_decay': 0.0}}
    ... )
    >>> 
    >>> # From config
    >>> config = {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.01}
    >>> optimizer = create_optimizer_from_config(config, model.parameters())
"""

from .base import (
    OptimizerComponent,
    OptimizerSpec,
    BaseOptimizerWrapper,
    create_param_groups,
    get_layer_wise_lr_groups
)

from .muon import MuonOptimizer
from .ademamix import AdEMAMixOptimizer
from .standard import AdamWOptimizer, SGDOptimizer, LionOptimizer

from .factory import (
    create_optimizer,
    create_optimizer_from_config,
    create_optimizer_from_spec,
    create_optimizer_with_param_groups,
    create_optimizer_with_llrd,
    get_optimizer_info,
    get_learning_rates,
    set_learning_rate,
    OPTIMIZER_REGISTRY
)


__all__ = [
    # Protocol and base classes
    'OptimizerComponent',
    'OptimizerSpec',
    'BaseOptimizerWrapper',
    
    # Optimizers
    'AdamWOptimizer',
    'MuonOptimizer',
    'AdEMAMixOptimizer',
    'SGDOptimizer',
    'LionOptimizer',
    
    # Factory functions
    'create_optimizer',
    'create_optimizer_from_config',
    'create_optimizer_from_spec',
    'create_optimizer_with_param_groups',
    'create_optimizer_with_llrd',
    
    # Utilities
    'get_optimizer_info',
    'get_learning_rates',
    'set_learning_rate',
    'create_param_groups',
    'get_layer_wise_lr_groups',
    
    # Registry
    'OPTIMIZER_REGISTRY',
]