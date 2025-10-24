"""
Optimizer factory and creation utilities.

This module provides functions for creating optimizers from config
and managing optimizer instances.
"""

from typing import Union, Dict, Any, List, Iterable, Optional
import torch
from torch.optim import Optimizer

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


# ============================================================================
# OPTIMIZER REGISTRY
# ============================================================================

OPTIMIZER_REGISTRY = {
    'adamw': AdamWOptimizer,
    'adam_w': AdamWOptimizer,
    'muon': MuonOptimizer,
    'ademamix': AdEMAMixOptimizer,
    'adema_mix': AdEMAMixOptimizer,
    'sgd': SGDOptimizer,
    'lion': LionOptimizer,
}


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_optimizer(
    name: str,
    params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip_type: Optional[str] = None,
    grad_clip_value: float = 1.0,
    **kwargs
) -> OptimizerComponent:
    """
    Create an optimizer by name.
    
    Args:
        name: Optimizer name ('adamw', 'muon', 'ademamix', 'sgd', 'lion')
        params: Model parameters or parameter groups
        lr: Learning rate
        weight_decay: Weight decay coefficient
        grad_clip_type: Type of gradient clipping ('norm', 'value', 'adaptive')
        grad_clip_value: Maximum value for gradient clipping
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        Optimizer wrapped with OptimizerComponent interface
    
    Example:
        >>> # Simple usage
        >>> optimizer = create_optimizer('adamw', model.parameters(), lr=1e-3)
        >>> 
        >>> # With parameter groups
        >>> param_groups = [
        ...     {'params': model.embedding.parameters(), 'lr': 1e-4},
        ...     {'params': model.transformer.parameters(), 'lr': 1e-3}
        ... ]
        >>> optimizer = create_optimizer('muon', param_groups)
        >>> 
        >>> # With gradient clipping
        >>> optimizer = create_optimizer(
        ...     'adamw',
        ...     model.parameters(),
        ...     lr=1e-3,
        ...     grad_clip_type='norm',
        ...     grad_clip_value=1.0
        ... )
    """
    name_lower = name.lower().strip()
    
    if name_lower not in OPTIMIZER_REGISTRY:
        available = ', '.join(OPTIMIZER_REGISTRY.keys())
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Available optimizers: {available}"
        )
    
    optimizer_class = OPTIMIZER_REGISTRY[name_lower]
    
    # Create optimizer with appropriate parameters
    if name_lower in ['adamw', 'adam_w']:
        optimizer = optimizer_class(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            amsgrad=kwargs.get('amsgrad', False),
            foreach=kwargs.get('foreach', True),
            fused=kwargs.get('fused', False)
        )
    
    elif name_lower == 'muon':
        optimizer = optimizer_class(
            params,
            lr=lr if lr != 1e-3 else 0.02,  # Default to 0.02 for Muon
            momentum=kwargs.get('momentum', 0.95),
            weight_decay=weight_decay,
            ns_steps=kwargs.get('ns_steps', 5),
            dampening=kwargs.get('dampening', 0.0),
            backend=kwargs.get('backend', 'newtonschulz5'),
            foreach=kwargs.get('foreach', True)
        )
    
    elif name_lower in ['ademamix', 'adema_mix']:
        optimizer = optimizer_class(
            params,
            lr=lr,
            beta1=kwargs.get('beta1', 0.9),
            beta2=kwargs.get('beta2', 0.999),
            beta3=kwargs.get('beta3', 0.9999),
            alpha=kwargs.get('alpha', 5.0),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay,
            foreach=kwargs.get('foreach', True)
        )
    
    elif name_lower == 'sgd':
        optimizer = optimizer_class(
            params,
            lr=lr if lr != 1e-3 else 0.01,  # Default to 0.01 for SGD
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
            dampening=kwargs.get('dampening', 0.0),
            nesterov=kwargs.get('nesterov', False),
            foreach=kwargs.get('foreach', True)
        )
    
    elif name_lower == 'lion':
        optimizer = optimizer_class(
            params,
            lr=lr if lr != 1e-3 else 1e-4,  # Default to 1e-4 for Lion
            betas=kwargs.get('betas', (0.9, 0.99)),
            weight_decay=weight_decay,
            foreach=kwargs.get('foreach', True)
        )
    
    # Wrap optimizer with component interface
    wrapped = BaseOptimizerWrapper(
        optimizer,
        component_name=name_lower,
        grad_clip_type=grad_clip_type,
        grad_clip_value=grad_clip_value
    )
    
    return wrapped


def create_optimizer_from_config(
    config: Dict[str, Any],
    params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]]
) -> OptimizerComponent:
    """
    Create optimizer from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'name' key and optional parameters
        params: Model parameters or parameter groups
    
    Returns:
        Optimizer wrapped with OptimizerComponent interface
    
    Example:
        >>> config = {
        ...     'name': 'adamw',
        ...     'lr': 1e-3,
        ...     'weight_decay': 0.01,
        ...     'betas': [0.9, 0.999],
        ...     'grad_clip': {
        ...         'type': 'norm',
        ...         'value': 1.0
        ...     }
        ... }
        >>> optimizer = create_optimizer_from_config(config, model.parameters())
    """
    config = config.copy()
    
    # Extract name
    name = config.pop('name')
    
    # Extract gradient clipping config
    grad_clip = config.pop('grad_clip', None)
    grad_clip_type = None
    grad_clip_value = 1.0
    
    if grad_clip:
        grad_clip_type = grad_clip.get('type')
        grad_clip_value = grad_clip.get('value', 1.0)
    
    # Create optimizer
    return create_optimizer(
        name,
        params,
        grad_clip_type=grad_clip_type,
        grad_clip_value=grad_clip_value,
        **config
    )


def create_optimizer_from_spec(
    spec: OptimizerSpec,
    params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]]
) -> OptimizerComponent:
    """
    Create optimizer from OptimizerSpec.
    
    Args:
        spec: OptimizerSpec instance
        params: Model parameters or parameter groups
    
    Returns:
        Optimizer wrapped with OptimizerComponent interface
    
    Example:
        >>> spec = OptimizerSpec(
        ...     name='muon',
        ...     lr=0.02,
        ...     momentum=0.95,
        ...     weight_decay=0.1
        ... )
        >>> optimizer = create_optimizer_from_spec(spec, model.parameters())
    """
    return create_optimizer_from_config(spec.to_dict(), params)


# ============================================================================
# PARAMETER GROUP HELPERS
# ============================================================================

def create_optimizer_with_param_groups(
    name: str,
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float = 0.0,
    rules: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs
) -> OptimizerComponent:
    """
    Create optimizer with automatic parameter grouping.
    
    Args:
        name: Optimizer name
        model: PyTorch model
        base_lr: Base learning rate
        weight_decay: Base weight decay
        rules: Dictionary mapping parameter name patterns to hyperparameters
               Example: {
                   'bias': {'weight_decay': 0.0},
                   'norm': {'weight_decay': 0.0},
                   'embedding': {'lr': base_lr * 0.1}
               }
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer with parameter groups
    
    Example:
        >>> optimizer = create_optimizer_with_param_groups(
        ...     'adamw',
        ...     model,
        ...     base_lr=1e-3,
        ...     rules={
        ...         'bias': {'weight_decay': 0.0},
        ...         'norm': {'weight_decay': 0.0}
        ...     }
        ... )
    """
    # Create parameter groups
    param_groups = create_param_groups(
        model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        rules=rules
    )
    
    # Create optimizer
    return create_optimizer(
        name,
        param_groups,
        lr=base_lr,
        weight_decay=weight_decay,
        **kwargs
    )


def create_optimizer_with_llrd(
    name: str,
    model: torch.nn.Module,
    base_lr: float,
    decay_rate: float = 0.95,
    num_layers: Optional[int] = None,
    **kwargs
) -> OptimizerComponent:
    """
    Create optimizer with layer-wise learning rate decay (LLRD).
    
    Useful for fine-tuning: lower layers get smaller learning rates.
    
    Args:
        name: Optimizer name
        model: PyTorch model
        base_lr: Learning rate for top layer
        decay_rate: LR decay per layer (0.95 = 5% decay per layer)
        num_layers: Number of layers (auto-detected if None)
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer with layer-wise learning rates
    
    Example:
        >>> # Top layer gets base_lr, each lower layer gets 0.95x
        >>> optimizer = create_optimizer_with_llrd(
        ...     'adamw',
        ...     model,
        ...     base_lr=1e-3,
        ...     decay_rate=0.95
        ... )
    """
    # Create layer-wise parameter groups
    param_groups = get_layer_wise_lr_groups(
        model,
        base_lr=base_lr,
        decay_rate=decay_rate,
        num_layers=num_layers
    )
    
    # Create optimizer
    return create_optimizer(
        name,
        param_groups,
        lr=base_lr,
        **kwargs
    )


# ============================================================================
# OPTIMIZER UTILITIES
# ============================================================================

def get_optimizer_info(optimizer: Union[Optimizer, OptimizerComponent]) -> Dict[str, Any]:
    """
    Get information about an optimizer.
    
    Args:
        optimizer: Optimizer instance
    
    Returns:
        Dictionary containing optimizer information
    
    Example:
        >>> info = get_optimizer_info(optimizer)
        >>> print(info['type'], info['lr'], info['num_param_groups'])
    """
    # Unwrap if needed
    if isinstance(optimizer, BaseOptimizerWrapper):
        actual_optimizer = optimizer.optimizer
        component_name = optimizer.component_name
    else:
        actual_optimizer = optimizer
        component_name = type(optimizer).__name__
    
    return {
        'type': component_name,
        'num_param_groups': len(actual_optimizer.param_groups),
        'lr': [group['lr'] for group in actual_optimizer.param_groups],
        'weight_decay': [group.get('weight_decay', 0.0) for group in actual_optimizer.param_groups]
    }


def get_learning_rates(optimizer: Union[Optimizer, OptimizerComponent]) -> List[float]:
    """
    Get current learning rates from optimizer.
    
    Args:
        optimizer: Optimizer instance
    
    Returns:
        List of learning rates for each parameter group
    
    Example:
        >>> lrs = get_learning_rates(optimizer)
        >>> print(f"Learning rates: {lrs}")
    """
    if isinstance(optimizer, BaseOptimizerWrapper):
        return optimizer.get_last_lr()
    else:
        return [group['lr'] for group in optimizer.param_groups]


def set_learning_rate(
    optimizer: Union[Optimizer, OptimizerComponent],
    lr: float,
    param_group_idx: Optional[int] = None
) -> None:
    """
    Set learning rate for optimizer.
    
    Args:
        optimizer: Optimizer instance
        lr: New learning rate
        param_group_idx: If specified, only set LR for this group
    
    Example:
        >>> # Set LR for all groups
        >>> set_learning_rate(optimizer, 1e-4)
        >>> 
        >>> # Set LR for specific group
        >>> set_learning_rate(optimizer, 1e-4, param_group_idx=0)
    """
    # Unwrap if needed
    if isinstance(optimizer, BaseOptimizerWrapper):
        actual_optimizer = optimizer.optimizer
    else:
        actual_optimizer = optimizer
    
    if param_group_idx is not None:
        actual_optimizer.param_groups[param_group_idx]['lr'] = lr
    else:
        for group in actual_optimizer.param_groups:
            group['lr'] = lr