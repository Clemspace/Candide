"""
Base classes and protocols for optimizers in Candide framework.

This module defines the OptimizerComponent protocol that all optimizers
must implement, similar to the LossComponent protocol.
"""

from typing import Protocol, Dict, List, Optional, Any, Callable, Iterable
import torch
from torch.optim import Optimizer
from dataclasses import dataclass


# ============================================================================
# OPTIMIZER PROTOCOL
# ============================================================================

class OptimizerComponent(Protocol):
    """
    Protocol for optimizer components in Candide framework.
    
    All optimizers must implement this interface to be compatible with
    the training system.
    """
    
    component_type: str = 'optimizer'
    component_name: str
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure for computing the loss
            
        Returns:
            Loss value if closure is provided
        """
        ...
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Zero out gradients.
        
        Args:
            set_to_none: If True, set gradients to None instead of zero
                        (more memory efficient)
        """
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get optimizer state for checkpointing.
        
        Returns:
            Dictionary containing optimizer state
        """
        ...
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load optimizer state from checkpoint.
        
        Args:
            state_dict: Dictionary containing optimizer state
        """
        ...
    
    def get_last_lr(self) -> List[float]:
        """
        Get last learning rates for logging.
        
        Returns:
            List of learning rates for each parameter group
        """
        ...


# ============================================================================
# OPTIMIZER SPECIFICATION
# ============================================================================

@dataclass
class OptimizerSpec:
    """
    Specification for creating an optimizer.
    
    This class holds all configuration needed to instantiate an optimizer,
    similar to LossSpec for losses.
    
    Example:
        >>> spec = OptimizerSpec(
        ...     name='muon',
        ...     lr=1e-3,
        ...     momentum=0.95,
        ...     weight_decay=0.1
        ... )
    """
    
    name: str
    lr: float = 1e-3
    weight_decay: float = 0.0
    
    # Common optimizer parameters
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False
    
    # Muon specific
    ns_steps: int = 5
    backend: str = 'newtonschulz5'
    
    # AdEMAMix specific
    beta1: float = 0.9
    beta2: float = 0.999
    beta3: float = 0.9999
    alpha: float = 5.0
    
    # Lion specific
    use_triton: bool = False
    
    # Sophia specific
    rho: float = 0.04
    
    # Gradient clipping
    grad_clip_type: Optional[str] = None  # 'norm', 'value', 'adaptive'
    grad_clip_value: float = 1.0
    
    # Advanced features
    foreach: bool = True  # Faster multi-tensor operations
    fused: bool = False   # Fused kernel (if available)
    
    # Additional kwargs
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize config dict if not provided."""
        if self.config is None:
            self.config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        return {
            'name': self.name,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'eps': self.eps,
            'momentum': self.momentum,
            'dampening': self.dampening,
            'nesterov': self.nesterov,
            'ns_steps': self.ns_steps,
            'backend': self.backend,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'beta3': self.beta3,
            'alpha': self.alpha,
            'use_triton': self.use_triton,
            'rho': self.rho,
            'grad_clip_type': self.grad_clip_type,
            'grad_clip_value': self.grad_clip_value,
            'foreach': self.foreach,
            'fused': self.fused,
            **self.config
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'OptimizerSpec':
        """Create spec from dictionary."""
        # Extract known fields
        known_fields = {
            'name', 'lr', 'weight_decay', 'betas', 'eps', 'momentum',
            'dampening', 'nesterov', 'ns_steps', 'backend', 'beta1',
            'beta2', 'beta3', 'alpha', 'use_triton', 'rho',
            'grad_clip_type', 'grad_clip_value', 'foreach', 'fused'
        }
        
        spec_kwargs = {k: v for k, v in config.items() if k in known_fields}
        extra_kwargs = {k: v for k, v in config.items() if k not in known_fields}
        
        if extra_kwargs:
            spec_kwargs['config'] = extra_kwargs
        
        return cls(**spec_kwargs)


# ============================================================================
# BASE OPTIMIZER WRAPPER
# ============================================================================

class BaseOptimizerWrapper:
    """
    Base wrapper for PyTorch optimizers to add OptimizerComponent interface.
    
    This wrapper adds:
    - Component type/name attributes
    - get_last_lr() method
    - Gradient clipping support
    - Consistent interface across all optimizers
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        component_name: str,
        grad_clip_type: Optional[str] = None,
        grad_clip_value: float = 1.0
    ):
        """
        Initialize wrapper.
        
        Args:
            optimizer: PyTorch optimizer to wrap
            component_name: Name of the optimizer (e.g., 'adamw', 'muon')
            grad_clip_type: Type of gradient clipping ('norm', 'value', 'adaptive')
            grad_clip_value: Maximum value for gradient clipping
        """
        self.optimizer = optimizer
        self.component_type = 'optimizer'
        self.component_name = component_name
        self.grad_clip_type = grad_clip_type
        self.grad_clip_value = grad_clip_value
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform optimization step with optional gradient clipping."""
        # Apply gradient clipping if configured
        if self.grad_clip_type is not None:
            self._clip_gradients()
        
        # Perform optimizer step
        return self.optimizer.step(closure)
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def _clip_gradients(self) -> None:
        """Apply gradient clipping based on configured type."""
        # Get all parameters from all param groups
        params = []
        for group in self.optimizer.param_groups:
            params.extend(group['params'])
        
        if self.grad_clip_type == 'norm':
            # Clip by global norm
            torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=self.grad_clip_value,
                norm_type=2.0
            )
        
        elif self.grad_clip_type == 'value':
            # Clip by value
            torch.nn.utils.clip_grad_value_(
                params,
                clip_value=self.grad_clip_value
            )
        
        elif self.grad_clip_type == 'adaptive':
            # Adaptive clipping based on parameter norm
            for p in params:
                if p.grad is not None:
                    param_norm = p.norm()
                    grad_norm = p.grad.norm()
                    if grad_norm > self.grad_clip_value * param_norm:
                        p.grad.mul_(self.grad_clip_value * param_norm / (grad_norm + 1e-8))
    
    @property
    def param_groups(self):
        """Access parameter groups."""
        return self.optimizer.param_groups
    
    @property
    def state(self):
        """Access optimizer state."""
        return self.optimizer.state
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.component_name}, groups={len(self.param_groups)})"


# ============================================================================
# PARAMETER GROUPING UTILITIES
# ============================================================================

def create_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float = 0.0,
    rules: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different hyperparameters.
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate
        weight_decay: Base weight decay
        rules: Dictionary mapping parameter name patterns to hyperparameters
               Example: {
                   'bias': {'weight_decay': 0.0},
                   'norm': {'weight_decay': 0.0},
                   'embedding': {'lr': base_lr * 0.1}
               }
    
    Returns:
        List of parameter group dictionaries
    
    Example:
        >>> param_groups = create_param_groups(
        ...     model,
        ...     base_lr=1e-3,
        ...     rules={
        ...         'bias': {'weight_decay': 0.0},
        ...         'embedding': {'lr': 1e-4}
        ...     }
        ... )
    """
    if rules is None:
        # Default: single group with all parameters
        return [{
            'params': model.parameters(),
            'lr': base_lr,
            'weight_decay': weight_decay
        }]
    
    # Categorize parameters
    param_dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Find matching rule
        matched_rule = None
        for pattern, rule_config in rules.items():
            if pattern in name.lower():
                matched_rule = pattern
                break
        
        # Add to appropriate group
        if matched_rule not in param_dict:
            param_dict[matched_rule] = []
        param_dict[matched_rule].append(param)
    
    # Create param groups
    param_groups = []
    
    # Add groups with specific rules
    for pattern, params in param_dict.items():
        if pattern is None:
            continue
        
        group_config = {
            'params': params,
            'lr': base_lr,
            'weight_decay': weight_decay
        }
        
        # Apply rule overrides
        if pattern in rules:
            group_config.update(rules[pattern])
        
        param_groups.append(group_config)
    
    # Add default group for unmatched params
    if None in param_dict:
        param_groups.append({
            'params': param_dict[None],
            'lr': base_lr,
            'weight_decay': weight_decay
        })
    
    return param_groups


def get_layer_wise_lr_groups(
    model: torch.nn.Module,
    base_lr: float,
    decay_rate: float = 0.95,
    num_layers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with layer-wise learning rate decay.
    
    Useful for fine-tuning: lower layers get smaller learning rates.
    
    Args:
        model: PyTorch model
        base_lr: Learning rate for top layer
        decay_rate: LR decay per layer (0.95 = 5% decay per layer)
        num_layers: Number of layers (auto-detected if None)
    
    Returns:
        List of parameter groups with decaying learning rates
    
    Example:
        >>> # Top layer gets base_lr, each lower layer gets 0.95x of previous
        >>> param_groups = get_layer_wise_lr_groups(
        ...     model,
        ...     base_lr=1e-3,
        ...     decay_rate=0.95
        ... )
    """
    # Group parameters by layer
    layer_params = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Extract layer number from name (assuming format like 'layer.0.weight')
        parts = name.split('.')
        layer_id = None
        
        for part in parts:
            if part.isdigit():
                layer_id = int(part)
                break
        
        if layer_id is None:
            layer_id = 0  # Default to layer 0
        
        if layer_id not in layer_params:
            layer_params[layer_id] = []
        layer_params[layer_id].append(param)
    
    # Create param groups with decaying LRs
    if num_layers is None:
        num_layers = max(layer_params.keys()) + 1
    
    param_groups = []
    for layer_id in sorted(layer_params.keys(), reverse=True):
        # Higher layers get higher LR
        lr = base_lr * (decay_rate ** (num_layers - layer_id - 1))
        
        param_groups.append({
            'params': layer_params[layer_id],
            'lr': lr
        })
    
    return param_groups