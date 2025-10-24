"""Loss factory and registry."""

from typing import Dict, Any, Union
from .base import LossComponent, LossSpec
from .cross_entropy import CrossEntropyLoss
from .semantic_entropy import SemanticEntropyProbe
from .kl_divergence import KLDivergenceLoss
from .composite import CompositeLoss


LOSS_REGISTRY = {
    'cross_entropy': CrossEntropyLoss,
    'semantic_entropy': SemanticEntropyProbe,
    'kl_divergence': KLDivergenceLoss,
    'composite': CompositeLoss,
}


def create_loss(name: str, config: Dict[str, Any] = None, **kwargs) -> LossComponent:
    """
    Create a loss function by name.
    
    Args:
        name: Loss name
        config: Optional config dict (alternative to **kwargs)
        **kwargs: Loss-specific arguments
    
    Returns:
        Loss instance
    
    Example:
        >>> loss_fn = create_loss('cross_entropy', vocab_size=32000)
        >>> # Or with config dict
        >>> loss_fn = create_loss('cross_entropy', {'vocab_size': 32000})
    """
    name_lower = name.lower().strip()
    
    if name_lower not in LOSS_REGISTRY:
        available = ', '.join(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss: {name}. Available: {available}")
    
    loss_class = LOSS_REGISTRY[name_lower]
    
    # Support both config dict and **kwargs
    if config is not None:
        kwargs.update(config)
    
    return loss_class(**kwargs)


def create_loss_from_config(config: Dict[str, Any]) -> LossComponent:
    """
    Create loss from config dict.
    
    Args:
        config: Config with 'name' or 'type' and optional parameters
    
    Returns:
        Loss instance
    
    Example:
        >>> config = {'name': 'cross_entropy', 'vocab_size': 32000}
        >>> loss_fn = create_loss_from_config(config)
        >>> # Or with 'type' instead of 'name'
        >>> config = {'type': 'cross_entropy', 'config': {'vocab_size': 32000}}
        >>> loss_fn = create_loss_from_config(config)
    """
    config = config.copy()
    
    # Support both 'name' and 'type'
    name = config.pop('name', None) or config.pop('type', None)
    if name is None:
        raise ValueError("Config must contain 'name' or 'type' key")
    
    # Handle nested 'config' key (e.g., {'type': 'ce', 'config': {...}})
    nested_config = config.pop('config', {})
    
    # Handle 'losses' key for composite
    if 'losses' in config:
        nested_config['loss_specs'] = [
            LossSpec(
                name=loss_config['name'],
                weight=loss_config.get('weight', 1.0),
                config=loss_config.get('config', {})
            )
            for loss_config in config.pop('losses')
        ]
    
    # Merge all configs
    nested_config.update(config)
    
    return create_loss(name, nested_config)


def create_loss_from_spec(spec: LossSpec) -> LossComponent:
    """
    Create loss from LossSpec.
    
    Args:
        spec: LossSpec instance
    
    Returns:
        Loss instance
    """
    return create_loss(spec.name, **spec.config)