"""Scheduler factory and registry."""

from typing import Dict, Any, Optional
from torch.optim import Optimizer
from .base import SchedulerComponent, SchedulerSpec
from .warmup import WarmupScheduler
from .cosine import CosineScheduler
from .constant import ConstantScheduler


SCHEDULER_REGISTRY = {
    'warmup': WarmupScheduler,
    'cosine': CosineScheduler,
    'constant': ConstantScheduler,
}


def create_scheduler(
    name: str,
    optimizer: Optimizer,
    **kwargs
) -> SchedulerComponent:
    """
    Create a scheduler by name.
    
    Args:
        name: Scheduler name ('warmup', 'cosine', 'constant')
        optimizer: PyTorch optimizer
        **kwargs: Scheduler-specific arguments
    
    Returns:
        Scheduler instance
    
    Example:
        >>> scheduler = create_scheduler(
        ...     'warmup',
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     decay_style='cosine'
        ... )
    """
    name_lower = name.lower().strip()
    
    if name_lower not in SCHEDULER_REGISTRY:
        available = ', '.join(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler: {name}. Available: {available}")
    
    scheduler_class = SCHEDULER_REGISTRY[name_lower]
    return scheduler_class(optimizer, **kwargs)


def create_scheduler_from_config(
    config: Dict[str, Any],
    optimizer: Optimizer
) -> SchedulerComponent:
    """
    Create scheduler from config dict.
    
    Args:
        config: Config with 'name' or 'type' and optional parameters
        optimizer: PyTorch optimizer
    
    Returns:
        Scheduler instance
    
    Example:
        >>> config = {
        ...     'name': 'warmup',
        ...     'warmup_steps': 1000,
        ...     'total_steps': 10000,
        ...     'decay_style': 'cosine'
        ... }
        >>> scheduler = create_scheduler_from_config(config, optimizer)
    """
    config = config.copy()
    
    # Support both 'name' and 'type'
    name = config.pop('name', None) or config.pop('type', None)
    if name is None:
        raise ValueError("Config must contain 'name' or 'type' key")
    
    return create_scheduler(name, optimizer, **config)


def create_scheduler_from_spec(
    spec: SchedulerSpec,
    optimizer: Optimizer
) -> SchedulerComponent:
    """
    Create scheduler from SchedulerSpec.
    
    Args:
        spec: SchedulerSpec instance
        optimizer: PyTorch optimizer
    
    Returns:
        Scheduler instance
    """
    config = spec.to_dict()
    name = config.pop('name')
    return create_scheduler(name, optimizer, **config)