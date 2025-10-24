"""Learning rate schedulers for Candide framework."""

from .base import SchedulerComponent, SchedulerSpec
from .warmup import WarmupScheduler
from .cosine import CosineScheduler
from .constant import ConstantScheduler
from .factory import (
    create_scheduler,
    create_scheduler_from_config,
    create_scheduler_from_spec,
    SCHEDULER_REGISTRY
)


__all__ = [
    # Protocol and base
    'SchedulerComponent',
    'SchedulerSpec',
    
    # Scheduler implementations
    'WarmupScheduler',
    'CosineScheduler',
    'ConstantScheduler',
    
    # Factory functions
    'create_scheduler',
    'create_scheduler_from_config',
    'create_scheduler_from_spec',
    
    # Registry
    'SCHEDULER_REGISTRY',
]