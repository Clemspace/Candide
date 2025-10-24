"""Base classes and protocols for learning rate schedulers."""

from typing import Protocol, Dict, Any, Optional, List
from dataclasses import dataclass, field
from torch.optim import Optimizer


class SchedulerComponent(Protocol):
    """Protocol for scheduler components."""
    
    component_type: str = 'scheduler'
    component_name: str
    
    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate."""
        ...
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates."""
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        ...
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        ...


@dataclass
class SchedulerSpec:
    """Specification for creating a scheduler."""
    
    name: str
    warmup_steps: int = 0
    total_steps: int = 1000
    min_lr: float = 0.0
    warmup_init_lr: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'warmup_init_lr': self.warmup_init_lr,
            **self.config
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SchedulerSpec':
        """Create from dictionary."""
        known_fields = {'name', 'warmup_steps', 'total_steps', 'min_lr', 'warmup_init_lr'}
        spec_kwargs = {k: v for k, v in d.items() if k in known_fields}
        config_kwargs = {k: v for k, v in d.items() if k not in known_fields}
        if config_kwargs:
            spec_kwargs['config'] = config_kwargs
        return cls(**spec_kwargs)