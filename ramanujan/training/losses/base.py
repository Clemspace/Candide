"""Base classes and protocols for losses."""

from typing import Protocol, Tuple, Dict, Any
import torch
from dataclasses import dataclass, field


class LossComponent(Protocol):
    """Protocol for loss components."""
    
    component_type: str = 'loss'
    component_name: str
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.
        
        Returns:
            loss: Scalar tensor
            metrics: Dictionary of metrics
        """
        ...


@dataclass
class LossSpec:
    """Specification for creating a loss."""
    
    name: str
    weight: float = 1.0
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'weight': self.weight,
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LossSpec':
        """Create from dictionary."""
        return cls(
            name=d['name'],
            weight=d.get('weight', 1.0),
            config=d.get('config', {})
        )