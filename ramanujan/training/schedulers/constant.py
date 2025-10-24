"""Constant learning rate scheduler (no changes)."""

from typing import Optional, List, Dict, Any
from torch.optim import Optimizer


class ConstantScheduler:
    """
    Constant learning rate scheduler (no LR changes).
    
    Useful as a baseline or when you don't want scheduling.
    
    Args:
        optimizer: PyTorch optimizer
    """
    
    component_type: str = 'scheduler'
    component_name: str = 'constant'
    
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._last_lr = self.base_lrs.copy()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            'base_lrs': self.base_lrs,
            '_last_lr': self._last_lr
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.base_lrs = state_dict['base_lrs']
        self._last_lr = state_dict['_last_lr']
    
    def step(self, epoch: Optional[int] = None) -> None:
        """No-op, learning rate stays constant."""
        pass
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates."""
        return self._last_lr
    
    def __repr__(self) -> str:
        return f"ConstantScheduler(lr={self.base_lrs})"