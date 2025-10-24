"""Cosine annealing scheduler with warmup and restarts."""

import math
from typing import Optional, List, Dict, Any
from torch.optim import Optimizer


class CosineScheduler:
    """
    Cosine annealing learning rate scheduler with optional warmup.
    
    Args:
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        warmup_steps: Number of warmup steps (default: 0)
        min_lr: Minimum learning rate (default: 0.0)
        warmup_init_lr: Initial warmup LR (default: 0.0)
        num_cycles: Number of cosine cycles (default: 1)
    """
    
    component_type: str = 'scheduler'
    component_name: str = 'cosine'
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        warmup_init_lr: float = 0.0,
        num_cycles: int = 1
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.warmup_init_lr = warmup_init_lr
        self.num_cycles = num_cycles
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._step_count = 0
        self._last_lr = self.base_lrs.copy()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'min_lr': self.min_lr,
            'warmup_init_lr': self.warmup_init_lr,
            'num_cycles': self.num_cycles,
            'base_lrs': self.base_lrs,
            '_step_count': self._step_count,
            '_last_lr': self._last_lr
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.total_steps = state_dict['total_steps']
        self.warmup_steps = state_dict['warmup_steps']
        self.min_lr = state_dict['min_lr']
        self.warmup_init_lr = state_dict['warmup_init_lr']
        self.num_cycles = state_dict['num_cycles']
        self.base_lrs = state_dict['base_lrs']
        self._step_count = state_dict['_step_count']
        self._last_lr = state_dict['_last_lr']
    
    def get_lr_factor(self, step: int) -> float:
        """
        Compute cosine learning rate factor.
        
        Args:
            step: Current training step
        
        Returns:
            Learning rate multiplier
        """
        if step < self.warmup_steps:
            # Linear warmup
            if self.warmup_steps == 0:
                return 1.0
            return step / self.warmup_steps
        
        # Cosine annealing after warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        # Apply cycles
        cosine_arg = math.pi * (progress * self.num_cycles % 1.0)
        return 0.5 * (1.0 + math.cos(cosine_arg))
    
    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate."""
        self._step_count += 1
        
        lr_factor = self.get_lr_factor(self._step_count)
        
        new_lrs = []
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            # During warmup
            if self._step_count <= self.warmup_steps:
                if self.warmup_steps == 0:
                    new_lr = base_lr
                else:
                    warmup_progress = self._step_count / self.warmup_steps
                    new_lr = self.warmup_init_lr + (base_lr - self.warmup_init_lr) * warmup_progress
            else:
                # After warmup, apply cosine
                new_lr = self.min_lr + (base_lr - self.min_lr) * lr_factor
            
            param_group['lr'] = new_lr
            new_lrs.append(new_lr)
        
        self._last_lr = new_lrs
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates."""
        return self._last_lr
    
    def __repr__(self) -> str:
        return (
            f"CosineScheduler("
            f"total_steps={self.total_steps}, "
            f"warmup_steps={self.warmup_steps}, "
            f"num_cycles={self.num_cycles})"
        )