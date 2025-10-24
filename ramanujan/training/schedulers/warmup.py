"""Warmup scheduler with linear warmup and optional decay."""

import math
from typing import Optional, List, Dict, Any
from torch.optim import Optimizer


class WarmupScheduler:
    """
    Learning rate scheduler with warmup and optional decay.
    
    Supports:
    - Linear warmup
    - Constant LR after warmup
    - Linear decay after warmup
    - Cosine decay after warmup
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate (default: 0.0)
        warmup_init_lr: Initial warmup LR (default: 0.0)
        decay_style: Decay after warmup ('constant', 'linear', 'cosine')
    """
    
    component_type: str = 'scheduler'
    component_name: str = 'warmup'
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_init_lr: float = 0.0,
        decay_style: str = 'cosine'
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_init_lr = warmup_init_lr
        self.decay_style = decay_style
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._step_count = 0
        self._last_lr = self.base_lrs.copy()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'warmup_init_lr': self.warmup_init_lr,
            'decay_style': self.decay_style,
            'base_lrs': self.base_lrs,
            '_step_count': self._step_count,
            '_last_lr': self._last_lr
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        self.warmup_init_lr = state_dict['warmup_init_lr']
        self.decay_style = state_dict['decay_style']
        self.base_lrs = state_dict['base_lrs']
        self._step_count = state_dict['_step_count']
        self._last_lr = state_dict['_last_lr']
    
    def get_lr_factor(self, step: int) -> float:
        """
        Compute learning rate factor for given step.
        
        Args:
            step: Current training step
        
        Returns:
            Learning rate multiplier
        """
        if step < self.warmup_steps:
            # Linear warmup
            if self.warmup_steps == 0:
                return 1.0
            return (step / self.warmup_steps)
        
        # After warmup
        if self.decay_style == 'constant':
            return 1.0
        
        elif self.decay_style == 'linear':
            # Linear decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return max(0.0, 1.0 - progress)
        
        elif self.decay_style == 'cosine':
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        
        else:
            raise ValueError(f"Unknown decay_style: {self.decay_style}")
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Update learning rate.
        
        Args:
            epoch: Optional epoch number (not used, for API compatibility)
        """
        self._step_count += 1
        
        # Compute learning rates
        lr_factor = self.get_lr_factor(self._step_count)
        
        new_lrs = []
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            # During warmup, interpolate from warmup_init_lr to base_lr
            if self._step_count <= self.warmup_steps:
                if self.warmup_steps == 0:
                    new_lr = base_lr
                else:
                    warmup_progress = self._step_count / self.warmup_steps
                    new_lr = self.warmup_init_lr + (base_lr - self.warmup_init_lr) * warmup_progress
            else:
                # After warmup, apply decay
                new_lr = self.min_lr + (base_lr - self.min_lr) * lr_factor
            
            param_group['lr'] = new_lr
            new_lrs.append(new_lr)
        
        self._last_lr = new_lrs
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates."""
        return self._last_lr
    
    def __repr__(self) -> str:
        return (
            f"WarmupScheduler("
            f"warmup_steps={self.warmup_steps}, "
            f"total_steps={self.total_steps}, "
            f"decay_style={self.decay_style})"
        )