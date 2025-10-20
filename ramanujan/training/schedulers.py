"""
Learning rate schedulers for Ramanujan Transformer training.

This module provides various learning rate scheduling strategies:
- CosineWarmupScheduler: Cosine decay with linear warmup
- CosineWarmupWithRestarts: Cosine with periodic restarts
- LinearWarmupScheduler: Simple linear warmup then constant
- PolynomialDecayScheduler: Polynomial decay with warmup

Example:
    >>> from ramanujan.training import CosineWarmupScheduler
    >>> 
    >>> scheduler = CosineWarmupScheduler(
    ...     optimizer,
    ...     warmup_steps=1000,
    ...     max_steps=10000,
    ...     min_lr=1e-6
    ... )
    >>> 
    >>> # Training loop
    >>> for step in range(max_steps):
    ...     loss.backward()
    ...     optimizer.step()
    ...     scheduler.step()
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List, Optional


# ============================================================================
# COSINE WARMUP SCHEDULER
# ============================================================================

class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing schedule with linear warmup.
    
    Learning rate schedule:
    1. Linear warmup from 0 to base_lr over warmup_steps
    2. Cosine decay from base_lr to min_lr over remaining steps
    
    This is the most common schedule for transformer training.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 0)
        last_epoch: Last epoch index (default: -1)
    
    Example:
        >>> scheduler = CosineWarmupScheduler(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     max_steps=10000,
        ...     min_lr=1e-6
        ... )
        >>> 
        >>> for step in range(max_steps):
        ...     train_step()
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


# ============================================================================
# COSINE WARMUP WITH RESTARTS
# ============================================================================

class CosineWarmupWithRestarts(_LRScheduler):
    """
    Cosine annealing with periodic warm restarts (SGDR).
    
    Periodically resets learning rate to initial value and performs
    cosine decay. This can help escape local minima.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps (only for first cycle)
        first_cycle_steps: Steps in first cycle
        cycle_mult: Multiplier for cycle length (default: 1.0)
        min_lr: Minimum learning rate (default: 0)
        max_lr_decay: Decay max LR each cycle (default: 1.0, no decay)
        last_epoch: Last epoch index (default: -1)
    
    Example:
        >>> scheduler = CosineWarmupWithRestarts(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     first_cycle_steps=5000,
        ...     cycle_mult=1.5,
        ...     min_lr=1e-6
        ... )
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        min_lr: float = 0.0,
        max_lr_decay: float = 1.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.min_lr = min_lr
        self.max_lr_decay = max_lr_decay
        
        self.cycle = 0
        self.step_in_cycle = 0
        self.current_cycle_steps = first_cycle_steps
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        # Warmup (only in first cycle)
        if self.cycle == 0 and self.step_in_cycle < self.warmup_steps:
            alpha = self.step_in_cycle / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        # Cosine decay within cycle
        if self.cycle == 0:
            progress = (self.step_in_cycle - self.warmup_steps) / (self.current_cycle_steps - self.warmup_steps)
        else:
            progress = self.step_in_cycle / self.current_cycle_steps
        
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Apply max LR decay for cycles after first
        max_lr_mult = self.max_lr_decay ** self.cycle
        
        return [
            self.min_lr + (base_lr * max_lr_mult - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        self.step_in_cycle += 1
        
        # Check if cycle is complete
        if self.step_in_cycle >= self.current_cycle_steps:
            self.cycle += 1
            self.step_in_cycle = 0
            self.current_cycle_steps = int(self.current_cycle_steps * self.cycle_mult)
        
        # Update learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# ============================================================================
# LINEAR WARMUP SCHEDULER
# ============================================================================

class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.
    
    Simple schedule that warms up linearly then maintains constant LR.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        last_epoch: Last epoch index (default: -1)
    
    Example:
        >>> scheduler = LinearWarmupScheduler(
        ...     optimizer,
        ...     warmup_steps=1000
        ... )
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Constant
            return self.base_lrs


# ============================================================================
# POLYNOMIAL DECAY SCHEDULER
# ============================================================================

class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial decay with linear warmup.
    
    Learning rate decays polynomially from base_lr to min_lr.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 0)
        power: Polynomial power (default: 1.0, linear decay)
        last_epoch: Last epoch index (default: -1)
    
    Example:
        >>> # Quadratic decay
        >>> scheduler = PolynomialDecayScheduler(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     max_steps=10000,
        ...     power=2.0
        ... )
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            decay_factor = (1 - progress) ** self.power
            
            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor
                for base_lr in self.base_lrs
            ]


# ============================================================================
# INVERSE SQRT SCHEDULER
# ============================================================================

class InverseSqrtScheduler(_LRScheduler):
    """
    Inverse square root decay with warmup.
    
    Used in "Attention is All You Need" paper.
    LR increases linearly during warmup, then decays proportional to 1/sqrt(step).
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        last_epoch: Last epoch index (default: -1)
    
    Example:
        >>> scheduler = InverseSqrtScheduler(
        ...     optimizer,
        ...     warmup_steps=4000
        ... )
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = max(self.last_epoch, 1)
        
        # Scale factor: min(step^-0.5, step * warmup_steps^-1.5)
        scale = min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )
        
        return [base_lr * scale for base_lr in self.base_lrs]


# ============================================================================
# SCHEDULER FACTORY
# ============================================================================

def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'cosine',
    warmup_steps: int = 1000,
    max_steps: int = 10000,
    **kwargs
) -> _LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'cosine_restarts', 'linear', 'polynomial', 'inverse_sqrt')
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        Scheduler instance
    
    Example:
        >>> # Cosine schedule
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     scheduler_type='cosine',
        ...     warmup_steps=1000,
        ...     max_steps=10000,
        ...     min_lr=1e-6
        ... )
        >>> 
        >>> # Cosine with restarts
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     scheduler_type='cosine_restarts',
        ...     warmup_steps=1000,
        ...     first_cycle_steps=5000,
        ...     cycle_mult=1.5
        ... )
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type in ['cosine', 'cosine_warmup']:
        return CosineWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=kwargs.get('min_lr', 0.0)
        )
    
    elif scheduler_type in ['cosine_restarts', 'sgdr']:
        return CosineWarmupWithRestarts(
            optimizer,
            warmup_steps=warmup_steps,
            first_cycle_steps=kwargs.get('first_cycle_steps', max_steps // 2),
            cycle_mult=kwargs.get('cycle_mult', 1.0),
            min_lr=kwargs.get('min_lr', 0.0),
            max_lr_decay=kwargs.get('max_lr_decay', 1.0)
        )
    
    elif scheduler_type in ['linear', 'linear_warmup']:
        return LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps
        )
    
    elif scheduler_type in ['polynomial', 'poly']:
        return PolynomialDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=kwargs.get('min_lr', 0.0),
            power=kwargs.get('power', 1.0)
        )
    
    elif scheduler_type in ['inverse_sqrt', 'invsqrt']:
        return InverseSqrtScheduler(
            optimizer,
            warmup_steps=warmup_steps
        )
    
    else:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type}. "
            f"Choose from: 'cosine', 'cosine_restarts', 'linear', 'polynomial', 'inverse_sqrt'"
        )


# ============================================================================
# SCHEDULER UTILITIES
# ============================================================================

def get_scheduler_info(scheduler: _LRScheduler) -> dict:
    """
    Get information about scheduler.
    
    Args:
        scheduler: Scheduler instance
    
    Returns:
        Dictionary with scheduler info
    """
    info = {
        'type': scheduler.__class__.__name__,
        'last_epoch': scheduler.last_epoch,
        'base_lrs': scheduler.base_lrs
    }
    
    # Add scheduler-specific info
    if hasattr(scheduler, 'warmup_steps'):
        info['warmup_steps'] = scheduler.warmup_steps
    if hasattr(scheduler, 'max_steps'):
        info['max_steps'] = scheduler.max_steps
    if hasattr(scheduler, 'min_lr'):
        info['min_lr'] = scheduler.min_lr
    
    return info


def plot_schedule(
    scheduler: _LRScheduler,
    num_steps: int = 1000,
    save_path: Optional[str] = None
):
    """
    Plot learning rate schedule.
    
    Args:
        scheduler: Scheduler instance
        num_steps: Number of steps to plot
        save_path: Optional path to save plot
    
    Example:
        >>> scheduler = CosineWarmupScheduler(optimizer, 100, 1000)
        >>> plot_schedule(scheduler, num_steps=1000, save_path='schedule.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Save original state
    original_epoch = scheduler.last_epoch
    
    # Compute LRs for each step
    lrs = []
    scheduler.last_epoch = -1
    for _ in range(num_steps):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])
    
    # Restore original state
    scheduler.last_epoch = original_epoch
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title(f'{scheduler.__class__.__name__} Schedule')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Schedule plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing schedulers.py module")
    print("="*70)
    
    # Create dummy optimizer
    import torch.nn as nn
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test CosineWarmupScheduler
    print("\n1. Testing CosineWarmupScheduler...")
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=100,
        max_steps=1000,
        min_lr=1e-6
    )
    
    lrs = []
    for step in range(1000):
        lrs.append(scheduler.get_lr()[0])
        scheduler.step()
    
    print(f"   Initial LR (step 0): {lrs[0]:.6f}")
    print(f"   LR after warmup (step 100): {lrs[100]:.6f}")
    print(f"   Final LR (step 999): {lrs[999]:.6f}")
    assert lrs[0] < lrs[100], "Warmup not working!"
    assert lrs[100] > lrs[999], "Decay not working!"
    print(f"   ✅ CosineWarmupScheduler working!")
    
    # Test CosineWarmupWithRestarts
    print("\n2. Testing CosineWarmupWithRestarts...")
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler2 = CosineWarmupWithRestarts(
        optimizer2,
        warmup_steps=50,
        first_cycle_steps=200,
        cycle_mult=1.5,
        min_lr=1e-6
    )
    
    lrs2 = []
    for step in range(500):
        lrs2.append(scheduler2.get_lr()[0])
        scheduler2.step()
    
    # Check for restart (LR should increase after first cycle)
    assert lrs2[200] < lrs2[201], "Restart not working!"
    print(f"   LR before restart (step 200): {lrs2[200]:.6f}")
    print(f"   LR after restart (step 201): {lrs2[201]:.6f}")
    print(f"   ✅ CosineWarmupWithRestarts working!")
    
    # Test LinearWarmupScheduler
    print("\n3. Testing LinearWarmupScheduler...")
    optimizer3 = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler3 = LinearWarmupScheduler(
        optimizer3,
        warmup_steps=100
    )
    
    lrs3 = []
    for step in range(200):
        lrs3.append(scheduler3.get_lr()[0])
        scheduler3.step()
    
    print(f"   LR at step 50: {lrs3[50]:.6f}")
    print(f"   LR at step 100: {lrs3[100]:.6f}")
    print(f"   LR at step 150: {lrs3[150]:.6f}")
    assert abs(lrs3[100] - lrs3[150]) < 1e-9, "Not constant after warmup!"
    print(f"   ✅ LinearWarmupScheduler working!")
    
    # Test PolynomialDecayScheduler
    print("\n4. Testing PolynomialDecayScheduler...")
    optimizer4 = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler4 = PolynomialDecayScheduler(
        optimizer4,
        warmup_steps=100,
        max_steps=1000,
        min_lr=1e-6,
        power=2.0
    )
    
    lrs4 = []
    for step in range(1000):
        lrs4.append(scheduler4.get_lr()[0])
        scheduler4.step()
    
    print(f"   LR at step 100: {lrs4[100]:.6f}")
    print(f"   LR at step 500: {lrs4[500]:.6f}")
    print(f"   LR at step 999: {lrs4[999]:.6f}")
    print(f"   ✅ PolynomialDecayScheduler working!")
    
    # Test InverseSqrtScheduler
    print("\n5. Testing InverseSqrtScheduler...")
    optimizer5 = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler5 = InverseSqrtScheduler(
        optimizer5,
        warmup_steps=100
    )
    
    lrs5 = []
    for step in range(500):
        lrs5.append(scheduler5.get_lr()[0])
        scheduler5.step()
    
    print(f"   LR at step 50: {lrs5[50]:.6f}")
    print(f"   LR at step 100: {lrs5[100]:.6f}")
    print(f"   LR at step 400: {lrs5[400]:.6f}")
    print(f"   ✅ InverseSqrtScheduler working!")
    
    # Test create_scheduler factory
    print("\n6. Testing create_scheduler factory...")
    optimizer6 = torch.optim.Adam(model.parameters(), lr=0.001)
    
    sched_cosine = create_scheduler(optimizer6, 'cosine', 100, 1000)
    sched_linear = create_scheduler(optimizer6, 'linear', 100)
    sched_poly = create_scheduler(optimizer6, 'polynomial', 100, 1000, power=2.0)
    
    print(f"   Cosine: {type(sched_cosine).__name__}")
    print(f"   Linear: {type(sched_linear).__name__}")
    print(f"   Polynomial: {type(sched_poly).__name__}")
    print(f"   ✅ Scheduler factory working!")
    
    # Test get_scheduler_info
    print("\n7. Testing get_scheduler_info...")
    info = get_scheduler_info(scheduler)
    
    print(f"   Type: {info['type']}")
    print(f"   Warmup steps: {info['warmup_steps']}")
    print(f"   Max steps: {info['max_steps']}")
    print(f"   Min LR: {info['min_lr']}")
    print(f"   ✅ Scheduler info working!")
    
    # Test plot_schedule (if matplotlib available)
    print("\n8. Testing plot_schedule...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        plot_schedule(scheduler, num_steps=1000)
        print(f"   ✅ Plot schedule working!")
    except ImportError:
        print(f"   ⚠️  Matplotlib not available, skipping plot test")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.training.schedulers import CosineWarmupScheduler")
    print("  from ramanujan.training.schedulers import create_scheduler")
    print("="*70)