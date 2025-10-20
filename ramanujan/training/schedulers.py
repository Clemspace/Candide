"""
Learning rate schedulers for Ramanujan Transformer training.

Modern scheduling strategies based on latest research:
- WSD (Warmup-Stable-Decay): State-of-the-art 3-phase schedule
- Cosine with Warmup: Standard GPT-3/LLaMA style
- Cosine with Restarts: Periodic resets for long training
- Linear, Polynomial, InverseSqrt: Classic schedules

Factory pattern for easy creation and configuration.

Example:
    >>> from ramanujan.training import SchedulerFactory, SchedulerConfig
    >>> 
    >>> # Using factory with config
    >>> config = SchedulerConfig(
    ...     scheduler_type='wsd',
    ...     warmup_steps=1000,
    ...     stable_steps=4500,
    ...     max_steps=10000,
    ...     min_lr=1e-5
    ... )
    >>> scheduler = SchedulerFactory.create(optimizer, config)
    >>> 
    >>> # Or directly
    >>> scheduler = create_scheduler(
    ...     optimizer,
    ...     scheduler_type='wsd',
    ...     warmup_steps=1000,
    ...     max_steps=10000
    ... )
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


# ============================================================================
# SCHEDULER CONFIGURATION
# ============================================================================

@dataclass
class SchedulerConfig:
    """
    Configuration for learning rate schedulers.
    
    Args:
        scheduler_type: Type of scheduler ('wsd', 'cosine', 'cosine_restarts', etc.)
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr: Minimum learning rate (default: 1e-6)
        
        # WSD-specific
        stable_steps: Steps at peak LR (for WSD scheduler)
        
        # Restart-specific
        first_cycle_steps: Steps in first cycle (for restarts)
        cycle_mult: Cycle length multiplier (for restarts)
        max_lr_decay: Max LR decay per cycle (for restarts)
        
        # Polynomial-specific
        power: Polynomial power (for polynomial decay)
        
        # Re-warming (experimental)
        use_rewarm: Enable periodic re-warming
        rewarm_every: Steps between re-warms
        rewarm_steps: Duration of re-warm
        rewarm_lr_ratio: LR ratio for re-warm (0.5 = 50% of base_lr)
    """
    scheduler_type: str = 'cosine'
    warmup_steps: int = 1000
    max_steps: int = 10000
    min_lr: float = 1e-6
    
    # WSD
    stable_steps: Optional[int] = None
    
    # Restarts
    first_cycle_steps: Optional[int] = None
    cycle_mult: float = 1.0
    max_lr_decay: float = 1.0
    
    # Polynomial
    power: float = 1.0
    
    # Re-warming
    use_rewarm: bool = False
    rewarm_every: Optional[int] = None
    rewarm_steps: int = 1000
    rewarm_lr_ratio: float = 0.5


# ============================================================================
# WSD SCHEDULER (State-of-the-Art)
# ============================================================================

class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay scheduler.
    
    Modern 3-phase schedule used in Mistral, Mixtral, and recent LLMs.
    Outperforms 2-phase cosine in practice.
    
    Phase 1: Linear warmup (0 → max_lr)
    Phase 2: Stable at max_lr (NEW!)
    Phase 3: Cosine decay (max_lr → min_lr)
    
    The stable phase allows the model to fully explore at peak LR
    before gradually annealing, leading to better final performance.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        stable_steps: Number of steps at peak LR
        max_steps: Total training steps
        min_lr: Minimum learning rate
        last_epoch: Last epoch (for resuming)
    
    Example:
        >>> # 5% warmup, 45% stable, 50% decay
        >>> scheduler = WSDScheduler(
        ...     optimizer,
        ...     warmup_steps=500,
        ...     stable_steps=4500,
        ...     max_steps=10000,
        ...     min_lr=1e-5
        ... )
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        stable_steps: int,
        max_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = max_steps - warmup_steps - stable_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        
        if self.decay_steps <= 0:
            raise ValueError(
                f"decay_steps must be > 0. Got: {self.decay_steps}. "
                f"max_steps ({max_steps}) must be > warmup_steps ({warmup_steps}) + stable_steps ({stable_steps})"
            )
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate with WSD schedule."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Phase 1: Linear warmup
            alpha = step / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        elif step < self.warmup_steps + self.stable_steps:
            # Phase 2: Stable at peak LR
            return self.base_lrs
        
        elif step < self.max_steps:
            # Phase 3: Cosine decay
            decay_step = step - self.warmup_steps - self.stable_steps
            progress = decay_step / self.decay_steps
            progress = min(1.0, progress)
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
        else:
            # Past max_steps: return min_lr
            return [self.min_lr for _ in self.base_lrs]


# ============================================================================
# COSINE WARMUP SCHEDULER
# ============================================================================

class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing schedule with linear warmup.
    
    The standard schedule for transformer training (GPT-3, LLaMA).
    
    Phase 1: Linear warmup (0 → max_lr)
    Phase 2: Cosine decay (max_lr → min_lr)
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr: Minimum learning rate
        last_epoch: Last epoch index
    
    Example:
        >>> scheduler = CosineWarmupScheduler(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     max_steps=10000,
        ...     min_lr=1e-6
        ... )
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        
        if max_steps <= warmup_steps:
            raise ValueError(
                f"max_steps ({max_steps}) must be > warmup_steps ({warmup_steps})"
            )
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        elif step < self.max_steps:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
        else:
            # Past max_steps
            return [self.min_lr for _ in self.base_lrs]


# ============================================================================
# COSINE WARMUP WITH RESTARTS
# ============================================================================

class CosineWarmupWithRestarts(_LRScheduler):
    """
    Cosine annealing with periodic warm restarts (SGDR).
    
    Periodically resets learning rate to initial value and performs
    cosine decay. Can help escape local minima in long training runs.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps (only for first cycle)
        first_cycle_steps: Steps in first cycle
        cycle_mult: Multiplier for cycle length
        min_lr: Minimum learning rate
        max_lr_decay: Decay max LR each cycle
        last_epoch: Last epoch index
    
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
        min_lr: float = 1e-6,
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
    
    Simple schedule: warms up linearly then maintains constant LR.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        last_epoch: Last epoch index
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
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
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
        min_lr: Minimum learning rate
        power: Polynomial power (1.0 = linear decay, 2.0 = quadratic)
        last_epoch: Last epoch index
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-6,
        power: float = 1.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.power = power
        
        if max_steps <= warmup_steps:
            raise ValueError(f"max_steps must be > warmup_steps")
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        elif step < self.max_steps:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)
            
            decay_factor = (1 - progress) ** self.power
            
            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor
                for base_lr in self.base_lrs
            ]
        else:
            return [self.min_lr for _ in self.base_lrs]


# ============================================================================
# INVERSE SQRT SCHEDULER
# ============================================================================

class InverseSqrtScheduler(_LRScheduler):
    """
    Inverse square root decay with warmup.
    
    Used in original "Attention is All You Need" paper.
    LR increases linearly during warmup, then decays ∝ 1/sqrt(step).
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        last_epoch: Last epoch index
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
        
        # Scale: min(step^-0.5, step * warmup_steps^-1.5)
        scale = min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )
        
        return [base_lr * scale for base_lr in self.base_lrs]


# ============================================================================
# SCHEDULER FACTORY
# ============================================================================

class SchedulerFactory:
    """
    Factory for creating learning rate schedulers.
    
    Provides unified interface for all scheduler types with
    configuration-based creation.
    
    Example:
        >>> from ramanujan.training import SchedulerFactory, SchedulerConfig
        >>> 
        >>> # Create with config object
        >>> config = SchedulerConfig(
        ...     scheduler_type='wsd',
        ...     warmup_steps=500,
        ...     stable_steps=4500,
        ...     max_steps=10000
        ... )
        >>> scheduler = SchedulerFactory.create(optimizer, config)
        >>> 
        >>> # Create from dictionary
        >>> config_dict = {
        ...     'scheduler_type': 'cosine',
        ...     'warmup_steps': 1000,
        ...     'max_steps': 10000,
        ...     'min_lr': 1e-6
        ... }
        >>> scheduler = SchedulerFactory.create_from_dict(optimizer, config_dict)
    """
    
    @staticmethod
    def create(
        optimizer: Optimizer,
        config: SchedulerConfig
    ) -> _LRScheduler:
        """
        Create scheduler from configuration.
        
        Args:
            optimizer: PyTorch optimizer
            config: SchedulerConfig instance
        
        Returns:
            Scheduler instance
        """
        scheduler_type = config.scheduler_type.lower()
        
        if scheduler_type in ['wsd', 'warmup_stable_decay']:
            # Auto-compute stable_steps if not provided
            stable_steps = config.stable_steps
            if stable_steps is None:
                # Default: 45% of total steps for stable phase
                stable_steps = int(0.45 * config.max_steps)
            
            return WSDScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                stable_steps=stable_steps,
                max_steps=config.max_steps,
                min_lr=config.min_lr
            )
        
        elif scheduler_type in ['cosine', 'cosine_warmup']:
            return CosineWarmupScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                min_lr=config.min_lr
            )
        
        elif scheduler_type in ['cosine_restarts', 'sgdr']:
            first_cycle_steps = config.first_cycle_steps
            if first_cycle_steps is None:
                first_cycle_steps = config.max_steps // 2
            
            return CosineWarmupWithRestarts(
                optimizer,
                warmup_steps=config.warmup_steps,
                first_cycle_steps=first_cycle_steps,
                cycle_mult=config.cycle_mult,
                min_lr=config.min_lr,
                max_lr_decay=config.max_lr_decay
            )
        
        elif scheduler_type in ['linear', 'linear_warmup']:
            return LinearWarmupScheduler(
                optimizer,
                warmup_steps=config.warmup_steps
            )
        
        elif scheduler_type in ['polynomial', 'poly']:
            return PolynomialDecayScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                min_lr=config.min_lr,
                power=config.power
            )
        
        elif scheduler_type in ['inverse_sqrt', 'invsqrt']:
            return InverseSqrtScheduler(
                optimizer,
                warmup_steps=config.warmup_steps
            )
        
        else:
            raise ValueError(
                f"Unknown scheduler_type: {scheduler_type}. "
                f"Choose from: 'wsd', 'cosine', 'cosine_restarts', "
                f"'linear', 'polynomial', 'inverse_sqrt'"
            )
    
    @staticmethod
    def create_from_dict(
        optimizer: Optimizer,
        config_dict: Dict[str, Any]
    ) -> _LRScheduler:
        """
        Create scheduler from dictionary.
        
        Useful for loading from YAML/JSON configs.
        
        Args:
            optimizer: PyTorch optimizer
            config_dict: Dictionary with scheduler parameters
        
        Returns:
            Scheduler instance
        """
        config = SchedulerConfig(**config_dict)
        return SchedulerFactory.create(optimizer, config)


# ============================================================================
# LEGACY FUNCTION (for backward compatibility)
# ============================================================================

def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'cosine',
    warmup_steps: int = 1000,
    max_steps: int = 10000,
    **kwargs
) -> _LRScheduler:
    """
    Create learning rate scheduler (legacy function).
    
    Use SchedulerFactory.create() for new code.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        Scheduler instance
    """
    config = SchedulerConfig(
        scheduler_type=scheduler_type,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        **kwargs
    )
    return SchedulerFactory.create(optimizer, config)


# ============================================================================
# SCHEDULER UTILITIES
# ============================================================================

def get_scheduler_info(scheduler: _LRScheduler) -> Dict[str, Any]:
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
        'base_lrs': scheduler.base_lrs,
        'current_lr': scheduler.get_lr()[0] if scheduler.base_lrs else None
    }
    
    # Add scheduler-specific info
    if hasattr(scheduler, 'warmup_steps'):
        info['warmup_steps'] = scheduler.warmup_steps
    if hasattr(scheduler, 'max_steps'):
        info['max_steps'] = scheduler.max_steps
    if hasattr(scheduler, 'min_lr'):
        info['min_lr'] = scheduler.min_lr
    if hasattr(scheduler, 'stable_steps'):
        info['stable_steps'] = scheduler.stable_steps
    
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
        >>> scheduler = WSDScheduler(optimizer, 500, 4500, 10000)
        >>> plot_schedule(scheduler, num_steps=10000, save_path='wsd_schedule.png')
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
    plt.figure(figsize=(12, 6))
    plt.plot(lrs, linewidth=2)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(f'{scheduler.__class__.__name__} Schedule', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add phase annotations for WSD
    if isinstance(scheduler, WSDScheduler):
        plt.axvline(scheduler.warmup_steps, color='r', linestyle='--', alpha=0.5, label='Warmup end')
        plt.axvline(scheduler.warmup_steps + scheduler.stable_steps, color='g', linestyle='--', alpha=0.5, label='Stable end')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Schedule plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_schedules(
    optimizer: Optimizer,
    configs: Dict[str, SchedulerConfig],
    num_steps: int = 10000,
    save_path: Optional[str] = None
):
    """
    Compare multiple scheduler configurations.
    
    Args:
        optimizer: Optimizer instance
        configs: Dictionary mapping names to SchedulerConfig instances
        num_steps: Number of steps to plot
        save_path: Optional path to save plot
    
    Example:
        >>> configs = {
        ...     'Cosine': SchedulerConfig('cosine', 1000, 10000),
        ...     'WSD': SchedulerConfig('wsd', 500, 10000, stable_steps=4500)
        ... }
        >>> compare_schedules(optimizer, configs, save_path='comparison.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available")
        return
    
    plt.figure(figsize=(14, 7))
    
    for name, config in configs.items():
        scheduler = SchedulerFactory.create(optimizer, config)
        
        lrs = []
        scheduler.last_epoch = -1
        for _ in range(num_steps):
            scheduler.step()
            lrs.append(scheduler.get_lr()[0])
        
        plt.plot(lrs, label=name, linewidth=2)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Schedulers
    'WSDScheduler',
    'CosineWarmupScheduler',
    'CosineWarmupWithRestarts',
    'LinearWarmupScheduler',
    'PolynomialDecayScheduler',
    'InverseSqrtScheduler',
    
    # Factory
    'SchedulerFactory',
    'SchedulerConfig',
    'create_scheduler',
    
    # Utilities
    'get_scheduler_info',
    'plot_schedule',
    'compare_schedules',
]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing improved schedulers.py module")
    print("="*70)
    
    # Create dummy optimizer
    import torch.nn as nn
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Test WSDScheduler
    print("\n1. Testing WSDScheduler (State-of-the-Art)...")
    wsd = WSDScheduler(
        optimizer,
        warmup_steps=500,
        stable_steps=4500,
        max_steps=10000,
        min_lr=1e-5
    )
    
    lrs_wsd = []
    for step in range(10000):
        lrs_wsd.append(wsd.get_lr()[0])
        wsd.step()
    
    print(f"   Initial LR (step 0): {lrs_wsd[0]:.6f}")
    print(f"   After warmup (step 500): {lrs_wsd[500]:.6f}")
    print(f"   During stable (step 3000): {lrs_wsd[3000]:.6f}")
    print(f"   After stable (step 5000): {lrs_wsd[5000]:.6f}")
    print(f"   Final LR (step 9999): {lrs_wsd[9999]:.6f}")
    assert lrs_wsd[500] == lrs_wsd[3000], "Stable phase not working!"
    assert lrs_wsd[5000] > lrs_wsd[9999], "Decay not working!"
    print(f"   ✅ WSDScheduler working!")
    
    # Test SchedulerFactory
    print("\n2. Testing SchedulerFactory...")
    
    config_wsd = SchedulerConfig(
        scheduler_type='wsd',
        warmup_steps=500,
        stable_steps=4500,
        max_steps=10000,
        min_lr=1e-5
    )
    sched_wsd = SchedulerFactory.create(optimizer, config_wsd)
    print(f"   WSD created: {type(sched_wsd).__name__}")
    
    config_cosine = SchedulerConfig(
        scheduler_type='cosine',
        warmup_steps=1000,
        max_steps=10000,
        min_lr=1e-6
    )
    sched_cosine = SchedulerFactory.create(optimizer, config_cosine)
    print(f"   Cosine created: {type(sched_cosine).__name__}")
    
    print(f"   ✅ SchedulerFactory working!")
    
    # Test from dict
    print("\n3. Testing create_from_dict...")
    config_dict = {
        'scheduler_type': 'wsd',
        'warmup_steps': 500,
        'max_steps': 10000,
        'min_lr': 1e-5
    }
    sched_dict = SchedulerFactory.create_from_dict(optimizer, config_dict)
    print(f"   Created from dict: {type(sched_dict).__name__}")
    print(f"   ✅ create_from_dict working!")
    
    # Test get_scheduler_info
    print("\n4. Testing get_scheduler_info...")
    info = get_scheduler_info(wsd)
    print(f"   Type: {info['type']}")
    print(f"   Warmup: {info['warmup_steps']}")
    print(f"   Stable: {info['stable_steps']}")
    print(f"   Max steps: {info['max_steps']}")
    print(f"   ✅ Scheduler info working!")
    
    # Test edge cases
    print("\n5. Testing edge cases...")
    try:
        bad_config = SchedulerConfig(
            scheduler_type='cosine',
            warmup_steps=10000,
            max_steps=10000  # Equal to warmup!
        )
        SchedulerFactory.create(optimizer, bad_config)
        print(f"   ❌ Should have raised error!")
    except ValueError as e:
        print(f"   ✅ Correctly caught error: {str(e)[:50]}...")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nUsage examples:")
    print("  # Modern WSD schedule (recommended)")
    print("  config = SchedulerConfig('wsd', 500, 10000, stable_steps=4500)")
    print("  scheduler = SchedulerFactory.create(optimizer, config)")
    print()
    print("  # Classic cosine schedule")
    print("  config = SchedulerConfig('cosine', 1000, 10000)")
    print("  scheduler = SchedulerFactory.create(optimizer, config)")
    print("="*70)
