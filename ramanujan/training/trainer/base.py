"""Base classes and protocols for training."""

from typing import Protocol, Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import torch
from torch.utils.data import DataLoader


class CallbackProtocol(Protocol):
    """Protocol for training callbacks."""
    
    def on_train_begin(self, trainer: 'Trainer') -> None:
        """Called at the start of training."""
        ...
    
    def on_train_end(self, trainer: 'Trainer') -> None:
        """Called at the end of training."""
        ...
    
    def on_epoch_begin(self, trainer: 'Trainer', epoch: int) -> None:
        """Called at the start of each epoch."""
        ...
    
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        ...
    
    def on_step_begin(self, trainer: 'Trainer', step: int) -> None:
        """Called at the start of each step."""
        ...
    
    def on_step_end(self, trainer: 'Trainer', step: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each step."""
        ...
    
    def on_validation_begin(self, trainer: 'Trainer') -> None:
        """Called at the start of validation."""
        ...
    
    def on_validation_end(self, trainer: 'Trainer', metrics: Dict[str, float]) -> None:
        """Called at the end of validation."""
        ...


@dataclass
class TrainingConfig:
    """
    Configuration for training.
    
    Example:
        >>> config = TrainingConfig(
        ...     output_dir='runs/experiment_001',
        ...     max_steps=10000,
        ...     batch_size=32,
        ...     learning_rate=1e-3,
        ...     eval_every=500,
        ...     save_every=1000
        ... )
    """
    
    # Output
    output_dir: str = 'runs/default'
    experiment_name: Optional[str] = None
    
    # Training
    max_steps: int = 10000
    max_epochs: Optional[int] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimizer config
    optimizer_name: str = 'adamw'
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler config
    scheduler_name: str = 'warmup'
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    
    # Loss config
    loss_name: str = 'cross_entropy'
    loss_config: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation
    eval_every: int = 500
    eval_steps: Optional[int] = None  # None = full validation set
    
    # Checkpointing
    save_every: int = 1000
    save_total_limit: int = 3  # Keep only N best checkpoints
    resume_from: Optional[str] = None
    
    # Logging
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True
    distributed: bool = False
    
    # Data
    num_workers: int = 4
    pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Debug
    detect_anomaly: bool = False
    profile: bool = False
    
    def __post_init__(self):
        """Validate and setup config."""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = Path(self.output_dir).name
        
        # Validate
        if self.max_steps <= 0 and self.max_epochs is None:
            raise ValueError("Either max_steps or max_epochs must be set")
        
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
            'max_steps': self.max_steps,
            'max_epochs': self.max_epochs,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'optimizer_name': self.optimizer_name,
            'optimizer_config': self.optimizer_config,
            'scheduler_name': self.scheduler_name,
            'scheduler_config': self.scheduler_config,
            'loss_name': self.loss_name,
            'loss_config': self.loss_config,
            'eval_every': self.eval_every,
            'eval_steps': self.eval_steps,
            'save_every': self.save_every,
            'save_total_limit': self.save_total_limit,
            'log_every': self.log_every,
            'use_wandb': self.use_wandb,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'distributed': self.distributed,
            'seed': self.seed
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config)


@dataclass
class TrainingState:
    """
    Current state of training.
    
    Tracks all stateful information needed for resuming.
    """
    
    step: int = 0
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    best_step: Optional[int] = None
    metrics_history: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step': self.step,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'best_step': self.best_step,
            'metrics_history': self.metrics_history
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'TrainingState':
        """Create from dictionary."""
        return cls(**state_dict)