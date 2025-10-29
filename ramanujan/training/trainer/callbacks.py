"""Enhanced training callbacks with proper integration."""

import torch
from typing import Dict, Any, Optional
from pathlib import Path
import json
import math

# Import from your existing utilities
from ...utils.logging import WandBLogger
from ...utils.metrics import MetricsTracker, compute_sparsity_stats


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_step_begin(self, trainer, step: int) -> None:
        """Called at the start of each step."""
        pass
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each step."""
        pass
    
    def on_validation_begin(self, trainer) -> None:
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Called at the end of validation."""
        pass


class WandBCallback(Callback):
    """
    Weights & Biases logging callback.
    
    Uses the existing WandBLogger from utils.logging.
    
    Args:
        project: WandB project name
        entity: WandB entity (username/team)
        name: Run name
        config: Additional config to log
        watch_model: Whether to watch model gradients
        watch_freq: Frequency for watching (in steps)
    """
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        watch_model: bool = True,
        watch_freq: int = 100,
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.watch_model = watch_model
        self.watch_freq = watch_freq
        self.wandb_logger = None
    
    def on_train_begin(self, trainer) -> None:
        """Initialize WandB run."""
        # Merge trainer config with additional config
        full_config = {
            **trainer.config.to_dict(),
            **self.config
        }
        
        # Add model info
        full_config['model_params'] = sum(p.numel() for p in trainer.model.parameters())
        full_config['trainable_params'] = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        
        # Initialize WandB logger
        self.wandb_logger = WandBLogger(
            project=self.project,
            name=self.name or trainer.config.experiment_name,
            config=full_config,
            entity=self.entity,
            enabled=True
        )
        
        # Watch model if requested
        if self.watch_model and self.wandb_logger.enabled:
            self.wandb_logger.watch(
                trainer.model,
                log='all',
                log_freq=self.watch_freq
            )
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Log step metrics."""
        if self.wandb_logger is not None and step % trainer.config.log_every == 0:
            self.wandb_logger.log(metrics, step=trainer.state.global_step)
    
    def on_validation_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Log validation metrics."""
        if self.wandb_logger is not None:
            self.wandb_logger.log(metrics, step=trainer.state.global_step)
    
    def on_train_end(self, trainer) -> None:
        """Finish WandB run."""
        if self.wandb_logger is not None:
            self.wandb_logger.finish()


class MetricsCallback(Callback):
    """
    Track and compute advanced metrics.
    
    Uses MetricsTracker from utils.metrics.
    
    Args:
        window_size: Window size for rolling averages
        log_every: Log metrics every N steps
    """
    
    def __init__(self, window_size: int = 100, log_every: int = 10):
        self.window_size = window_size
        self.log_every = log_every
        self.tracker = MetricsTracker(window_size=window_size)
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Update metrics tracker."""
        self.tracker.update(metrics)
        
        # Log rolling averages periodically
        if step % self.log_every == 0:
            averages = self.tracker.get_averages()
            
            # Add to trainer's metrics for other callbacks
            for key, value in averages.items():
                metrics[f'avg_{key}'] = value
    
    def on_train_end(self, trainer) -> None:
        """Print final statistics."""
        print("\n" + "="*70)
        print("📊 Final Training Statistics")
        print("="*70)
        
        for key in ['loss', 'perplexity', 'accuracy']:
            if key in self.tracker.metrics:
                stats = self.tracker.get_stats(key)
                print(f"{key}:")
                print(f"  Average: {stats['average']:.4f}")
                print(f"  Min: {stats['min']:.4f}")
                print(f"  Max: {stats['max']:.4f}")
                print(f"  Std: {stats['std']:.4f}")
        
        print("="*70)


class EarlyStoppingCallback(Callback):
    """
    Early stopping based on validation metric.
    
    Args:
        monitor: Metric to monitor (e.g., 'val_loss')
        patience: Number of evaluations to wait
        mode: 'min' or 'max'
        min_delta: Minimum change to qualify as improvement
        save_best: Whether to save best checkpoint
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        mode: str = 'min',
        min_delta: float = 0.0,
        save_best: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.save_best = save_best
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_step = 0
    
    def on_validation_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Check if should stop."""
        if self.monitor not in metrics:
            return
        
        current = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current
            self.wait = 0
            trainer.state.best_metric = current
            trainer.state.best_step = trainer.state.global_step
            
            # Save best checkpoint
            if self.save_best:
                best_path = Path(trainer.config.output_dir) / 'checkpoints' / 'best.pt'
                trainer.save_checkpoint(str(best_path))
                print(f"✅ New best {self.monitor}: {current:.4f}")
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                print(f"⚠️  Early stopping triggered after {self.wait} evaluations without improvement")
                self.stopped_step = trainer.state.global_step
                raise KeyboardInterrupt("Early stopping")


class CheckpointCallback(Callback):
    """
    Periodic checkpoint saving.
    
    Args:
        save_every: Save every N steps
        keep_last_n: Keep only N most recent checkpoints
        save_optimizer: Whether to save optimizer state
    """
    
    def __init__(
        self,
        save_every: int = 1000,
        keep_last_n: int = 3,
        save_optimizer: bool = True,
    ):
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Save checkpoint if needed."""
        if trainer.state.global_step % self.save_every == 0:
            trainer.save_checkpoint()


class ProgressCallback(Callback):
    """Print training progress and information."""
    
    def on_train_begin(self, trainer) -> None:
        """Print training start info."""
        print("="*70)
        print("🚀 Training Started")
        print("="*70)
        print(f"Model: {trainer.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        print(f"Device: {trainer.device}")
        print(f"Mixed Precision: {trainer.use_mixed_precision}")
        
        if trainer.is_distributed:
            print(f"Distributed: Rank {trainer.local_rank}")
        
        print(f"\nTraining Configuration:")
        print(f"  Max Steps: {trainer.config.max_steps}")
        print(f"  Batch Size: {trainer.config.batch_size}")
        print(f"  Gradient Accumulation: {trainer.config.gradient_accumulation_steps}")
        print(f"  Learning Rate: {trainer.config.learning_rate}")
        print(f"  Weight Decay: {trainer.config.weight_decay}")
        print(f"  Max Grad Norm: {trainer.config.max_grad_norm}")
        
        print(f"\nOutput Directory: {trainer.config.output_dir}")
        print("="*70 + "\n")
    
    def on_train_end(self, trainer) -> None:
        """Print training end info."""
        print("\n" + "="*70)
        print("✅ Training Completed")
        print("="*70)
        print(f"Final Step: {trainer.state.global_step}")
        print(f"Total Tokens Processed: {trainer.tokens_processed:,}")
        
        if trainer.state.best_metric is not None:
            print(f"Best Metric: {trainer.state.best_metric:.4f} (step {trainer.state.best_step})")
        
        print("="*70 + "\n")


class MetricsSaverCallback(Callback):
    """
    Save metrics to JSON file.
    
    Args:
        save_path: Path to save metrics (default: <output_dir>/metrics.json)
        save_every: Save every N steps (optional, None = only at end)
    """
    
    def __init__(
        self,
        save_path: Optional[str] = None,
        save_every: Optional[int] = None,
    ):
        self.save_path = save_path
        self.save_every = save_every
    
    def _save_metrics(self, trainer):
        """Save metrics to file."""
        save_path = self.save_path or (Path(trainer.config.output_dir) / 'metrics.json')
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        
        with open(save_path, 'w') as f:
            json.dump(trainer.state.metrics_history, f, indent=2)
        
        print(f"💾 Metrics saved: {save_path}")
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Save metrics periodically if requested."""
        if self.save_every and trainer.state.global_step % self.save_every == 0:
            self._save_metrics(trainer)
    
    def on_train_end(self, trainer) -> None:
        """Save all metrics history."""
        self._save_metrics(trainer)


class SparsityTrackerCallback(Callback):
    """
    Track sparsity statistics for Ramanujan models.
    
    Uses compute_sparsity_stats from utils.metrics.
    
    Args:
        log_every: Log sparsity every N steps
        verbose: Print detailed layer-wise stats
    """
    
    def __init__(self, log_every: int = 500, verbose: bool = False):
        self.log_every = log_every
        self.verbose = verbose
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Track sparsity."""
        if trainer.state.global_step % self.log_every != 0:
            return
        
        # Compute sparsity stats
        sparsity_stats = compute_sparsity_stats(trainer.model)
        
        if sparsity_stats['num_sparse_layers'] == 0:
            return  # No sparse layers
        
        # Add to metrics
        metrics['sparsity/overall'] = sparsity_stats['overall']
        metrics['sparsity/attention'] = sparsity_stats['attention']
        metrics['sparsity/ffn'] = sparsity_stats['ffn']
        metrics['sparsity/num_layers'] = sparsity_stats['num_sparse_layers']
        
        # Print summary
        print(f"   📊 Sparsity: {sparsity_stats['overall']:.2%} overall "
              f"(Attn: {sparsity_stats['attention']:.2%}, FFN: {sparsity_stats['ffn']:.2%})")
        
        # Print detailed stats if verbose
        if self.verbose and sparsity_stats['layers']:
            print("   Layer-wise sparsity:")
            for layer_info in sparsity_stats['layers'][:5]:  # Show first 5
                print(f"     {layer_info['name']}: {layer_info['sparsity']:.2%}")


class LearningRateMonitorCallback(Callback):
    """
    Monitor and log learning rate changes.
    
    Args:
        log_every: Log LR every N steps
    """
    
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self.last_lr = None
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Log learning rate."""
        if trainer.state.global_step % self.log_every != 0:
            return
        
        current_lr = trainer._get_lr()
        
        # Add to metrics
        metrics['learning_rate'] = current_lr
        
        # Detect significant changes
        if self.last_lr is not None:
            lr_change = abs(current_lr - self.last_lr) / self.last_lr
            if lr_change > 0.1:  # More than 10% change
                print(f"   📉 LR changed: {self.last_lr:.6f} → {current_lr:.6f}")
        
        self.last_lr = current_lr


class GradientNormMonitorCallback(Callback):
    """
    Monitor gradient norms to detect training instabilities.
    
    Args:
        log_every: Log grad norm every N steps
        warn_threshold: Warn if grad norm exceeds this value
    """
    
    def __init__(self, log_every: int = 10, warn_threshold: float = 10.0):
        self.log_every = log_every
        self.warn_threshold = warn_threshold
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Monitor gradient norm."""
        if 'grad_norm' not in metrics:
            return
        
        grad_norm = metrics['grad_norm']
        
        # Warn if too high
        if grad_norm > self.warn_threshold:
            print(f"   ⚠️  High gradient norm detected: {grad_norm:.2f}")
        
        # Log periodically
        if trainer.state.global_step % self.log_every == 0:
            metrics['gradient_norm'] = grad_norm


class MemoryMonitorCallback(Callback):
    """
    Monitor GPU memory usage.
    
    Args:
        log_every: Log memory every N steps
    """
    
    def __init__(self, log_every: int = 100):
        self.log_every = log_every
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Log memory usage."""
        if not torch.cuda.is_available():
            return
        
        if trainer.state.global_step % self.log_every != 0:
            return
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(trainer.device) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(trainer.device) / (1024 ** 3)  # GB
        max_allocated = torch.cuda.max_memory_allocated(trainer.device) / (1024 ** 3)  # GB
        
        metrics['memory/allocated_gb'] = allocated
        metrics['memory/reserved_gb'] = reserved
        metrics['memory/max_allocated_gb'] = max_allocated


# ============================================================================
# CALLBACK FACTORY
# ============================================================================

def create_default_callbacks(config: Dict[str, Any]) -> list[Callback]:
    """
    Create default set of callbacks from config.
    
    Args:
        config: Training configuration dict
    
    Returns:
        List of callbacks
    
    Example:
        >>> callbacks = create_default_callbacks({
        ...     'use_wandb': True,
        ...     'wandb_project': 'my-project',
        ...     'output_dir': 'runs/exp1'
        ... })
    """
    callbacks = [
        ProgressCallback(),
        MetricsCallback(window_size=100),
    ]
    
    # WandB if requested
    if config.get('use_wandb', False):
        callbacks.append(WandBCallback(
            project=config.get('wandb_project', 'ramanujan-training'),
            entity=config.get('wandb_entity'),
            name=config.get('wandb_run_name'),
        ))
    
    # Checkpointing
    callbacks.append(CheckpointCallback(
        save_every=config.get('save_every', 1000),
        keep_last_n=config.get('save_total_limit', 3),
    ))
    
    # Metrics saver
    callbacks.append(MetricsSaverCallback())
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitorCallback(log_every=10))
    
    # Gradient norm monitor
    if config.get('max_grad_norm', 0) > 0:
        callbacks.append(GradientNormMonitorCallback(
            log_every=10,
            warn_threshold=config.get('max_grad_norm', 1.0) * 2
        ))
    
    # Memory monitor for GPU training
    if config.get('device', 'cpu') == 'cuda':
        callbacks.append(MemoryMonitorCallback(log_every=100))
    
    # Early stopping if validation is configured
    if config.get('val_dataloader') is not None and config.get('early_stopping', False):
        callbacks.append(EarlyStoppingCallback(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 5),
            mode='min',
        ))
    
    # Sparsity tracking for Ramanujan models
    if config.get('track_sparsity', False):
        callbacks.append(SparsityTrackerCallback(
            log_every=500,
            verbose=config.get('verbose_sparsity', False)
        ))
    
    return callbacks