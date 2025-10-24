"""Training callbacks for monitoring and control."""

import torch
from typing import Dict, Any, Optional
from pathlib import Path
import json


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
    
    Args:
        project: WandB project name
        entity: WandB entity (username/team)
        name: Run name
        config: Additional config to log
    """
    
    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.run = None
    
    def on_train_begin(self, trainer) -> None:
        """Initialize WandB run."""
        try:
            import wandb
            
            self.run = wandb.init(
                project=self.project or trainer.config.wandb_project,
                entity=self.entity or trainer.config.wandb_entity,
                name=self.name or trainer.config.wandb_run_name or trainer.config.experiment_name,
                config={**trainer.config.to_dict(), **self.config}
            )
            
            # Watch model
            wandb.watch(trainer.model, log='all', log_freq=trainer.config.log_every)
            
            print("✅ WandB initialized")
        except ImportError:
            print("⚠️  WandB not installed, skipping logging")
            self.run = None
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Log step metrics."""
        if self.run is not None and step % trainer.config.log_every == 0:
            import wandb
            wandb.log(metrics, step=trainer.state.global_step)
    
    def on_validation_end(self, trainer, metrics: Dict[str, float]) -> None:
        """Log validation metrics."""
        if self.run is not None:
            import wandb
            wandb.log(metrics, step=trainer.state.global_step)
    
    def on_train_end(self, trainer) -> None:
        """Finish WandB run."""
        if self.run is not None:
            import wandb
            wandb.finish()
            print("✅ WandB run finished")


class EarlyStoppingCallback(Callback):
    """
    Early stopping based on validation metric.
    
    Args:
        monitor: Metric to monitor (e.g., 'val_loss')
        patience: Number of evaluations to wait
        mode: 'min' or 'max'
        min_delta: Minimum change to qualify as improvement
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        mode: str = 'min',
        min_delta: float = 0.0
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
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
    Save checkpoints periodically.
    
    Args:
        save_every: Save every N steps
        keep_last_n: Keep only N most recent checkpoints
    """
    
    def __init__(self, save_every: int = 1000, keep_last_n: int = 3):
        self.save_every = save_every
        self.keep_last_n = keep_last_n
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Save checkpoint if needed."""
        if trainer.state.global_step % self.save_every == 0:
            trainer.save_checkpoint()


class ProgressCallback(Callback):
    """Print training progress."""
    
    def on_train_begin(self, trainer) -> None:
        """Print training start info."""
        print("="*70)
        print("🚀 Training started")
        print(f"   Model: {trainer.model.__class__.__name__}")
        print(f"   Device: {trainer.device}")
        print(f"   Max steps: {trainer.config.max_steps}")
        print(f"   Batch size: {trainer.config.batch_size}")
        print(f"   Learning rate: {trainer.config.learning_rate}")
        print(f"   Output: {trainer.config.output_dir}")
        print("="*70)
    
    def on_train_end(self, trainer) -> None:
        """Print training end info."""
        print("="*70)
        print("✅ Training completed")
        print(f"   Final step: {trainer.state.global_step}")
        if trainer.state.best_metric is not None:
            print(f"   Best metric: {trainer.state.best_metric:.4f} (step {trainer.state.best_step})")
        print("="*70)


class MetricsSaverCallback(Callback):
    """
    Save metrics to JSON file.
    
    Args:
        save_path: Path to save metrics (default: <output_dir>/metrics.json)
    """
    
    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path
    
    def on_train_end(self, trainer) -> None:
        """Save all metrics history."""
        save_path = self.save_path or (Path(trainer.config.output_dir) / 'metrics.json')
        
        with open(save_path, 'w') as f:
            json.dump(trainer.state.metrics_history, f, indent=2)
        
        print(f"💾 Metrics saved: {save_path}")


class SparsityTrackerCallback(Callback):
    """
    Track sparsity statistics for Ramanujan models.
    
    Logs:
    - Attention sparsity per layer
    - FFN sparsity per layer
    - Average sparsity
    """
    
    def __init__(self, log_every: int = 500):
        self.log_every = log_every
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        """Track sparsity."""
        if step % self.log_every != 0:
            return
        
        if not hasattr(trainer.model, 'get_sparsity_stats'):
            return
        
        # Get sparsity stats from model
        sparsity_stats = trainer.model.get_sparsity_stats()
        
        # Log to WandB if available
        for callback in trainer.callbacks:
            if isinstance(callback, WandBCallback) and callback.run is not None:
                import wandb
                wandb.log(sparsity_stats, step=trainer.state.global_step)
        
        # Print summary
        if 'mean_sparsity' in sparsity_stats:
            print(f"   Sparsity: {sparsity_stats['mean_sparsity']:.2%}")