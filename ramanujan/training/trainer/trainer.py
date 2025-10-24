"""Main Trainer class for training models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import time

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

# Mixed precision imports (compatible with PyTorch 2.0+)
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    # Fallback for older PyTorch
    from torch.amp import autocast, GradScaler

from .base import TrainingConfig, TrainingState, CallbackProtocol
from ..losses import create_loss_from_config
from ..optimizers import create_optimizer_from_config
from ..schedulers import create_scheduler_from_config


class Trainer:
    """
    Main trainer class for training language models.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing and resuming
    - Validation
    - Callbacks (WandB, early stopping, etc.)
    - Metrics tracking
    
    Example:
        >>> config = TrainingConfig(
        ...     output_dir='runs/exp1',
        ...     max_steps=10000,
        ...     batch_size=32
        ... )
        >>> trainer = Trainer(
        ...     model=model,
        ...     config=config,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List[CallbackProtocol]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            callbacks: List of callbacks (optional)
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks or []
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Create loss function
        loss_config = {'name': config.loss_name, **config.loss_config}
        self.loss_fn = create_loss_from_config(loss_config)
        
        # Create optimizer
        optimizer_config = {
            'name': config.optimizer_name,
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay,
            **config.optimizer_config
        }
        self.optimizer = create_optimizer_from_config(
            optimizer_config,
            self.model.parameters()
        )
        
        # Create scheduler
        scheduler_config = {
            'name': config.scheduler_name,
            **config.scheduler_config
        }
        
        # Add required parameters for warmup scheduler if not provided
        if config.scheduler_name == 'warmup':
            if 'warmup_steps' not in scheduler_config:
                scheduler_config['warmup_steps'] = config.max_steps // 10  # 10% warmup
            if 'total_steps' not in scheduler_config:
                scheduler_config['total_steps'] = config.max_steps
        
        # Add required parameters for cosine scheduler if not provided
        elif config.scheduler_name == 'cosine':
            if 'total_steps' not in scheduler_config:
                scheduler_config['total_steps'] = config.max_steps
        
        self.scheduler = create_scheduler_from_config(
            scheduler_config,
            self.optimizer
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.state = TrainingState()
        
        # Metrics
        self.running_loss = 0.0
        self.running_steps = 0
        
        # Set seed
        self._set_seed(config.seed)
        
        # Resume if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def train(self):
        """Main training loop."""
        # Callback: train begin
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        self.model.train()
        
        try:
            if self.config.max_epochs is not None:
                self._train_epochs()
            else:
                self._train_steps()
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        finally:
            # Callback: train end
            for callback in self.callbacks:
                callback.on_train_end(self)
    
    def _train_epochs(self):
        """Train for a fixed number of epochs."""
        for epoch in range(self.state.epoch, self.config.max_epochs):
            self.state.epoch = epoch
            
            # Callback: epoch begin
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)
            
            # Train one epoch
            epoch_metrics = self._train_one_epoch()
            
            # Callback: epoch end
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, epoch_metrics)
            
            # Validate
            if (epoch + 1) % self.config.eval_every == 0 and self.val_dataloader:
                val_metrics = self.validate()
                epoch_metrics.update(val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint()
    
    def _train_steps(self):
        """Train for a fixed number of steps."""
        epoch_iterator = iter(self.train_dataloader)
        
        pbar = tqdm(
            total=self.config.max_steps,
            initial=self.state.global_step,
            desc="Training"
        )
        
        while self.state.global_step < self.config.max_steps:
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                epoch_iterator = iter(self.train_dataloader)
                batch = next(epoch_iterator)
                self.state.epoch += 1
            
            # Train step
            metrics = self._train_step(batch)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{metrics.get('loss', 0):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Log
            if self.state.global_step % self.config.log_every == 0:
                self._log_metrics(metrics)
            
            # Validate
            if self.state.global_step % self.config.eval_every == 0 and self.val_dataloader:
                val_metrics = self.validate()
                self._log_metrics(val_metrics, prefix='val')
            
            # Save checkpoint
            if self.state.global_step % self.config.save_every == 0:
                self.save_checkpoint()
        
        pbar.close()
    
    def _train_one_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.train_dataloader, desc=f"Epoch {self.state.epoch}"):
            metrics = self._train_step(batch)
            epoch_loss += metrics['loss']
            num_batches += 1
        
        return {'loss': epoch_loss / num_batches}
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Dictionary with 'input_ids', 'labels', etc.
        
        Returns:
            Dictionary of metrics
        """
        # Callback: step begin
        for callback in self.callbacks:
            callback.on_step_begin(self, self.state.step)
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        if self.config.mixed_precision:
            # PyTorch 2.1+ requires device_type, older versions don't support it
            try:
                with autocast(device_type=self.device.type):
                    # Get model outputs
                    outputs = self.model(**batch)
                    
                    # Compute loss
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    targets = batch.get('labels', batch.get('target_ids'))
                    loss, loss_metrics = self.loss_fn.compute(logits, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
            except TypeError:
                # Fallback for older PyTorch versions
                with autocast():
                    # Get model outputs
                    outputs = self.model(**batch)
                    
                    # Compute loss
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    targets = batch.get('labels', batch.get('target_ids'))
                    loss, loss_metrics = self.loss_fn.compute(logits, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
        else:
            # Get model outputs
            outputs = self.model(**batch)
            
            # Compute loss
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs
            
            targets = batch.get('labels', batch.get('target_ids'))
            loss, loss_metrics = self.loss_fn.compute(logits, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        self.state.step += 1
        
        if self.state.step % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            if self.config.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Increment global step
            self.state.global_step += 1
        
        # Metrics
        metrics = {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'lr': self.scheduler.get_last_lr()[0],
            'step': self.state.global_step,
            **loss_metrics
        }
        
        # Callback: step end
        for callback in self.callbacks:
            callback.on_step_end(self, self.state.step, metrics)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        # Callback: validation begin
        for callback in self.callbacks:
            callback.on_validation_begin(self)
        
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_metrics = {}
        
        eval_steps = self.config.eval_steps or len(self.val_dataloader)
        
        for i, batch in enumerate(self.val_dataloader):
            if i >= eval_steps:
                break
            
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.config.mixed_precision:
                try:
                    with autocast(device_type=self.device.type):
                        outputs = self.model(**batch)
                        
                        if isinstance(outputs, dict) and 'logits' in outputs:
                            logits = outputs['logits']
                        else:
                            logits = outputs
                        
                        targets = batch.get('labels', batch.get('target_ids'))
                        loss, loss_metrics = self.loss_fn.compute(logits, targets)
                except TypeError:
                    # Fallback for older PyTorch
                    with autocast():
                        outputs = self.model(**batch)
                        
                        if isinstance(outputs, dict) and 'logits' in outputs:
                            logits = outputs['logits']
                        else:
                            logits = outputs
                        
                        targets = batch.get('labels', batch.get('target_ids'))
                        loss, loss_metrics = self.loss_fn.compute(logits, targets)
            else:
                outputs = self.model(**batch)
                
                if isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                targets = batch.get('labels', batch.get('target_ids'))
                loss, loss_metrics = self.loss_fn.compute(logits, targets)
            
            # Accumulate
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Accumulate metrics
            for key, value in loss_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = 0.0
                all_metrics[key] += value * batch_size
        
        # Average metrics
        val_metrics = {
            'val_loss': total_loss / total_samples,
            **{f'val_{k}': v / total_samples for k, v in all_metrics.items()}
        }
        
        self.model.train()
        
        # Callback: validation end
        for callback in self.callbacks:
            callback.on_validation_end(self, val_metrics)
        
        return val_metrics
    
    def save_checkpoint(self, path: Optional[str] = None):
        """
        Save checkpoint.
        
        Args:
            path: Path to save checkpoint (optional, auto-generated if None)
        """
        if path is None:
            checkpoint_dir = Path(self.config.output_dir) / 'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            path = checkpoint_dir / f'checkpoint_step_{self.state.global_step}.pt'
        else:
            # Ensure parent directory exists
            Path(path).parent.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'state': self.state.to_dict(),
            'config': self.config.to_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, path: str):
        """
        Load checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.state = TrainingState.from_dict(checkpoint['state'])
        
        if self.scaler and checkpoint['scaler']:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        print(f"📂 Checkpoint loaded: {path}")
        print(f"   Resuming from step {self.state.global_step}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the N best."""
        checkpoint_dir = Path(self.config.output_dir) / 'checkpoints'
        if not checkpoint_dir.exists():
            return
        
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_step_*.pt'))
        
        if len(checkpoints) > self.config.save_total_limit:
            for ckpt in checkpoints[:-self.config.save_total_limit]:
                ckpt.unlink()
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics to console and callbacks."""
        # Add prefix
        if prefix:
            metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}
        
        # Store in history
        self.state.metrics_history.append(metrics)
        
        # Print to console
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        print(f"Step {self.state.global_step}: {metrics_str}")