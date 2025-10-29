"""Refactored Trainer - Clean component injection and better architecture."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import time
import math

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Mixed precision
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    from torch.amp import autocast, GradScaler

from .base import TrainingConfig, TrainingState, CallbackProtocol


class Trainer:
    """
    Refactored trainer with component injection.
    
    Key improvements:
    - Components (optimizer, loss, scheduler) are injected, not created internally
    - DDP support built-in
    - Gradient checkpointing support
    - Better metrics computation
    - Cleaner separation of concerns
    
    Example:
        >>> # Create components externally
        >>> model = create_model(model_config)
        >>> optimizer = create_optimizer('adamw', model.parameters(), lr=1e-3)
        >>> loss_fn = create_loss('cross_entropy', vocab_size=32000)
        >>> scheduler = create_scheduler('warmup', optimizer, warmup_steps=1000)
        >>> 
        >>> # Inject into trainer
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=loss_fn,
        ...     config=training_config,
        ...     train_dataloader=train_loader,
        ...     scheduler=scheduler,
        ...     callbacks=[wandb_callback, checkpoint_callback]
        ... )
        >>> 
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,  # Accept any optimizer
        loss_fn: Union[nn.Module, Any],    # Accept any loss function
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,    # Accept any scheduler
        callbacks: Optional[List[CallbackProtocol]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer with injected components.
        
        Args:
            model: Model to train
            optimizer: Optimizer (pre-configured)
            loss_fn: Loss function (pre-configured)
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            scheduler: Learning rate scheduler (optional)
            callbacks: List of callbacks (optional)
            device: Device to use (optional, defaults to config.device)
        """
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks or []
        
        # Device
        self.device = device or torch.device(config.device)
        self.is_distributed = config.distributed
        self.local_rank = 0
        
        # Setup distributed if needed
        if self.is_distributed:
            self._setup_distributed()
        
        # Model setup
        self.model = model.to(self.device)
        
        # Gradient checkpointing
        if hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        # Components (injected, not created)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(self.device) if hasattr(loss_fn, 'to') else loss_fn
        self.scheduler = scheduler
        
        # Mixed precision
        self.use_mixed_precision = config.mixed_precision
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training state
        self.state = TrainingState()
        
        # Metrics tracking
        self.step_start_time = None
        self.tokens_processed = 0
        
        # Set seed
        self._set_seed(config.seed)
        
        # Resume if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        
        self.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        print(f"🌐 Distributed training: Rank {self.local_rank}/{torch.distributed.get_world_size()}")
    
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
            desc="Training",
            disable=self.is_distributed and self.local_rank != 0
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
            if not self.is_distributed or self.local_rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{metrics.get('loss', 0):.4f}",
                    'ppl': f"{metrics.get('perplexity', 0):.1f}",
                    'lr': f"{self._get_lr():.2e}"
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
                if not self.is_distributed or self.local_rank == 0:
                    self.save_checkpoint()
        
        pbar.close()
    
    def _train_one_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.state.epoch}",
            disable=self.is_distributed and self.local_rank != 0
        ):
            metrics = self._train_step(batch)
            epoch_loss += metrics['loss']
            num_batches += 1
        
        return {'loss': epoch_loss / num_batches}
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Dictionary with 'input_ids' and optionally 'labels'
        
        Returns:
            Dictionary of metrics
        """
        # Start timing
        if self.step_start_time is None:
            self.step_start_time = time.time()
        
        # Callback: step begin
        for callback in self.callbacks:
            callback.on_step_begin(self, self.state.step)
        
        # Prepare batch
        batch = self._prepare_batch(batch)
        
        # Forward pass with mixed precision
        loss, logits = self._forward_pass(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        self._backward_pass(loss)
        
        # Update weights
        self.state.step += 1
        
        if self.state.step % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            grad_norm = self._clip_gradients()
            
            # Optimizer step
            self._optimizer_step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Increment global step
            self.state.global_step += 1
            
            # Compute metrics
            metrics = self._compute_metrics(
                loss=loss.item() * self.config.gradient_accumulation_steps,
                logits=logits,
                targets=batch.get('labels'),
                grad_norm=grad_norm
            )
            
            # Callback: step end
            for callback in self.callbacks:
                callback.on_step_end(self, self.state.step, metrics)
            
            # Reset timing
            self.step_start_time = None
            
            return metrics
        
        # Return partial metrics for accumulation steps
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'step': self.state.global_step
        }
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for forward pass.
        
        Can be overridden for custom batch handling.
        """
        # Move to device
        prepared = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.device, non_blocking=True)
            else:
                prepared[k] = v
        
        return prepared
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through model and loss computation.
        
        Returns:
            Tuple of (loss, logits)
        """
        if self.use_mixed_precision:
            try:
                with autocast(device_type=self.device.type):
                    return self._compute_loss(batch)
            except TypeError:
                # Fallback for older PyTorch
                with autocast():
                    return self._compute_loss(batch)
        else:
            return self._compute_loss(batch)
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss from batch.
        
        Returns:
            Tuple of (loss, logits)
        """
        # Extract labels and remove from batch
        labels = batch.pop('labels', None)
        
        # Remove attention_mask if present (model will auto-generate if needed)
        # This avoids shape mismatches
        batch.pop('attention_mask', None)
        
        # Get model outputs
        outputs = self.model(**batch)
        
        # Put labels back for potential reuse
        if labels is not None:
            batch['labels'] = labels
        
        # Extract logits
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('output'))
        else:
            logits = outputs
        
        # If no targets, can't compute loss (skip)
        if labels is None:
            return torch.tensor(0.0, device=logits.device), logits
        
        # Compute loss
        if hasattr(self.loss_fn, 'compute'):
            # LossComponent interface
            loss, _ = self.loss_fn.compute(logits, labels)
        else:
            # Standard PyTorch loss
            # Handle both language modeling (3D logits) and classification (2D logits)
            if logits.dim() == 3:
                # Language modeling: (B, S, V) -> (B*S, V) and (B, S) -> (B*S,)
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            else:
                # Classification: (B, C) and (B,)
                loss = self.loss_fn(logits, labels)
        
        return loss, logits
    
    def _backward_pass(self, loss: torch.Tensor):
        """Backward pass with mixed precision support."""
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _clip_gradients(self) -> float:
        """
        Clip gradients and return norm.
        
        Returns:
            Gradient norm before clipping
        """
        if self.config.max_grad_norm <= 0:
            return 0.0
        
        # Unscale if using mixed precision
        if self.use_mixed_precision:
            self.scaler.unscale_(self.optimizer)
        
        # Get gradient norm and clip
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        ).item()
        
        return grad_norm
    
    def _optimizer_step(self):
        """Optimizer step with mixed precision support."""
        if self.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
    
    def _compute_metrics(
        self,
        loss: float,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        grad_norm: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute training metrics.
        
        Args:
            loss: Loss value
            logits: Model logits
            targets: Target tokens
            grad_norm: Gradient norm
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'loss': loss,
            'perplexity': math.exp(min(loss, 20)),  # Cap to avoid overflow
            'lr': self._get_lr(),
            'step': self.state.global_step,
        }
        
        # Add gradient norm
        if grad_norm > 0:
            metrics['grad_norm'] = grad_norm
        
        # Compute throughput if timing available
        if self.step_start_time is not None:
            elapsed = time.time() - self.step_start_time
            if targets is not None:
                num_tokens = targets.numel()
                metrics['tokens_per_sec'] = num_tokens / elapsed
                self.tokens_processed += num_tokens
        
        # Compute accuracy if targets available
        if targets is not None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                # Mask padding tokens if needed
                if hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None:
                    mask = targets != self.config.pad_token_id
                    accuracy = (preds == targets)[mask].float().mean().item()
                else:
                    accuracy = (preds == targets).float().mean().item()
                metrics['accuracy'] = accuracy
        
        return metrics
    
    def _get_lr(self) -> float:
        """Get current learning rate."""
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'get_last_lr'):
                lrs = self.scheduler.get_last_lr()
                return lrs[0] if lrs else 0.0
        
        # Fallback to optimizer
        return self.optimizer.param_groups[0]['lr']
    
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
        total_tokens = 0
        all_metrics = {}
        
        eval_steps = self.config.eval_steps or len(self.val_dataloader)
        
        for i, batch in enumerate(self.val_dataloader):
            if i >= eval_steps:
                break
            
            # Prepare batch
            batch = self._prepare_batch(batch)
            
            # Forward pass
            loss, logits = self._forward_pass(batch)
            
            # Accumulate
            targets = batch.get('labels')
            if targets is not None:
                num_tokens = targets.numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
                # Compute accuracy
                preds = logits.argmax(dim=-1)
                if hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None:
                    mask = targets != self.config.pad_token_id
                    acc = (preds == targets)[mask].float().mean().item()
                else:
                    acc = (preds == targets).float().mean().item()
                
                if 'accuracy' not in all_metrics:
                    all_metrics['accuracy'] = 0.0
                all_metrics['accuracy'] += acc * num_tokens
        
        # Average metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        val_metrics = {
            'val_loss': avg_loss,
            'val_perplexity': math.exp(min(avg_loss, 20)),
        }
        
        # Add other averaged metrics
        for key, value in all_metrics.items():
            val_metrics[f'val_{key}'] = value / total_tokens if total_tokens > 0 else 0.0
        
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
            Path(path).parent.mkdir(exist_ok=True, parents=True)
        
        # Get model state dict (unwrap DDP if needed)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        
        checkpoint = {
            'model': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state': self.state.to_dict(),
            'config': self.config.to_dict(),
        }
        
        # Add scheduler if present
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        # Add scaler if using mixed precision
        if self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if not self.is_distributed or self.local_rank == 0:
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
        
        # Load model (handle DDP wrapper)
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(checkpoint['model'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler if present
        if 'scheduler' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load state
        self.state = TrainingState.from_dict(checkpoint['state'])
        
        # Load scaler if using mixed precision
        if 'scaler' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        print(f"📂 Checkpoint loaded: {path}")
        print(f"   Resuming from step {self.state.global_step}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the N most recent."""
        checkpoint_dir = Path(self.config.output_dir) / 'checkpoints'
        if not checkpoint_dir.exists():
            return
        
        checkpoints = sorted(
            checkpoint_dir.glob('checkpoint_step_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        if len(checkpoints) > self.config.save_total_limit:
            for ckpt in checkpoints[:-self.config.save_total_limit]:
                ckpt.unlink()
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics to console and callbacks."""
        # Add prefix
        if prefix:
            metrics = {f'{prefix}/{k}' if not k.startswith(prefix) else k: v 
                      for k, v in metrics.items()}
        
        # Store in history
        self.state.metrics_history.append(metrics)
        
        # Print to console (only on main process)
        if not self.is_distributed or self.local_rank == 0:
            metrics_str = ', '.join([f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' 
                                    for k, v in metrics.items()])
            print(f"Step {self.state.global_step}: {metrics_str}")