"""
Training utilities for Ramanujan Transformer.

This module provides the main Trainer class and ablation study support:
- Trainer: Main training loop with logging, checkpointing, evaluation
- AblationStudy: Run systematic ablation experiments
- Training utilities and metrics

Example:
    >>> from ramanujan.training import Trainer
    >>> from ramanujan.architecture import create_model, ModelConfig
    >>> 
    >>> # Create model
    >>> config = ModelConfig(...)
    >>> model = create_model(config)
    >>> 
    >>> # Create trainer
    >>> trainer = Trainer(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     eval_loader=eval_loader,
    ...     optimizer=optimizer,
    ...     scheduler=scheduler,
    ...     config=training_config
    ... )
    >>> 
    >>> # Train
    >>> trainer.train()
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import os
import time
import json
from pathlib import Path

from .losses import create_loss, LossTracker, compute_perplexity, compute_bits_per_byte
from .optimizers import create_optimizer, clip_grad_norm
from .schedulers import SchedulerFactory, SchedulerConfig



# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training dynamics
    max_steps: int = 10000
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100
    
    # Logging
    log_every: int = 10
    save_every: int = 1000
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Loss
    loss_type: str = 'semantic_entropy'  # 'ce' or 'semantic_entropy'
    label_smoothing: float = 0.0
    semantic_entropy_alpha: float = 0.1
    
    # Optimizer
    optimizer_type: str = 'adamw'
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    

    # Scheduler
    scheduler_type: str = 'cosine'
    warmup_steps: int = 1000
    min_lr: float = 0.0
    
    # WSD-specific
    stable_steps: Optional[int] = None  # For WSD scheduler
    
    # Restart-specific (optional, for future)
    first_cycle_steps: Optional[int] = None
    cycle_mult: float = 1.0
    max_lr_decay: float = 1.0
    
    # Polynomial-specific (optional, for future)
    power: float = 1.0
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = False  # Use AMP
    
    # Reproducibility
    seed: int = 42
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "ramanujan-transformer"
    wandb_run_name: Optional[str] = None
    
    # Resume
    resume_from_checkpoint: Optional[str] = None


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """
    Main trainer class for Ramanujan Transformer.
    
    Handles:
    - Training loop with gradient accumulation
    - Evaluation and metrics tracking
    - Checkpointing and resume
    - Logging (console + W&B)
    - Mixed precision training
    
    Args:
        model: Model to train
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        optimizer: Optimizer instance
        scheduler: LR scheduler instance
        loss_fn: Loss function
        config: Training configuration
    
    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     eval_loader=eval_loader,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     config=config
        ... )
        >>> 
        >>> # Train model
        >>> trainer.train()
        >>> 
        >>> # Evaluate
        >>> metrics = trainer.evaluate()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_fn: Optional[nn.Module] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainingConfig()
        
        # Setup device
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # Setup loss function
        if loss_fn is None:
            self.loss_fn = create_loss(
                self.config.loss_type,
                vocab_size=model.vocab_size,
                label_smoothing=self.config.label_smoothing,
                alpha=self.config.semantic_entropy_alpha
            ).to(self.device)
        else:
            self.loss_fn = loss_fn.to(self.device)

        
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
        
        # Create directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Metrics tracking
        self.loss_tracker = LossTracker(window_size=100)
        self.train_metrics = []
        self.eval_metrics = []
        
        # Setup W&B if requested
        self.wandb_run = None
        if self.config.use_wandb:
            self._setup_wandb()
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config)
            )
            print("W&B logging enabled")
        except ImportError:
            print("W&B not available. Install with: pip install wandb")
            self.config.use_wandb = False
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data with 'input_ids' and optionally 'attention_mask'
        
        Returns:
            Dictionary with step metrics
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        with torch.autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
            # Get model outputs
            if hasattr(self.model, 'forward') and 'return_hidden_states' in self.model.forward.__code__.co_varnames:
                logits, hidden_states = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_hidden_states=True
                )
                # Use last layer hidden states for semantic entropy
                last_hidden = hidden_states[-1] if hidden_states else None
            else:
                logits = self.model(input_ids, attention_mask=attention_mask)
                last_hidden = None
            
            # Compute loss (causal LM: predict next token)
            targets = input_ids[:, 1:]
            logits = logits[:, :-1, :]
            
            if last_hidden is not None:
                last_hidden = last_hidden[:, :-1, :]
            
            loss = self.loss_fn(
                logits,
                targets,
                hidden_states=last_hidden,
                return_metrics=False
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Compute metrics
        with torch.no_grad():
            perplexity = compute_perplexity(loss * self.config.gradient_accumulation_steps)
            bpb = compute_bits_per_byte(
                loss * self.config.gradient_accumulation_steps,
                vocab_size=self.model.vocab_size,
                sequence_length=targets.shape[1]
            )
        
        metrics = {
            'loss': (loss * self.config.gradient_accumulation_steps).item(),
            'perplexity': perplexity.item(),
            'bpb': bpb,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        # Unscale gradients if using mixed precision
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Clip gradients
        grad_norm = clip_grad_norm(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        return grad_norm
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*70}")
        print(f"Starting training for {self.config.max_steps} steps")
        print(f"{'='*70}\n")
        
        self.model.train()
        accumulation_step = 0
        start_time = time.time()
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                # Training step
                step_metrics = self.train_step(batch)
                accumulation_step += 1
                
                # Update weights after accumulation
                if accumulation_step % self.config.gradient_accumulation_steps == 0:
                    grad_norm = self.optimizer_step()
                    step_metrics['grad_norm'] = grad_norm
                    
                    self.global_step += 1
                    self.loss_tracker.update(step_metrics['loss'])
                    
                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        self._log_metrics(step_metrics, prefix='train')
                    
                    # Evaluation
                    if self.eval_loader is not None and self.global_step % self.config.eval_every == 0:
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics, prefix='eval')
                        
                        # Save best model
                        if eval_metrics['loss'] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics['loss']
                            self.save_checkpoint('best.pt')
                            print(f"   → New best model saved! Loss: {self.best_eval_loss:.4f}")
                        
                        self.model.train()
                    
                    # Checkpointing
                    if self.global_step % self.config.save_every == 0:
                        self.save_checkpoint(f'step_{self.global_step}.pt')
                    
                    # Check if done
                    if self.global_step >= self.config.max_steps:
                        break
            
            self.epoch += 1
        
        # Final evaluation and save
        if self.eval_loader is not None:
            final_metrics = self.evaluate()
            print(f"\nFinal evaluation metrics:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")
        
        self.save_checkpoint('final.pt')
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Steps: {self.global_step}")
        print(f"{'='*70}\n")
        
        if self.wandb_run:
            self.wandb_run.finish()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on evaluation set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        for i, batch in enumerate(self.eval_loader):
            if i >= self.config.eval_steps:
                break
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask=attention_mask)
            
            # Compute loss
            targets = input_ids[:, 1:]
            logits = logits[:, :-1, :]
            
            loss = self.loss_fn(logits, targets, return_metrics=False)
            
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
            num_batches += 1
        
        # Compute metrics
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        bpb = compute_bits_per_byte(
            torch.tensor(avg_loss),
            vocab_size=self.model.vocab_size,
            sequence_length=1
        )
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity.item(),
            'bpb': bpb,
            'num_batches': num_batches
        }
        
        self.eval_metrics.append(metrics)
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = 'train'):
        """Log metrics to console and W&B."""
        # Console logging
        step_info = f"Step {self.global_step}"
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[{prefix.upper()}] {step_info} | {metrics_str}")
        
        # W&B logging
        if self.config.use_wandb and self.wandb_run:
            log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
            log_dict['step'] = self.global_step
            self.wandb_run.log(log_dict)
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'config': vars(self.config)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")


# ============================================================================
# ABLATION STUDY
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    baseline_config: Dict[str, Any]
    ablations: List[Dict[str, Any]]
    output_dir: str = "ablations"
    num_seeds: int = 3
    eval_steps: int = 100


class AblationStudy:
    """
    Run systematic ablation experiments.
    
    Compares different model configurations to understand
    the contribution of each component.
    
    Args:
        base_config: Base training configuration
        ablation_config: Ablation study configuration
    
    Example:
        >>> ablations = [
        ...     {'name': 'baseline', 'changes': {}},
        ...     {'name': 'no_semantic_entropy', 'changes': {'loss_type': 'ce'}},
        ...     {'name': 'no_sliding_window', 'changes': {'use_sliding_window': False}},
        ...     {'name': 'all_improvements', 'changes': {
        ...         'loss_type': 'semantic_entropy',
        ...         'use_sliding_window': True,
        ...         'attention_sparsity': 0.82,
        ...         'ffn_sparsity': 0.88
        ...     }}
        ... ]
        >>> 
        >>> study = AblationStudy(base_config, ablation_config)
        >>> results = study.run()
    """
    
    def __init__(
        self,
        base_config: TrainingConfig,
        ablation_config: AblationConfig
    ):
        self.base_config = base_config
        self.ablation_config = ablation_config
        self.results = {}
    
    def run(self) -> Dict[str, Any]:
        """
        Run all ablation experiments.
        
        Returns:
            Dictionary with results for each ablation
        """
        print(f"\n{'='*70}")
        print(f"Starting Ablation Study")
        print(f"Number of ablations: {len(self.ablation_config.ablations)}")
        print(f"Seeds per ablation: {self.ablation_config.num_seeds}")
        print(f"{'='*70}\n")
        
        for ablation in self.ablation_config.ablations:
            ablation_name = ablation['name']
            print(f"\n{'='*70}")
            print(f"Running ablation: {ablation_name}")
            print(f"{'='*70}\n")
            
            ablation_results = self._run_ablation(ablation)
            self.results[ablation_name] = ablation_results
            
            # Save intermediate results
            self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _run_ablation(self, ablation: Dict[str, Any]) -> Dict[str, Any]:
        """Run single ablation with multiple seeds."""
        results = {
            'seeds': [],
            'avg_loss': 0.0,
            'avg_perplexity': 0.0,
            'avg_bpb': 0.0
        }
        
        for seed in range(self.ablation_config.num_seeds):
            print(f"\nSeed {seed + 1}/{self.ablation_config.num_seeds}")
            
            # Create config with changes
            config = self._create_ablation_config(ablation, seed)
            
            # Train model (simplified - in practice you'd create model, trainer, etc.)
            seed_results = self._train_ablation(config, ablation['name'], seed)
            
            results['seeds'].append(seed_results)
            results['avg_loss'] += seed_results['final_loss']
            results['avg_perplexity'] += seed_results['final_perplexity']
            results['avg_bpb'] += seed_results['final_bpb']
        
        # Compute averages
        num_seeds = self.ablation_config.num_seeds
        results['avg_loss'] /= num_seeds
        results['avg_perplexity'] /= num_seeds
        results['avg_bpb'] /= num_seeds
        
        return results
    
    def _create_ablation_config(
        self,
        ablation: Dict[str, Any],
        seed: int
    ) -> TrainingConfig:
        """Create config for ablation experiment."""
        config_dict = vars(self.base_config).copy()
        config_dict.update(ablation.get('changes', {}))
        config_dict['seed'] = seed
        config_dict['output_dir'] = os.path.join(
            self.ablation_config.output_dir,
            ablation['name'],
            f'seed_{seed}'
        )
        return TrainingConfig(**config_dict)
    
    def _train_ablation(
        self,
        config: TrainingConfig,
        ablation_name: str,
        seed: int
    ) -> Dict[str, float]:
        """
        Train model for ablation.
        
        Note: This is a simplified version. In practice, you would:
        1. Create model from config
        2. Create data loaders
        3. Create optimizer and scheduler
        4. Create trainer
        5. Train and evaluate
        """
        # Placeholder - implement full training logic
        return {
            'final_loss': 3.5,  # Placeholder
            'final_perplexity': 33.1,  # Placeholder
            'final_bpb': 1.2  # Placeholder
        }
    
    def _save_results(self):
        """Save ablation results to JSON."""
        output_path = os.path.join(
            self.ablation_config.output_dir,
            'ablation_results.json'
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def _print_summary(self):
        """Print summary of ablation results."""
        print(f"\n{'='*70}")
        print(f"Ablation Study Results Summary")
        print(f"{'='*70}\n")
        
        # Sort by average loss
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['avg_loss']
        )
        
        print(f"{'Ablation':<30} {'Loss':<12} {'Perplexity':<12} {'BPB':<8}")
        print(f"{'-'*70}")
        
        for name, results in sorted_results:
            print(
                f"{name:<30} "
                f"{results['avg_loss']:<12.4f} "
                f"{results['avg_perplexity']:<12.2f} "
                f"{results['avg_bpb']:<8.4f}"
            )
        
        print(f"\n{'='*70}\n")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing trainer.py module")
    print("="*70)
    
    # Create dummy model and data
    from torch.utils.data import TensorDataset, DataLoader
    
    class DummyModel(nn.Module):
        def __init__(self, vocab_size=1000, dim=256):
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(vocab_size, dim)
            self.fc = nn.Linear(dim, vocab_size)
        
        def forward(self, input_ids, attention_mask=None, return_hidden_states=False):
            x = self.embedding(input_ids)
            logits = self.fc(x)
            
            if return_hidden_states:
                return logits, [x]
            return logits
    
    model = DummyModel()
    
    # Create dummy data
    input_ids = torch.randint(0, 1000, (100, 32))
    dataset = TensorDataset(input_ids)
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )
    
    # Test TrainingConfig
    print("\n1. Testing TrainingConfig...")
    config = TrainingConfig(
        max_steps=100,
        batch_size=4,
        eval_every=50,
        log_every=10,
        device='cpu'
    )
    print(f"   Config created: {config.max_steps} steps")
    print(f"   ✅ TrainingConfig working!")
    
    # Test Trainer initialization
    print("\n2. Testing Trainer initialization...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = None
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    print(f"   Trainer device: {trainer.device}")
    print(f"   Global step: {trainer.global_step}")
    print(f"   ✅ Trainer initialization working!")
    
    # Test train_step
    print("\n3. Testing train_step...")
    batch = {'input_ids': input_ids[:4]}
    metrics = trainer.train_step(batch)
    
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Perplexity: {metrics['perplexity']:.2f}")
    print(f"   ✅ train_step working!")
    
    # Test optimizer_step
    print("\n4. Testing optimizer_step...")
    grad_norm = trainer.optimizer_step()
    
    print(f"   Gradient norm: {grad_norm:.4f}")
    print(f"   ✅ optimizer_step working!")
    
    # Test evaluate
    print("\n5. Testing evaluate...")
    eval_metrics = trainer.evaluate()
    
    print(f"   Eval loss: {eval_metrics['loss']:.4f}")
    print(f"   Eval perplexity: {eval_metrics['perplexity']:.2f}")
    print(f"   ✅ evaluate working!")
    
    # Test save/load checkpoint
    print("\n6. Testing checkpoint save/load...")
    trainer.save_checkpoint('test_checkpoint.pt')
    
    # Modify state
    old_step = trainer.global_step
    trainer.global_step = 999
    
    # Load checkpoint
    trainer.load_checkpoint(
        os.path.join(config.checkpoint_dir, 'test_checkpoint.pt')
    )
    
    assert trainer.global_step == old_step, "Checkpoint load failed!"
    print(f"   Checkpoint restored to step {trainer.global_step}")
    print(f"   ✅ Checkpoint save/load working!")
    
    # Test short training run
    print("\n7. Testing short training run...")
    short_config = TrainingConfig(
        max_steps=10,
        batch_size=4,
        eval_every=5,
        log_every=2,
        device='cpu',
        gradient_accumulation_steps=2
    )
    
    short_trainer = Trainer(
        model=DummyModel(),
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        scheduler=None,
        config=short_config
    )
    
    short_trainer.train()
    
    print(f"   Training completed!")
    print(f"   Final step: {short_trainer.global_step}")
    print(f"   ✅ Training loop working!")
    
    # Test AblationConfig
    print("\n8. Testing AblationConfig...")
    ablation_config = AblationConfig(
        baseline_config={},
        ablations=[
            {'name': 'baseline', 'changes': {}},
            {'name': 'test', 'changes': {'learning_rate': 0.0001}}
        ],
        num_seeds=2
    )
    
    print(f"   Ablations: {len(ablation_config.ablations)}")
    print(f"   Seeds: {ablation_config.num_seeds}")
    print(f"   ✅ AblationConfig working!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.training import Trainer, TrainingConfig")
    print("  from ramanujan.training import AblationStudy, AblationConfig")
    print("="*70)