"""
TTS Phase 1 Training Script
===========================

Train the complete TTS pipeline:
- StyleSystemPhase1 (speaker embedding only, fixed for single speaker)
- ProsodySystem (duration, F0, energy prediction)
- FlowMatchingDecoder (mel generation)

Usage:
    python train_phase1.py --data_dir ./data/siwis_processed --output_dir ./checkpoints/phase1

With wandb:
    python train_phase1.py --data_dir ./data/siwis_processed --wandb --wandb_project candide-tts

For multi-GPU:
    torchrun --nproc_per_node=2 train_phase1.py --data_dir ./data/siwis_processed
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ramanujan.data.datasets.tts_datasets import TTSDataset, collate_tts_batch, create_dataloaders

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")


# =============================================================================
# TRAINING CONFIG
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    data_dir: str = './data/siwis_processed'
    output_dir: str = './checkpoints/phase1'
    
    # Model (should match TTSConfigPhase1)
    n_phonemes: int = 42
    phoneme_dim: int = 256
    n_speakers: int = 1
    speaker_dim: int = 256
    style_dim: int = 256
    prosody_hidden_dim: int = 256
    n_mels: int = 80
    acoustic_hidden_channels: int = 256
    acoustic_channel_mults: Tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    
    # Training
    batch_size: int = 16
    num_workers: int = 4
    max_frames: int = 800  # Max frames per sample
    max_phonemes: int = 150  # Max phonemes per sample
    
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    
    # Loss weights
    loss_mel: float = 1.0
    loss_duration: float = 0.1
    loss_f0: float = 0.1
    loss_energy: float = 0.1
    
    # Checkpointing
    save_every: int = 5  # Save every N epochs
    log_every: int = 50  # Log every N steps
    eval_every: int = 1  # Eval every N epochs
    
    # Hardware
    use_amp: bool = True  # Mixed precision
    device: str = 'cuda'
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = 'candide-tts'
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ['tts', 'phase1'])
    log_audio_every: int = 10  # Log audio samples every N epochs
    log_spectrograms: bool = True
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['acoustic_channel_mults'] = list(d['acoustic_channel_mults'])
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        d = d.copy()
        if 'acoustic_channel_mults' in d:
            d['acoustic_channel_mults'] = tuple(d['acoustic_channel_mults'])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# WANDB LOGGER
# =============================================================================

class WandbLogger:
    """Wrapper for wandb logging with graceful fallback."""
    
    def __init__(self, config: TrainingConfig, model: nn.Module):
        self.enabled = config.use_wandb and WANDB_AVAILABLE
        self.config = config
        
        if self.enabled:
            # Initialize wandb
            run_name = config.wandb_run_name or f"tts-phase1-{time.strftime('%Y%m%d-%H%M%S')}"
            
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                tags=config.wandb_tags,
                config=config.to_dict(),
                dir=config.output_dir,
            )
            
            # Watch model for gradient tracking
            wandb.watch(model, log='gradients', log_freq=100)
            
            print(f"Wandb initialized: {wandb.run.url}")
        else:
            if config.use_wandb and not WANDB_AVAILABLE:
                print("Warning: wandb requested but not installed")
    
    def log(self, metrics: Dict, step: int, prefix: str = ''):
        """Log metrics to wandb."""
        if not self.enabled:
            return
        
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        wandb.log(metrics, step=step)
    
    def log_audio(self, audio: np.ndarray, sample_rate: int, name: str, step: int):
        """Log audio to wandb."""
        if not self.enabled:
            return
        
        wandb.log({
            name: wandb.Audio(audio, sample_rate=sample_rate)
        }, step=step)
    
    def log_spectrogram(self, spec: np.ndarray, name: str, step: int, caption: str = ''):
        """Log spectrogram as image to wandb."""
        if not self.enabled:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Frames')
        ax.set_ylabel('Mel bins')
        ax.set_title(caption or name)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        wandb.log({name: wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def log_f0_contour(self, f0_pred: np.ndarray, f0_target: np.ndarray, 
                       name: str, step: int):
        """Log F0 contour comparison."""
        if not self.enabled:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 3))
        frames = np.arange(len(f0_target))
        
        # Plot target
        ax.plot(frames, f0_target, 'b-', alpha=0.7, label='Target', linewidth=1.5)
        # Plot prediction
        ax.plot(frames, f0_pred, 'r--', alpha=0.7, label='Predicted', linewidth=1.5)
        
        ax.set_xlabel('Frames')
        ax.set_ylabel('F0 (Hz)')
        ax.legend()
        ax.set_title(name)
        plt.tight_layout()
        
        wandb.log({name: wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def log_duration_comparison(self, dur_pred: np.ndarray, dur_target: np.ndarray,
                                 name: str, step: int):
        """Log duration comparison as bar chart."""
        if not self.enabled:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 4))
        n_phones = len(dur_target)
        x = np.arange(n_phones)
        width = 0.35
        
        ax.bar(x - width/2, dur_target, width, label='Target', alpha=0.7)
        ax.bar(x + width/2, dur_pred, width, label='Predicted', alpha=0.7)
        
        ax.set_xlabel('Phoneme index')
        ax.set_ylabel('Duration (frames)')
        ax.legend()
        ax.set_title(name)
        plt.tight_layout()
        
        wandb.log({name: wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def log_table(self, data: List[Dict], name: str, step: int):
        """Log table to wandb."""
        if not self.enabled:
            return
        
        table = wandb.Table(columns=list(data[0].keys()))
        for row in data:
            table.add_data(*row.values())
        
        wandb.log({name: table}, step=step)
    
    def log_histogram(self, values: np.ndarray, name: str, step: int):
        """Log histogram to wandb."""
        if not self.enabled:
            return
        
        wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled:
            wandb.finish()


# =============================================================================
# TRAINING LOOP
# =============================================================================

class TTSTrainer:
    """Trainer for TTS Phase 1."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            max_frames=config.max_frames,
            max_phonemes=config.max_phonemes,
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler: warmup + cosine decay
        warmup_steps = config.warmup_epochs * len(self.train_loader)
        total_steps = config.epochs * len(self.train_loader)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_amp else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Wandb logger
        self.logger = WandbLogger(config, self.model)
        
        # Log initial info
        self.logger.log({
            'total_params': total_params,
            'trainable_params': trainable_params,
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'train_batches': len(self.train_loader),
            'val_batches': len(self.val_loader),
        }, step=0, prefix='info')
    
    def _build_model(self):
        """Build TTS model."""
        # Import model classes
        from ramanujan.models.architectures.tts_model import TTSModelPhase1, TTSConfigPhase1
        
        model_config = TTSConfigPhase1(
            n_phonemes=self.config.n_phonemes,
            phoneme_dim=self.config.phoneme_dim,
            n_speakers=self.config.n_speakers,
            speaker_dim=self.config.speaker_dim,
            style_dim=self.config.style_dim,
            prosody_hidden_dim=self.config.prosody_hidden_dim,
            n_mels=self.config.n_mels,
            acoustic_hidden_channels=self.config.acoustic_hidden_channels,
            acoustic_channel_mults=self.config.acoustic_channel_mults,
            dropout=self.config.dropout,
        )
        
        return TTSModelPhase1(model_config)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        # Track both formats: component_loss and velocity_mse
        loss_components = {'mel': 0.0, 'duration': 0.0, 'f0': 0.0, 'energy': 0.0, 'voicing': 0.0}
        extra_metrics = {'velocity_mse': 0.0}
        n_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast('cuda', enabled=self.config.use_amp):
                outputs = self.model(
                    phoneme_ids=batch['phoneme_ids'],
                    speaker_id=torch.zeros(batch['phoneme_ids'].shape[0], dtype=torch.long, device=self.device),
                    target_durations=batch['durations'],
                    target_mel=batch['mel'],
                    target_f0=batch['f0'],
                    target_energy=batch['energy'],
                    target_voiced=batch['voiced'],
                )
                
                loss = outputs['loss']
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            self.scheduler.step()
            self.global_step += 1
            
            # Accumulate losses
            total_loss += loss.item()
            step_losses = {'loss': loss.item()}
            
            # Debug: print output keys on first batch
            if self.global_step == 1:
                print(f"  [DEBUG] Model output keys: {list(outputs.keys())}")
            
            # Check if model returns component losses
            has_component_losses = 'mel_loss' in outputs or 'duration_loss' in outputs
            
            if has_component_losses:
                # New model format with explicit loss components
                for k in loss_components:
                    key = f'{k}_loss'
                    if key in outputs:
                        val = outputs[key]
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        loss_components[k] += val
                        step_losses[key] = val
            else:
                # Old model format - compute losses manually for logging
                if 'velocity_mse' in outputs:
                    mse = outputs['velocity_mse']
                    if isinstance(mse, torch.Tensor):
                        mse = mse.item()
                    loss_components['mel'] += mse
                    step_losses['mel_loss'] = mse
                    extra_metrics['velocity_mse'] += mse
                
                # Compute duration loss manually
                if 'log_durations' in outputs and 'durations' in batch:
                    pred_log_dur = outputs['log_durations']
                    target_dur = batch['durations']
                    target_log_dur = torch.log(target_dur.clamp(min=1))
                    dur_loss = F.mse_loss(pred_log_dur, target_log_dur).item()
                    loss_components['duration'] += dur_loss
                    step_losses['duration_loss'] = dur_loss
                
                # Compute F0 loss manually
                if 'f0' in outputs and 'f0' in batch:
                    pred_f0 = outputs['f0']
                    target_f0 = batch['f0']
                    min_len = min(pred_f0.shape[1], target_f0.shape[1])
                    f0_loss = F.mse_loss(pred_f0[:, :min_len], target_f0[:, :min_len]).item()
                    loss_components['f0'] += f0_loss
                    step_losses['f0_loss'] = f0_loss
                
                # Compute energy loss manually
                if 'energy' in outputs and 'energy' in batch:
                    pred_energy = outputs['energy']
                    target_energy = batch['energy']
                    min_len = min(pred_energy.shape[1], target_energy.shape[1])
                    energy_loss = F.mse_loss(pred_energy[:, :min_len], target_energy[:, :min_len]).item()
                    loss_components['energy'] += energy_loss
                    step_losses['energy_loss'] = energy_loss
            
            n_batches += 1
            
            # Log to wandb every step
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.log({
                **step_losses,
                'lr': lr,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            }, step=self.global_step, prefix='train')
            
            # Console log
            if batch_idx % self.config.log_every == 0:
                print(f"  Step {self.global_step} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
        
        # Average losses
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return {'loss': avg_loss, **avg_components}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {'mel': 0.0, 'duration': 0.0, 'f0': 0.0, 'energy': 0.0, 'voicing': 0.0}
        n_batches = 0
        
        # Store one batch for visualization
        vis_batch = None
        vis_outputs = None
        
        for batch_idx, batch in enumerate(self.val_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            with autocast('cuda', enabled=self.config.use_amp):
                outputs = self.model(
                    phoneme_ids=batch['phoneme_ids'],
                    speaker_id=torch.zeros(batch['phoneme_ids'].shape[0], dtype=torch.long, device=self.device),
                    target_durations=batch['durations'],
                    target_mel=batch['mel'],
                    target_f0=batch['f0'],
                    target_energy=batch['energy'],
                    target_voiced=batch['voiced'],
                )
            
            total_loss += outputs['loss'].item()
            
            # Check if model returns component losses
            has_component_losses = 'mel_loss' in outputs or 'duration_loss' in outputs
            
            if has_component_losses:
                for k in loss_components:
                    key = f'{k}_loss'
                    if key in outputs:
                        val = outputs[key]
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        loss_components[k] += val
            else:
                # Compute losses manually
                if 'velocity_mse' in outputs:
                    mse = outputs['velocity_mse']
                    if isinstance(mse, torch.Tensor):
                        mse = mse.item()
                    loss_components['mel'] += mse
                
                if 'log_durations' in outputs and 'durations' in batch:
                    pred_log_dur = outputs['log_durations']
                    target_dur = batch['durations']
                    target_log_dur = torch.log(target_dur.clamp(min=1))
                    dur_loss = F.mse_loss(pred_log_dur, target_log_dur).item()
                    loss_components['duration'] += dur_loss
                
                if 'f0' in outputs and 'f0' in batch:
                    pred_f0 = outputs['f0']
                    target_f0 = batch['f0']
                    min_len = min(pred_f0.shape[1], target_f0.shape[1])
                    f0_loss = F.mse_loss(pred_f0[:, :min_len], target_f0[:, :min_len]).item()
                    loss_components['f0'] += f0_loss
                
                if 'energy' in outputs and 'energy' in batch:
                    pred_energy = outputs['energy']
                    target_energy = batch['energy']
                    min_len = min(pred_energy.shape[1], target_energy.shape[1])
                    energy_loss = F.mse_loss(pred_energy[:, :min_len], target_energy[:, :min_len]).item()
                    loss_components['energy'] += energy_loss
            
            n_batches += 1
            
            # Save first batch for visualization
            if vis_batch is None:
                vis_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()}
                vis_outputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                               for k, v in outputs.items()}
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
        
        # Log visualizations
        if self.config.log_spectrograms and vis_batch is not None:
            self._log_visualizations(vis_batch, vis_outputs)
        
        return {'loss': avg_loss, **avg_components}
    
    def _log_visualizations(self, batch: Dict, outputs: Dict):
        """Log spectrograms and other visualizations."""
        if not self.logger.enabled:
            return
        
        # Get first sample from batch
        idx = 0
        
        # Target mel
        target_mel = batch['mel'][idx].numpy()  # (n_mels, T)
        self.logger.log_spectrogram(
            target_mel,
            'val/target_mel',
            self.global_step,
            caption='Target Mel Spectrogram'
        )
        
        # F0 contours - model outputs 'f0' not 'pred_f0'
        f0_key = 'pred_f0' if 'pred_f0' in outputs else 'f0'
        if f0_key in outputs and 'f0' in batch:
            f0_pred = outputs[f0_key][idx].numpy()
            f0_target = batch['f0'][idx].numpy()
            # Align lengths
            min_len = min(len(f0_pred), len(f0_target))
            self.logger.log_f0_contour(
                f0_pred[:min_len], f0_target[:min_len],
                'val/f0_comparison',
                self.global_step
            )
        
        # Duration comparison - model outputs 'durations' not 'pred_durations'
        dur_key = 'pred_durations' if 'pred_durations' in outputs else 'durations'
        if dur_key in outputs and 'durations' in batch:
            dur_pred = outputs[dur_key][idx].numpy()
            dur_target = batch['durations'][idx].numpy()
            # Trim to actual length (non-padded)
            n_phones = int((dur_target > 0).sum())
            if n_phones > 0:
                self.logger.log_duration_comparison(
                    dur_pred[:n_phones], dur_target[:n_phones],
                    'val/duration_comparison',
                    self.global_step
                )
        
        # Energy comparison
        energy_key = 'pred_energy' if 'pred_energy' in outputs else 'energy'
        if energy_key in outputs and 'energy' in batch:
            energy_pred = outputs[energy_key][idx].numpy()
            energy_target = batch['energy'][idx].numpy()
            min_len = min(len(energy_pred), len(energy_target))
            
            # Log as line plot
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 3))
            frames = np.arange(min_len)
            ax.plot(frames, energy_target[:min_len], 'b-', alpha=0.7, label='Target', linewidth=1.5)
            ax.plot(frames, energy_pred[:min_len], 'r--', alpha=0.7, label='Predicted', linewidth=1.5)
            ax.set_xlabel('Frames')
            ax.set_ylabel('Energy')
            ax.legend()
            ax.set_title('Energy Comparison')
            plt.tight_layout()
            
            if self.logger.enabled:
                wandb.log({'val/energy_comparison': wandb.Image(fig)}, step=self.global_step)
            plt.close(fig)
    
    def save_checkpoint(self, name: str = 'latest'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.output_dir / f'checkpoint_{name}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Log checkpoint to wandb
        if self.logger.enabled and name == 'best':
            wandb.save(str(path))
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.config.use_amp}")
        print(f"Wandb: {self.logger.enabled}")
        print()
        
        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch
                start_time = time.time()
                
                print(f"Epoch {epoch + 1}/{self.config.epochs}")
                print("-" * 50)
                
                # Train
                train_metrics = self.train_epoch()
                
                # Log epoch-level train metrics
                self.logger.log(train_metrics, step=self.global_step, prefix='train_epoch')
                
                # Validate
                if (epoch + 1) % self.config.eval_every == 0:
                    val_metrics = self.validate()
                    
                    # Log validation metrics
                    self.logger.log(val_metrics, step=self.global_step, prefix='val')
                    
                    # Check for improvement
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint('best')
                    
                    val_str = " | ".join(f"val_{k}: {v:.4f}" for k, v in val_metrics.items())
                else:
                    val_str = ""
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(f'epoch_{epoch + 1}')
                self.save_checkpoint('latest')
                
                # Log epoch
                elapsed = time.time() - start_time
                train_str = " | ".join(f"train_{k}: {v:.4f}" for k, v in train_metrics.items())
                print(f"{train_str}")
                if val_str:
                    print(f"{val_str}")
                print(f"Time: {elapsed:.1f}s")
                print()
                
                # Log epoch time
                self.logger.log({'epoch_time': elapsed}, step=self.global_step, prefix='timing')
            
            print("Training complete!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        finally:
            # Always finish wandb
            self.logger.finish()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TTS Phase 1")
    
    # Data args
    parser.add_argument('--data_dir', type=str, default='./data/siwis_processed')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/phase1')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Hardware args
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Wandb args
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='candide-tts')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None, help='Run name')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=['tts', 'phase1', 'siwis'])
    
    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Build config
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
    )
    
    # Create trainer
    trainer = TTSTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train!
    trainer.train()


if __name__ == '__main__':
    main()