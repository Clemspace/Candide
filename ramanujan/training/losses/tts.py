"""
TTS Loss Functions
==================

Loss functions for Phase 1 TTS training:
- MelLoss: L1/MSE on mel spectrograms
- DurationLoss: MSE on log-durations
- F0Loss: Voicing BCE + F0 regression
- EnergyLoss: MSE on log-energy
- TTSLoss: Combined loss with configurable weights

Candide Protocol compliant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ramanujan.core.interface import TensorSpec
from ramanujan.core.registry import register_component
from ramanujan.tts.base import TimeEmbedding
# =============================================================================
# BASE LOSS
# =============================================================================

class BaseLoss(nn.Module):
    """Base class for TTS losses."""
    
    def __init__(self, weight: float = 1.0, name: str = ""):
        super().__init__()
        self.weight = weight
        self.name = name or self.__class__.__name__
    
    @property
    def component_type(self) -> str:
        return 'loss'
    
    def get_config(self) -> Dict[str, Any]:
        return {'weight': self.weight, 'name': self.name}


# =============================================================================
# MEL LOSS
# =============================================================================

@register_component('tts.loss', 'mel')
class MelLoss(BaseLoss):
    """
    Loss on mel spectrograms.
    
    L1 produces sharper spectrograms, MSE is smoother.
    """
    
    def __init__(self, loss_type: str = 'l1', weight: float = 1.0):
        super().__init__(weight=weight, name='mel_loss')
        self.loss_type = loss_type
        self.loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'pred_mel': TensorSpec(shape=('batch', 'n_mels', 'frames')),
            'target_mel': TensorSpec(shape=('batch', 'n_mels', 'frames'))
        }
    
    def forward(
        self,
        pred_mel: Tensor,
        target_mel: Tensor,
        mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, frames)
            loss = (self.loss_fn(pred_mel, target_mel, reduction='none') * mask).sum()
            loss = loss / mask.sum().clamp(min=1)
        else:
            loss = self.loss_fn(pred_mel, target_mel)
        
        return {
            'loss': loss * self.weight,
            'mel_loss': loss.detach()
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {'loss_type': self.loss_type, 'weight': self.weight}


# =============================================================================
# DURATION LOSS
# =============================================================================

@register_component('tts.loss', 'duration')
class DurationLoss(BaseLoss):
    """MSE loss on log-durations."""
    
    def __init__(self, weight: float = 0.1):
        super().__init__(weight=weight, name='duration_loss')
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'pred_log_dur': TensorSpec(shape=('batch', 'seq')),
            'target_dur': TensorSpec(shape=('batch', 'seq'))
        }
    
    def forward(
        self,
        pred_log_dur: Tensor,
        target_dur: Tensor,
        mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        target_log_dur = torch.log(target_dur.clamp(min=1.0))
        
        if mask is not None:
            valid = ~mask
            loss = (F.mse_loss(pred_log_dur, target_log_dur, reduction='none') * valid.float()).sum()
            loss = loss / valid.sum().clamp(min=1)
        else:
            loss = F.mse_loss(pred_log_dur, target_log_dur)
        
        return {
            'loss': loss * self.weight,
            'duration_loss': loss.detach()
        }


# =============================================================================
# F0 LOSS
# =============================================================================

@register_component('tts.loss', 'f0')
class F0Loss(BaseLoss):
    """
    F0 prediction loss:
    - Voicing classification (BCE)
    - F0 regression on voiced frames (MSE on log-F0)
    """
    
    def __init__(self, voicing_weight: float = 0.5, f0_weight: float = 0.5, weight: float = 0.1):
        super().__init__(weight=weight, name='f0_loss')
        self.voicing_weight = voicing_weight
        self.f0_weight = f0_weight
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'pred_voicing': TensorSpec(shape=('batch', 'frames')),
            'pred_log_f0_norm': TensorSpec(shape=('batch', 'frames')),
            'target_voiced_mask': TensorSpec(shape=('batch', 'frames'), dtype=torch.bool),
            'target_f0': TensorSpec(shape=('batch', 'frames'))
        }
    
    def forward(
        self,
        pred_voicing: Tensor,
        pred_log_f0_norm: Tensor,
        target_voiced_mask: Tensor,
        target_f0: Tensor,
        frame_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        # Voicing loss
        target_voicing = target_voiced_mask.float()
        voicing_loss = F.binary_cross_entropy(pred_voicing, target_voicing, reduction='mean')
        
        # F0 loss (voiced frames only)
        if target_voiced_mask.any():
            # Normalize target F0 to [0, 1]
            target_log_f0 = torch.log(target_f0.clamp(min=50.0))
            target_norm = (target_log_f0 - 3.91) / (6.68 - 3.91)  # log(50) to log(800)
            target_norm = target_norm.clamp(0, 1)
            
            f0_loss = F.mse_loss(
                pred_log_f0_norm[target_voiced_mask],
                target_norm[target_voiced_mask]
            )
        else:
            f0_loss = torch.tensor(0.0, device=pred_voicing.device)
        
        total = voicing_loss * self.voicing_weight + f0_loss * self.f0_weight
        
        return {
            'loss': total * self.weight,
            'voicing_loss': voicing_loss.detach(),
            'f0_regression_loss': f0_loss.detach()
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'voicing_weight': self.voicing_weight,
            'f0_weight': self.f0_weight,
            'weight': self.weight
        }


# =============================================================================
# ENERGY LOSS
# =============================================================================

@register_component('tts.loss', 'energy')
class EnergyLoss(BaseLoss):
    """MSE loss on log-energy."""
    
    def __init__(self, weight: float = 0.05):
        super().__init__(weight=weight, name='energy_loss')
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'pred_log_energy': TensorSpec(shape=('batch', 'frames')),
            'target_energy': TensorSpec(shape=('batch', 'frames'))
        }
    
    def forward(
        self,
        pred_log_energy: Tensor,
        target_energy: Tensor,
        mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        target_log = torch.log(target_energy.clamp(min=1e-5))
        
        if mask is not None:
            valid = ~mask
            loss = (F.mse_loss(pred_log_energy, target_log, reduction='none') * valid.float()).sum()
            loss = loss / valid.sum().clamp(min=1)
        else:
            loss = F.mse_loss(pred_log_energy, target_log)
        
        return {
            'loss': loss * self.weight,
            'energy_loss': loss.detach()
        }


# =============================================================================
# COMBINED TTS LOSS
# =============================================================================

@dataclass
class TTSLossConfig:
    """Configuration for TTS loss weights."""
    mel_weight: float = 1.0
    mel_type: str = 'l1'
    duration_weight: float = 0.1
    f0_weight: float = 0.1
    energy_weight: float = 0.05


@register_component('tts.loss', 'tts_combined')
class TTSLoss(nn.Module):
    """
    Combined TTS training loss for Phase 1.
    
    Computes all loss components and returns weighted sum.
    """
    
    def __init__(
        self,
        mel_weight: float = 1.0,
        mel_type: str = 'l1',
        duration_weight: float = 0.1,
        f0_weight: float = 0.1,
        energy_weight: float = 0.05
    ):
        super().__init__()
        
        self.mel_loss = MelLoss(loss_type=mel_type, weight=mel_weight)
        self.duration_loss = DurationLoss(weight=duration_weight)
        self.f0_loss = F0Loss(weight=f0_weight)
        self.energy_loss = EnergyLoss(weight=energy_weight)
        
        self._config = TTSLossConfig(
            mel_weight=mel_weight,
            mel_type=mel_type,
            duration_weight=duration_weight,
            f0_weight=f0_weight,
            energy_weight=energy_weight
        )
    
    @property
    def component_type(self) -> str:
        return 'loss'
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'mel_weight': self._config.mel_weight,
            'mel_type': self._config.mel_type,
            'duration_weight': self._config.duration_weight,
            'f0_weight': self._config.f0_weight,
            'energy_weight': self._config.energy_weight
        }
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        masks: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total loss.
        
        Args:
            predictions: Model outputs with keys like 'mel', 'log_durations', 'voicing', etc.
            targets: Ground truth with keys like 'mel', 'durations', 'voiced_mask', 'f0', 'energy'
            masks: Optional masks with keys 'phoneme_mask', 'frame_mask'
        
        Returns:
            (total_loss, loss_dict)
        """
        masks = masks or {}
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # Mel / velocity loss
        if 'velocity_mse' in predictions:
            # Flow matching mode
            losses['velocity_mse'] = predictions['velocity_mse']
            total = total + predictions['loss']
        elif 'mel' in predictions and 'mel' in targets:
            mel_out = self.mel_loss(predictions['mel'], targets['mel'], masks.get('frame_mask'))
            losses.update(mel_out)
            total = total + mel_out['loss']
        
        # Duration loss
        if 'log_durations' in predictions and 'durations' in targets:
            dur_out = self.duration_loss(
                predictions['log_durations'],
                targets['durations'],
                masks.get('phoneme_mask')
            )
            losses.update(dur_out)
            total = total + dur_out['loss']
        
        # F0 loss
        if 'voicing' in predictions and 'voiced_mask' in targets:
            f0_out = self.f0_loss(
                predictions['voicing'],
                predictions.get('log_f0_norm', predictions.get('log_f0_normalized', torch.zeros_like(predictions['voicing']))),
                targets['voiced_mask'],
                targets['f0'],
                masks.get('frame_mask')
            )
            losses.update(f0_out)
            total = total + f0_out['loss']
        
        # Energy loss
        if 'log_energy' in predictions and 'energy' in targets:
            energy_out = self.energy_loss(
                predictions['log_energy'],
                targets['energy'],
                masks.get('frame_mask')
            )
            losses.update(energy_out)
            total = total + energy_out['loss']
        
        losses['total_loss'] = total.detach()
        
        return total, losses