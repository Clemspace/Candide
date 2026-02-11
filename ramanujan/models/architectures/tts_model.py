"""
TTS Model (Phase 1) - Full Training Support
============================================

Complete TTS model with proper loss computation for:
- Duration prediction (MSE on log-durations)
- F0 prediction (MSE + voicing BCE)
- Energy prediction (MSE)
- Mel generation (flow matching)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Adjust imports based on your structure
try:
    from ramanujan.tts.phonemes import N_PHONEMES
except ImportError:
    N_PHONEMES = 42

try:
    from ramanujan.core.interface import TensorSpec
    from ramanujan.core.cost_estimation import ComputeCost
    from ramanujan.core.registry import register_component
except ImportError:
    # Fallbacks
    TensorSpec = None
    ComputeCost = None
    def register_component(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator


@dataclass
class TTSConfigPhase1:
    """Configuration for Phase 1 TTS model."""
    # Phoneme
    n_phonemes: int = N_PHONEMES
    phoneme_dim: int = 256
    
    # Style (Phase 1: speaker only)
    n_speakers: int = 1
    speaker_dim: int = 256
    style_dim: int = 256
    
    # Prosody
    prosody_hidden_dim: int = 256
    n_duration_layers: int = 4
    n_f0_layers: int = 4
    n_energy_layers: int = 2
    
    # Acoustic
    n_mels: int = 80
    acoustic_hidden_channels: int = 256
    acoustic_channel_mults: Tuple[int, ...] = (1, 2, 4)
    n_flow_steps: int = 10
    
    # General
    dropout: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, tuple):
                v = list(v)
            d[k] = v
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TTSConfigPhase1':
        d = d.copy()
        if 'acoustic_channel_mults' in d and isinstance(d['acoustic_channel_mults'], list):
            d['acoustic_channel_mults'] = tuple(d['acoustic_channel_mults'])
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@register_component('tts.model', 'tts_phase1')
class TTSModelPhase1(nn.Module):
    """
    Complete TTS Model for Phase 1 training.
    
    Pipeline:
        phonemes + speaker_id 
            → StyleSystem → style embeddings
            → ProsodySystem → durations, F0, energy, frame_features
            → AcousticDecoder → mel spectrogram
    
    Training: Uses target durations for teacher forcing, computes all losses
    Inference: Uses predicted durations, generates mel via ODE integration
    """
    
    def __init__(self, config: TTSConfigPhase1):
        super().__init__()
        self.config = config
        
        # Import components
        from ramanujan.models.components.tts.style_encoders import StyleSystemPhase1
        from ramanujan.models.components.tts.prosody_predictors import ProsodySystem
        from ramanujan.models.components.tts.acoustic_decoder import FlowMatchingDecoder
        
        # Build components
        self.style_system = StyleSystemPhase1(
            n_speakers=config.n_speakers,
            speaker_dim=config.speaker_dim,
            style_dim=config.style_dim,
            dropout=config.dropout
        )
        
        self.prosody_system = ProsodySystem(
            n_phonemes=config.n_phonemes,
            phoneme_dim=config.phoneme_dim,
            hidden_dim=config.prosody_hidden_dim,
            style_dim=config.style_dim,
            n_duration_layers=config.n_duration_layers,
            n_f0_layers=config.n_f0_layers,
            n_energy_layers=config.n_energy_layers,
            dropout=config.dropout
        )
        
        self.acoustic_decoder = FlowMatchingDecoder(
            n_mels=config.n_mels,
            hidden_channels=config.acoustic_hidden_channels,
            time_dim=256,
            style_dim=config.style_dim,
            prosody_dim=config.prosody_hidden_dim,
            channel_mults=config.acoustic_channel_mults,
            dropout=config.dropout
        )
        
        self.n_flow_steps = config.n_flow_steps
    
    @property
    def component_type(self) -> str:
        return 'model'
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            'style_system': count(self.style_system),
            'prosody_system': count(self.prosody_system),
            'acoustic_decoder': count(self.acoustic_decoder),
            'total': count(self)
        }
    
    def forward(
        self,
        phoneme_ids: Tensor,
        speaker_id: Tensor,
        target_durations: Optional[Tensor] = None,
        target_mel: Optional[Tensor] = None,
        target_f0: Optional[Tensor] = None,
        target_energy: Optional[Tensor] = None,
        target_voiced: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for training.
        
        Args:
            phoneme_ids: (B, S) phoneme indices
            speaker_id: (B,) speaker indices
            target_durations: (B, S) target durations in frames
            target_mel: (B, n_mels, T) target mel spectrogram
            target_f0: (B, T) target F0 contour
            target_energy: (B, T) target energy contour
            target_voiced: (B, T) target voiced mask
        
        Returns:
            Dictionary with 'loss' and component losses
        """
        batch_size = phoneme_ids.shape[0]
        device = phoneme_ids.device
        
        # 1. Style encoding
        style_out = self.style_system(speaker_id=speaker_id)
        style_for_prosody = style_out['style_for_prosody']  # (B, style_dim)
        style_for_acoustic = style_out['style_for_acoustic']  # (B, style_dim)
        
        # 2. Prosody prediction
        prosody_out = self.prosody_system(
            phoneme_ids=phoneme_ids,
            style=style_for_prosody,
            target_durations=target_durations  # Teacher forcing
        )
        
        # Get prosody outputs
        pred_durations = prosody_out['durations']  # (B, S)
        pred_log_durations = prosody_out.get('log_durations', torch.log(pred_durations.clamp(min=1)))
        frame_features = prosody_out['frame_features']  # (B, hidden_dim, T)
        pred_f0 = prosody_out.get('f0', None)  # (B, T)
        pred_voicing = prosody_out.get('voicing', None)  # (B, T)
        pred_energy = prosody_out.get('energy', None)  # (B, T)
        pred_log_energy = prosody_out.get('log_energy', None)
        
        # Initialize outputs
        outputs = {}
        total_loss = 0.0
        
        # 3. Compute losses
        
        # Duration loss (MSE on log-durations)
        if target_durations is not None:
            target_log_dur = torch.log(target_durations.clamp(min=1))
            duration_loss = F.mse_loss(pred_log_durations, target_log_dur)
            outputs['duration_loss'] = duration_loss
            total_loss = total_loss + duration_loss * 0.1
        
        # F0 loss
        if target_f0 is not None and pred_f0 is not None:
            # Align lengths
            min_len = min(pred_f0.shape[1], target_f0.shape[1])
            pred_f0_aligned = pred_f0[:, :min_len]
            target_f0_aligned = target_f0[:, :min_len]
            
            if target_voiced is not None:
                voiced_aligned = target_voiced[:, :min_len]
                # F0 loss only on voiced frames
                if voiced_aligned.any():
                    f0_loss = F.mse_loss(
                        torch.log(pred_f0_aligned[voiced_aligned].clamp(min=1)),
                        torch.log(target_f0_aligned[voiced_aligned].clamp(min=1))
                    )
                else:
                    f0_loss = torch.tensor(0.0, device=device)
                
                # Voicing loss
                if pred_voicing is not None:
                    voicing_aligned = pred_voicing[:, :min_len]
                    voicing_loss = F.binary_cross_entropy_with_logits(
                        voicing_aligned,
                        voiced_aligned.float()
                    )
                    outputs['voicing_loss'] = voicing_loss
                    total_loss = total_loss + voicing_loss * 0.1
            else:
                f0_loss = F.mse_loss(pred_f0_aligned, target_f0_aligned)
            
            outputs['f0_loss'] = f0_loss
            total_loss = total_loss + f0_loss * 0.1
        
        # Energy loss
        if target_energy is not None and pred_energy is not None:
            min_len = min(pred_energy.shape[1], target_energy.shape[1])
            energy_loss = F.mse_loss(
                pred_energy[:, :min_len],
                target_energy[:, :min_len]
            )
            outputs['energy_loss'] = energy_loss
            total_loss = total_loss + energy_loss * 0.1
        
        # 4. Acoustic decoder (flow matching loss)
        if target_mel is not None:
            frame_features = frame_features.transpose(1, 2)  # (B, T, D) -> (B, D, T)

            # Align frame_features with target_mel
            target_frames = target_mel.shape[2]
            pred_frames = frame_features.shape[2]
            
            if pred_frames != target_frames:
                # Interpolate frame_features to match target
                frame_features = F.interpolate(
                    frame_features,
                    size=target_frames,
                    mode='linear',
                    align_corners=False
                )
            
            # Get F0 and energy for conditioning
            if target_f0 is not None:
                f0_cond = target_f0[:, :target_frames]
                if f0_cond.shape[1] < target_frames:
                    f0_cond = F.pad(f0_cond, (0, target_frames - f0_cond.shape[1]))
            else:
                f0_cond = torch.zeros(batch_size, target_frames, device=device)
            
            if target_energy is not None:
                energy_cond = target_energy[:, :target_frames]
                if energy_cond.shape[1] < target_frames:
                    energy_cond = F.pad(energy_cond, (0, target_frames - energy_cond.shape[1]))
            else:
                energy_cond = torch.ones(batch_size, target_frames, device=device)
            
            # Compute flow matching loss
            acoustic_out = self.acoustic_decoder.compute_loss(
                target_mel=target_mel,
                frame_features=frame_features,
                f0=f0_cond,
                energy=energy_cond,
                style=style_for_acoustic
            )
            
            mel_loss = acoustic_out['loss']
            outputs['mel_loss'] = mel_loss
            total_loss = total_loss + mel_loss * 1.0
        
        outputs['loss'] = total_loss
        
        # Store predictions for logging
        outputs['pred_durations'] = pred_durations
        if pred_f0 is not None:
            outputs['pred_f0'] = pred_f0
        if pred_energy is not None:
            outputs['pred_energy'] = pred_energy
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        phoneme_ids: Tensor,
        speaker_id: Tensor,
        n_flow_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Generate mel spectrogram from phonemes.
        
        Args:
            phoneme_ids: (B, S) phoneme indices
            speaker_id: (B,) speaker indices
            n_flow_steps: Number of flow matching steps (default: config value)
            temperature: Sampling temperature
        
        Returns:
            Dictionary with 'mel', 'durations', 'f0', 'energy'
        """
        if n_flow_steps is None:
            n_flow_steps = self.n_flow_steps
        
        # 1. Style encoding
        style_out = self.style_system(speaker_id=speaker_id)
        style_for_prosody = style_out['style_for_prosody']
        style_for_acoustic = style_out['style_for_acoustic']

        
        # 2. Prosody prediction (no teacher forcing)
        prosody_out = self.prosody_system(
            phoneme_ids=phoneme_ids,
            style=style_for_prosody,
            target_durations=None  # Use predicted durations
        )
        
        durations = prosody_out['durations']
        frame_features = prosody_out['frame_features'].transpose(1, 2)  # (B, T, D) -> (B, D, T)
        f0 = prosody_out.get('f0', torch.zeros(frame_features.shape[0], frame_features.shape[2], device=frame_features.device))
        energy = prosody_out.get('energy', torch.ones(frame_features.shape[0], frame_features.shape[2], device=frame_features.device))
        
        # 3. Generate mel
        acoustic_out = self.acoustic_decoder.generate(
            frame_features=frame_features,
            f0=f0,
            energy=energy,
            style=style_for_acoustic,
            n_steps=n_flow_steps
        )
        
        return {
            'mel': acoustic_out['mel'],
            'durations': durations,
            'f0': f0,
            'energy': energy
        }