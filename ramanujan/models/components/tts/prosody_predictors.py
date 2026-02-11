"""
TTS Prosody Predictors
======================

Predict prosodic features: duration, F0 (pitch), and energy.
These control the timing and melody of synthesized speech.

Components:
- DurationPredictor: phoneme → duration in frames
- F0Predictor: phoneme + duration → pitch contour
- EnergyPredictor: phoneme + duration → energy contour
- ProsodySystem: combines all three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple
import math
from ramanujan.core.registry import register_component



# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ConvBlock(nn.Module):
    """1D convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, channels, seq)"""
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq, channels)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (batch, channels, seq)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    
    Modulates features using conditioning: y = γ * x + β
    Used to inject style information into predictions.
    """
    
    def __init__(self, feature_dim: int, conditioning_dim: int):
        super().__init__()
        self.projection = nn.Linear(conditioning_dim, feature_dim * 2)
    
    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq, dim) or (batch, dim)
            conditioning: (batch, cond_dim)
        """
        params = self.projection(conditioning)
        gamma, beta = params.chunk(2, dim=-1)
        
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        return gamma * x + beta


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq, dim)"""
        return x + self.pe[:, :x.size(1)]


# =============================================================================
# DURATION PREDICTOR
# =============================================================================

@register_component('tts_predictor', 'duration')
class DurationPredictor(nn.Module):
    """
    Predicts phoneme durations (in frames).
    
    Architecture: Conv stack with FiLM conditioning from style.
    Output: log-duration (positive values via exp at inference).
    
    Args:
        n_phonemes: Size of phoneme vocabulary
        phoneme_dim: Phoneme embedding dimension
        hidden_dim: Hidden layer dimension
        style_dim: Style conditioning dimension
        n_layers: Number of conv layers
        kernel_size: Conv kernel size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_phonemes: int = 50,
        phoneme_dim: int = 256,
        hidden_dim: int = 256,
        style_dim: int = 128,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.phoneme_dim = phoneme_dim
        self.hidden_dim = hidden_dim
        
        # Phoneme embedding
        self.phoneme_embedding = nn.Embedding(n_phonemes, phoneme_dim)
        self.pos_encoding = PositionalEncoding(phoneme_dim)
        
        # Input projection
        self.input_proj = nn.Linear(phoneme_dim, hidden_dim)
        
        # FiLM conditioning
        self.film = FiLMLayer(hidden_dim, style_dim)
        
        # Conv stack
        self.conv_layers = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection (to log-duration)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        phoneme_ids: Tensor,
        style: Tensor,
        phoneme_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Args:
            phoneme_ids: (batch, seq_len) phoneme indices
            style: (batch, style_dim) style conditioning
            phoneme_mask: (batch, seq_len) True for padding
            
        Returns:
            Dict with 'durations' and 'log_durations'
        """
        # Embed phonemes
        x = self.phoneme_embedding(phoneme_ids)  # (batch, seq, phoneme_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Project and condition
        x = self.input_proj(x)
        x = self.film(x, style)
        
        # Conv layers (need transpose for conv1d)
        x = x.transpose(1, 2)  # (batch, hidden, seq)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)  # (batch, seq, hidden)
        
        # Output
        log_durations = self.output_proj(x).squeeze(-1)  # (batch, seq)
        
        # Convert to actual durations (min 1 frame)
        durations = torch.exp(log_durations).clamp(min=1.0)
        
        # Apply mask
        if phoneme_mask is not None:
            durations = durations.masked_fill(phoneme_mask, 0.0)
            log_durations = log_durations.masked_fill(phoneme_mask, 0.0)
        
        return {
            'durations': durations,
            'log_durations': log_durations
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'phoneme_dim': self.phoneme_dim,
            'hidden_dim': self.hidden_dim
        }


# =============================================================================
# LENGTH REGULATOR
# =============================================================================

class LengthRegulator(nn.Module):
    """
    Expands phoneme-level features to frame-level using durations.
    
    Each phoneme embedding is repeated according to its duration.
    This is the bridge between phoneme and frame domains.
    """
    
    def forward(
        self,
        x: Tensor,
        durations: Tensor
    ) -> Tuple[Tensor, int]:
        """
        Args:
            x: (batch, seq_len, dim) phoneme features
            durations: (batch, seq_len) durations in frames
            
        Returns:
            Tuple of:
                - (batch, total_frames, dim) expanded features
                - total_frames (int)
        """
        batch_size, seq_len, dim = x.shape
        durations_int = durations.round().long().clamp(min=0)
        
        # Compute max frames
        total_frames = durations_int.sum(dim=-1).max().item()
        
        # Expand each sample
        outputs = []
        for b in range(batch_size):
            frames = []
            for i in range(seq_len):
                dur = durations_int[b, i].item()
                if dur > 0:
                    frames.append(x[b, i:i+1].repeat(dur, 1))
            
            if frames:
                sample = torch.cat(frames, dim=0)
                # Pad to max_frames
                if sample.shape[0] < total_frames:
                    pad = torch.zeros(total_frames - sample.shape[0], dim, device=x.device)
                    sample = torch.cat([sample, pad], dim=0)
                sample = sample[:total_frames]
            else:
                sample = torch.zeros(total_frames, dim, device=x.device)
            
            outputs.append(sample)
        
        return torch.stack(outputs), total_frames


# =============================================================================
# F0 PREDICTOR
# =============================================================================

@register_component('tts_predictor', 'f0')
class F0Predictor(nn.Module):
    """
    Predicts F0 (pitch) contour at frame level.
    
    Two outputs:
    - Voicing probability (is this frame voiced?)
    - Log-F0 value (pitch in log-Hz)
    
    F0 only exists for voiced sounds (vowels, voiced consonants).
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        style_dim: Style conditioning dimension
        n_layers: Number of conv layers
        f0_min: Minimum F0 in Hz
        f0_max: Maximum F0 in Hz
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        style_dim: int = 128,
        n_layers: int = 4,
        f0_min: float = 50.0,
        f0_max: float = 800.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.log_f0_min = math.log(f0_min)
        self.log_f0_max = math.log(f0_max)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # FiLM conditioning
        self.film = FiLMLayer(hidden_dim, style_dim)
        
        # Conv stack
        self.conv_layers = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, kernel_size=5, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Voicing head (binary classification)
        self.voicing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # F0 head (normalized log-F0)
        self.f0_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1], scale to log-F0 range
        )
    
    def forward(
        self,
        frame_features: Tensor,
        style: Tensor,
        frame_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Args:
            frame_features: (batch, n_frames, dim) frame-level features
            style: (batch, style_dim) style conditioning
            frame_mask: (batch, n_frames) True for padding
            
        Returns:
            Dict with 'f0', 'voicing', 'log_f0_normalized'
        """
        x = self.input_proj(frame_features)
        x = self.film(x, style)
        
        # Conv layers
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)
        
        # Predictions
        voicing = self.voicing_head(x).squeeze(-1)  # (batch, frames)
        log_f0_norm = self.f0_head(x).squeeze(-1)   # (batch, frames) in [0, 1]
        
        # Denormalize F0
        log_f0 = log_f0_norm * (self.log_f0_max - self.log_f0_min) + self.log_f0_min
        f0 = torch.exp(log_f0) * (voicing > 0.5).float()  # Zero for unvoiced
        
        # Apply mask
        if frame_mask is not None:
            f0 = f0.masked_fill(frame_mask, 0.0)
            voicing = voicing.masked_fill(frame_mask, 0.0)
        
        return {
            'f0': f0,
            'voicing': voicing,
            'log_f0_normalized': log_f0_norm
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'f0_min': self.f0_min,
            'f0_max': self.f0_max
        }


# =============================================================================
# ENERGY PREDICTOR
# =============================================================================

@register_component('tts_predictor', 'energy')
class EnergyPredictor(nn.Module):
    """
    Predicts energy (loudness) contour at frame level.
    
    Simpler than F0 since energy exists for all frames.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        style_dim: Style conditioning dimension
        n_layers: Number of conv layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        style_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # FiLM conditioning
        self.film = FiLMLayer(hidden_dim, style_dim)
        
        # Conv stack
        self.conv_layers = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Output (log-energy)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        frame_features: Tensor,
        style: Tensor,
        frame_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Args:
            frame_features: (batch, n_frames, dim)
            style: (batch, style_dim)
            frame_mask: (batch, n_frames) True for padding
            
        Returns:
            Dict with 'energy' and 'log_energy'
        """
        x = self.input_proj(frame_features)
        x = self.film(x, style)
        
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)
        
        log_energy = self.output_proj(x).squeeze(-1)
        energy = torch.exp(log_energy)
        
        if frame_mask is not None:
            energy = energy.masked_fill(frame_mask, 0.0)
            log_energy = log_energy.masked_fill(frame_mask, 0.0)
        
        return {
            'energy': energy,
            'log_energy': log_energy
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {}


# =============================================================================
# COMPLETE PROSODY SYSTEM
# =============================================================================

@register_component('tts_module', 'prosody_system')
class ProsodySystem(nn.Module):
    """
    Complete prosody prediction system.
    
    Pipeline: phonemes → durations → expand → F0 + energy
    
    Args:
        n_phonemes: Phoneme vocabulary size
        phoneme_dim: Phoneme embedding dimension
        hidden_dim: Hidden dimension for predictors
        style_dim: Style conditioning dimension
        n_duration_layers: Layers in duration predictor
        n_f0_layers: Layers in F0 predictor
        n_energy_layers: Layers in energy predictor
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_phonemes: int = 50,
        phoneme_dim: int = 256,
        hidden_dim: int = 256,
        style_dim: int = 128,
        n_duration_layers: int = 4,
        n_f0_layers: int = 4,
        n_energy_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.duration_predictor = DurationPredictor(
            n_phonemes=n_phonemes,
            phoneme_dim=phoneme_dim,
            hidden_dim=hidden_dim,
            style_dim=style_dim,
            n_layers=n_duration_layers,
            dropout=dropout
        )
        
        self.length_regulator = LengthRegulator()
        
        self.f0_predictor = F0Predictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            style_dim=style_dim,
            n_layers=n_f0_layers,
            dropout=dropout
        )
        
        self.energy_predictor = EnergyPredictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            style_dim=style_dim,
            n_layers=n_energy_layers,
            dropout=dropout
        )
        
        # Shared phoneme embedding with duration predictor
        self.phoneme_embedding = self.duration_predictor.phoneme_embedding
        
        # Frame-level feature projection
        self.frame_proj = nn.Linear(phoneme_dim, hidden_dim)
    
    def forward(
        self,
        phoneme_ids: Tensor,
        style: Tensor,
        target_durations: Optional[Tensor] = None,
        phoneme_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Predict all prosodic features.
        
        If target_durations provided (training), uses those.
        Otherwise uses predicted durations.
        
        Args:
            phoneme_ids: (batch, seq_len) phoneme indices
            style: (batch, style_dim) style conditioning
            target_durations: (batch, seq_len) ground truth durations
            phoneme_mask: (batch, seq_len) True for padding
            
        Returns:
            Dict with durations, f0, energy, and intermediate features
        """
        # Predict durations
        dur_out = self.duration_predictor(phoneme_ids, style, phoneme_mask)
        
        # Use target or predicted durations
        durations = target_durations if target_durations is not None else dur_out['durations']
        
        # Get phoneme embeddings and expand to frame level
        phoneme_embs = self.phoneme_embedding(phoneme_ids)
        frame_features, total_frames = self.length_regulator(phoneme_embs, durations)
        frame_features = self.frame_proj(frame_features)
        
        # Predict F0 and energy
        f0_out = self.f0_predictor(frame_features, style)
        energy_out = self.energy_predictor(frame_features, style)
        
        return {
            **dur_out,
            **f0_out,
            **energy_out,
            'frame_features': frame_features,
            'total_frames': total_frames
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'duration': self.duration_predictor.get_config(),
            'f0': self.f0_predictor.get_config(),
            'energy': self.energy_predictor.get_config()
        }