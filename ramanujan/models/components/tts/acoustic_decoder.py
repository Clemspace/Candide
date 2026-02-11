"""
TTS Acoustic Decoder
====================

Flow Matching decoder: converts prosodic features to mel spectrograms.

Flow Matching is more efficient than diffusion:
- Linear interpolation path (not noisy diffusion)
- Fewer steps needed (10 vs 1000)
- Better for small-scale models

Candide Protocol compliant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple
import math

from ramanujan.core.interface import TensorSpec
from ramanujan.core.cost_estimation import ComputeCost
from ramanujan.core.registry import register_component
from ramanujan.tts.base import TimeEmbedding


# =============================================================================
# UNET BUILDING BLOCKS
# =============================================================================

class ResBlock(nn.Module):
    """Residual block with time and style conditioning."""
    
    def __init__(self, channels: int, time_dim: int, style_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels * 2)
        self.style_proj = nn.Linear(style_dim, channels * 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, time_emb: Tensor, style_emb: Tensor) -> Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        # Conditioning
        t_s, t_b = self.time_proj(time_emb).chunk(2, dim=-1)
        s_s, s_b = self.style_proj(style_emb).chunk(2, dim=-1)
        scale = (t_s + s_s).unsqueeze(-1)
        bias = (t_b + s_b).unsqueeze(-1)
        h = h * (1 + scale) + bias
        
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class DownBlock(nn.Module):
    """Downsample block with channel projection."""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, style_dim: int, 
                 n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        # Project channels first, then apply ResBlocks
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.res_blocks = nn.ModuleList([
            ResBlock(out_ch, time_dim, style_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.down = nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)
    
    def forward(self, x: Tensor, t: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.proj(x)  # Project to out_ch first
        for block in self.res_blocks:
            x = block(x, t, s)
        skip = x  # Skip connection before downsampling
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsample block with skip connection."""
    
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, style_dim: int,
                 n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)
        # After concat with skip: in_ch + skip_ch channels
        self.proj = nn.Conv1d(in_ch + skip_ch, out_ch, 1)
        self.res_blocks = nn.ModuleList([
            ResBlock(out_ch, time_dim, style_dim, dropout)
            for _ in range(n_blocks)
        ])
    
    def forward(self, x: Tensor, skip: Tensor, t: Tensor, s: Tensor) -> Tensor:
        x = self.up(x)
        # Handle size mismatch from downsampling
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
        x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
        x = self.proj(x)  # Project to out_ch
        for block in self.res_blocks:
            x = block(x, t, s)
        return x


class MiddleBlock(nn.Module):
    """Middle block with self-attention."""
    
    def __init__(self, channels: int, time_dim: int, style_dim: int, 
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.res1 = ResBlock(channels, time_dim, style_dim, dropout)
        self.attn_norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.res2 = ResBlock(channels, time_dim, style_dim, dropout)
    
    def forward(self, x: Tensor, t: Tensor, s: Tensor) -> Tensor:
        x = self.res1(x, t, s)
        h = self.attn_norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        x = x + h.transpose(1, 2)
        return self.res2(x, t, s)


# =============================================================================
# FLOW MATCHING DECODER
# =============================================================================

@register_component('tts.decoder', 'flow_matching')
class FlowMatchingDecoder(nn.Module):
    """
    Flow Matching decoder for mel spectrogram generation.
    
    Training:
        x_t = t * x_target + (1-t) * noise
        v_pred = network(x_t, t, conditioning)
        loss = MSE(v_pred, x_target - noise)
    
    Inference:
        x_0 = noise
        for t in [0, 1] with n_steps:
            x_{t+dt} = x_t + v(x_t, t) * dt
        return x_1
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        hidden_channels: int = 256,
        time_dim: int = 256,
        style_dim: int = 256,
        prosody_dim: int = 256,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        n_res_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_channels = hidden_channels
        self.channel_mults = channel_mults
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        
        # Input: mel + prosody + f0 + energy
        input_ch = n_mels + prosody_dim + 2
        self.input_proj = nn.Conv1d(input_ch, hidden_channels, 3, padding=1)
        
        # Build encoder and track channels for skip connections
        self.down_blocks = nn.ModuleList()
        skip_channels = []  # Track skip connection channels
        ch = hidden_channels
        
        for mult in channel_mults:
            out_ch = hidden_channels * mult
            self.down_blocks.append(
                DownBlock(ch, out_ch, time_dim, style_dim, n_res_blocks, dropout)
            )
            skip_channels.append(out_ch)  # Skip has out_ch channels
            ch = out_ch
        
        # Middle block
        self.middle = MiddleBlock(ch, time_dim, style_dim, dropout=dropout)
        
        # Build decoder with correct skip channel sizes
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = hidden_channels * mult
            skip_ch = skip_channels[-(i+1)]  # Get corresponding skip channels
            self.up_blocks.append(
                UpBlock(ch, skip_ch, out_ch, time_dim, style_dim, n_res_blocks, dropout)
            )
            ch = out_ch
        
        # Output projection
        self.output_norm = nn.GroupNorm(8, hidden_channels)
        self.output_proj = nn.Conv1d(hidden_channels, n_mels, 3, padding=1)
    
    @property
    def component_type(self) -> str:
        return 'decoder'
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'noisy_mel': TensorSpec(shape=('batch', 'n_mels', 'frames')),
            'frame_features': TensorSpec(shape=('batch', 'prosody_dim', 'frames')),
            'f0': TensorSpec(shape=('batch', 'frames')),
            'energy': TensorSpec(shape=('batch', 'frames')),
            'style': TensorSpec(shape=('batch', 'style_dim')),
            't': TensorSpec(shape=('batch',))
        }
    
    @property
    def output_spec(self) -> Dict[str, TensorSpec]:
        return {
            'velocity': TensorSpec(shape=('batch', 'n_mels', 'frames'))
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'n_mels': self.n_mels,
            'hidden_channels': self.hidden_channels,
            'channel_mults': self.channel_mults
        }
    
    @staticmethod
    def estimate_cost(input_shapes: Dict[str, Tuple], config: Dict[str, Any]) -> ComputeCost:
        h = config['hidden_channels']
        mults = config.get('channel_mults', (1, 2, 4))
        # Rough estimate
        params = h * h * 10 * sum(m**2 for m in mults)
        return ComputeCost(flops=params * 2, params=params, memory_bytes=params * 4)
    
    def forward(
        self,
        noisy_mel: Tensor,
        frame_features: Tensor,
        f0: Tensor,
        energy: Tensor,
        style: Tensor,
        t: Tensor
    ) -> Dict[str, Tensor]:
        """Predict velocity for flow matching."""
        time_emb = self.time_embed(t)
        
        # Concatenate inputs: [mel, prosody, f0, energy]
        x = torch.cat([
            noisy_mel,
            frame_features,
            f0.unsqueeze(1),
            energy.unsqueeze(1)
        ], dim=1)
        x = self.input_proj(x)
        
        # Encoder with skip connections
        skips = []
        for down in self.down_blocks:
            x, skip = down(x, time_emb, style)
            skips.append(skip)
        
        # Middle
        x = self.middle(x, time_emb, style)
        
        # Decoder with skip connections (reversed order)
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, time_emb, style)
        
        # Output
        velocity = self.output_proj(F.silu(self.output_norm(x)))
        return {'velocity': velocity}
    
    def compute_loss(
        self,
        target_mel: Tensor,
        frame_features: Tensor,
        f0: Tensor,
        energy: Tensor,
        style: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute flow matching training loss.
        
        Returns dict with 'loss' and 'velocity_mse'.
        """
        batch_size = target_mel.shape[0]
        device = target_mel.device
        
        # Sample noise and timestep
        noise = torch.randn_like(target_mel)
        t = torch.rand(batch_size, device=device)
        t_exp = t.view(-1, 1, 1)
        
        # Interpolate: x_t = t * target + (1-t) * noise
        noisy_mel = t_exp * target_mel + (1 - t_exp) * noise
        
        # True velocity is (target - noise)
        true_velocity = target_mel - noise
        
        # Predict velocity
        out = self.forward(noisy_mel, frame_features, f0, energy, style, t)
        pred_velocity = out['velocity']
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, true_velocity)
        
        return {
            'loss': loss,
            'velocity_mse': loss.detach()
        }
    
    @torch.no_grad()
    def generate(
        self,
        frame_features: Tensor,
        f0: Tensor,
        energy: Tensor,
        style: Tensor,
        n_steps: int = 10
    ) -> Dict[str, Tensor]:
        """Generate mel spectrogram via Euler integration."""
        batch_size = frame_features.shape[0]
        n_frames = frame_features.shape[-1]
        device = frame_features.device
        
        # Start from noise
        x = torch.randn(batch_size, self.n_mels, n_frames, device=device)
        
        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            out = self.forward(x, frame_features, f0, energy, style, t)
            x = x + out['velocity'] * dt
        
        return {'mel': x}