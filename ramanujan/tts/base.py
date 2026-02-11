"""
TTS Base Classes and Utilities
==============================

Shared infrastructure for TTS components following Candide protocols.
"""

from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import math

# Handle optional torch import
try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    Tensor = Any
    TORCH_AVAILABLE = False

# =============================================================================
# TENSOR SPEC (Candide protocol)
# =============================================================================

@dataclass(frozen=True)
class TensorSpec:
    """Specification for input/output tensors matching Candide's interface."""
    shape: Tuple[str, ...]
    dtype: Any = None  # torch.dtype when available
    optional: bool = False
    description: str = ""
    
    def __post_init__(self):
        if self.dtype is None and TORCH_AVAILABLE:
            object.__setattr__(self, 'dtype', torch.float32)
    
    def resolve_shape(self, dim_values: Dict[str, int]) -> Tuple[int, ...]:
        return tuple(dim_values.get(s, -1) for s in self.shape)


@dataclass
class ComputeCost:
    """Computational cost estimate."""
    flops: int = 0
    params: int = 0
    memory_bytes: int = 0
    
    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 ** 2)
    
    def __add__(self, other: 'ComputeCost') -> 'ComputeCost':
        return ComputeCost(
            flops=self.flops + other.flops,
            params=self.params + other.params,
            memory_bytes=self.memory_bytes + other.memory_bytes
        )
    
    def summary(self) -> str:
        return f"FLOPs: {self.flops:,} | Params: {self.params:,} | Memory: {self.memory_mb:.2f} MB"


# =============================================================================
# REGISTRY
# =============================================================================

# _REGISTRY: Dict[Tuple[str, str], type] = {}

# def register_component(category: str, name: str):
#     """Register a component. Replace with ramanujan.core.register_component in integration."""
#     def decorator(cls):
#         _REGISTRY[(category, name)] = cls
#         cls._component_category = category
#         cls._component_name = name
#         return cls
#     return decorator

# def create_component(category: str, name: str, **kwargs):
#     """Create a registered component."""
#     key = (category, name)
#     if key not in _REGISTRY:
#         raise KeyError(f"Component {category}/{name} not registered. Available: {list(_REGISTRY.keys())}")
#     return _REGISTRY[key](**kwargs)

# def list_components(category: Optional[str] = None) -> List[Tuple[str, str]]:
#     """List registered components."""
#     if category:
#         return [(c, n) for (c, n) in _REGISTRY.keys() if c == category]
#     return list(_REGISTRY.keys())


# =============================================================================
# AUDIO CONFIG
# =============================================================================

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    f0_min: float = 50.0
    f0_max: float = 800.0
    
    @property
    def fps(self) -> float:
        """Frames per second."""
        return self.sample_rate / self.hop_length
    
    @property
    def frame_duration_ms(self) -> float:
        """Duration of one frame in milliseconds."""
        return 1000 * self.hop_length / self.sample_rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_AUDIO_CONFIG = AudioConfig()


# =============================================================================
# COMMON BUILDING BLOCKS (require torch)
# =============================================================================

if TORCH_AVAILABLE:
    class ConvBlock(nn.Module):
        """1D convolution block with normalization and activation."""
        
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1):
            super().__init__()
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
            self.norm = nn.LayerNorm(out_ch)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x: Tensor) -> Tensor:
            """x: (batch, channels, seq)"""
            x = self.conv(x)
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
            x = self.act(x)
            return self.dropout(x)


    class FiLMLayer(nn.Module):
        """Feature-wise Linear Modulation for conditioning."""
        
        def __init__(self, feature_dim: int, cond_dim: int):
            super().__init__()
            self.proj = nn.Linear(cond_dim, feature_dim * 2)
        
        def forward(self, x: Tensor, cond: Tensor) -> Tensor:
            gamma, beta = self.proj(cond).chunk(2, dim=-1)
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
            return x + self.pe[:, :x.size(1)]


    class TimeEmbedding(nn.Module):
        """Sinusoidal time embedding for flow matching."""
        
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        
        def forward(self, t: Tensor) -> Tensor:
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
            args = t.unsqueeze(-1) * freqs
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            return self.mlp(emb)
else:
    ConvBlock = None
    FiLMLayer = None
    PositionalEncoding = None
    TimeEmbedding = None