"""
TTS Style Encoders - Candide Compliant
======================================

Style encoding components following Candide's protocol-based composition:
- TensorSpec for input/output specifications
- estimate_cost() for computational cost estimation
- get_config() for serialization
- Proper registration with @register_component

Components:
- AccentEncoder: Regional accent embedding
- EmotionEncoder: Emotion with intensity
- SpeakerEncoder: Speaker identity (lookup or encoder)
- StyleFusion: Fuses all style factors
- StyleSystem: Complete style encoding pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import math

from ramanujan.core.interface import TensorSpec
from ramanujan.core.cost_estimation import ComputeCost
from ramanujan.core.registry import register_component

# =============================================================================
# ACCENT ENCODER
# =============================================================================

@register_component('tts.style', 'accent_encoder')
class AccentEncoder(nn.Module):
    """
    Encodes regional accent to embedding vector.
    
    Accents affect pronunciation patterns, vowel quality, and prosodic contours.
    For French: Parisian, Southern, Quebec, Belgian, Swiss, West African, Neutral.
    """
    
    def __init__(
        self,
        n_accents: int = 7,
        embedding_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_accents = n_accents
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(n_accents, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
    
    # ─── Candide Protocol Implementation ───────────────────────────────────
    
    @property
    def component_type(self) -> str:
        return 'encoder'
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'accent_id': TensorSpec(
                shape=('batch',),
                dtype=torch.long,
                description="Accent category index"
            )
        }
    
    @property
    def output_spec(self) -> Dict[str, TensorSpec]:
        return {
            'accent_emb': TensorSpec(
                shape=('batch', 'accent_dim'),
                dtype=torch.float32,
                description="Accent embedding"
            )
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        return {
            'n_accents': self.n_accents,
            'embedding_dim': self.embedding_dim,
            'dropout': self.dropout.p
        }
    
    @staticmethod
    def estimate_cost(input_shapes: Dict[str, Tuple], config: Dict[str, Any]) -> ComputeCost:
        """Estimate computational cost."""
        n_accents = config['n_accents']
        embedding_dim = config['embedding_dim']
        batch_size = input_shapes.get('accent_id', (1,))[0]
        
        params = n_accents * embedding_dim + embedding_dim  # embedding + layernorm
        flops = batch_size * embedding_dim * 2  # lookup + layernorm
        memory = params * 4  # float32
        
        return ComputeCost(flops=flops, params=params, memory_bytes=memory)
    
    # ─── Forward Pass ──────────────────────────────────────────────────────
    
    def forward(self, accent_id: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            accent_id: (batch,) accent indices
            
        Returns:
            Dict with 'accent_emb': (batch, embedding_dim)
        """
        emb = self.embedding(accent_id)
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return {'accent_emb': emb}


# =============================================================================
# EMOTION ENCODER
# =============================================================================

@register_component('tts.style', 'emotion_encoder')
class EmotionEncoder(nn.Module):
    """
    Encodes emotion with intensity control.
    
    Supports categorical emotions (joy, sadness, anger, etc.) 
    and continuous valence/arousal representation.
    """
    
    def __init__(
        self,
        n_emotions: int = 9,
        embedding_dim: int = 32,
        use_intensity: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_emotions = n_emotions
        self.embedding_dim = embedding_dim
        self.use_intensity = use_intensity
        
        # Categorical pathway
        cat_dim = embedding_dim - 8 if use_intensity else embedding_dim
        self.category_embedding = nn.Embedding(n_emotions, cat_dim)
        
        # Intensity pathway
        if use_intensity:
            self.intensity_encoder = nn.Sequential(
                nn.Linear(1, 8),
                nn.Tanh()
            )
        
        # Continuous V/A pathway
        self.continuous_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.normal_(self.category_embedding.weight, mean=0, std=0.1)
    
    @property
    def component_type(self) -> str:
        return 'encoder'
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'emotion_id': TensorSpec(
                shape=('batch',), dtype=torch.long, optional=True,
                description="Emotion category index"
            ),
            'intensity': TensorSpec(
                shape=('batch',), dtype=torch.float32, optional=True,
                description="Emotion intensity 0-1"
            ),
            'valence': TensorSpec(
                shape=('batch',), dtype=torch.float32, optional=True,
                description="Valence -1 to 1 (continuous mode)"
            ),
            'arousal': TensorSpec(
                shape=('batch',), dtype=torch.float32, optional=True,
                description="Arousal -1 to 1 (continuous mode)"
            )
        }
    
    @property
    def output_spec(self) -> Dict[str, TensorSpec]:
        return {
            'emotion_emb': TensorSpec(
                shape=('batch', 'emotion_dim'),
                description="Emotion embedding"
            )
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'n_emotions': self.n_emotions,
            'embedding_dim': self.embedding_dim,
            'use_intensity': self.use_intensity
        }
    
    @staticmethod
    def estimate_cost(input_shapes: Dict[str, Tuple], config: Dict[str, Any]) -> ComputeCost:
        n_emotions = config['n_emotions']
        embedding_dim = config['embedding_dim']
        batch_size = input_shapes.get('emotion_id', (1,))[0]
        
        params = n_emotions * embedding_dim + 3 * embedding_dim * 2 + embedding_dim
        flops = batch_size * embedding_dim * 4
        
        return ComputeCost(flops=flops, params=params, memory_bytes=params * 4)
    
    def forward(
        self,
        emotion_id: Optional[Tensor] = None,
        intensity: Optional[Tensor] = None,
        valence: Optional[Tensor] = None,
        arousal: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Encode emotion to embedding.
        
        Continuous mode (valence/arousal) takes precedence if provided.
        """
        # Continuous mode
        if valence is not None and arousal is not None:
            if intensity is None:
                intensity = torch.ones_like(valence) * 0.5
            continuous = torch.stack([valence, arousal, intensity], dim=-1)
            emb = self.continuous_encoder(continuous)
        
        # Categorical mode
        elif emotion_id is not None:
            cat_emb = self.category_embedding(emotion_id)
            
            if self.use_intensity:
                if intensity is None:
                    intensity = torch.ones(emotion_id.shape[0], device=emotion_id.device) * 0.5
                int_emb = self.intensity_encoder(intensity.unsqueeze(-1))
                emb = torch.cat([cat_emb, int_emb], dim=-1)
            else:
                emb = cat_emb
        else:
            raise ValueError("Must provide emotion_id or (valence, arousal)")
        
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return {'emotion_emb': emb}


# =============================================================================
# SPEAKER ENCODER
# =============================================================================

@register_component('tts.style', 'speaker_encoder')
class SpeakerEncoder(nn.Module):
    """
    Encodes speaker identity.
    
    Two modes:
    - 'lookup': Embedding table for known speakers
    - 'encoder': Extract speaker embedding from reference audio (voice cloning)
    """
    
    def __init__(
        self,
        n_speakers: int = 1,
        embedding_dim: int = 128,
        mode: str = 'lookup',  # 'lookup' or 'encoder'
        dropout: float = 0.1,
        n_mels: int = 80
    ):
        super().__init__()
        self.n_speakers = n_speakers
        self.embedding_dim = embedding_dim
        self.mode = mode
        
        if mode == 'lookup':
            self.embedding = nn.Embedding(n_speakers, embedding_dim)
            nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        elif mode == 'encoder':
            # Simple mel encoder - use pretrained ECAPA-TDNN in production
            self.encoder = nn.Sequential(
                nn.Conv1d(n_mels, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=5, padding=2, stride=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=5, padding=2, stride=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, embedding_dim)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    @property
    def component_type(self) -> str:
        return 'encoder'
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        if self.mode == 'lookup':
            return {
                'speaker_id': TensorSpec(
                    shape=('batch',), dtype=torch.long,
                    description="Speaker index"
                )
            }
        else:
            return {
                'reference_mel': TensorSpec(
                    shape=('batch', 'n_mels', 'ref_frames'),
                    description="Reference mel spectrogram for voice cloning"
                )
            }
    
    @property
    def output_spec(self) -> Dict[str, TensorSpec]:
        return {
            'speaker_emb': TensorSpec(
                shape=('batch', 'speaker_dim'),
                description="Speaker embedding"
            )
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'n_speakers': self.n_speakers,
            'embedding_dim': self.embedding_dim,
            'mode': self.mode
        }
    
    @staticmethod
    def estimate_cost(input_shapes: Dict[str, Tuple], config: Dict[str, Any]) -> ComputeCost:
        mode = config['mode']
        embedding_dim = config['embedding_dim']
        
        if mode == 'lookup':
            params = config['n_speakers'] * embedding_dim
            flops = embedding_dim
        else:
            # Rough estimate for encoder
            params = 128 * 80 * 5 + 256 * 128 * 5 + 256 * 256 * 5 + 256 * embedding_dim
            flops = params * 2  # Approximate
        
        return ComputeCost(flops=flops, params=params, memory_bytes=params * 4)
    
    def forward(
        self,
        speaker_id: Optional[Tensor] = None,
        reference_mel: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Encode speaker identity."""
        if self.mode == 'lookup':
            if speaker_id is None:
                raise ValueError("speaker_id required for lookup mode")
            emb = self.embedding(speaker_id)
        else:
            if reference_mel is None:
                raise ValueError("reference_mel required for encoder mode")
            emb = self.encoder(reference_mel)
        
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return {'speaker_emb': emb}


# =============================================================================
# STYLE FUSION
# =============================================================================

@register_component('tts.style', 'style_fusion')
class StyleFusion(nn.Module):
    """
    Fuses accent, emotion, and speaker embeddings.
    
    Uses attention to learn interactions between style factors.
    Produces separate projections for different pipeline stages.
    """
    
    def __init__(
        self,
        accent_dim: int = 64,
        emotion_dim: int = 32,
        speaker_dim: int = 128,
        hidden_dim: int = 256,
        output_dims: Optional[Dict[str, int]] = None,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.accent_dim = accent_dim
        self.emotion_dim = emotion_dim
        self.speaker_dim = speaker_dim
        self.hidden_dim = hidden_dim
        
        self.output_dims = output_dims or {
            'prosody': 128,
            'acoustic': 256,
            'combined': 256
        }
        
        total_input = accent_dim + emotion_dim + speaker_dim
        
        self.input_proj = nn.Linear(total_input, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.output_projections = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim)
            for name, dim in self.output_dims.items()
        })
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def component_type(self) -> str:
        return 'fusion'
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'accent_emb': TensorSpec(shape=('batch', 'accent_dim')),
            'emotion_emb': TensorSpec(shape=('batch', 'emotion_dim')),
            'speaker_emb': TensorSpec(shape=('batch', 'speaker_dim'))
        }
    
    @property
    def output_spec(self) -> Dict[str, TensorSpec]:
        return {
            f'style_for_{name}': TensorSpec(shape=('batch', f'{name}_dim'))
            for name in self.output_dims.keys()
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'accent_dim': self.accent_dim,
            'emotion_dim': self.emotion_dim,
            'speaker_dim': self.speaker_dim,
            'hidden_dim': self.hidden_dim,
            'output_dims': self.output_dims
        }
    
    @staticmethod
    def estimate_cost(input_shapes: Dict[str, Tuple], config: Dict[str, Any]) -> ComputeCost:
        hidden = config['hidden_dim']
        total_in = config['accent_dim'] + config['emotion_dim'] + config['speaker_dim']
        
        params = (
            total_in * hidden +  # input proj
            4 * hidden * hidden +  # attention
            2 * hidden * hidden * 4 +  # ffn
            sum(hidden * d for d in config['output_dims'].values())  # output projs
        )
        return ComputeCost(flops=params * 2, params=params, memory_bytes=params * 4)
    
    def forward(
        self,
        accent_emb: Tensor,
        emotion_emb: Tensor,
        speaker_emb: Tensor
    ) -> Dict[str, Tensor]:
        """Fuse style embeddings."""
        combined = torch.cat([accent_emb, emotion_emb, speaker_emb], dim=-1)
        x = self.input_proj(combined).unsqueeze(1)
        
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        x = x.squeeze(1)
        
        return {
            f'style_for_{name}': proj(x)
            for name, proj in self.output_projections.items()
        }


# =============================================================================
# COMPLETE STYLE SYSTEM
# =============================================================================

@register_component('tts.module', 'style_system')
class StyleSystem(nn.Module):
    """
    Complete style encoding system combining all encoders with fusion.
    
    This is a composite component that wraps the individual encoders.
    Can be used as a single node in a ComputationGraph.
    """
    
    def __init__(
        self,
        n_accents: int = 7,
        n_emotions: int = 9,
        n_speakers: int = 1,
        accent_dim: int = 64,
        emotion_dim: int = 32,
        speaker_dim: int = 128,
        fusion_hidden_dim: int = 256,
        speaker_mode: str = 'lookup',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.accent_encoder = AccentEncoder(n_accents, accent_dim, dropout)
        self.emotion_encoder = EmotionEncoder(n_emotions, emotion_dim, dropout=dropout)
        self.speaker_encoder = SpeakerEncoder(n_speakers, speaker_dim, speaker_mode, dropout)
        self.fusion = StyleFusion(
            accent_dim, emotion_dim, speaker_dim, fusion_hidden_dim, dropout=dropout
        )
        
        # Store config for serialization
        self._config = {
            'n_accents': n_accents,
            'n_emotions': n_emotions,
            'n_speakers': n_speakers,
            'accent_dim': accent_dim,
            'emotion_dim': emotion_dim,
            'speaker_dim': speaker_dim,
            'fusion_hidden_dim': fusion_hidden_dim,
            'speaker_mode': speaker_mode,
            'dropout': dropout
        }
    
    @property
    def component_type(self) -> str:
        return 'system'
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'accent_id': TensorSpec(shape=('batch',), dtype=torch.long),
            'speaker_id': TensorSpec(shape=('batch',), dtype=torch.long, optional=True),
            'emotion_id': TensorSpec(shape=('batch',), dtype=torch.long, optional=True),
            'emotion_intensity': TensorSpec(shape=('batch',), optional=True),
            'reference_mel': TensorSpec(shape=('batch', 'n_mels', 'ref_frames'), optional=True)
        }
    
    @property
    def output_spec(self) -> Dict[str, TensorSpec]:
        return {
            'accent_emb': TensorSpec(shape=('batch', 'accent_dim')),
            'emotion_emb': TensorSpec(shape=('batch', 'emotion_dim')),
            'speaker_emb': TensorSpec(shape=('batch', 'speaker_dim')),
            'style_for_prosody': TensorSpec(shape=('batch', 'prosody_style_dim')),
            'style_for_acoustic': TensorSpec(shape=('batch', 'acoustic_style_dim')),
            'style_for_combined': TensorSpec(shape=('batch', 'combined_style_dim'))
        }
    
    def get_config(self) -> Dict[str, Any]:
        return self._config.copy()
    
    @staticmethod
    def estimate_cost(input_shapes: Dict[str, Tuple], config: Dict[str, Any]) -> ComputeCost:
        # Sum of sub-component costs
        accent_cost = AccentEncoder.estimate_cost(input_shapes, {
            'n_accents': config['n_accents'],
            'embedding_dim': config['accent_dim']
        })
        emotion_cost = EmotionEncoder.estimate_cost(input_shapes, {
            'n_emotions': config['n_emotions'],
            'embedding_dim': config['emotion_dim']
        })
        speaker_cost = SpeakerEncoder.estimate_cost(input_shapes, {
            'n_speakers': config['n_speakers'],
            'embedding_dim': config['speaker_dim'],
            'mode': config.get('speaker_mode', 'lookup')
        })
        fusion_cost = StyleFusion.estimate_cost(input_shapes, {
            'accent_dim': config['accent_dim'],
            'emotion_dim': config['emotion_dim'],
            'speaker_dim': config['speaker_dim'],
            'hidden_dim': config['fusion_hidden_dim'],
            'output_dims': {'prosody': 128, 'acoustic': 256, 'combined': 256}
        })
        
        return accent_cost + emotion_cost + speaker_cost + fusion_cost
    
    def forward(
        self,
        accent_id: Tensor,
        speaker_id: Optional[Tensor] = None,
        emotion_id: Optional[Tensor] = None,
        emotion_intensity: Optional[Tensor] = None,
        reference_mel: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Encode all style factors and fuse.
        
        Returns dict with individual embeddings plus fused style vectors.
        """
        accent_out = self.accent_encoder(accent_id)
        
        emotion_out = self.emotion_encoder(
            emotion_id=emotion_id,
            intensity=emotion_intensity
        )
        
        if self.speaker_encoder.mode == 'lookup':
            speaker_out = self.speaker_encoder(speaker_id=speaker_id)
        else:
            speaker_out = self.speaker_encoder(reference_mel=reference_mel)
        
        fused = self.fusion(
            accent_out['accent_emb'],
            emotion_out['emotion_emb'],
            speaker_out['speaker_emb']
        )
        
        return {
            **accent_out,
            **emotion_out,
            **speaker_out,
            **fused
        }

# =============================================================================
# PHASE 1 STYLE SYSTEM
# =============================================================================

@register_component('tts.module', 'style_system_phase1')
class StyleSystemPhase1(nn.Module):
    """
    Simplified style system for Phase 1 training.
    
    Only speaker embedding, no emotion/accent control.
    Style vector conditions prosody and acoustic systems.
    """
    
    def __init__(
        self,
        n_speakers: int = 1,
        speaker_dim: int = 256,
        style_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_speakers = n_speakers
        self.speaker_dim = speaker_dim
        self.style_dim = style_dim
        
        self.speaker_encoder = SpeakerEncoder(n_speakers, speaker_dim, dropout=dropout)
        
        # Project to style vectors for different stages
        self.proj_prosody = nn.Linear(speaker_dim, style_dim)
        self.proj_acoustic = nn.Linear(speaker_dim, style_dim)
    
    @property
    def component_type(self) -> str:
        return 'system'
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        return {
            'speaker_id': TensorSpec(shape=('batch',), dtype=torch.long)
        }
    
    @property
    def output_spec(self) -> Dict[str, TensorSpec]:
        return {
            'speaker_emb': TensorSpec(shape=('batch', 'speaker_dim')),
            'style_for_prosody': TensorSpec(shape=('batch', 'style_dim')),
            'style_for_acoustic': TensorSpec(shape=('batch', 'style_dim'))
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'n_speakers': self.n_speakers,
            'speaker_dim': self.speaker_dim,
            'style_dim': self.style_dim
        }
    
    @staticmethod
    def estimate_cost(input_shapes: Dict[str, Tuple], config: Dict[str, Any]) -> ComputeCost:
        speaker_cost = SpeakerEncoder.estimate_cost(input_shapes, {
            'n_speakers': config['n_speakers'],
            'embedding_dim': config['speaker_dim']
        })
        proj_params = 2 * config['speaker_dim'] * config['style_dim']
        return speaker_cost + ComputeCost(params=proj_params, memory_bytes=proj_params * 4)
    
    def forward(self, speaker_id: Tensor) -> Dict[str, Tensor]:
        speaker_out = self.speaker_encoder(speaker_id=speaker_id)
        speaker_emb = speaker_out['speaker_emb']
        
        return {
            'speaker_emb': speaker_emb,
            'style_for_prosody': self.proj_prosody(speaker_emb),
            'style_for_acoustic': self.proj_acoustic(speaker_emb)
        }