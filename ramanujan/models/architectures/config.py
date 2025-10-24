"""Model configuration system."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
import yaml


@dataclass
class TransformerConfig:
    """
    Configuration for Ramanujan Transformer models.
    
    This configuration supports:
    - Standard transformers (GPT, BERT-style)
    - Modern efficient transformers (LLaMA-style)
    - Custom configurations
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feedforward dimension (default: None, auto-calculated)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        
        # Architecture options
        norm_first: Use pre-norm (True) or post-norm (False)
        norm_type: Normalization type ('rms', 'layer')
        attention_type: Attention type ('mha', 'gqa', 'mqa')
        ffn_type: Feedforward type ('swiglu', 'gelu')
        n_kv_heads: Number of KV heads for GQA (None = auto)
        use_rope: Whether to use RoPE for position encoding
        rope_theta: RoPE theta parameter
        bias: Whether to use bias in linear layers
        
        # Special tokens
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        
        # Model behavior
        tie_word_embeddings: Tie input and output embeddings
    
    Example:
        >>> # LLaMA-style config
        >>> config = TransformerConfig.llama(
        ...     vocab_size=50000,
        ...     d_model=4096,
        ...     n_layers=32,
        ...     n_heads=32,
        ... )
        
        >>> # Custom config
        >>> config = TransformerConfig(
        ...     vocab_size=50000,
        ...     d_model=768,
        ...     n_layers=12,
        ...     n_heads=12,
        ... )
    """
    
    # Model architecture
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: Optional[int] = None
    max_seq_len: int = 2048
    dropout: float = 0.0
    
    # Architecture options
    norm_first: bool = True
    norm_type: str = 'rms'
    attention_type: str = 'mha'
    ffn_type: str = 'swiglu'
    n_kv_heads: Optional[int] = None
    use_rope: bool = True
    rope_theta: float = 10000.0
    bias: bool = False
    
    # Special tokens
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Model behavior
    tie_word_embeddings: bool = False
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Validate divisibility
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        
        # Set default d_ff based on ffn_type
        if self.d_ff is None:
            if self.ffn_type == 'swiglu':
                # SwiGLU uses ~8/3 * d_model
                self.d_ff = int(8 * self.d_model / 3)
                self.d_ff = ((self.d_ff + 255) // 256) * 256  # Round to 256
            else:
                # Standard FFN uses 4 * d_model
                self.d_ff = 4 * self.d_model
        
        # Set default n_kv_heads for GQA
        if self.attention_type in ['gqa', 'grouped_query']:
            if self.n_kv_heads is None:
                self.n_kv_heads = max(1, self.n_heads // 4)
        elif self.attention_type in ['mqa', 'multi_query']:
            self.n_kv_heads = 1
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'TransformerConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TransformerConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    # ========================================================================
    # PRESET CONFIGURATIONS
    # ========================================================================
    
    @classmethod
    def llama(
        cls,
        vocab_size: int,
        d_model: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048,
        **kwargs
    ) -> 'TransformerConfig':
        """
        LLaMA-style configuration.
        
        Features:
        - Pre-norm with RMSNorm
        - Grouped Query Attention (GQA)
        - SwiGLU feedforward
        - RoPE position encoding
        - No bias
        """
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 4)
        
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            norm_first=True,
            norm_type='rms',
            attention_type='gqa',
            ffn_type='swiglu',
            use_rope=True,
            bias=False,
            **kwargs
        )
    
    @classmethod
    def gpt(
        cls,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 1024,
        **kwargs
    ) -> 'TransformerConfig':
        """
        GPT-style configuration.
        
        Features:
        - Pre-norm with LayerNorm
        - Multi-Head Attention (MHA)
        - GELU feedforward
        - Learned position encoding
        - With bias
        """
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            norm_first=True,
            norm_type='layer',
            attention_type='mha',
            ffn_type='gelu',
            use_rope=False,  # GPT uses learned positions
            bias=True,
            **kwargs
        )
    
    @classmethod
    def bert(
        cls,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 512,
        **kwargs
    ) -> 'TransformerConfig':
        """
        BERT-style configuration.
        
        Features:
        - Post-norm with LayerNorm
        - Multi-Head Attention (MHA)
        - GELU feedforward
        - Learned position encoding
        - With bias
        """
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            norm_first=False,  # BERT uses post-norm
            norm_type='layer',
            attention_type='mha',
            ffn_type='gelu',
            use_rope=False,
            bias=True,
            **kwargs
        )
    
    @classmethod
    def tiny(cls, vocab_size: int, **kwargs) -> 'TransformerConfig':
        """Tiny model for testing (124M params)."""
        return cls.llama(
            vocab_size=vocab_size,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            **kwargs
        )
    
    @classmethod
    def small(cls, vocab_size: int, **kwargs) -> 'TransformerConfig':
        """Small model (350M params)."""
        return cls.llama(
            vocab_size=vocab_size,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            n_kv_heads=4,
            **kwargs
        )
    
    @classmethod
    def medium(cls, vocab_size: int, **kwargs) -> 'TransformerConfig':
        """Medium model (1B params)."""
        return cls.llama(
            vocab_size=vocab_size,
            d_model=2048,
            n_layers=24,
            n_heads=32,
            n_kv_heads=8,
            **kwargs
        )
    
    @classmethod
    def large(cls, vocab_size: int, **kwargs) -> 'TransformerConfig':
        """Large model (7B params)."""
        return cls.llama(
            vocab_size=vocab_size,
            d_model=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            **kwargs
        )