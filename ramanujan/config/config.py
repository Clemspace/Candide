"""
Candide Configuration System
=============================

A unified, scalable configuration system for the entire framework.

Design Principles:
1. Single source of truth for all configurations
2. Modular: Model, Training, Data configs are separate but composable
3. Extensible: Easy to add new config types without breaking existing code
4. Type-safe: Use dataclasses with validation
5. Architecture-agnostic: Works with any model architecture
6. Backward compatible: Handles multiple naming conventions

Usage:
    # Simple usage
    config = ExperimentConfig.from_yaml('config.yaml')
    
    # Access model config
    model = create_model(config.model)
    
    # Access training config
    optimizer = setup_optimizer(model, config.training)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union, Type
from pathlib import Path
import json
import yaml
from abc import ABC, abstractmethod


# ============================================================================
# BASE CLASSES (Foundation for extensibility)
# ============================================================================

class BaseConfig(ABC):
    """
    Base class for all configurations.
    
    All config classes should inherit from this to ensure consistency.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        pass
    
    def to_json(self, path: Union[str, Path]):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_yaml(self, path: Union[str, Path]):
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'BaseConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'BaseConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


# ============================================================================
# MODEL CONFIGURATION (Architecture-specific)
# ============================================================================

@dataclass
class ModelConfig(BaseConfig):
    """
    Unified model configuration.
    
    This config uses a canonical naming convention internally but accepts
    multiple naming conventions via aliases. This makes it user-friendly
    while maintaining consistency internally.
    
    Canonical names (used internally):
        - d_model: model dimension
        - n_layers: number of layers
        - n_heads: number of attention heads
        - n_kv_heads: number of key-value heads (for GQA)
        - d_ff: feedforward dimension
    
    Accepted aliases:
        - dim, hidden_size → d_model
        - num_layers → n_layers
        - num_heads → n_heads
        - num_kv_heads → n_kv_heads
        - ffn_dim, hidden_dim → d_ff
    
    This allows configs from different sources (papers, frameworks) to work
    seamlessly with our system.
    """
    
    # Core architecture (required)
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    
    # Optional architecture parameters
    d_ff: Optional[int] = None  # Auto-calculated if None
    n_kv_heads: Optional[int] = None  # Auto-set based on attention_type
    max_seq_len: int = 2048
    dropout: float = 0.0
    
    # Architecture options (modular components)
    norm_first: bool = True
    norm_type: str = 'rms'  # 'rms', 'layer', 'none'
    attention_type: str = 'mha'  # 'mha', 'gqa', 'mqa', 'flash'
    ffn_type: str = 'swiglu'  # 'swiglu', 'gelu', 'geglu', 'relu'
    use_rope: bool = True
    rope_theta: float = 10000.0
    bias: bool = False
    
    # Special tokens
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Model behavior
    tie_word_embeddings: bool = False
    
    # Extensibility
    custom_components: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set smart defaults."""
        # Validate divisibility
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        
        # Auto-calculate d_ff based on ffn_type
        if self.d_ff is None:
            if self.ffn_type in ['swiglu', 'geglu']:
                # SwiGLU uses ~8/3 * d_model
                self.d_ff = int(8 * self.d_model / 3)
                # Round to 256 for better hardware utilization
                self.d_ff = ((self.d_ff + 255) // 256) * 256
            else:
                # Standard FFN uses 4 * d_model
                self.d_ff = 4 * self.d_model
        
        # Auto-set n_kv_heads based on attention_type
        if self.n_kv_heads is None:
            if self.attention_type in ['gqa', 'grouped_query']:
                self.n_kv_heads = max(1, self.n_heads // 4)
            elif self.attention_type in ['mqa', 'multi_query']:
                self.n_kv_heads = 1
            else:
                self.n_kv_heads = self.n_heads
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads
    
    @property
    def is_gqa(self) -> bool:
        """Check if using Grouped Query Attention."""
        return self.n_kv_heads is not None and self.n_kv_heads < self.n_heads
    
    @property
    def is_mqa(self) -> bool:
        """Check if using Multi-Query Attention."""
        return self.n_kv_heads == 1
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary with alias handling."""
        config_dict = cls._handle_aliases(config_dict.copy())
        return cls(**config_dict)
    
    @classmethod
    def _handle_aliases(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle different naming conventions.
        
        This makes the framework flexible and user-friendly while maintaining
        internal consistency.
        """
        aliases = {
            # Model dimension
            'dim': 'd_model',
            'hidden_size': 'd_model',
            'hidden_dim': 'd_model',
            
            # Layer count
            'num_layers': 'n_layers',
            'depth': 'n_layers',
            
            # Attention heads
            'num_heads': 'n_heads',
            'num_attention_heads': 'n_heads',
            
            # KV heads
            'num_kv_heads': 'n_kv_heads',
            'num_key_value_heads': 'n_kv_heads',
            
            # Feedforward dimension
            'ffn_dim': 'd_ff',
            'hidden_dim': 'd_ff',
            'intermediate_size': 'd_ff',
            
            # Sequence length
            'seq_len': 'max_seq_len',
            'context_length': 'max_seq_len',
            'max_position_embeddings': 'max_seq_len',
        }
        
        for old_name, new_name in aliases.items():
            if old_name in config_dict and new_name not in config_dict:
                config_dict[new_name] = config_dict.pop(old_name)
        
        return config_dict
    
    # Preset configurations for common architectures
    @classmethod
    def llama_style(
        cls,
        vocab_size: int,
        d_model: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        **kwargs
    ) -> 'ModelConfig':
        """LLaMA-style config (modern, efficient)."""
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            norm_first=True,
            norm_type='rms',
            attention_type='gqa',
            ffn_type='swiglu',
            use_rope=True,
            bias=False,
            **kwargs
        )
    
    @classmethod
    def gpt_style(
        cls,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        **kwargs
    ) -> 'ModelConfig':
        """GPT-style config (classic transformer)."""
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            norm_first=True,
            norm_type='layer',
            attention_type='mha',
            ffn_type='gelu',
            use_rope=False,
            bias=True,
            **kwargs
        )


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig(BaseConfig):
    """
    Training configuration.
    
    Covers all aspects of the training process.
    """
    
    # Training duration
    max_steps: int = 10000
    max_epochs: Optional[int] = None  # Alternative to max_steps
    
    # Batch configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Optimization
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd', 'muon'
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Learning rate schedule
    lr_scheduler: str = 'warmup_stable_decay'  # 'warmup_cosine', 'constant', 'linear'
    warmup_steps: int = 500
    min_lr: float = 3e-5
    
    # Gradient management
    grad_clip: Optional[float] = 1.0
    grad_clip_norm_type: float = 2.0
    
    # Precision and efficiency
    mixed_precision: Optional[str] = None  # 'fp16', 'bf16', or None
    gradient_checkpointing: bool = False
    compile: bool = False  # torch.compile
    
    # Regularization
    dropout: float = 0.0
    label_smoothing: float = 0.0
    
    # Logging frequency
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    
    # Validation
    val_steps: Optional[int] = None  # Number of validation steps
    
    # Early stopping
    early_stopping: bool = False
    patience: int = 5
    min_delta: float = 1e-4
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.batch_size * self.gradient_accumulation_steps
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

@dataclass
class DataConfig(BaseConfig):
    """
    Data configuration.
    
    Handles all data loading and preprocessing settings.
    """
    
    # Dataset
    dataset: str = 'fineweb'  # 'fineweb', 'wikitext', 'custom'
    dataset_path: Optional[str] = None  # For custom datasets
    
    # Dataset-specific settings
    subset: Optional[str] = 'sample-10BT'  # For datasets with subsets
    split: str = 'train'  # Dataset split
    streaming: bool = False  # Streaming mode
    
    # Tokenization
    tokenizer: Optional[str] = None  # Path to tokenizer or name
    tokenizer_type: str = 'gpt2'  # 'gpt2', 'sentencepiece', 'custom'
    sequence_length: int = 1024
    
    # DataLoader settings
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    
    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True
    
    # Validation data
    val_dataset: Optional[str] = None
    val_split: str = 'validation'
    val_size: Optional[float] = None  # Proportion of data for validation
    
    # Data augmentation
    augmentation: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data config."""
        if self.streaming and self.num_workers > 0:
            # Streaming datasets typically require num_workers=0
            import warnings
            warnings.warn(
                "Streaming datasets often work best with num_workers=0. "
                f"You have num_workers={self.num_workers}"
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig(BaseConfig):
    """
    System and hardware configuration.
    """
    
    # Device
    device: str = 'cuda'  # 'cuda', 'cpu', 'mps'
    device_ids: Optional[List[int]] = None  # For multi-GPU
    
    # Distributed training
    distributed: bool = False
    backend: str = 'nccl'  # 'nccl', 'gloo'
    world_size: int = 1
    local_rank: int = 0
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Debugging
    detect_anomaly: bool = False
    profile: bool = False
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_optimizer: bool = True
    save_scheduler: bool = True
    keep_last_n: int = 3
    
    # Logging
    log_dir: str = './logs'
    log_level: str = 'INFO'
    
    # Monitoring
    use_wandb: bool = False
    use_tensorboard: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# ============================================================================
# EXPERIMENT CONFIGURATION (Top-level)
# ============================================================================

@dataclass
class ExperimentConfig(BaseConfig):
    """
    Complete experiment configuration.
    
    This is the single entry point for all configurations. It composes
    all sub-configs into a unified configuration object.
    
    Usage:
        config = ExperimentConfig.from_yaml('config.yaml')
        model = create_model(config.model)
        trainer = Trainer(config)
    """
    
    # Sub-configurations
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    system: SystemConfig
    
    # Experiment metadata
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Custom settings (for extensibility)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create config from dictionary.
        
        Handles nested configs and provides helpful error messages.
        """
        try:
            # Extract sub-configs
            model_config = ModelConfig.from_dict(
                config_dict.get('model', {})
            )
            training_config = TrainingConfig.from_dict(
                config_dict.get('training', {})
            )
            data_config = DataConfig.from_dict(
                config_dict.get('data', {})
            )
            system_config = SystemConfig.from_dict(
                config_dict.get('system', {})
            )
            
            # Extract top-level fields
            name = config_dict.get('name')
            description = config_dict.get('description')
            tags = config_dict.get('tags', [])
            custom = config_dict.get('custom', {})
            
            return cls(
                model=model_config,
                training=training_config,
                data=data_config,
                system=system_config,
                name=name,
                description=description,
                tags=tags,
                custom=custom,
            )
        
        except Exception as e:
            raise ValueError(
                f"Failed to create ExperimentConfig from dict. "
                f"Error: {e}\n"
                f"Config dict keys: {list(config_dict.keys())}"
            ) from e
    
    def validate(self):
        """
        Validate the complete configuration.
        
        Checks for common issues and inconsistencies.
        """
        # Check vocab size consistency
        if hasattr(self.model, 'vocab_size'):
            # Could validate against tokenizer if loaded
            pass
        
        # Check sequence length consistency
        if self.data.sequence_length > self.model.max_seq_len:
            raise ValueError(
                f"data.sequence_length ({self.data.sequence_length}) "
                f"exceeds model.max_seq_len ({self.model.max_seq_len})"
            )
        
        # Check streaming + num_workers
        if self.data.streaming and self.data.num_workers > 0:
            import warnings
            warnings.warn(
                "Streaming with num_workers > 0 may cause issues. "
                "Consider setting num_workers=0"
            )
        
        # Check distributed config
        if self.system.distributed:
            if self.system.world_size <= 1:
                raise ValueError(
                    "distributed=True but world_size <= 1"
                )
        
        return True
    
    @property
    def experiment_id(self) -> str:
        """Generate unique experiment ID."""
        import hashlib
        import time
        
        if self.name:
            base = self.name
        else:
            base = f"exp_{int(time.time())}"
        
        # Add hash of config for uniqueness
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{base}_{config_hash}"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_config(
    path: Union[str, Path],
    config_class: Type[BaseConfig] = ExperimentConfig
) -> BaseConfig:
    """
    Load configuration from file.
    
    Args:
        path: Path to config file (YAML or JSON)
        config_class: Config class to use (default: ExperimentConfig)
    
    Returns:
        Config instance
    
    Example:
        config = load_config('config.yaml')
        model_config = load_config('model.yaml', ModelConfig)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if path.suffix in ['.yaml', '.yml']:
        return config_class.from_yaml(path)
    elif path.suffix == '.json':
        return config_class.from_json(path)
    else:
        raise ValueError(f"Unknown file type: {path.suffix}")


def create_default_config(
    vocab_size: int,
    model_preset: str = 'llama',
    **kwargs
) -> ExperimentConfig:
    """
    Create a default configuration with sensible defaults.
    
    Args:
        vocab_size: Vocabulary size
        model_preset: Model architecture preset ('llama', 'gpt')
        **kwargs: Override any defaults
    
    Returns:
        Complete experiment configuration
    
    Example:
        config = create_default_config(
            vocab_size=50000,
            model_preset='llama',
            training={'max_steps': 5000}
        )
    """
    # Create model config
    if model_preset == 'llama':
        model_config = ModelConfig.llama_style(vocab_size)
    elif model_preset == 'gpt':
        model_config = ModelConfig.gpt_style(vocab_size)
    else:
        raise ValueError(f"Unknown preset: {model_preset}")
    
    # Override model config if provided
    if 'model' in kwargs:
        model_dict = {**model_config.to_dict(), **kwargs.pop('model')}
        model_config = ModelConfig.from_dict(model_dict)
    
    # Create other configs with defaults
    training_config = TrainingConfig(**kwargs.pop('training', {}))
    data_config = DataConfig(**kwargs.pop('data', {}))
    system_config = SystemConfig(**kwargs.pop('system', {}))
    
    return ExperimentConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        system=system_config,
        **kwargs
    )


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_config_file(path: Union[str, Path]) -> bool:
    """
    Validate a config file without creating config object.
    
    Useful for CI/CD and config linting.
    """
    try:
        config = load_config(path)
        config.validate()
        return True
    except Exception as e:
        print(f"Config validation failed: {e}")
        return False