"""
Configuration management for Ramanujan Transformer.

This module provides configuration classes and utilities:
- ModelConfig: Model architecture configuration
- SparsityConfig: Ramanujan sparsity configuration
- TrainingConfig: Training dynamics configuration (imported from training module)
- ExperimentConfig: Complete experiment configuration
- YAML loading/saving utilities

Example:
    >>> from ramanujan.utils import load_config, ExperimentConfig
    >>> 
    >>> # Load from YAML
    >>> config = load_config('configs/optimal.yaml')
    >>> 
    >>> # Create programmatically
    >>> config = ExperimentConfig(
    ...     model=ModelConfig(dim=890, num_layers=6),
    ...     sparsity=SparsityConfig(attention=0.82, ffn=0.88)
    ... )
"""

import yaml
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """
    Model architecture configuration.
    
    Args:
        vocab_size: Vocabulary size
        dim: Model dimension (hidden size)
        num_layers: Number of transformer blocks
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (for GQA)
        hidden_dim: FFN hidden dimension (default: None, auto-computed)
        max_seq_len: Maximum sequence length (default: 2048)
        dropout: Dropout probability (default: 0.0)
        attention_dropout: Attention-specific dropout (default: 0.0)
        pad_token_id: Padding token ID (default: 0)
        tie_embeddings: Tie input/output embeddings (default: True)
        model_type: Type of model ('standard', 'enhanced', 'baseline')
        ffn_type: FFN type ('swiglu', 'standard')
        norm_type: Normalization type ('rms', 'layer')
    
    Example:
        >>> config = ModelConfig(
        ...     vocab_size=32000,
        ...     dim=890,
        ...     num_layers=6,
        ...     num_heads=10,
        ...     num_kv_heads=5
        ... )
    """
    
    # Core architecture
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    
    # Optional architecture
    hidden_dim: Optional[int] = None
    max_seq_len: int = 2048
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Special tokens
    pad_token_id: int = 0
    
    # Model variants
    tie_embeddings: bool = True
    model_type: str = 'standard'  # 'standard', 'enhanced', 'baseline'
    ffn_type: str = 'swiglu'  # 'swiglu', 'standard'
    norm_type: str = 'rms'  # 'rms', 'layer'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
        
        # Auto-compute hidden_dim if not provided
        if self.hidden_dim is None:
            if self.ffn_type == 'swiglu':
                self.hidden_dim = int(8 * self.dim / 3)
                # Round to nearest multiple of 256 for efficiency
                self.hidden_dim = ((self.hidden_dim + 255) // 256) * 256
            else:
                self.hidden_dim = 4 * self.dim


# ============================================================================
# SPARSITY CONFIGURATION
# ============================================================================

@dataclass
class SparsityConfig:
    """
    Ramanujan sparsity configuration.
    
    Args:
        attention_sparsity: Target sparsity for attention projections (0.0-1.0)
        ffn_sparsity: Target sparsity for FFN layers (0.0-1.0)
        max_prime: Maximum prime for Ramanujan graph construction
        force_method: Force specific construction method ('lps', 'biregular', None)
        use_sliding_window: Enable sliding window attention
        window_size: Sliding window size
        num_global_tokens: Number of global attention tokens
    
    Example:
        >>> config = SparsityConfig(
        ...     attention_sparsity=0.82,
        ...     ffn_sparsity=0.88,
        ...     use_sliding_window=True,
        ...     window_size=512
        ... )
    """
    
    # Sparsity levels
    attention_sparsity: float = 0.0
    ffn_sparsity: float = 0.0
    
    # Ramanujan graph settings
    max_prime: int = 1000
    force_method: Optional[str] = "lps"  # 'lps', 'biregular', None
    
    # Sliding window settings
    use_sliding_window: bool = False
    window_size: int = 512
    num_global_tokens: int = 64
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.attention_sparsity <= 1.0:
            raise ValueError(f"attention_sparsity must be in [0, 1], got {self.attention_sparsity}")
        if not 0.0 <= self.ffn_sparsity <= 1.0:
            raise ValueError(f"ffn_sparsity must be in [0, 1], got {self.ffn_sparsity}")
        if self.force_method not in ['lps', 'biregular', None]:
            raise ValueError(f"force_method must be 'lps', 'biregular', or None, got {self.force_method}")


# ============================================================================
# TRAINING CONFIGURATION (reference to training module)
# ============================================================================

# Note: TrainingConfig is defined in ramanujan/training/trainer.py
# We import it here for convenience
from ..training.trainer import TrainingConfig


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    Combines model, sparsity, and training configurations.
    
    Args:
        model: Model configuration
        sparsity: Sparsity configuration
        training: Training configuration
        name: Experiment name
        description: Experiment description
        tags: Tags for organization
    
    Example:
        >>> config = ExperimentConfig(
        ...     model=ModelConfig(dim=890, num_layers=6, ...),
        ...     sparsity=SparsityConfig(attention_sparsity=0.82, ...),
        ...     training=TrainingConfig(max_steps=10000, ...),
        ...     name="optimal-890d-6l",
        ...     description="Optimal configuration with 890 dim"
        ... )
    """
    
    model: ModelConfig
    sparsity: SparsityConfig
    training: TrainingConfig
    
    # Metadata
    name: str = "experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model': asdict(self.model),
            'sparsity': asdict(self.sparsity),
            'training': asdict(self.training),
            'name': self.name,
            'description': self.description,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(
            model=ModelConfig(**config_dict['model']),
            sparsity=SparsityConfig(**config_dict['sparsity']),
            training=TrainingConfig(**config_dict['training']),
            name=config_dict.get('name', 'experiment'),
            description=config_dict.get('description', ''),
            tags=config_dict.get('tags', [])
        )


# ============================================================================
# YAML UTILITIES
# ============================================================================

def load_config(config_path: str) -> ExperimentConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        ExperimentConfig instance
    
    Example:
        >>> config = load_config('configs/optimal.yaml')
        >>> print(config.model.dim)  # 890
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ExperimentConfig.from_dict(config_dict)


def save_config(config: ExperimentConfig, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: ExperimentConfig instance
        config_path: Path to save YAML file
    
    Example:
        >>> save_config(config, 'configs/my_config.yaml')
    """
    config_dict = config.to_dict()
    
    # Create directory if needed
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config saved to: {config_path}")


def create_ablation_configs(
    base_config: ExperimentConfig,
    ablations: List[Dict[str, Any]],
    output_dir: str = "configs/ablations"
) -> Dict[str, ExperimentConfig]:
    """
    Create ablation configs from base config.
    
    Args:
        base_config: Base experiment configuration
        ablations: List of ablation specifications
        output_dir: Directory to save configs
    
    Returns:
        Dictionary mapping ablation names to configs
    
    Example:
        >>> ablations = [
        ...     {'name': 'no_se', 'changes': {'training.loss_type': 'ce'}},
        ...     {'name': 'no_sw', 'changes': {'sparsity.use_sliding_window': False}}
        ... ]
        >>> configs = create_ablation_configs(base_config, ablations)
    """
    configs = {}
    
    for ablation in ablations:
        ablation_name = ablation['name']
        
        # Create copy of base config
        config_dict = base_config.to_dict()
        
        # Apply changes
        for key, value in ablation.get('changes', {}).items():
            # Handle nested keys like 'training.loss_type'
            parts = key.split('.')
            target = config_dict
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        
        # Update metadata
        config_dict['name'] = ablation_name
        config_dict['description'] = ablation.get('description', f'Ablation: {ablation_name}')
        config_dict['tags'] = ['ablation'] + config_dict.get('tags', [])
        
        # Create config
        ablation_config = ExperimentConfig.from_dict(config_dict)
        configs[ablation_name] = ablation_config
        
        # Save to file
        output_path = Path(output_dir) / f'{ablation_name}.yaml'
        save_config(ablation_config, str(output_path))
    
    return configs


# ============================================================================
# CONFIG TEMPLATES
# ============================================================================

def create_baseline_config() -> ExperimentConfig:
    """
    Create baseline configuration.
    
    Simple transformer without any improvements.
    
    Returns:
        ExperimentConfig for baseline
    """
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=32000,
            dim=512,
            num_layers=6,
            num_heads=8,
            num_kv_heads=8,  # No GQA
            model_type='baseline',
            ffn_type='standard',
            norm_type='layer'
        ),
        sparsity=SparsityConfig(
            attention_sparsity=0.0,
            ffn_sparsity=0.0,
            use_sliding_window=False
        ),
        training=TrainingConfig(
            max_steps=10000,
            batch_size=8,
            learning_rate=0.0003,
            loss_type='ce'
        ),
        name='baseline',
        description='Baseline transformer without improvements'
    )


def create_optimal_config() -> ExperimentConfig:
    """
    Create optimal configuration.
    
    890-dim model with all improvements.
    
    Returns:
        ExperimentConfig for optimal model
    """
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=31980,
            dim=890,
            num_layers=6,
            num_heads=10,
            num_kv_heads=5,
            model_type='enhanced',
            ffn_type='swiglu',
            norm_type='rms'
        ),
        sparsity=SparsityConfig(
            attention_sparsity=0.82,
            ffn_sparsity=0.88,
            use_sliding_window=True,
            window_size=512,
            num_global_tokens=64
        ),
        training=TrainingConfig(
            max_steps=10000,
            batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=0.0003,
            loss_type='semantic_entropy',
            semantic_entropy_alpha=0.1,
            warmup_steps=1000
        ),
        name='optimal-890d-6l',
        description='Optimal configuration with all improvements'
    )


def create_small_config() -> ExperimentConfig:
    """
    Create small configuration for testing.
    
    Returns:
        ExperimentConfig for small model
    """
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=10000,
            dim=256,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            model_type='standard'
        ),
        sparsity=SparsityConfig(
            attention_sparsity=0.0,
            ffn_sparsity=0.0
        ),
        training=TrainingConfig(
            max_steps=1000,
            batch_size=4,
            learning_rate=0.001
        ),
        name='small-test',
        description='Small model for testing'
    )


# ============================================================================
# CONFIG UTILITIES
# ============================================================================

def merge_configs(base: ExperimentConfig, override: Dict[str, Any]) -> ExperimentConfig:
    """
    Merge override values into base config.
    
    Args:
        base: Base configuration
        override: Dictionary with override values
    
    Returns:
        New ExperimentConfig with overrides applied
    
    Example:
        >>> base = create_baseline_config()
        >>> overrides = {'model': {'dim': 1024}, 'training': {'batch_size': 16}}
        >>> config = merge_configs(base, overrides)
    """
    config_dict = base.to_dict()
    
    # Deep merge
    def deep_merge(d1, d2):
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                deep_merge(d1[key], value)
            else:
                d1[key] = value
    
    deep_merge(config_dict, override)
    
    return ExperimentConfig.from_dict(config_dict)


def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate experiment configuration.
    
    Args:
        config: Configuration to validate
    
    Returns:
        List of validation warnings (empty if all OK)
    
    Example:
        >>> config = create_optimal_config()
        >>> warnings = validate_config(config)
        >>> if warnings:
        ...     for w in warnings:
        ...         print(f"Warning: {w}")
    """
    warnings = []
    
    # Check model config
    if config.model.dim < 128:
        warnings.append(f"Model dim ({config.model.dim}) is very small")
    if config.model.num_layers < 2:
        warnings.append(f"num_layers ({config.model.num_layers}) is very small")
    
    # Check sparsity config
    if config.sparsity.attention_sparsity > 0.95:
        warnings.append(f"attention_sparsity ({config.sparsity.attention_sparsity}) is very high")
    if config.sparsity.ffn_sparsity > 0.95:
        warnings.append(f"ffn_sparsity ({config.sparsity.ffn_sparsity}) is very high")
    
    # Check training config
    if config.training.learning_rate > 0.01:
        warnings.append(f"learning_rate ({config.training.learning_rate}) is very high")
    if config.training.warmup_steps > config.training.max_steps * 0.5:
        warnings.append(f"warmup_steps is > 50% of max_steps")
    
    # Check compatibility
    if config.model.model_type == 'enhanced' and config.sparsity.attention_sparsity < 0.05:
        warnings.append("Enhanced model typically uses attention sparsity")
    if config.sparsity.use_sliding_window and config.model.model_type == 'baseline':
        warnings.append("Sliding window not available in baseline model")
    
    return warnings


def print_config_summary(config: ExperimentConfig):
    """
    Print human-readable config summary.
    
    Args:
        config: Configuration to summarize
    
    Example:
        >>> config = create_optimal_config()
        >>> print_config_summary(config)
    """
    print("=" * 70)
    print(f"Experiment: {config.name}")
    print("=" * 70)
    
    if config.description:
        print(f"\n{config.description}\n")
    
    print("Model Configuration:")
    print(f"  Type: {config.model.model_type}")
    print(f"  Dimension: {config.model.dim}")
    print(f"  Layers: {config.model.num_layers}")
    print(f"  Heads: {config.model.num_heads} (KV: {config.model.num_kv_heads})")
    print(f"  Vocab size: {config.model.vocab_size}")
    print(f"  Max seq len: {config.model.max_seq_len}")
    
    print("\nSparsity Configuration:")
    print(f"  Attention: {config.sparsity.attention_sparsity:.1%}")
    print(f"  FFN: {config.sparsity.ffn_sparsity:.1%}")
    print(f"  Sliding window: {config.sparsity.use_sliding_window}")
    if config.sparsity.use_sliding_window:
        print(f"  Window size: {config.sparsity.window_size}")
    
    print("\nTraining Configuration:")
    print(f"  Max steps: {config.training.max_steps}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Loss type: {config.training.loss_type}")
    print(f"  Optimizer: {config.training.optimizer_type}")
    print(f"  Scheduler: {config.training.scheduler_type}")
    
    print("=" * 70)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing config.py module")
    print("="*70)
    
    # Test ModelConfig
    print("\n1. Testing ModelConfig...")
    model_config = ModelConfig(
        vocab_size=32000,
        dim=512,
        num_layers=6,
        num_heads=8,
        num_kv_heads=4
    )
    print(f"   Dim: {model_config.dim}")
    print(f"   Hidden dim (auto): {model_config.hidden_dim}")
    print(f"   ✅ ModelConfig working!")
    
    # Test SparsityConfig
    print("\n2. Testing SparsityConfig...")
    sparsity_config = SparsityConfig(
        attention_sparsity=0.82,
        ffn_sparsity=0.88,
        use_sliding_window=True
    )
    print(f"   Attention sparsity: {sparsity_config.attention_sparsity}")
    print(f"   Sliding window: {sparsity_config.use_sliding_window}")
    print(f"   ✅ SparsityConfig working!")
    
    # Test ExperimentConfig
    print("\n3. Testing ExperimentConfig...")
    exp_config = ExperimentConfig(
        model=model_config,
        sparsity=sparsity_config,
        training=TrainingConfig(max_steps=10000),
        name="test-experiment"
    )
    print(f"   Name: {exp_config.name}")
    print(f"   Model dim: {exp_config.model.dim}")
    print(f"   ✅ ExperimentConfig working!")
    
    # Test to_dict/from_dict
    print("\n4. Testing dict conversion...")
    config_dict = exp_config.to_dict()
    exp_config2 = ExperimentConfig.from_dict(config_dict)
    assert exp_config2.model.dim == exp_config.model.dim
    print(f"   Roundtrip successful!")
    print(f"   ✅ Dict conversion working!")
    
    # Test save/load YAML
    print("\n5. Testing YAML save/load...")
    test_path = "/tmp/test_config.yaml"
    save_config(exp_config, test_path)
    loaded_config = load_config(test_path)
    assert loaded_config.model.dim == exp_config.model.dim
    print(f"   YAML roundtrip successful!")
    print(f"   ✅ YAML save/load working!")
    
    # Test config templates
    print("\n6. Testing config templates...")
    baseline = create_baseline_config()
    optimal = create_optimal_config()
    small = create_small_config()
    
    print(f"   Baseline dim: {baseline.model.dim}")
    print(f"   Optimal dim: {optimal.model.dim}")
    print(f"   Small dim: {small.model.dim}")
    print(f"   ✅ Config templates working!")
    
    # Test ablation configs
    print("\n7. Testing ablation configs...")
    ablations = [
        {'name': 'no_se', 'changes': {'training': {'loss_type': 'ce'}}},
        {'name': 'no_sw', 'changes': {'sparsity': {'use_sliding_window': False}}}
    ]
    ablation_configs = create_ablation_configs(optimal, ablations, "/tmp/ablations")
    print(f"   Created {len(ablation_configs)} ablation configs")
    print(f"   ✅ Ablation configs working!")
    
    # Test merge_configs
    print("\n8. Testing merge_configs...")
    override = {'model': {'dim': 1024}, 'training': {'batch_size': 16}}
    merged = merge_configs(baseline, override)
    assert merged.model.dim == 1024
    assert merged.training.batch_size == 16
    print(f"   Merged dim: {merged.model.dim}")
    print(f"   Merged batch size: {merged.training.batch_size}")
    print(f"   ✅ merge_configs working!")
    
    # Test validate_config
    print("\n9. Testing validate_config...")
    warnings = validate_config(optimal)
    print(f"   Warnings: {len(warnings)}")
    for w in warnings:
        print(f"     - {w}")
    print(f"   ✅ validate_config working!")
    
    # Test print_config_summary
    print("\n10. Testing print_config_summary...")
    print_config_summary(optimal)
    print(f"   ✅ print_config_summary working!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.utils import load_config, ExperimentConfig")
    print("  from ramanujan.utils import create_baseline_config, create_optimal_config")
    print("="*70)