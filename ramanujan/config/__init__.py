"""
Candide Configuration System
=============================

A unified, scalable configuration system for the Candide framework.

Public API:
    # Configuration Classes
    - ExperimentConfig: Complete experiment configuration
    - ModelConfig: Model architecture configuration
    - TrainingConfig: Training hyperparameters
    - DataConfig: Data loading and preprocessing
    - SystemConfig: Hardware and system settings
    
    # Loading Functions
    - load_config: Load from file (YAML/JSON/TOML)
    - load_config_with_overrides: Load with CLI overrides
    - merge_configs: Merge multiple configs
    
    # Creation Helpers
    - create_default_config: Create config with sensible defaults
    
    # Validation
    - validate_config_file: Validate config file

Usage:
    ```python
    from ramanujan.config import ExperimentConfig, load_config
    
    # Load from file
    config = load_config('config.yaml')
    
    # Create programmatically
    config = ExperimentConfig(
        model=ModelConfig.llama_style(vocab_size=50000),
        training=TrainingConfig(max_steps=10000),
        data=DataConfig(dataset='fineweb'),
        system=SystemConfig()
    )
    
    # Access sub-configs
    model = create_model(config.model)
    trainer = Trainer(config)
    ```
"""

# Core config classes
from .config import (
    BaseConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    SystemConfig,
    ExperimentConfig,
)

# Loading utilities
from .loaders import (
    load_config,
    load_config_with_overrides,
    merge_configs,
    save_config,
)

# Creation helpers
from .config import (
    create_default_config,
)

# Validation
from .config import (
    validate_config_file,
)

from .validators import (
    validate_hardware_compatibility,
    validate_dataset_model_compatibility,
    ConfigValidator,
)

__all__ = [
    # Config classes
    'BaseConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'SystemConfig',
    'ExperimentConfig',
    
    # Loading
    'load_config',
    'load_config_with_overrides',
    'merge_configs',
    'save_config',
    
    # Creation
    'create_default_config',
    
    # Validation
    'validate_config_file',
    'validate_hardware_compatibility',
    'validate_dataset_model_compatibility',
    'ConfigValidator',
]

__version__ = '0.1.0'