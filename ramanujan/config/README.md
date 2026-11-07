# Candide Configuration System

A unified, scalable configuration system for the Candide framework.

## Features

- **Unified Configuration**: Single source of truth for all experiment settings
- **Type-Safe**: Dataclass-based with full type annotations
- **Flexible Loading**: Support for YAML, JSON, and TOML
- **Environment Variables**: Automatic expansion of environment variables
- **Config Merging**: Compose configs from multiple sources
- **CLI Overrides**: Override any config value from command line
- **Validation**: Comprehensive validation including hardware compatibility
- **Presets**: Built-in presets for common architectures (LLaMA, GPT)
- **Extensible**: Easy to add custom configuration sections

## Installation

The config system is part of the Candide framework:

```python
from ramanujan.config import ExperimentConfig, load_config
```

## Quick Start

### Creating a Config

```python
from ramanujan.config import create_default_config

# Create with LLaMA preset
config = create_default_config(
    vocab_size=50000,
    model_preset='llama',
)

# Create with custom settings
config = create_default_config(
    vocab_size=50000,
    model_preset='llama',
    model={'d_model': 1024, 'n_layers': 24},
    training={'max_steps': 10000, 'batch_size': 32},
)
```

### Loading from File

```python
from ramanujan.config import load_config

# Load from YAML
config = load_config('config.yaml')

# Load from JSON
config = load_config('config.json')
```

### Saving a Config

```python
from ramanujan.config import save_config

# Save to YAML
save_config(config, 'config.yaml')

# Save to JSON
save_config(config, 'config.json')
```

## Configuration Structure

```yaml
# config.yaml
model:
  vocab_size: 50000
  d_model: 768
  n_layers: 12
  n_heads: 12
  norm_type: rms
  attention_type: gqa
  ffn_type: swiglu
  use_rope: true

training:
  max_steps: 10000
  batch_size: 16
  learning_rate: 3e-4
  optimizer: adamw
  grad_clip: 1.0
  mixed_precision: bf16

data:
  dataset: fineweb
  sequence_length: 1024
  num_workers: 4

system:
  device: cuda
  seed: 42
  checkpoint_dir: ./checkpoints
```

## Advanced Features

### Environment Variables

Use environment variables in your config:

```yaml
# config.yaml
model:
  vocab_size: ${VOCAB_SIZE:50000}  # Default: 50000
  d_model: ${MODEL_DIM}

training:
  max_steps: ${MAX_STEPS}
```

```python
import os
os.environ['VOCAB_SIZE'] = '60000'
os.environ['MODEL_DIM'] = '1024'
os.environ['MAX_STEPS'] = '20000'

config = load_config('config.yaml')
# vocab_size=60000, d_model=1024, max_steps=20000
```

### CLI Overrides

Override config values from command line:

```python
config = load_config_with_overrides(
    'config.yaml',
    cli_args=[
        'model.d_model=512',
        'training.batch_size=32',
        'training.learning_rate=1e-4',
    ]
)
```

### Config Merging

Compose configs from multiple sources:

```python
from ramanujan.config import merge_configs, load_config

# Base configuration
base = load_config('base.yaml')

# Experiment-specific overrides
experiment = load_config('experiment.yaml')

# Merge (experiment overrides base)
config = merge_configs(base, experiment)
```

### Validation

Comprehensive validation including hardware checks:

```python
from ramanujan.config import ConfigValidator

validator = ConfigValidator(config)
report = validator.validate()

# Check report
if not report['valid']:
    print("Issues:", report['issues'])
    print("Warnings:", report['warnings'])

print(f"Estimated memory: {report['estimated_memory_gb']:.2f} GB")
print(f"Estimated parameters: {report['estimated_parameters']:,}")
```

## Configuration Classes

### ModelConfig

Architecture configuration:

```python
from ramanujan.config import ModelConfig

config = ModelConfig(
    vocab_size=50000,
    d_model=768,
    n_layers=12,
    n_heads=12,
    # Auto-calculated:
    # - d_ff based on ffn_type
    # - n_kv_heads based on attention_type
)
```

**Presets:**

```python
# LLaMA-style (RMSNorm, GQA, SwiGLU, RoPE)
config = ModelConfig.llama_style(vocab_size=50000)

# GPT-style (LayerNorm, MHA, GELU, learned positions)
config = ModelConfig.gpt_style(vocab_size=50000)
```

**Alias Handling:**

The config accepts multiple naming conventions:

```python
# These are all equivalent:
ModelConfig(d_model=768)
ModelConfig(dim=768)
ModelConfig(hidden_size=768)
```

### TrainingConfig

Training hyperparameters:

```python
from ramanujan.config import TrainingConfig

config = TrainingConfig(
    max_steps=10000,
    batch_size=16,
    gradient_accumulation_steps=4,  # Effective batch: 64
    learning_rate=3e-4,
    optimizer='adamw',
    grad_clip=1.0,
    mixed_precision='bf16',
)
```

### DataConfig

Data loading configuration:

```python
from ramanujan.config import DataConfig

config = DataConfig(
    dataset='fineweb',
    sequence_length=1024,
    num_workers=4,
    streaming=False,
)
```

### SystemConfig

Hardware and system settings:

```python
from ramanujan.config import SystemConfig

config = SystemConfig(
    device='cuda',
    seed=42,
    distributed=True,
    world_size=4,
    checkpoint_dir='./checkpoints',
    use_wandb=True,
)
```

## Validation

### Hardware Compatibility

```python
from ramanujan.config.validators import validate_hardware_compatibility

is_compatible = validate_hardware_compatibility(
    config,
    available_memory_gb=24.0,  # Auto-detect if None
)
```

### Memory Estimation

```python
from ramanujan.config.validators import estimate_memory_usage

memory_gb = estimate_memory_usage(config)
print(f"Estimated memory: {memory_gb:.2f} GB")
```

### Parameter Estimation

```python
from ramanujan.config.validators import estimate_parameters

n_params = estimate_parameters(config.model)
print(f"Parameters: {n_params:,}")
```

## Real-World Examples

### Experiment Setup

```python
# base_config.yaml
model:
  vocab_size: 50000
  d_model: 768
  n_layers: 12

training:
  max_steps: 10000
  
# experiment_1.yaml (only overrides)
model:
  d_model: 1024
training:
  learning_rate: 1e-4
```

```python
config = load_config_with_overrides(
    'base_config.yaml',
    overrides=load_config('experiment_1.yaml').to_dict(),
)
```

### Distributed Training

```python
config = create_default_config(vocab_size=50000)

config.system.distributed = True
config.system.world_size = 4
config.system.device_ids = [0, 1, 2, 3]
config.training.batch_size = 8  # Per device

# Effective batch size: 8 * 4 = 32
print(f"Effective batch: {config.training.effective_batch_size * config.system.world_size}")
```

### Continual Pretraining

```python
# Load checkpoint config
checkpoint_config = load_config('checkpoint/config.yaml')

# Continue with new data
config = merge_configs(
    checkpoint_config,
    {
        'training': {'max_steps': 20000},
        'data': {'dataset': 'new_dataset'},
    }
)
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/config/ -v

# With coverage
pytest tests/config/ --cov=ramanujan.config --cov-report=html

# Specific test file
pytest tests/config/test_config.py -v
```

## Migration from Old Config System

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for details on migrating from the old config system.

## API Reference

### Loading Functions

- `load_config(path, config_class=ExperimentConfig)`: Load from file
- `load_config_with_overrides(path, overrides=None, cli_args=None)`: Load with overrides
- `merge_configs(*configs)`: Merge multiple configs
- `save_config(config, path, format=None)`: Save to file

### Creation Functions

- `create_default_config(vocab_size, model_preset='llama')`: Create with defaults
- `ModelConfig.llama_style(vocab_size)`: LLaMA preset
- `ModelConfig.gpt_style(vocab_size)`: GPT preset

### Validation Functions

- `validate_config_file(path)`: Validate config file
- `validate_hardware_compatibility(config)`: Check hardware compatibility
- `validate_dataset_model_compatibility(config)`: Check data compatibility
- `validate_training_stability(config)`: Check training settings
- `estimate_memory_usage(config)`: Estimate memory requirements
- `estimate_parameters(model_config)`: Estimate parameter count

## Contributing

When adding new configuration options:

1. Add field to appropriate config class (`ModelConfig`, `TrainingConfig`, etc.)
2. Add validation logic in `validators.py` if needed
3. Add tests in `tests/config/test_*.py`
4. Update this README

## License

Part of the Candide framework.