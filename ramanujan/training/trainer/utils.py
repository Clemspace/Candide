"""Training utilities for setup and validation."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml


# ============================================================================
# CONFIG VALIDATION
# ============================================================================

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate training configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required sections
    required_sections = ['model', 'training']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate model config
    if 'model' in config:
        model_config = config['model']
        required_model_fields = ['vocab_size', 'd_model', 'n_layers', 'n_heads']
        for field in required_model_fields:
            if field not in model_config:
                errors.append(f"Missing required model field: {field}")
        
        # Validate dimensions
        if 'd_model' in model_config and 'n_heads' in model_config:
            if model_config['d_model'] % model_config['n_heads'] != 0:
                errors.append(
                    f"d_model ({model_config['d_model']}) must be divisible by "
                    f"n_heads ({model_config['n_heads']})"
                )
        
        # Validate GQA setup
        if 'n_kv_heads' in model_config and 'n_heads' in model_config:
            if model_config['n_heads'] % model_config['n_kv_heads'] != 0:
                errors.append(
                    f"n_heads ({model_config['n_heads']}) must be divisible by "
                    f"n_kv_heads ({model_config['n_kv_heads']})"
                )
    
    # Validate training config
    if 'training' in config:
        training_config = config['training']
        
        # Check at least one stopping criterion
        if 'max_steps' not in training_config and 'max_epochs' not in training_config:
            errors.append("Must specify either max_steps or max_epochs")
        
        # Validate positive values
        positive_fields = ['batch_size', 'learning_rate']
        for field in positive_fields:
            if field in training_config and training_config[field] <= 0:
                errors.append(f"{field} must be positive, got {training_config[field]}")
        
        # Validate gradient accumulation
        if 'gradient_accumulation_steps' in training_config:
            if training_config['gradient_accumulation_steps'] < 1:
                errors.append("gradient_accumulation_steps must be >= 1")
    
    return errors


def print_config_summary(config: Dict[str, Any]):
    """
    Print a summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("="*70)
    print("📋 Configuration Summary")
    print("="*70)
    
    # Model
    if 'model' in config:
        print("\n🧠 Model:")
        model_config = config['model']
        print(f"  Architecture: {model_config.get('attention_type', 'standard')} attention, "
              f"{model_config.get('ffn_type', 'standard')} FFN")
        print(f"  Layers: {model_config.get('n_layers')}")
        print(f"  Dimensions: d_model={model_config.get('d_model')}, "
              f"n_heads={model_config.get('n_heads')}, "
              f"d_ff={model_config.get('d_ff', 'auto')}")
        if model_config.get('n_kv_heads'):
            print(f"  GQA: n_kv_heads={model_config['n_kv_heads']}")
        print(f"  Vocab size: {model_config.get('vocab_size'):,}")
        print(f"  Max sequence length: {model_config.get('max_seq_len')}")
        print(f"  Dropout: {model_config.get('dropout', 0.0)}")
        print(f"  RoPE: {model_config.get('use_rope', False)}")
    
    # Training
    if 'training' in config:
        print("\n🏋️  Training:")
        training_config = config['training']
        print(f"  Steps: {training_config.get('max_steps', 'N/A')}")
        print(f"  Batch size: {training_config.get('batch_size')}")
        print(f"  Gradient accumulation: {training_config.get('gradient_accumulation_steps', 1)}")
        effective_batch = (
            training_config.get('batch_size', 1) * 
            training_config.get('gradient_accumulation_steps', 1)
        )
        print(f"  Effective batch size: {effective_batch}")
        print(f"  Learning rate: {training_config.get('learning_rate')}")
        print(f"  Weight decay: {training_config.get('weight_decay', 0.0)}")
        print(f"  Optimizer: {training_config.get('optimizer', 'adamw')}")
        print(f"  Scheduler: {training_config.get('lr_scheduler', 'warmup')}")
        print(f"  Mixed precision: {training_config.get('mixed_precision', 'none')}")
        print(f"  Gradient checkpointing: {training_config.get('gradient_checkpointing', False)}")
    
    # Data
    if 'data' in config:
        print("\n📚 Data:")
        data_config = config['data']
        print(f"  Dataset: {data_config.get('dataset')}")
        print(f"  Sequence length: {data_config.get('sequence_length')}")
        print(f"  Streaming: {data_config.get('streaming', False)}")
        print(f"  Num workers: {data_config.get('num_workers', 0)}")
    
    # Hardware
    if 'hardware' in config:
        print("\n💻 Hardware:")
        hardware_config = config['hardware']
        print(f"  Device: {hardware_config.get('device', 'auto')}")
        print(f"  Seed: {hardware_config.get('seed', 42)}")
    
    # Logging
    if 'logging' in config:
        print("\n📊 Logging:")
        logging_config = config['logging']
        print(f"  WandB: {logging_config.get('use_wandb', False)}")
        if logging_config.get('use_wandb'):
            print(f"  Project: {logging_config.get('wandb_project')}")
        print(f"  Log every: {logging_config.get('log_every', 10)} steps")
        print(f"  Eval every: {logging_config.get('eval_every', 500)} steps")
    
    print("="*70)


# ============================================================================
# MODEL VERIFICATION
# ============================================================================

def verify_model(model: nn.Module, config: Dict[str, Any]) -> bool:
    """
    Verify model is correctly configured.
    
    Args:
        model: Model to verify
        config: Configuration dictionary
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check parameter count
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {n_params:,} parameters")
        
        # Check device
        device = next(model.parameters()).device
        print(f"✓ Model is on device: {device}")
        
        # Try forward pass with dummy data
        model.eval()
        vocab_size = config['model']['vocab_size']
        batch_size = 2
        seq_len = 16
        
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
            
            # Extract logits
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs
            
            expected_shape = (batch_size, seq_len, vocab_size)
            if logits.shape != expected_shape:
                print(f"✗ Output shape mismatch: expected {expected_shape}, got {logits.shape}")
                return False
            
            print(f"✓ Forward pass successful: {logits.shape}")
        
        model.train()
        return True
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


def estimate_memory_usage(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate memory usage for training.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary with memory estimates in GB
    """
    model_config = config['model']
    training_config = config['training']
    
    # Rough parameter count estimation
    d_model = model_config['d_model']
    n_layers = model_config['n_layers']
    vocab_size = model_config['vocab_size']
    d_ff = model_config.get('d_ff') or (4 * d_model)
    
    # Embedding parameters
    embed_params = vocab_size * d_model
    
    # Per-layer parameters (attention + FFN)
    attn_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = 2 * d_model * d_ff  # Up and down projections (SwiGLU uses 3x)
    if model_config.get('ffn_type') == 'swiglu':
        ffn_params = 3 * d_model * d_ff
    
    layer_params = attn_params + ffn_params
    total_params = embed_params + (n_layers * layer_params) + (d_model * vocab_size)  # LM head
    
    # Bytes per parameter (assume fp32 or mixed precision)
    bytes_per_param = 4 if training_config.get('mixed_precision') == 'none' else 2
    
    # Model memory
    model_memory = (total_params * bytes_per_param) / (1024 ** 3)  # GB
    
    # Optimizer state (AdamW: 2x parameters for momentum and variance)
    optimizer_memory = model_memory * 2
    
    # Gradient memory (same as model)
    gradient_memory = model_memory
    
    # Activation memory (rough estimate based on batch size and sequence length)
    batch_size = training_config['batch_size']
    seq_len = config.get('data', {}).get('sequence_length', 512)
    
    # Activations per layer: roughly batch * seq * d_model
    activation_memory = (batch_size * seq_len * d_model * n_layers * bytes_per_param) / (1024 ** 3)
    
    # Total training memory
    total_training = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    # Add 20% buffer for PyTorch overhead
    total_training *= 1.2
    
    return {
        'model_gb': model_memory,
        'optimizer_gb': optimizer_memory,
        'gradients_gb': gradient_memory,
        'activations_gb': activation_memory,
        'total_training_gb': total_training,
        'total_inference_gb': model_memory + activation_memory,
    }


def print_memory_estimate(config: Dict[str, Any]):
    """
    Print memory usage estimate.
    
    Args:
        config: Configuration dictionary
    """
    memory = estimate_memory_usage(config)
    
    print("\n💾 Estimated Memory Usage:")
    print(f"  Model: {memory['model_gb']:.2f} GB")
    print(f"  Optimizer: {memory['optimizer_gb']:.2f} GB")
    print(f"  Gradients: {memory['gradients_gb']:.2f} GB")
    print(f"  Activations: {memory['activations_gb']:.2f} GB")
    print(f"  Total (training): {memory['total_training_gb']:.2f} GB")
    print(f"  Total (inference): {memory['total_inference_gb']:.2f} GB")
    
    # Warn if likely to OOM
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"\n  Available GPU memory: {available_memory:.2f} GB")
        
        if memory['total_training_gb'] > available_memory * 0.9:
            print("  ⚠️  WARNING: Estimated memory usage exceeds available GPU memory!")
            print("     Consider:")
            print("       - Reducing batch size")
            print("       - Enabling gradient checkpointing")
            print("       - Using gradient accumulation")
            print("       - Reducing sequence length")


# ============================================================================
# SETUP HELPERS
# ============================================================================

def setup_output_directory(output_dir: str):
    """
    Setup output directory structure.
    
    Args:
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'checkpoints').mkdir(exist_ok=True)
    (output_path / 'logs').mkdir(exist_ok=True)
    (output_path / 'samples').mkdir(exist_ok=True)
    
    print(f"✅ Output directory setup: {output_dir}")


def save_config(config: Dict[str, Any], output_dir: str):
    """
    Save configuration to output directory.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
    """
    config_path = Path(output_dir) / 'config.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Config saved: {config_path}")


def check_dependencies():
    """Check if required dependencies are available."""
    dependencies = {
        'torch': 'PyTorch',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm (progress bars)',
    }
    
    optional = {
        'wandb': 'Weights & Biases (logging)',
    }
    
    print("\n🔍 Checking dependencies...")
    
    # Check required
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - REQUIRED")
            missing.append(module)
    
    # Check optional
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ○ {name} - optional")
    
    if missing:
        print(f"\n❌ Missing required dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing training utilities")
    print("="*70)
    
    # Test config validation
    print("\n1. Testing config validation...")
    valid_config = {
        'model': {
            'vocab_size': 32000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
        },
        'training': {
            'max_steps': 10000,
            'batch_size': 32,
            'learning_rate': 1e-3,
        }
    }
    
    errors = validate_config(valid_config)
    if errors:
        print(f"  Errors found: {errors}")
    else:
        print("  ✓ Config is valid")
    
    # Test config summary
    print("\n2. Testing config summary...")
    print_config_summary(valid_config)
    
    # Test memory estimation
    print("\n3. Testing memory estimation...")
    print_memory_estimate(valid_config)
    
    # Test dependency check
    print("\n4. Checking dependencies...")
    check_dependencies()
    
    print("\n" + "="*70)
    print("✅ All utility tests passed!")
    print("="*70)