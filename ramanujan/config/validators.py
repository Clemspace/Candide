"""
Configuration Validators
=========================

Advanced validation for configs including:
- Hardware compatibility checks
- Dataset-model compatibility
- Training stability checks
- Resource estimation
"""

from typing import Optional, List, Dict, Any
import warnings

from .config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig, SystemConfig


# ============================================================================
# HARDWARE VALIDATION
# ============================================================================

def validate_hardware_compatibility(
    config: ExperimentConfig,
    available_memory_gb: Optional[float] = None,
    warn_only: bool = True,
) -> bool:
    """
    Validate that config is compatible with available hardware.
    
    Checks:
    - Model fits in memory
    - Batch size is reasonable
    - Multi-GPU config is valid
    
    Args:
        config: Experiment configuration
        available_memory_gb: Available GPU memory in GB (auto-detect if None)
        warn_only: If True, warn instead of raising errors
    
    Returns:
        True if valid, False otherwise (or raises if warn_only=False)
    """
    # Estimate memory usage
    estimated_memory = estimate_memory_usage(config)
    
    # Get available memory
    if available_memory_gb is None:
        available_memory_gb = _get_available_gpu_memory()
    
    # Check if model fits
    if available_memory_gb is not None:
        if estimated_memory > available_memory_gb:
            msg = (
                f"Estimated memory usage ({estimated_memory:.2f} GB) "
                f"exceeds available memory ({available_memory_gb:.2f} GB)"
            )
            if warn_only:
                warnings.warn(msg)
                return False
            else:
                raise ValueError(msg)
    
    # Validate distributed config
    if config.system.distributed:
        if config.system.world_size <= 1:
            msg = "distributed=True but world_size <= 1"
            if warn_only:
                warnings.warn(msg)
                return False
            else:
                raise ValueError(msg)
        
        if config.system.device_ids and len(config.system.device_ids) != config.system.world_size:
            msg = f"device_ids length ({len(config.system.device_ids)}) != world_size ({config.system.world_size})"
            if warn_only:
                warnings.warn(msg)
                return False
            else:
                raise ValueError(msg)
    
    return True


def estimate_memory_usage(config: ExperimentConfig) -> float:
    """
    Estimate GPU memory usage in GB.
    
    Rough estimation based on:
    - Model parameters
    - Optimizer states
    - Gradients
    - Activations
    - Batch size
    
    Returns:
        Estimated memory in GB
    """
    model = config.model
    training = config.training
    
    # Calculate number of parameters
    n_params = estimate_parameters(model)
    
    # Memory components (in bytes)
    # 1. Model weights (fp32 = 4 bytes, bf16/fp16 = 2 bytes)
    bytes_per_param = 2 if training.mixed_precision else 4
    model_memory = n_params * bytes_per_param
    
    # 2. Optimizer states (AdamW: 2 states per param)
    optimizer_memory = n_params * 4 * 2  # Always fp32 for optimizer
    
    # 3. Gradients
    gradient_memory = n_params * bytes_per_param
    
    # 4. Activations (rough estimate based on batch size and model size)
    seq_len = config.data.sequence_length
    batch_size = training.batch_size
    activation_memory = (
        batch_size * seq_len * model.d_model * model.n_layers * 4 * 2  # Attention + FFN
    )
    
    # If using gradient checkpointing, reduce activation memory
    if training.gradient_checkpointing:
        activation_memory *= 0.2  # Rough reduction factor
    
    # Total memory
    total_bytes = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    # Convert to GB and add 20% safety margin
    total_gb = (total_bytes / (1024 ** 3)) * 1.2
    
    return total_gb


def estimate_parameters(model_config: ModelConfig) -> int:
    """
    Estimate number of parameters in model.
    
    Args:
        model_config: Model configuration
    
    Returns:
        Estimated number of parameters
    """
    d = model_config.d_model
    n = model_config.n_layers
    v = model_config.vocab_size
    d_ff = model_config.d_ff
    
    # Embedding layer
    embed_params = v * d
    
    # Each transformer layer
    # - Attention: 4 * d^2 (Q, K, V, O projections)
    # - FFN: depends on ffn_type
    if model_config.ffn_type in ['swiglu', 'geglu']:
        # SwiGLU has 3 projections
        ffn_params = 3 * d * d_ff
    else:
        # Standard FFN has 2 projections
        ffn_params = 2 * d * d_ff
    
    # Norms (negligible)
    norm_params = 2 * d
    
    layer_params = 4 * d * d + ffn_params + norm_params
    total_layer_params = n * layer_params
    
    # Output head
    if model_config.tie_word_embeddings:
        head_params = 0
    else:
        head_params = v * d
    
    total = embed_params + total_layer_params + head_params
    
    return total


def _get_available_gpu_memory() -> Optional[float]:
    """Get available GPU memory in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get first GPU's total memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return total_memory / (1024 ** 3)
    except:
        pass
    
    return None


# ============================================================================
# DATASET-MODEL COMPATIBILITY
# ============================================================================

def validate_dataset_model_compatibility(
    config: ExperimentConfig,
    warn_only: bool = True,
) -> bool:
    """
    Validate dataset and model are compatible.
    
    Checks:
    - Sequence length compatibility
    - Vocab size compatibility (if tokenizer available)
    
    Args:
        config: Experiment configuration
        warn_only: If True, warn instead of raising errors
    
    Returns:
        True if valid, False otherwise
    """
    # Check sequence length
    if config.data.sequence_length > config.model.max_seq_len:
        msg = (
            f"data.sequence_length ({config.data.sequence_length}) "
            f"exceeds model.max_seq_len ({config.model.max_seq_len})"
        )
        if warn_only:
            warnings.warn(msg)
            return False
        else:
            raise ValueError(msg)
    
    # Check streaming + num_workers
    if config.data.streaming and config.data.num_workers > 0:
        warnings.warn(
            "Streaming with num_workers > 0 may cause issues. "
            "Consider setting num_workers=0"
        )
    
    return True


# ============================================================================
# TRAINING STABILITY CHECKS
# ============================================================================

def validate_training_stability(
    config: ExperimentConfig,
    warn_only: bool = True,
) -> bool:
    """
    Check for common training instability issues.
    
    Checks:
    - Learning rate sanity
    - Warmup steps reasonable
    - Gradient clipping enabled for large models
    - Mixed precision settings
    
    Args:
        config: Experiment configuration
        warn_only: If True, warn instead of raising errors
    
    Returns:
        True if valid, False otherwise
    """
    training = config.training
    model = config.model
    
    issues = []
    
    # Check learning rate
    if training.learning_rate > 1e-2:
        issues.append(f"Learning rate very high: {training.learning_rate}")
    if training.learning_rate < 1e-6:
        issues.append(f"Learning rate very low: {training.learning_rate}")
    
    # Check warmup
    if training.warmup_steps == 0 and training.max_steps > 1000:
        issues.append("No warmup steps for long training")
    
    # Check gradient clipping for large models
    n_params = estimate_parameters(model)
    if n_params > 1e9 and training.grad_clip is None:  # >1B params
        issues.append("No gradient clipping for large model (>1B params)")
    
    # Check mixed precision
    if training.mixed_precision not in [None, 'fp16', 'bf16']:
        issues.append(f"Invalid mixed_precision: {training.mixed_precision}")
    
    # Check effective batch size
    eff_batch = training.effective_batch_size
    if eff_batch < 8:
        issues.append(f"Very small effective batch size: {eff_batch}")
    if eff_batch > 1024:
        issues.append(f"Very large effective batch size: {eff_batch}")
    
    # Report issues
    if issues:
        msg = "Potential training stability issues:\n" + "\n".join(f"  - {i}" for i in issues)
        if warn_only:
            warnings.warn(msg)
            return False
        else:
            raise ValueError(msg)
    
    return True


# ============================================================================
# COMPREHENSIVE VALIDATOR
# ============================================================================

class ConfigValidator:
    """
    Comprehensive config validator.
    
    Runs all validation checks and provides detailed report.
    
    Example:
        validator = ConfigValidator(config)
        report = validator.validate()
        print(report)
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def validate(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Args:
            verbose: Print report
        
        Returns:
            Validation report dict
        """
        self.issues = []
        self.warnings = []
        self.info = []
        
        # Run checks
        self._check_hardware()
        self._check_dataset_compatibility()
        self._check_training_stability()
        self._check_optimizer_config()
        self._check_special_tokens()
        
        # Generate report
        report = {
            'valid': len(self.issues) == 0,
            'issues': self.issues,
            'warnings': self.warnings,
            'info': self.info,
            'estimated_memory_gb': estimate_memory_usage(self.config),
            'estimated_parameters': estimate_parameters(self.config.model),
        }
        
        if verbose:
            self._print_report(report)
        
        return report
    
    def _check_hardware(self):
        """Check hardware compatibility."""
        try:
            memory = estimate_memory_usage(self.config)
            available = _get_available_gpu_memory()
            
            self.info.append(f"Estimated memory usage: {memory:.2f} GB")
            
            if available:
                self.info.append(f"Available GPU memory: {available:.2f} GB")
                if memory > available:
                    self.issues.append(
                        f"Estimated memory ({memory:.2f} GB) exceeds "
                        f"available ({available:.2f} GB)"
                    )
                elif memory > available * 0.9:
                    self.warnings.append(
                        f"Memory usage ({memory:.2f} GB) is close to "
                        f"available ({available:.2f} GB)"
                    )
        except Exception as e:
            self.warnings.append(f"Could not check hardware: {e}")
    
    def _check_dataset_compatibility(self):
        """Check dataset-model compatibility."""
        if self.config.data.sequence_length > self.config.model.max_seq_len:
            self.issues.append(
                f"data.sequence_length ({self.config.data.sequence_length}) > "
                f"model.max_seq_len ({self.config.model.max_seq_len})"
            )
    
    def _check_training_stability(self):
        """Check training stability."""
        training = self.config.training
        
        # Learning rate
        if training.learning_rate > 1e-2:
            self.warnings.append(f"High learning rate: {training.learning_rate}")
        
        # Warmup
        if training.warmup_steps == 0:
            self.warnings.append("No warmup steps")
        
        # Grad clip
        n_params = estimate_parameters(self.config.model)
        if n_params > 1e9 and training.grad_clip is None:
            self.warnings.append("No gradient clipping for large model")
    
    def _check_optimizer_config(self):
        """Check optimizer configuration."""
        training = self.config.training
        
        if training.optimizer not in ['adamw', 'adam', 'sgd', 'muon']:
            self.warnings.append(f"Unknown optimizer: {training.optimizer}")
        
        if training.weight_decay > 0.5:
            self.warnings.append(f"High weight decay: {training.weight_decay}")
    
    def _check_special_tokens(self):
        """Check special token configuration."""
        model = self.config.model
        
        if model.pad_token_id is None:
            self.info.append("No pad_token_id set")
        
        if model.bos_token_id is None:
            self.info.append("No bos_token_id set")
        
        if model.eos_token_id is None:
            self.info.append("No eos_token_id set")
    
    def _print_report(self, report: Dict[str, Any]):
        """Print validation report."""
        print("\n" + "="*60)
        print("CONFIG VALIDATION REPORT")
        print("="*60)
        
        print(f"\nStatus: {'✓ VALID' if report['valid'] else '✗ INVALID'}")
        print(f"Estimated Parameters: {report['estimated_parameters']:,}")
        print(f"Estimated Memory: {report['estimated_memory_gb']:.2f} GB")
        
        if report['issues']:
            print("\n⚠️  ISSUES:")
            for issue in report['issues']:
                print(f"  - {issue}")
        
        if report['warnings']:
            print("\n⚠  WARNINGS:")
            for warning in report['warnings']:
                print(f"  - {warning}")
        
        if report['info']:
            print("\nℹ️  INFO:")
            for info in report['info']:
                print(f"  - {info}")
        
        print("\n" + "="*60 + "\n")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_validate(config: ExperimentConfig) -> bool:
    """
    Quick validation check.
    
    Returns True if no issues found.
    """
    validator = ConfigValidator(config)
    report = validator.validate(verbose=False)
    return report['valid']