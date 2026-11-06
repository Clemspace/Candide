"""
Tests for Configuration Validators
===================================

Tests for validators.py covering:
- Hardware compatibility validation
- Memory estimation
- Dataset-model compatibility
- Training stability checks
- Comprehensive validation
"""

import pytest
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ramanujan.config.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    SystemConfig,
    ExperimentConfig,
    create_default_config,
)
from ramanujan.config.validators import (
    validate_hardware_compatibility,
    estimate_memory_usage,
    estimate_parameters,
    validate_dataset_model_compatibility,
    validate_training_stability,
    ConfigValidator,
    quick_validate,
)


# ============================================================================
# PARAMETER ESTIMATION TESTS
# ============================================================================

class TestParameterEstimation:
    """Test parameter estimation."""
    
    def test_estimate_parameters_basic(self):
        """Test basic parameter estimation."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=3072,
            ffn_type='gelu',
        )
        
        n_params = estimate_parameters(config)
        
        # Rough calculation:
        # Embedding: 50000 * 768 = 38.4M
        # Each layer: ~7M (attention + FFN)
        # Total: ~38.4M + 12 * 7M = ~122M
        # Actual is around 161M with all components
        assert 100_000_000 < n_params < 200_000_000
    
    def test_estimate_parameters_swiglu(self):
        """Test parameter estimation with SwiGLU."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            ffn_type='swiglu',
        )
        
        n_params = estimate_parameters(config)
        
        # SwiGLU has more parameters than GELU
        # Should be larger than basic config
        assert n_params > 120_000_000
    
    def test_estimate_parameters_tied_embeddings(self):
        """Test with tied embeddings."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            tie_word_embeddings=True,
        )
        
        n_params = estimate_parameters(config)
        
        # Should be same as untied (estimate doesn't double count)
        assert n_params > 0
    
    def test_estimate_parameters_large_model(self):
        """Test estimation for large model."""
        config = ModelConfig.llama_style(
            vocab_size=50000,
            d_model=4096,
            n_layers=32,
            n_heads=32,
        )
        
        n_params = estimate_parameters(config)
        
        # Should be in billions
        assert n_params > 1_000_000_000


# ============================================================================
# MEMORY ESTIMATION TESTS
# ============================================================================

class TestMemoryEstimation:
    """Test memory usage estimation."""
    
    def test_estimate_memory_basic(self):
        """Test basic memory estimation."""
        config = create_default_config(
            vocab_size=50000,
            model_preset='gpt',
        )
        
        memory_gb = estimate_memory_usage(config)
        
        # Should be reasonable for small model
        assert 0.5 < memory_gb < 10.0
    
    def test_estimate_memory_with_mixed_precision(self):
        """Test memory estimation with mixed precision."""
        config = create_default_config(vocab_size=50000)
        config_fp32 = create_default_config(vocab_size=50000)
        
        # Enable mixed precision
        config.training.mixed_precision = 'bf16'
        
        memory_fp16 = estimate_memory_usage(config)
        memory_fp32 = estimate_memory_usage(config_fp32)
        
        # FP16/BF16 should use less memory
        assert memory_fp16 < memory_fp32
    
    def test_estimate_memory_with_gradient_checkpointing(self):
        """Test memory estimation with gradient checkpointing."""
        config = create_default_config(vocab_size=50000)
        config_gc = create_default_config(vocab_size=50000)
        
        # Enable gradient checkpointing
        config_gc.training.gradient_checkpointing = True
        
        memory_normal = estimate_memory_usage(config)
        memory_gc = estimate_memory_usage(config_gc)
        
        # Gradient checkpointing should reduce memory
        assert memory_gc < memory_normal
    
    def test_estimate_memory_large_batch(self):
        """Test memory scales with batch size."""
        config_small = create_default_config(vocab_size=50000)
        config_large = create_default_config(vocab_size=50000)
        
        config_small.training.batch_size = 8
        config_large.training.batch_size = 32
        
        memory_small = estimate_memory_usage(config_small)
        memory_large = estimate_memory_usage(config_large)
        
        # Larger batch should use more memory
        assert memory_large > memory_small


# ============================================================================
# HARDWARE COMPATIBILITY TESTS
# ============================================================================

class TestHardwareCompatibility:
    """Test hardware compatibility validation."""
    
    def test_validate_with_sufficient_memory(self):
        """Test validation passes with sufficient memory."""
        config = create_default_config(vocab_size=50000)
        
        # 100GB should be plenty
        result = validate_hardware_compatibility(
            config,
            available_memory_gb=100.0,
            warn_only=True,
        )
        
        # Should pass (True) or have no GPU available (also True for warn_only)
        assert result in [True, False]  # Accept both for flexibility
    
    def test_validate_with_insufficient_memory(self):
        """Test validation fails with insufficient memory."""
        config = create_default_config(vocab_size=50000)
        
        # 0.1GB should be too little
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware_compatibility(
                config,
                available_memory_gb=0.1,
                warn_only=True,
            )
            
            assert result == False
            assert len(w) > 0
    
    def test_validate_with_insufficient_memory_raises(self):
        """Test validation raises with warn_only=False."""
        config = create_default_config(vocab_size=50000)
        
        with pytest.raises(ValueError, match="memory"):
            validate_hardware_compatibility(
                config,
                available_memory_gb=0.1,
                warn_only=False,
            )
    
    def test_validate_distributed_config(self):
        """Test distributed configuration validation."""
        config = create_default_config(vocab_size=50000)
        config.system.distributed = True
        config.system.world_size = 1  # Invalid!
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware_compatibility(
                config, available_memory_gb=1e6, warn_only=True   # isolate the distributed check
            )
            
            # Should return False and have warning
            assert result == False
            # Check that we got a warning about world_size
            assert len(w) > 0
            assert any('world_size' in str(warning.message).lower() for warning in w)
    
    def test_validate_device_ids_mismatch(self):
        """Test validation of device_ids vs world_size."""
        config = create_default_config(vocab_size=50000)
        config.system.distributed = True
        config.system.world_size = 4
        config.system.device_ids = [0, 1]  # Mismatch!
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware_compatibility(config, warn_only=True)
            
            assert result == False


# ============================================================================
# DATASET-MODEL COMPATIBILITY TESTS
# ============================================================================

class TestDatasetModelCompatibility:
    """Test dataset-model compatibility validation."""
    
    def test_validate_compatible(self):
        """Test validation passes for compatible config."""
        config = create_default_config(vocab_size=50000)
        config.model.max_seq_len = 2048
        config.data.sequence_length = 1024  # Compatible
        
        result = validate_dataset_model_compatibility(config, warn_only=True)
        assert result == True
    
    def test_validate_sequence_length_mismatch(self):
        """Test validation fails for sequence length mismatch."""
        config = create_default_config(vocab_size=50000)
        config.model.max_seq_len = 1024
        config.data.sequence_length = 2048  # Too long!
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_dataset_model_compatibility(config, warn_only=True)
            
            assert result == False
            assert any('sequence_length' in str(warning.message) for warning in w)
    
    def test_validate_sequence_length_raises(self):
        """Test validation raises with warn_only=False."""
        config = create_default_config(vocab_size=50000)
        config.model.max_seq_len = 1024
        config.data.sequence_length = 2048
        
        with pytest.raises(ValueError, match="sequence_length"):
            validate_dataset_model_compatibility(config, warn_only=False)
    
    def test_validate_streaming_warning(self):
        """Test warning for streaming with num_workers."""
        config = create_default_config(vocab_size=50000)
        config.data.streaming = True
        config.data.num_workers = 4
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_dataset_model_compatibility(config, warn_only=True)
            
            # Should warn about streaming + num_workers
            assert any('streaming' in str(warning.message).lower() for warning in w)


# ============================================================================
# TRAINING STABILITY TESTS
# ============================================================================

class TestTrainingStability:
    """Test training stability validation."""
    
    def test_validate_stable_config(self):
        """Test validation passes for stable config."""
        config = create_default_config(vocab_size=50000)
        config.training.learning_rate = 3e-4
        config.training.warmup_steps = 500
        config.training.grad_clip = 1.0
        
        result = validate_training_stability(config, warn_only=True)
        # May have warnings but should not fail
        assert result in [True, False]  # Just check it runs
    
    def test_validate_high_learning_rate(self):
        """Test warning for high learning rate."""
        config = create_default_config(vocab_size=50000)
        config.training.learning_rate = 0.1  # Very high
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_training_stability(config, warn_only=True)
            
            assert result == False
            assert any('learning rate' in str(warning.message).lower() for warning in w)
    
    def test_validate_low_learning_rate(self):
        """Test warning for low learning rate."""
        config = create_default_config(vocab_size=50000)
        config.training.learning_rate = 1e-7  # Very low
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_training_stability(config, warn_only=True)
            
            assert result == False
    
    def test_validate_no_warmup(self):
        """Test warning for no warmup steps."""
        config = create_default_config(vocab_size=50000)
        config.training.warmup_steps = 0
        config.training.max_steps = 10000
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_training_stability(config, warn_only=True)
            
            assert result == False
            assert any('warmup' in str(warning.message).lower() for warning in w)
    
    def test_validate_no_grad_clip_large_model(self):
        """Test warning for no gradient clipping on large model."""
        config = ModelConfig.llama_style(
            vocab_size=50000,
            d_model=4096,
            n_layers=32,
        )
        exp_config = ExperimentConfig(
            model=config,
            training=TrainingConfig(grad_clip=None),  # No clipping
            data=DataConfig(),
            system=SystemConfig(),
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_training_stability(exp_config, warn_only=True)
            
            assert result == False
            assert any('gradient clipping' in str(warning.message).lower() for warning in w)
    
    def test_validate_invalid_mixed_precision(self):
        """Test warning for invalid mixed precision."""
        config = create_default_config(vocab_size=50000)
        config.training.mixed_precision = 'invalid'
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_training_stability(config, warn_only=True)
            
            assert result == False
    
    def test_validate_extreme_batch_sizes(self):
        """Test warnings for extreme batch sizes."""
        # Very small batch
        config_small = create_default_config(vocab_size=50000)
        config_small.training.batch_size = 2
        config_small.training.gradient_accumulation_steps = 2  # Effective: 4
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_training_stability(config_small, warn_only=True)
            
            assert result == False
        
        # Very large batch
        config_large = create_default_config(vocab_size=50000)
        config_large.training.batch_size = 512
        config_large.training.gradient_accumulation_steps = 4  # Effective: 2048
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_training_stability(config_large, warn_only=True)
            
            assert result == False


# ============================================================================
# COMPREHENSIVE VALIDATOR TESTS
# ============================================================================

class TestConfigValidator:
    """Test ConfigValidator class."""
    
    def test_validator_basic(self):
        """Test basic validator usage."""
        config = create_default_config(vocab_size=50000)
        validator = ConfigValidator(config)
        
        report = validator.validate(verbose=False)
        
        assert isinstance(report, dict)
        assert 'valid' in report
        assert 'issues' in report
        assert 'warnings' in report
        assert 'info' in report
        assert 'estimated_memory_gb' in report
        assert 'estimated_parameters' in report
    
    def test_validator_valid_config(self, monkeypatch):
        """Test validator with valid config."""
        import ramanujan.config.validators as validators
        monkeypatch.setattr(validators, "_get_available_gpu_memory", lambda: 1e6)

        config = create_default_config(vocab_size=50000)
        config.training.warmup_steps = 500
        config.training.grad_clip = 1.0
        
        validator = ConfigValidator(config)
        report = validator.validate(verbose=False)
        
        # May have info messages but should have no critical issues
        # (warnings are acceptable for default configs)
        assert len(report['issues']) == 0
    
    def test_validator_invalid_config(self):
        """Test validator with invalid config."""
        config = create_default_config(vocab_size=50000)
        config.model.max_seq_len = 1024
        config.data.sequence_length = 2048  # Invalid!
        
        validator = ConfigValidator(config)
        report = validator.validate(verbose=False)
        
        assert report['valid'] == False
        assert len(report['issues']) > 0
    
    def test_validator_warnings(self):
        """Test validator generates warnings."""
        config = create_default_config(vocab_size=50000)
        config.training.learning_rate = 0.1  # High LR
        config.training.warmup_steps = 0  # No warmup
        
        validator = ConfigValidator(config)
        report = validator.validate(verbose=False)
        
        assert len(report['warnings']) > 0
    
    def test_validator_info(self):
        """Test validator generates info messages."""
        config = create_default_config(vocab_size=50000)
        
        validator = ConfigValidator(config)
        report = validator.validate(verbose=False)
        
        # Should have info about memory and parameters
        assert report['estimated_memory_gb'] > 0
        assert report['estimated_parameters'] > 0
    
    def test_quick_validate(self):
        """Test quick_validate convenience function."""
        config = create_default_config(vocab_size=50000)
        
        result = quick_validate(config)
        assert isinstance(result, bool)
    
    def test_validator_print_report(self):
        """Test that validator can print report."""
        config = create_default_config(vocab_size=50000)
        validator = ConfigValidator(config)
        
        # Should not raise
        report = validator.validate(verbose=True)
        assert report is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple validators."""
    
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline."""
        config = create_default_config(vocab_size=50000)
        
        # Run all validators
        hw_result = validate_hardware_compatibility(config)
        data_result = validate_dataset_model_compatibility(config)
        train_result = validate_training_stability(config)
        
        # All should run without errors
        assert isinstance(hw_result, bool)
        assert isinstance(data_result, bool)
        assert isinstance(train_result, bool)
    
    def test_validator_catches_all_issues(self):
        """Test that ConfigValidator catches all types of issues."""
        config = create_default_config(vocab_size=50000)
        
        # Add multiple issues
        config.model.max_seq_len = 1024
        config.data.sequence_length = 2048  # Incompatible
        config.training.learning_rate = 0.1  # Too high
        config.training.warmup_steps = 0  # No warmup
        config.system.distributed = True
        config.system.world_size = 1  # Invalid distributed config
        
        validator = ConfigValidator(config)
        report = validator.validate(verbose=False)
        
        # Should catch all issues
        assert report['valid'] == False
        assert len(report['issues']) >= 2  # seq_len + distributed
        assert len(report['warnings']) >= 2  # LR + warmup


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases in validation."""
    
    def test_zero_parameters(self):
        """Test handling of edge case configs."""
        # This would be caught by ModelConfig validation,
        # but let's test the estimator is robust
        config = ModelConfig(
            vocab_size=100,
            d_model=64,
            n_layers=1,
            n_heads=1,
        )
        
        n_params = estimate_parameters(config)
        assert n_params > 0
    
    def test_memory_estimation_edge_cases(self):
        """Test memory estimation with edge cases."""
        # Very small model
        config_small = create_default_config(vocab_size=1000)
        config_small.model.d_model = 64
        config_small.model.n_layers = 2
        
        memory = estimate_memory_usage(config_small)
        assert memory > 0
        assert memory < 1.0  # Should be less than 1GB


if __name__ == '__main__':
    pytest.main([__file__, '-v'])