"""
Tests for Configuration System
===============================

Comprehensive tests for config.py covering:
- All config classes
- Alias handling
- Validation
- Serialization
- Preset configurations
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ramanujan.config.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    SystemConfig,
    ExperimentConfig,
    create_default_config,
    validate_config_file,
)


# ============================================================================
# MODEL CONFIG TESTS
# ============================================================================

class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_basic_creation(self):
        """Test creating a basic model config."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            n_layers=12,
            n_heads=12,
        )
        
        assert config.vocab_size == 50000
        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.n_heads == 12
    
    def test_auto_ff_calculation_swiglu(self):
        """Test automatic feedforward dimension calculation for SwiGLU."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            ffn_type='swiglu',
        )
        
        # SwiGLU: ~8/3 * d_model, rounded to 256
        expected = ((int(8 * 768 / 3) + 255) // 256) * 256
        assert config.d_ff == expected
    
    def test_auto_ff_calculation_gelu(self):
        """Test automatic feedforward dimension calculation for GELU."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            ffn_type='gelu',
        )
        
        # Standard FFN: 4 * d_model
        assert config.d_ff == 4 * 768
    
    def test_manual_ff_override(self):
        """Test manual feedforward dimension override."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            d_ff=2048,
        )
        
        assert config.d_ff == 2048
    
    def test_gqa_kv_heads(self):
        """Test GQA automatically sets n_kv_heads."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            n_heads=12,
            attention_type='gqa',
        )
        
        assert config.n_kv_heads == 3  # 12 // 4
        assert config.is_gqa
        assert not config.is_mqa
    
    def test_mqa_kv_heads(self):
        """Test MQA sets n_kv_heads to 1."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            n_heads=12,
            attention_type='mqa',
        )
        
        assert config.n_kv_heads == 1
        assert config.is_mqa
        assert config.is_gqa  # MQA is a special case of GQA
    
    def test_mha_kv_heads(self):
        """Test MHA sets n_kv_heads equal to n_heads."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            n_heads=12,
            attention_type='mha',
        )
        
        assert config.n_kv_heads == 12
        assert not config.is_gqa
        assert not config.is_mqa
    
    def test_head_dim_property(self):
        """Test head_dim property."""
        config = ModelConfig(
            vocab_size=50000,
            d_model=768,
            n_heads=12,
        )
        
        assert config.head_dim == 64  # 768 // 12
    
    def test_validation_divisibility(self):
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            ModelConfig(
                vocab_size=50000,
                d_model=769,  # Not divisible by 12
                n_heads=12,
            )
    
    def test_alias_handling_d_model(self):
        """Test alias handling for d_model."""
        # Test 'dim' alias (use 768 which is divisible by 12)
        config_dict = {'vocab_size': 50000, 'dim': 768}
        config = ModelConfig.from_dict(config_dict)
        assert config.d_model == 768
        
        # Test 'hidden_size' alias
        config_dict = {'vocab_size': 50000, 'hidden_size': 768}
        config = ModelConfig.from_dict(config_dict)
        assert config.d_model == 768
    
    def test_alias_handling_n_layers(self):
        """Test alias handling for n_layers."""
        config_dict = {'vocab_size': 50000, 'num_layers': 24}
        config = ModelConfig.from_dict(config_dict)
        assert config.n_layers == 24
    
    def test_alias_handling_n_heads(self):
        """Test alias handling for n_heads."""
        config_dict = {'vocab_size': 50000, 'num_heads': 16}
        config = ModelConfig.from_dict(config_dict)
        assert config.n_heads == 16
    
    def test_alias_handling_max_seq_len(self):
        """Test alias handling for max_seq_len."""
        config_dict = {'vocab_size': 50000, 'context_length': 4096}
        config = ModelConfig.from_dict(config_dict)
        assert config.max_seq_len == 4096
    
    def test_llama_preset(self):
        """Test LLaMA-style preset."""
        config = ModelConfig.llama_style(vocab_size=50000)
        
        assert config.vocab_size == 50000
        assert config.norm_type == 'rms'
        assert config.attention_type == 'gqa'
        assert config.ffn_type == 'swiglu'
        assert config.use_rope == True
        assert config.bias == False
    
    def test_gpt_preset(self):
        """Test GPT-style preset."""
        config = ModelConfig.gpt_style(vocab_size=50000)
        
        assert config.vocab_size == 50000
        assert config.norm_type == 'layer'
        assert config.attention_type == 'mha'
        assert config.ffn_type == 'gelu'
        assert config.use_rope == False
        assert config.bias == True
    
    def test_preset_with_overrides(self):
        """Test preset with custom overrides."""
        config = ModelConfig.llama_style(
            vocab_size=50000,
            d_model=1024,
            n_layers=24,
        )
        
        assert config.d_model == 1024
        assert config.n_layers == 24
        # Preset settings should still apply
        assert config.norm_type == 'rms'
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ModelConfig(vocab_size=50000)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 50000
        assert 'd_model' in config_dict
    
    def test_serialization_json(self):
        """Test JSON serialization."""
        config = ModelConfig(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            temp_path = f.name
        
        try:
            loaded = ModelConfig.from_json(temp_path)
            assert loaded.vocab_size == config.vocab_size
            assert loaded.d_model == config.d_model
        finally:
            Path(temp_path).unlink()
    
    def test_serialization_yaml(self):
        """Test YAML serialization."""
        config = ModelConfig(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            temp_path = f.name
        
        try:
            loaded = ModelConfig.from_yaml(temp_path)
            assert loaded.vocab_size == config.vocab_size
            assert loaded.d_model == config.d_model
        finally:
            Path(temp_path).unlink()
    
    def test_custom_components(self):
        """Test custom components extensibility."""
        config = ModelConfig(
            vocab_size=50000,
            custom_components={
                'sparse_attention': True,
                'custom_param': 42,
            }
        )
        
        assert config.custom_components['sparse_attention'] == True
        assert config.custom_components['custom_param'] == 42


# ============================================================================
# TRAINING CONFIG TESTS
# ============================================================================

class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_basic_creation(self):
        """Test creating basic training config."""
        config = TrainingConfig(
            max_steps=10000,
            batch_size=16,
            learning_rate=1e-4,
        )
        
        assert config.max_steps == 10000
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
    
    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = TrainingConfig(
            batch_size=8,
            gradient_accumulation_steps=4,
        )
        
        assert config.effective_batch_size == 32
    
    def test_defaults(self):
        """Test default values."""
        config = TrainingConfig()
        
        assert config.max_steps == 10000
        assert config.optimizer == 'adamw'
        assert config.grad_clip == 1.0
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'max_steps': 5000,
            'batch_size': 32,
            'learning_rate': 3e-4,
        }
        
        config = TrainingConfig.from_dict(config_dict)
        assert config.max_steps == 5000
        assert config.batch_size == 32


# ============================================================================
# DATA CONFIG TESTS
# ============================================================================

class TestDataConfig:
    """Test DataConfig class."""
    
    def test_basic_creation(self):
        """Test creating basic data config."""
        config = DataConfig(
            dataset='fineweb',
            sequence_length=1024,
        )
        
        assert config.dataset == 'fineweb'
        assert config.sequence_length == 1024
    
    def test_streaming_warning(self):
        """Test warning for streaming with num_workers."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = DataConfig(
                streaming=True,
                num_workers=4,
            )
            
            assert len(w) == 1
            assert "streaming" in str(w[0].message).lower()
    
    def test_defaults(self):
        """Test default values."""
        config = DataConfig()
        
        assert config.dataset == 'fineweb'
        assert config.num_workers == 4
        assert config.use_cache == True


# ============================================================================
# SYSTEM CONFIG TESTS
# ============================================================================

class TestSystemConfig:
    """Test SystemConfig class."""
    
    def test_basic_creation(self):
        """Test creating basic system config."""
        config = SystemConfig(
            device='cuda',
            seed=42,
        )
        
        assert config.device == 'cuda'
        assert config.seed == 42
    
    def test_defaults(self):
        """Test default values."""
        config = SystemConfig()
        
        assert config.device == 'cuda'
        assert config.seed == 42
        assert config.distributed == False


# ============================================================================
# EXPERIMENT CONFIG TESTS
# ============================================================================

class TestExperimentConfig:
    """Test ExperimentConfig class."""
    
    def test_basic_creation(self):
        """Test creating complete experiment config."""
        config = ExperimentConfig(
            model=ModelConfig(vocab_size=50000),
            training=TrainingConfig(),
            data=DataConfig(),
            system=SystemConfig(),
        )
        
        assert config.model.vocab_size == 50000
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.system, SystemConfig)
    
    def test_from_dict_complete(self):
        """Test creation from complete dictionary."""
        config_dict = {
            'model': {
                'vocab_size': 50000,
                'd_model': 768,
            },
            'training': {
                'max_steps': 5000,
            },
            'data': {
                'dataset': 'wikitext',
            },
            'system': {
                'device': 'cpu',
            },
            'name': 'test_experiment',
        }
        
        config = ExperimentConfig.from_dict(config_dict)
        assert config.model.vocab_size == 50000
        assert config.training.max_steps == 5000
        assert config.data.dataset == 'wikitext'
        assert config.system.device == 'cpu'
        assert config.name == 'test_experiment'
    
    def test_from_dict_minimal(self):
        """Test creation from minimal dictionary."""
        config_dict = {
            'model': {'vocab_size': 50000},
            'training': {},
            'data': {},
            'system': {},
        }
        
        config = ExperimentConfig.from_dict(config_dict)
        assert config.model.vocab_size == 50000
        # Should have defaults for everything else
        assert config.training.max_steps == 10000
    
    def test_validation_sequence_length(self):
        """Test validation catches sequence length mismatch."""
        config = ExperimentConfig(
            model=ModelConfig(vocab_size=50000, max_seq_len=1024),
            training=TrainingConfig(),
            data=DataConfig(sequence_length=2048),  # Too long!
            system=SystemConfig(),
        )
        
        with pytest.raises(ValueError, match="sequence_length"):
            config.validate()
    
    def test_validation_distributed(self):
        """Test validation catches distributed config issues."""
        config = ExperimentConfig(
            model=ModelConfig(vocab_size=50000),
            training=TrainingConfig(),
            data=DataConfig(),
            system=SystemConfig(distributed=True, world_size=1),
        )
        
        with pytest.raises(ValueError, match="world_size"):
            config.validate()
    
    def test_experiment_id_with_name(self):
        """Test experiment ID generation with name."""
        config = ExperimentConfig(
            model=ModelConfig(vocab_size=50000),
            training=TrainingConfig(),
            data=DataConfig(),
            system=SystemConfig(),
            name='my_experiment',
        )
        
        exp_id = config.experiment_id
        assert exp_id.startswith('my_experiment_')
        assert len(exp_id.split('_')[-1]) == 8  # Hash length
    
    def test_experiment_id_without_name(self):
        """Test experiment ID generation without name."""
        config = ExperimentConfig(
            model=ModelConfig(vocab_size=50000),
            training=TrainingConfig(),
            data=DataConfig(),
            system=SystemConfig(),
        )
        
        exp_id = config.experiment_id
        assert exp_id.startswith('exp_')
    
    def test_serialization_roundtrip(self):
        """Test complete serialization roundtrip."""
        original = ExperimentConfig(
            model=ModelConfig.llama_style(vocab_size=50000),
            training=TrainingConfig(max_steps=5000),
            data=DataConfig(dataset='wikitext'),
            system=SystemConfig(device='cpu'),
            name='test',
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            original.to_yaml(f.name)
            temp_path = f.name
        
        try:
            loaded = ExperimentConfig.from_yaml(temp_path)
            assert loaded.model.vocab_size == original.model.vocab_size
            assert loaded.training.max_steps == original.training.max_steps
            assert loaded.data.dataset == original.data.dataset
            assert loaded.name == original.name
        finally:
            Path(temp_path).unlink()


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_default_config_llama(self):
        """Test creating default config with LLaMA preset."""
        config = create_default_config(
            vocab_size=50000,
            model_preset='llama',
        )
        
        assert config.model.vocab_size == 50000
        assert config.model.norm_type == 'rms'
        assert config.model.ffn_type == 'swiglu'
    
    def test_create_default_config_gpt(self):
        """Test creating default config with GPT preset."""
        config = create_default_config(
            vocab_size=50000,
            model_preset='gpt',
        )
        
        assert config.model.vocab_size == 50000
        assert config.model.norm_type == 'layer'
        assert config.model.ffn_type == 'gelu'
    
    def test_create_default_config_with_overrides(self):
        """Test creating default config with overrides."""
        config = create_default_config(
            vocab_size=32000,
            model_preset='llama',
            model={'d_model': 512},
            training={'max_steps': 5000},
        )
        
        assert config.model.d_model == 512
        assert config.training.max_steps == 5000
        # Preset should still apply
        assert config.model.norm_type == 'rms'
    
    def test_validate_config_file_valid(self):
        """Test validate_config_file with valid config."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            temp_path = f.name
        
        try:
            result = validate_config_file(temp_path)
            assert result == True
        finally:
            Path(temp_path).unlink()
    
    def test_validate_config_file_invalid(self):
        """Test validate_config_file with invalid config."""
        # Create invalid config (seq_len > max_seq_len)
        config_dict = {
            'model': {'vocab_size': 50000, 'max_seq_len': 1024},
            'training': {},
            'data': {'sequence_length': 2048},
            'system': {},
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            result = validate_config_file(temp_path)
            assert result == False
        finally:
            Path(temp_path).unlink()


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_preset(self):
        """Test invalid model preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_default_config(
                vocab_size=50000,
                model_preset='invalid',
            )
    
    def test_missing_required_field(self):
        """Test missing required field (vocab_size)."""
        with pytest.raises(TypeError):
            ModelConfig(d_model=768)  # Missing vocab_size
    
    def test_from_dict_error_handling(self):
        """Test error handling in from_dict."""
        with pytest.raises(ValueError, match="Failed to create"):
            ExperimentConfig.from_dict({})  # Missing required fields


if __name__ == '__main__':
    pytest.main([__file__, '-v'])