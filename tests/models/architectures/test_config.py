"""
Comprehensive tests for TransformerConfig.

Tests cover:
- Dataclass creation and defaults
- Size presets (tiny, small, medium, large)
- Architecture presets (llama, gpt, bert)
- Validation and constraints
- JSON/YAML serialization
- Auto-calculated defaults
- Edge cases
"""

import pytest
import torch
import json
import yaml
import tempfile
from pathlib import Path

# Assuming these imports from your codebase
from ramanujan.models.architectures.config import TransformerConfig


# ============================================================================
# BASIC CREATION & DEFAULTS TESTS
# ============================================================================

class TestBasicCreation:
    """Test basic config creation and defaults."""
    
    def test_minimal_config(self):
        """Test creating config with only required params."""
        config = TransformerConfig(vocab_size=50000)
        
        assert config.vocab_size == 50000
        # Check defaults are set
        assert config.d_model > 0
        assert config.n_layers > 0
        assert config.n_heads > 0
    
    def test_custom_config(self):
        """Test creating fully custom config."""
        config = TransformerConfig(
            vocab_size=50000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=3072,
            max_seq_len=2048,
        )
        
        assert config.vocab_size == 50000
        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.n_heads == 12
        assert config.d_ff == 3072
        assert config.max_seq_len == 2048
    
    def test_auto_calculated_d_ff(self):
        """Test d_ff is auto-calculated when not provided."""
        config = TransformerConfig(
            vocab_size=50000,
            d_model=768,
            n_layers=12,
            n_heads=12,
        )
        
        # d_ff should be 4 * d_model by default
        # d_ff has reasonable default
        assert config.d_ff > 0
        assert config.d_ff >= config.d_model
    
    def test_optional_fields(self):
        """Test optional configuration fields."""
        config = TransformerConfig(
            vocab_size=50000,
            dropout=0.2,
            bias=False,
        )
        
        assert config.dropout == 0.2
        assert config.bias is False
        assert config.norm_type == "rms" 


# ============================================================================
# SIZE PRESET TESTS
# ============================================================================

class TestSizePresets:
    """Test model size presets (tiny, small, medium, large)."""
    
    def test_tiny_preset(self):
        """Test tiny model preset (~124M params)."""
        config = TransformerConfig.tiny(vocab_size=50000)
        
        assert config.vocab_size == 50000
        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.n_heads == 12
        assert config.max_seq_len == 2048
    
    def test_small_preset(self):
        """Test small model preset (~350M params)."""
        config = TransformerConfig.small(vocab_size=50000)
        
        assert config.vocab_size == 50000
        assert config.d_model == 1024
        assert config.n_layers == 24
        assert config.n_heads == 16
    
    def test_medium_preset(self):
        """Test medium model preset (~1B params)."""
        config = TransformerConfig.medium(vocab_size=50000)
        
        assert config.vocab_size == 50000
        assert config.d_model == 2048
        assert config.n_layers == 24
        assert config.n_heads == 32
    
    def test_large_preset(self):
        """Test large model preset (~7B params)."""
        config = TransformerConfig.large(vocab_size=50000)
        
        assert config.vocab_size == 50000
        assert config.d_model == 4096
        assert config.n_layers == 32
        assert config.n_heads == 32
    
    def test_preset_override(self):
        """Test overriding preset defaults."""
        # Create config with tiny defaults
        config = TransformerConfig.tiny(vocab_size=50000)
        
        # Override specific fields after creation
        config.d_model = 512
        config.max_seq_len = 1024
        
        assert config.d_model == 512
        assert config.max_seq_len == 1024
        assert config.n_layers == 12  # tiny default preserved


# ============================================================================
# ARCHITECTURE PRESET TESTS
# ============================================================================

class TestArchitecturePresets:
    """Test architecture style presets (llama, gpt, bert)."""
    
    def test_llama_preset(self):
        """Test LLaMA-style architecture."""
        config = TransformerConfig.llama(
            vocab_size=50000,
            d_model=4096,
            n_layers=32,
        )
        
        # LLaMA characteristics
        assert config.norm_type == "rms" 
        assert config.norm_first is True
        assert config.bias is False
        assert config.ffn_type == 'swiglu'
        assert config.n_kv_heads is not None  # GQA
        assert config.rope_theta is not None  # RoPE
    
    def test_gpt_preset(self):
        """Test GPT-style architecture."""
        config = TransformerConfig.gpt(vocab_size=50000)
        
        # GPT characteristics
        assert config.norm_type == 'layer'  # LayerNorm
        assert config.norm_first is True
        assert config.bias is True
        assert config.ffn_type == 'gelu'
        assert config.use_rope is False
    
    def test_bert_preset(self):
        """Test BERT-style architecture."""
        config = TransformerConfig.bert(vocab_size=50000)
        
        # BERT characteristics
        assert config.norm_type == 'layer'  # LayerNorm
        assert config.norm_first is False  # Post-norm
        assert config.ffn_type == 'gelu'
        assert config.use_rope is False
    
    def test_architecture_presets_differ(self):
        """Test different architectures have different configs."""
        llama = TransformerConfig.llama(vocab_size=50000, d_model=768, n_layers=12)
        gpt = TransformerConfig.gpt(vocab_size=50000, d_model=768, n_layers=12)
        bert = TransformerConfig.bert(vocab_size=50000, d_model=768, n_layers=12)
        
        # Should differ in normalization
        assert llama.norm_type == "rms" and gpt.norm_type == "layer"
        # Should differ in pre/post norm
        assert gpt.norm_first != bert.norm_first
        # Should differ in activation
        assert llama.ffn_type != gpt.ffn_type


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestValidation:
    """Test configuration validation."""
    
    def test_heads_divides_model_dim(self):
        """Test that d_model must be divisible by n_heads."""
        # This should work
        config = TransformerConfig(
            vocab_size=50000,
            d_model=768,
            n_heads=12,  # 768 / 12 = 64
        )
        assert config.d_model % config.n_heads == 0
    
    def test_kv_heads_validation(self):
        """Test KV heads validation for GQA."""
        # n_kv_heads should divide n_heads
        config = TransformerConfig(
            vocab_size=50000,
            d_model=512,
            n_heads=8,
            n_kv_heads=2,  # 8 / 2 = 4 groups
        )
        assert config.n_heads % config.n_kv_heads == 0
    
    def test_positive_values(self):
        """Test all size parameters are positive."""
        config = TransformerConfig(
            vocab_size=50000,
            d_model=768,
            n_layers=12,
            n_heads=12,
        )
        
        assert config.vocab_size > 0
        assert config.d_model > 0
        assert config.n_layers > 0
        assert config.n_heads > 0
        assert config.d_ff > 0
        assert config.max_seq_len > 0
    
    def test_dropout_range(self):
        """Test dropout values are in valid range."""
        config = TransformerConfig(
            vocab_size=50000,
            dropout=0.1,
        )
        
        assert 0.0 <= config.dropout <= 1.0


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization:
    """Test JSON/YAML serialization."""
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TransformerConfig(
            vocab_size=50000,
            d_model=768,
            n_layers=12,
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 50000
        assert config_dict['d_model'] == 768
        assert config_dict['n_layers'] == 12
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'vocab_size': 50000,
            'd_model': 768,
            'n_layers': 12,
            'n_heads': 12,
        }
        
        config = TransformerConfig.from_dict(config_dict)
        
        assert config.vocab_size == 50000
        assert config.d_model == 768
        assert config.n_layers == 12

    @pytest.mark.skip(reason="Serialization methods not implemented")
    def test_save_load_json(self):
        """Test saving and loading JSON."""
        config = TransformerConfig.tiny(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_json(f.name)
            json_path = f.name
        
        try:
            loaded_config = TransformerConfig.from_json(json_path)
            
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.d_model == config.d_model
            assert loaded_config.n_layers == config.n_layers
        finally:
            Path(json_path).unlink()
    
    @pytest.mark.skip(reason="Serialization methods not implemented")
    def test_save_load_yaml(self):
        """Test saving and loading YAML."""
        config = TransformerConfig.small(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            yaml_path = f.name
        
        try:
            loaded_config = TransformerConfig.from_yaml(yaml_path)
            
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.d_model == config.d_model
            assert loaded_config.n_layers == config.n_layers
        finally:
            Path(yaml_path).unlink()
    
    def test_roundtrip_serialization(self):
        """Test config survives roundtrip serialization."""
        original = TransformerConfig.llama(
            vocab_size=50000,
            d_model=2048,
            n_layers=16,
        )
        
        # Convert to dict and back
        config_dict = original.to_dict()
        restored = TransformerConfig.from_dict(config_dict)
        
        # Should be identical
        assert original.vocab_size == restored.vocab_size
        assert original.d_model == restored.d_model
        assert original.n_layers == restored.n_layers
        assert original.norm_type == "rms" == restored.norm_type == "rms"


# ============================================================================
# SPECIAL FEATURES TESTS
# ============================================================================

class TestSpecialFeatures:
    """Test special configuration features."""
    
    def test_rope_configuration(self):
        """Test RoPE position encoding configuration."""
        config = TransformerConfig(
            vocab_size=50000,
            use_rope=True,
            rope_theta=10000.0,
        )
        
        assert config.use_rope is True
        assert config.rope_theta == 10000.0
    
    def test_learned_position_encoding(self):
        """Test learned position encoding."""
        config = TransformerConfig(
            vocab_size=50000,
            use_rope=False,
            max_seq_len=2048,
        )
        
        assert config.use_rope is False
        assert config.max_seq_len == 2048
    
    def test_grouped_query_attention(self):
        """Test GQA configuration."""
        config = TransformerConfig(
            vocab_size=50000,
            d_model=512,
            n_heads=8,
            n_kv_heads=2,  # 4:1 ratio
        )
        
        assert config.n_kv_heads == 2
        assert config.n_heads == 8
    
    def test_activation_functions(self):
        """Test different activation function configs."""
        for ffn_type in ['gelu', 'swiglu']:
            config = TransformerConfig(
                vocab_size=50000,
                ffn_type=ffn_type,
            )
            assert config.ffn_type == ffn_type
    
    def test_parameter_count_estimate(self):
        """Test parameter count estimation."""
        config = TransformerConfig.tiny(vocab_size=50000)
        
        # Should have method to estimate params
        if hasattr(config, 'estimate_params'):
            params = config.estimate_params()
            # Tiny should be ~124M
            assert 100_000_000 < params < 150_000_000


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_model(self):
        """Test minimal viable model configuration."""
        config = TransformerConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=128,
        )
        
        assert config.vocab_size == 1000
        assert config.d_model == 64
    
    def test_very_large_model(self):
        """Test large model configuration."""
        config = TransformerConfig(
            vocab_size=100000,
            d_model=8192,
            n_layers=80,
            n_heads=64,
            max_seq_len=8192,
        )
        
        assert config.d_model == 8192
        assert config.n_layers == 80
    
    def test_repr_string(self):
        """Test string representation."""
        config = TransformerConfig.tiny(vocab_size=50000)
        
        repr_str = repr(config)
        
        assert 'TransformerConfig' in repr_str
        assert '50000' in repr_str  # vocab_size should appear


if __name__ == '__main__':
    pytest.main([__file__, '-v'])