"""
Tests for Configuration Loaders
================================

Tests for loaders.py covering:
- Multiple file format loading (YAML, JSON, TOML)
- Environment variable expansion
- Config merging
- CLI overrides
- Batch loading
"""

import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ramanujan.config.config import ModelConfig, ExperimentConfig, create_default_config
from ramanujan.config.loaders import (
    load_config,
    save_config,
    expand_environment_variables,
    merge_configs,
    load_config_with_overrides,
    load_configs_from_directory,
    _auto_convert_type,
    _flatten_to_nested,
)


# ============================================================================
# FILE LOADING TESTS
# ============================================================================

class TestFileLoading:
    """Test loading configs from different file formats."""
    
    def test_load_yaml(self):
        """Test loading YAML config."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            temp_path = f.name
        
        try:
            loaded = load_config(temp_path)
            assert loaded.model.vocab_size == 50000
        finally:
            Path(temp_path).unlink()
    
    def test_load_json(self):
        """Test loading JSON config."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            temp_path = f.name
        
        try:
            loaded = load_config(temp_path)
            assert loaded.model.vocab_size == 50000
        finally:
            Path(temp_path).unlink()
    
    def test_load_yml_extension(self):
        """Test loading .yml extension (alternative to .yaml)."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config.to_dict(), f)
            temp_path = f.name
        
        try:
            loaded = load_config(temp_path)
            assert loaded.model.vocab_size == 50000
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent.yaml')
    
    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_specific_config_class(self):
        """Test loading specific config class."""
        model_config = ModelConfig(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            model_config.to_yaml(f.name)
            temp_path = f.name
        
        try:
            loaded = load_config(temp_path, config_class=ModelConfig)
            assert isinstance(loaded, ModelConfig)
            assert loaded.vocab_size == 50000
        finally:
            Path(temp_path).unlink()
    
    def test_load_with_validation_disabled(self):
        """Test loading with validation disabled."""
        # Create invalid config
        config_dict = {
            'model': {'vocab_size': 50000, 'max_seq_len': 1024},
            'training': {},
            'data': {'sequence_length': 2048},  # Invalid!
            'system': {},
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            # Should load without error when validate=False
            loaded = load_config(temp_path, validate=False)
            assert loaded.model.vocab_size == 50000
        finally:
            Path(temp_path).unlink()


# ============================================================================
# SAVE CONFIG TESTS
# ============================================================================

class TestSaveConfig:
    """Test saving configs to different formats."""
    
    def test_save_yaml(self):
        """Test saving to YAML."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path)
            assert Path(temp_path).exists()
            
            # Verify content
            loaded = load_config(temp_path)
            assert loaded.model.vocab_size == 50000
        finally:
            Path(temp_path).unlink()
    
    def test_save_json(self):
        """Test saving to JSON."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path)
            assert Path(temp_path).exists()
            
            loaded = load_config(temp_path)
            assert loaded.model.vocab_size == 50000
        finally:
            Path(temp_path).unlink()
    
    def test_save_creates_parent_dir(self):
        """Test that save_config creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_default_config(vocab_size=50000)
            nested_path = Path(tmpdir) / 'subdir' / 'config.yaml'
            
            save_config(config, nested_path)
            assert nested_path.exists()
    
    def test_save_explicit_format(self):
        """Test saving with explicit format."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path, format='json')
            
            # Should be JSON despite no extension
            with open(temp_path, 'r') as f:
                data = json.load(f)
            assert data['model']['vocab_size'] == 50000
        finally:
            Path(temp_path).unlink()


# ============================================================================
# ENVIRONMENT VARIABLE EXPANSION TESTS
# ============================================================================

class TestEnvironmentVariables:
    """Test environment variable expansion."""
    
    def test_expand_simple_var(self):
        """Test expanding simple ${VAR} format."""
        os.environ['TEST_VAR'] = '42'
        
        config_dict = {
            'value': '${TEST_VAR}',
        }
        
        expanded = expand_environment_variables(config_dict)
        assert expanded['value'] == 42  # Auto-converted to int
    
    def test_expand_with_default(self):
        """Test ${VAR:default} format."""
        # Variable doesn't exist
        config_dict = {
            'value': '${NONEXISTENT:100}',
        }
        
        expanded = expand_environment_variables(config_dict)
        assert expanded['value'] == 100
    
    def test_expand_dollar_var(self):
        """Test simple $VAR format."""
        os.environ['TEST_VAR'] = '42'
        
        config_dict = {
            'value': '$TEST_VAR',
        }
        
        expanded = expand_environment_variables(config_dict)
        assert expanded['value'] == 42
    
    def test_expand_nested_dict(self):
        """Test expansion in nested dictionary."""
        os.environ['MODEL_DIM'] = '768'
        
        config_dict = {
            'model': {
                'd_model': '${MODEL_DIM}',
            }
        }
        
        expanded = expand_environment_variables(config_dict)
        assert expanded['model']['d_model'] == 768
    
    def test_expand_list(self):
        """Test expansion in lists."""
        os.environ['VAL1'] = '1'
        os.environ['VAL2'] = '2'
        
        config_dict = {
            'values': ['${VAL1}', '${VAL2}'],
        }
        
        expanded = expand_environment_variables(config_dict)
        assert expanded['values'] == [1, 2]
    
    def test_expand_fail_on_missing(self):
        """Test fail_on_missing flag."""
        config_dict = {
            'value': '${NONEXISTENT_VAR}',
        }
        
        with pytest.raises(ValueError, match="not found"):
            expand_environment_variables(config_dict, fail_on_missing=True)
    
    def test_auto_convert_bool(self):
        """Test auto-conversion of boolean strings."""
        assert _auto_convert_type('true') == True
        assert _auto_convert_type('false') == False
        assert _auto_convert_type('yes') == True
        assert _auto_convert_type('no') == False
    
    def test_auto_convert_none(self):
        """Test auto-conversion of None."""
        assert _auto_convert_type('none') is None
        assert _auto_convert_type('null') is None
    
    def test_auto_convert_numbers(self):
        """Test auto-conversion of numbers."""
        assert _auto_convert_type('42') == 42
        assert _auto_convert_type('3.14') == 3.14
        assert _auto_convert_type('1e-4') == 1e-4
    
    def test_load_with_env_vars(self):
        """Test loading config with environment variables."""
        os.environ['VOCAB_SIZE'] = '50000'
        os.environ['D_MODEL'] = '768'
        
        config_dict = {
            'model': {
                'vocab_size': '${VOCAB_SIZE}',
                'd_model': '${D_MODEL}',
            },
            'training': {},
            'data': {},
            'system': {},
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.model.vocab_size == 50000
            assert config.model.d_model == 768
        finally:
            Path(temp_path).unlink()


# ============================================================================
# CONFIG MERGING TESTS
# ============================================================================

class TestConfigMerging:
    """Test config merging functionality."""
    
    def test_merge_two_configs(self):
        """Test merging two configs."""
        base = create_default_config(vocab_size=50000, model_preset='llama')
        override = create_default_config(
            vocab_size=50000,
            model_preset='llama',
            training={'max_steps': 5000},
        )
        
        merged = merge_configs(base, override)
        assert merged.training.max_steps == 5000
        # Base values should remain
        assert merged.model.vocab_size == 50000
    
    def test_merge_from_dicts(self):
        """Test merging from dictionaries."""
        base_dict = {
            'model': {'vocab_size': 50000, 'd_model': 768},
            'training': {'max_steps': 10000},
            'data': {},
            'system': {},
        }
        
        override_dict = {
            'model': {'d_model': 1024, 'n_heads': 16},  # Override d_model and adjust n_heads
            'training': {'batch_size': 32},  # Add batch_size
        }
        
        merged = merge_configs(base_dict, override_dict)
        assert merged.model.d_model == 1024  # Overridden
        assert merged.model.vocab_size == 50000  # Kept
        assert merged.training.batch_size == 32  # Added
    
    def test_merge_multiple_configs(self):
        """Test merging multiple configs."""
        base = {'model': {'vocab_size': 50000, 'd_model': 768}}
        override1 = {'model': {'d_model': 1024, 'n_heads': 16}}  # Adjust n_heads too
        override2 = {'model': {'n_layers': 24}}
        
        # Create complete configs
        base_config = create_default_config(vocab_size=50000)
        base_dict = base_config.to_dict()
        base_dict.update(base)
        
        merged = merge_configs(base_dict, override1, override2)
        assert merged.model.vocab_size == 50000
        assert merged.model.d_model == 1024
        assert merged.model.n_layers == 24
    
    def test_deep_merge(self):
        """Test deep merging of nested dicts."""
        base = create_default_config(vocab_size=50000)
        
        override_dict = {
            'model': {
                'd_model': 512,
                'custom_components': {'new_feature': True},
            }
        }
        
        merged = merge_configs(base, override_dict)
        assert merged.model.d_model == 512
        assert merged.model.custom_components['new_feature'] == True


# ============================================================================
# CLI OVERRIDES TESTS
# ============================================================================

class TestCLIOverrides:
    """Test CLI-style overrides."""
    
    def test_load_with_dict_overrides(self):
        """Test loading with dictionary overrides."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            temp_path = f.name
        
        try:
            loaded = load_config_with_overrides(
                temp_path,
                overrides={'model.d_model': 512, 'training.batch_size': 32},
            )
            assert loaded.model.d_model == 512
            assert loaded.training.batch_size == 32
        finally:
            Path(temp_path).unlink()
    
    def test_load_with_cli_args(self):
        """Test loading with CLI arguments."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            temp_path = f.name
        
        try:
            loaded = load_config_with_overrides(
                temp_path,
                cli_args=['model.d_model=512', 'training.batch_size=32'],
            )
            assert loaded.model.d_model == 512
            assert loaded.training.batch_size == 32
        finally:
            Path(temp_path).unlink()
    
    def test_flatten_to_nested(self):
        """Test conversion of flat to nested dict."""
        flat = {
            'model.d_model': 512,
            'training.batch_size': 32,
            'system.device': 'cpu',
        }
        
        nested = _flatten_to_nested(flat)
        assert nested == {
            'model': {'d_model': 512},
            'training': {'batch_size': 32},
            'system': {'device': 'cpu'},
        }
    
    def test_flatten_deep_nesting(self):
        """Test flattening with deep nesting."""
        flat = {
            'a.b.c.d': 'value',
        }
        
        nested = _flatten_to_nested(flat)
        assert nested == {
            'a': {
                'b': {
                    'c': {
                        'd': 'value'
                    }
                }
            }
        }
    
    def test_invalid_cli_arg(self):
        """Test invalid CLI argument format."""
        config = create_default_config(vocab_size=50000)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid override"):
                load_config_with_overrides(
                    temp_path,
                    cli_args=['invalid_format'],
                )
        finally:
            Path(temp_path).unlink()


# ============================================================================
# BATCH LOADING TESTS
# ============================================================================

class TestBatchLoading:
    """Test loading multiple configs from directory."""
    
    def test_load_configs_from_directory(self):
        """Test loading all configs from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple configs
            config1 = create_default_config(vocab_size=50000)
            config2 = create_default_config(vocab_size=60000)
            
            config1.to_yaml(Path(tmpdir) / 'exp1.yaml')
            config2.to_yaml(Path(tmpdir) / 'exp2.yaml')
            
            # Load all
            configs = load_configs_from_directory(tmpdir)
            
            assert len(configs) == 2
            assert 'exp1' in configs
            assert 'exp2' in configs
            assert configs['exp1'].model.vocab_size == 50000
            assert configs['exp2'].model.vocab_size == 60000
    
    def test_load_with_pattern(self):
        """Test loading with specific pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create configs with different extensions
            config = create_default_config(vocab_size=50000)
            config.to_yaml(Path(tmpdir) / 'config.yaml')
            config.to_json(Path(tmpdir) / 'config.json')
            
            # Load only YAML
            configs = load_configs_from_directory(tmpdir, pattern='*.yaml')
            assert len(configs) == 1
            assert 'config' in configs
    
    def test_load_ignores_invalid_files(self):
        """Test that invalid files are skipped with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid config
            config = create_default_config(vocab_size=50000)
            config.to_yaml(Path(tmpdir) / 'valid.yaml')
            
            # Create invalid config
            with open(Path(tmpdir) / 'invalid.yaml', 'w') as f:
                f.write('invalid: yaml: content:')
            
            # Should load valid one and skip invalid
            configs = load_configs_from_directory(tmpdir)
            assert len(configs) == 1
            assert 'valid' in configs


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_workflow(self):
        """Test complete config loading workflow."""
        # Set environment variables
        os.environ['VOCAB_SIZE'] = '50000'
        os.environ['MAX_STEPS'] = '5000'
        
        # Create base config with env vars
        config_dict = {
            'model': {
                'vocab_size': '${VOCAB_SIZE}',
                'd_model': 768,
            },
            'training': {
                'max_steps': '${MAX_STEPS}',
            },
            'data': {},
            'system': {},
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            # Load with CLI overrides
            config = load_config_with_overrides(
                temp_path,
                cli_args=['model.d_model=768', 'model.n_heads=12', 'training.batch_size=32'],
            )
            
            # Check everything
            assert config.model.vocab_size == 50000  # From env var
            assert config.model.d_model == 768  # From CLI
            assert config.training.max_steps == 5000  # From env var
            assert config.training.batch_size == 32  # From CLI
        finally:
            Path(temp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])