"""
Configuration Loading Utilities
================================

Enhanced config loading with:
- Multiple format support (YAML, JSON, TOML)
- Environment variable substitution
- Config merging and overrides
- CLI argument parsing
- Validation on load
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Type
import os
import re
import json
import yaml

from .config import BaseConfig, ExperimentConfig


# ============================================================================
# FILE FORMAT SUPPORT
# ============================================================================

def load_config(
    path: Union[str, Path],
    config_class: Type[BaseConfig] = ExperimentConfig,
    validate: bool = True,
    expand_env_vars: bool = True,
) -> BaseConfig:
    """
    Load configuration from file with enhanced features.
    
    Args:
        path: Path to config file (YAML, JSON, or TOML)
        config_class: Config class to instantiate
        validate: Whether to validate after loading
        expand_env_vars: Whether to expand environment variables
    
    Returns:
        Config instance
    
    Example:
        # Load with environment variables
        os.environ['VOCAB_SIZE'] = '50000'
        config = load_config('config.yaml')  # Can use ${VOCAB_SIZE}
        
        # Load specific config type
        model_config = load_config('model.yaml', ModelConfig)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Load raw dict based on file type
    if path.suffix in ['.yaml', '.yml']:
        config_dict = _load_yaml(path)
    elif path.suffix == '.json':
        config_dict = _load_json(path)
    elif path.suffix == '.toml':
        config_dict = _load_toml(path)
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported: .yaml, .yml, .json, .toml"
        )
    
    # Expand environment variables if requested
    if expand_env_vars:
        config_dict = expand_environment_variables(config_dict)
    
    # Create config instance
    config = config_class.from_dict(config_dict)
    
    # Validate if requested
    if validate and hasattr(config, 'validate'):
        config.validate()
    
    return config


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def _load_toml(path: Path) -> Dict[str, Any]:
    """Load TOML file."""
    try:
        import tomli
    except ImportError:
        try:
            import tomllib as tomli  # Python 3.11+
        except ImportError:
            raise ImportError(
                "TOML support requires 'tomli' package. "
                "Install with: pip install tomli"
            )
    
    with open(path, 'rb') as f:
        return tomli.load(f)


def save_config(
    config: BaseConfig,
    path: Union[str, Path],
    format: Optional[str] = None,
):
    """
    Save configuration to file.
    
    Args:
        config: Config instance to save
        path: Output file path
        format: File format ('yaml', 'json', 'toml'). Auto-detected if None.
    
    Example:
        save_config(config, 'config.yaml')
        save_config(config, 'config.json', format='json')
    """
    path = Path(path)
    
    # Auto-detect format from extension
    if format is None:
        format = path.suffix.lstrip('.')
        if format == 'yml':
            format = 'yaml'
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on format
    if format == 'yaml':
        config.to_yaml(path)
    elif format == 'json':
        config.to_json(path)
    elif format == 'toml':
        _save_toml(config, path)
    else:
        raise ValueError(
            f"Unsupported format: {format}. "
            f"Supported: yaml, json, toml"
        )


def _save_toml(config: BaseConfig, path: Path):
    """Save config to TOML file."""
    try:
        import tomli_w
    except ImportError:
        raise ImportError(
            "TOML writing requires 'tomli_w' package. "
            "Install with: pip install tomli-w"
        )
    
    with open(path, 'wb') as f:
        tomli_w.dump(config.to_dict(), f)


# ============================================================================
# ENVIRONMENT VARIABLE EXPANSION
# ============================================================================

def expand_environment_variables(
    config_dict: Dict[str, Any],
    fail_on_missing: bool = False,
) -> Dict[str, Any]:
    """
    Recursively expand environment variables in config.
    
    Supports formats:
        - ${VAR_NAME}
        - ${VAR_NAME:default_value}
        - $VAR_NAME (simple form)
    
    Args:
        config_dict: Configuration dictionary
        fail_on_missing: Whether to fail if env var not found
    
    Returns:
        Config dict with expanded variables
    
    Example:
        config = {
            'model': {
                'vocab_size': '${VOCAB_SIZE:50000}',
                'd_model': '${MODEL_DIM}'
            }
        }
        expanded = expand_environment_variables(config)
    """
    if isinstance(config_dict, dict):
        return {
            key: expand_environment_variables(value, fail_on_missing)
            for key, value in config_dict.items()
        }
    elif isinstance(config_dict, list):
        return [
            expand_environment_variables(item, fail_on_missing)
            for item in config_dict
        ]
    elif isinstance(config_dict, str):
        return _expand_string(config_dict, fail_on_missing)
    else:
        return config_dict


def _expand_string(value: str, fail_on_missing: bool) -> Any:
    """Expand environment variables in a string."""
    # Pattern: ${VAR_NAME:default} or ${VAR_NAME} or $VAR_NAME
    pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*?)(?::([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)'
    
    def replacer(match):
        # Handle ${VAR:default} format
        if match.group(1):
            var_name = match.group(1)
            default = match.group(2)
        # Handle $VAR format
        else:
            var_name = match.group(3)
            default = None
        
        # Get environment variable
        env_value = os.environ.get(var_name)
        
        if env_value is not None:
            return env_value
        elif default is not None:
            return default
        elif fail_on_missing:
            raise ValueError(
                f"Environment variable '{var_name}' not found and no default provided"
            )
        else:
            # Return original if not found
            return match.group(0)
    
    result = re.sub(pattern, replacer, value)
    
    # Try to convert to appropriate type
    return _auto_convert_type(result)


def _auto_convert_type(value: str) -> Any:
    """Automatically convert string to appropriate type."""
    # Try boolean
    if value.lower() in ('true', 'yes', 'on'):
        return True
    if value.lower() in ('false', 'no', 'off'):
        return False
    
    # Try None
    if value.lower() in ('none', 'null'):
        return None
    
    # Try integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


# ============================================================================
# CONFIG MERGING
# ============================================================================

def merge_configs(
    base_config: Union[BaseConfig, Dict[str, Any]],
    *override_configs: Union[BaseConfig, Dict[str, Any]],
    config_class: Type[BaseConfig] = ExperimentConfig,
) -> BaseConfig:
    """
    Merge multiple configs with override priority.
    
    Later configs override earlier ones. Useful for:
    - Base config + experiment-specific overrides
    - Shared settings + user-specific settings
    - Default config + command-line overrides
    
    Args:
        base_config: Base configuration
        *override_configs: Configs to merge (in priority order)
        config_class: Config class for final result
    
    Returns:
        Merged config
    
    Example:
        base = load_config('base.yaml')
        exp = load_config('experiment.yaml')
        merged = merge_configs(base, exp)
    """
    # Convert all to dicts
    base_dict = (
        base_config.to_dict() if isinstance(base_config, BaseConfig)
        else base_config
    )
    
    # Deep merge all override dicts
    for override in override_configs:
        override_dict = (
            override.to_dict() if isinstance(override, BaseConfig)
            else override
        )
        base_dict = _deep_merge(base_dict, override_dict)
    
    # Create final config
    return config_class.from_dict(base_dict)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Override values take precedence. Nested dicts are merged recursively.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = _deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value
    
    return result


# ============================================================================
# CLI OVERRIDES
# ============================================================================

def load_config_with_overrides(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    cli_args: Optional[List[str]] = None,
    config_class: Type[BaseConfig] = ExperimentConfig,
) -> BaseConfig:
    """
    Load config with CLI-style overrides.
    
    Supports dot notation for nested overrides:
        model.d_model=512
        training.learning_rate=1e-4
    
    Args:
        config_path: Path to base config file
        overrides: Dictionary of overrides
        cli_args: List of CLI-style override strings
        config_class: Config class to instantiate
    
    Returns:
        Config with overrides applied
    
    Example:
        # From dict
        config = load_config_with_overrides(
            'config.yaml',
            overrides={'model.d_model': 512}
        )
        
        # From CLI args
        config = load_config_with_overrides(
            'config.yaml',
            cli_args=['model.d_model=512', 'training.batch_size=16']
        )
    """
    # Load base config
    base_config = load_config(config_path, config_class, validate=False)
    base_dict = base_config.to_dict()
    
    # Apply dictionary overrides
    if overrides:
        override_dict = _flatten_to_nested(overrides)
        base_dict = _deep_merge(base_dict, override_dict)
    
    # Parse CLI args
    if cli_args:
        cli_overrides = _parse_cli_overrides(cli_args)
        base_dict = _deep_merge(base_dict, cli_overrides)
    
    # Create final config with validation
    config = config_class.from_dict(base_dict)
    if hasattr(config, 'validate'):
        config.validate()
    
    return config


def _parse_cli_overrides(cli_args: List[str]) -> Dict[str, Any]:
    """
    Parse CLI-style overrides.
    
    Format: key=value or key.nested=value
    """
    overrides = {}
    
    for arg in cli_args:
        if '=' not in arg:
            raise ValueError(f"Invalid override format: {arg}. Expected key=value")
        
        key, value = arg.split('=', 1)
        overrides[key] = _auto_convert_type(value)
    
    return _flatten_to_nested(overrides)


def _flatten_to_nested(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert flat dot notation dict to nested dict.
    
    Example:
        {'model.d_model': 512, 'training.lr': 1e-4}
        -> {'model': {'d_model': 512}, 'training': {'lr': 1e-4}}
    """
    result = {}
    
    for key, value in flat_dict.items():
        parts = key.split('.')
        current = result
        
        # Navigate/create nested structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set final value
        current[parts[-1]] = value
    
    return result


# ============================================================================
# BATCH LOADING
# ============================================================================

def load_configs_from_directory(
    directory: Union[str, Path],
    pattern: str = "*.yaml",
    config_class: Type[BaseConfig] = ExperimentConfig,
) -> Dict[str, BaseConfig]:
    """
    Load all configs from a directory.
    
    Useful for loading multiple experiment configs.
    
    Args:
        directory: Directory containing config files
        pattern: Glob pattern for files
        config_class: Config class to instantiate
    
    Returns:
        Dict mapping filename (without extension) to config
    
    Example:
        configs = load_configs_from_directory('experiments/')
        for name, config in configs.items():
            print(f"Experiment: {name}")
    """
    directory = Path(directory)
    configs = {}
    
    for config_file in directory.glob(pattern):
        name = config_file.stem
        try:
            config = load_config(config_file, config_class)
            configs[name] = config
        except Exception as e:
            print(f"Warning: Failed to load {config_file}: {e}")
    
    return configs