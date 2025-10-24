"""Model factory for easy instantiation."""

from typing import Optional, Dict, Any
from .architectures import TransformerConfig, RamanujanTransformer


def create_model(
    config: Optional[TransformerConfig | Dict[str, Any]] = None,
    preset: Optional[str] = None,
    **kwargs
) -> RamanujanTransformer:
    """
    Create a Ramanujan Transformer model.
    
    Args:
        config: TransformerConfig or config dict
        preset: Preset configuration ('tiny', 'small', 'medium', 'large', 'llama', 'gpt', 'bert')
        **kwargs: Override config values
    
    Returns:
        RamanujanTransformer model
    
    Example:
        >>> # Using preset
        >>> model = create_model(preset='tiny', vocab_size=50000)
        
        >>> # Using config dict
        >>> model = create_model(config={
        ...     'vocab_size': 50000,
        ...     'd_model': 768,
        ...     'n_layers': 12,
        ... })
        
        >>> # Using TransformerConfig
        >>> config = TransformerConfig.llama(vocab_size=50000)
        >>> model = create_model(config=config)
    """
    # Handle config
    if config is None:
        if preset is None:
            raise ValueError("Must provide either config or preset")
        
        # Create from preset
        preset_map = {
            'tiny': TransformerConfig.tiny,
            'small': TransformerConfig.small,
            'medium': TransformerConfig.medium,
            'large': TransformerConfig.large,
            'llama': TransformerConfig.llama,
            'gpt': TransformerConfig.gpt,
            'bert': TransformerConfig.bert,
        }
        
        if preset.lower() not in preset_map:
            raise ValueError(
                f"Unknown preset: {preset}. "
                f"Available: {list(preset_map.keys())}"
            )
        
        if 'vocab_size' not in kwargs:
            raise ValueError("vocab_size is required when using preset")
        
        config = preset_map[preset.lower()](**kwargs)
    
    elif isinstance(config, dict):
        # Create from dict
        config = TransformerConfig.from_dict(config)
    
    elif isinstance(config, TransformerConfig):
        # Already a config, optionally override
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = TransformerConfig.from_dict(config_dict)
    
    else:
        raise TypeError(
            f"config must be TransformerConfig or dict, got {type(config)}"
        )
    
    # Create model
    return RamanujanTransformer(config)


def get_config(preset: str, **kwargs) -> TransformerConfig:
    """
    Get a model configuration.
    
    Args:
        preset: Preset name
        **kwargs: Override values
    
    Returns:
        TransformerConfig
    
    Example:
        >>> config = get_config('tiny', vocab_size=50000, d_model=512)
    """
    preset_map = {
        'tiny': TransformerConfig.tiny,
        'small': TransformerConfig.small,
        'medium': TransformerConfig.medium,
        'large': TransformerConfig.large,
        'llama': TransformerConfig.llama,
        'gpt': TransformerConfig.gpt,
        'bert': TransformerConfig.bert,
    }
    
    if preset.lower() not in preset_map:
        raise ValueError(
            f"Unknown preset: {preset}. "
            f"Available: {list(preset_map.keys())}"
        )
    
    return preset_map[preset.lower()](**kwargs)