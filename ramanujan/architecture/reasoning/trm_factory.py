# reasoning/trm_factory.py
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn

@dataclass
class ReasoningConfig:
    """Configuration for reasoning modules."""
    # Type selection
    reasoning_type: str = 'trm'  # 'trm', 'hrm', 'none'
    
    # Core TRM parameters
    dim: int = 512
    num_layers: int = 2
    n_recursions: int = 6
    T_cycles: int = 3
    max_supervision_steps: int = 16
    
    # Architecture
    use_self_attention: bool = True
    window_size: Optional[int] = None
    dropout: float = 0.1
    ema_decay: float = 0.999
    
    # Vocabulary (inherited from model)
    vocab_size: int = 32000
    max_seq_len: int = 2048


@dataclass 
class AdapterConfig:
    """Configuration for LLM-TRM adapter."""
    llm_dim: int = 768
    trm_dim: int = 512
    fusion_type: str = 'gated'  # 'gated', 'cross_attn', 'concat', 'add'
    num_fusion_layers: int = 2
    dropout: float = 0.1


@dataclass
class RouterConfig:
    """Configuration for routing logic."""
    llm_dim: int = 768
    use_learned_routing: bool = True
    confidence_threshold: float = 0.8
    structured_task_threshold: float = 0.7


class ReasoningFactory:
    """
    Factory for creating reasoning modules.
    
    Example:
        >>> config = ReasoningConfig(
        ...     reasoning_type='trm',
        ...     dim=512,
        ...     n_recursions=6
        ... )
        >>> reasoning = ReasoningFactory.create(config)
    """
    
    @staticmethod
    def create(config: ReasoningConfig) -> Optional[nn.Module]:
        """Create reasoning module based on config."""
        if config.reasoning_type == 'none':
            return None
        elif config.reasoning_type == 'trm':
            from .trm_core import TinyRecursiveModel, TRMConfig
            trm_config = TRMConfig(
                dim=config.dim,
                num_layers=config.num_layers,
                n_recursions=config.n_recursions,
                T_cycles=config.T_cycles,
                max_supervision_steps=config.max_supervision_steps,
                use_self_attention=config.use_self_attention,
                window_size=config.window_size,
                dropout=config.dropout,
                ema_decay=config.ema_decay,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len
            )
            return TinyRecursiveModel(trm_config)
        elif config.reasoning_type == 'hrm':
            # Future: HRM implementation
            raise NotImplementedError("HRM not yet implemented")
        else:
            raise ValueError(f"Unknown reasoning_type: {config.reasoning_type}")
    
    @staticmethod
    def create_from_dict(config_dict: dict) -> Optional[nn.Module]:
        """Create from dictionary."""
        config = ReasoningConfig(**config_dict)
        return ReasoningFactory.create(config)