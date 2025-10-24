"""Loss functions for Candide framework."""

from .base import LossComponent, LossSpec
from .cross_entropy import CrossEntropyLoss
from .semantic_entropy import SemanticEntropyProbe
from .kl_divergence import KLDivergenceLoss
from .composite import CompositeLoss
from .factory import (
    create_loss,
    create_loss_from_config,
    create_loss_from_spec,
    LOSS_REGISTRY
)


__all__ = [
    # Protocol and base
    'LossComponent',
    'LossSpec',
    
    # Loss implementations
    'CrossEntropyLoss',
    'SemanticEntropyProbe',
    'KLDivergenceLoss',
    'CompositeLoss',
    
    # Factory functions
    'create_loss',
    'create_loss_from_config',
    'create_loss_from_spec',
    
    # Registry
    'LOSS_REGISTRY',
]