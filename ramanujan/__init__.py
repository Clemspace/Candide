"""
Ramanujan Transformer: Efficient sparse transformers using Ramanujan graph theory.

Main components:
- foundation: Ramanujan graph construction and sparse layers
- architecture: Model components (attention, FFN, blocks)
- training: Training utilities (losses, optimizers, trainer)
- data: Dataset loaders and preprocessing
- utils: Configuration, metrics, logging

Example usage:
    >>> from ramanujan import RamanujanFoundation, create_model, Trainer
    >>> from ramanujan.utils import load_config
    >>> 
    >>> config = load_config("configs/optimal.yaml")
    >>> model = create_model(config)
    >>> trainer = Trainer(model, config, train_loader, eval_loader)
    >>> trainer.train()
"""

__version__ = "0.1.0"
__author__ = "Clément Castellon"

# Core foundation
from .foundation import (
    RamanujanFoundation,
    RamanujanLinearLayer,
    RamanujanMath
)

# Architecture
from .architecture import (
    create_model,
    TransformerBlock,
    EnhancedPretrainingModel
)

# Training
from .training import (
    Trainer,
    SemanticEntropyLoss,
    create_optimizer,
    CosineWarmupScheduler
)

# Utils
from .utils import (
    load_config,
    ExperimentConfig,
    compute_sparsity_stats
)

__all__ = [
    # Foundation
    'RamanujanFoundation',
    'RamanujanLinearLayer',
    'RamanujanMath',
    
    # Architecture
    'create_model',
    'TransformerBlock',
    'EnhancedPretrainingModel',
    
    # Training
    'Trainer',
    'SemanticEntropyLoss',
    'create_optimizer',
    'CosineWarmupScheduler',
    
    # Utils
    'load_config',
    'ExperimentConfig',
    'compute_sparsity_stats',
]
