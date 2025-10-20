"""
Training: Components for training Ramanujan models.

This module provides training utilities including loss functions,
optimizers, schedulers, and the main Trainer class.

Components:
- Loss functions (semantic entropy, standard CE)
- Optimizers (Muon, AdEMAMix, AdamW, hybrid)
- Learning rate schedulers
- Trainer class for orchestrating training

Example:
    >>> from ramanujan.training import Trainer, create_optimizer
    >>> from ramanujan import create_model
    >>> 
    >>> model = create_model(config)
    >>> optimizer = create_optimizer(model, config)
    >>> trainer = Trainer(model, optimizer, train_loader, eval_loader, config)
    >>> trainer.train()
"""

from .losses import (
    SemanticEntropyLoss,
    SemanticEntropyProbe,
    standard_cross_entropy,
    create_loss,
)

from .optimizers import (
    MuonOptimizer,
    AdEMAMixOptimizer,
    HybridOptimizerManager,
    create_optimizer,
    create_layerwise_optimizer
)

from .schedulers import (
    CosineWarmupScheduler,
    CosineWarmupWithRestarts,
    create_scheduler
)

from .trainer import (
    Trainer,
    AblationStudy
)

__all__ = [
    # Losses
    'SemanticEntropyLoss',
    'SemanticEntropyProbe',
    'standard_cross_entropy',
    'create_loss',
    
    # Optimizers
    'MuonOptimizer',
    'AdEMAMixOptimizer',
    'HybridOptimizerManager',
    'create_optimizer',
    'create_layerwise_optimizer',
    
    # Schedulers
    'CosineWarmupScheduler',
    'CosineWarmupWithRestarts',
    'create_scheduler',
    
    # Training
    'Trainer',
    'AblationStudy',
]