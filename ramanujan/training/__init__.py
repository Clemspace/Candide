"""Training utilities for Candide framework."""

# Loss system
from .losses import (
    LossComponent,
    LossSpec,
    CrossEntropyLoss,
    SemanticEntropyProbe,
    KLDivergenceLoss,
    CompositeLoss,
    create_loss,
    create_loss_from_config,
    create_loss_from_spec,
    LOSS_REGISTRY
)

# Optimizer system
from .optimizers import (
    OptimizerComponent,
    OptimizerSpec,
    BaseOptimizerWrapper,
    AdamWOptimizer,
    MuonOptimizer,
    AdEMAMixOptimizer,
    SGDOptimizer,
    LionOptimizer,
    create_optimizer,
    create_optimizer_from_config,
    create_optimizer_from_spec,
    create_optimizer_with_param_groups,
    create_optimizer_with_llrd,
    get_optimizer_info,
    get_learning_rates,
    set_learning_rate,
    create_param_groups,
    get_layer_wise_lr_groups,
    OPTIMIZER_REGISTRY
)

# Scheduler system
from .schedulers import (
    SchedulerComponent,
    SchedulerSpec,
    WarmupScheduler,
    CosineScheduler,
    ConstantScheduler,
    create_scheduler,
    create_scheduler_from_config,
    create_scheduler_from_spec,
    SCHEDULER_REGISTRY
)

# Trainer system
from .trainer import (
    Trainer,
    TrainingConfig,
    TrainingState,
    CallbackProtocol,
    Callback,
    WandBCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    ProgressCallback,
    MetricsSaverCallback,
    SparsityTrackerCallback
)



__all__ = [
    # Losses
    'LossComponent',
    'LossSpec',
    'CrossEntropyLoss',
    'SemanticEntropyProbe',
    'KLDivergenceLoss',
    'CompositeLoss',
    'create_loss',
    'create_loss_from_config',
    'create_loss_from_spec',
    'LOSS_REGISTRY',
    
    # Optimizers
    'OptimizerComponent',
    'OptimizerSpec',
    'BaseOptimizerWrapper',
    'AdamWOptimizer',
    'MuonOptimizer',
    'AdEMAMixOptimizer',
    'SGDOptimizer',
    'LionOptimizer',
    'create_optimizer',
    'create_optimizer_from_config',
    'create_optimizer_from_spec',
    'create_optimizer_with_param_groups',
    'create_optimizer_with_llrd',
    'get_optimizer_info',
    'get_learning_rates',
    'set_learning_rate',
    'create_param_groups',
    'get_layer_wise_lr_groups',
    'OPTIMIZER_REGISTRY',
    
    # Schedulers
    'SchedulerComponent',
    'SchedulerSpec',
    'WarmupScheduler',
    'CosineScheduler',
    'ConstantScheduler',
    'create_scheduler',
    'create_scheduler_from_config',
    'create_scheduler_from_spec',
    'SCHEDULER_REGISTRY',

    # Trainer
    'Trainer',
    'TrainingConfig',
    'TrainingState',
    'CallbackProtocol',
    'Callback',
    'WandBCallback',
    'EarlyStoppingCallback',
    'CheckpointCallback',
    'ProgressCallback',
    'MetricsSaverCallback',
    'SparsityTrackerCallback',
]