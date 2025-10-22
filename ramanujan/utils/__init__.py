"""
Utilities: Configuration, metrics, and logging.

This module provides utility functions for configuration management,
metrics computation, and experiment logging.

Components:
- Configuration classes and YAML loading
- Metrics computation (sparsity, BPB, etc.)
- Logging utilities (WandB, console)
- Report generation

Example:
    >>> from ramanujan.utils import load_config, compute_sparsity_stats
    >>> 
    >>> config = load_config("configs/optimal.yaml")
    >>> stats = compute_sparsity_stats(model)
    >>> print(f"Overall sparsity: {stats['overall']:.2%}")
"""

from .config import (
    ModelConfig,
    SparsityConfig,
    TrainingConfig,
    ExperimentConfig,
    load_config,
    save_config,
    create_ablation_configs
)

from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_info
)

from .metrics import (
    compute_sparsity_stats,
    compute_bpb,
    compute_perplexity,
    MetricsTracker,
    count_parameters
)

from .logging import (
    setup_logging,
    WandBLogger,
    ConsoleLogger,
    ExperimentLogger
)

__all__ = [
    # Config
    'ModelConfig',
    'SparsityConfig',
    'TrainingConfig',
    'ExperimentConfig',
    'load_config',
    'save_config',
    'create_ablation_configs',

    # Checkpoint
    'load_checkpoint',
    'save_checkpoint',
    'get_checkpoint_info',
    
    # Metrics
    'compute_sparsity_stats',
    'compute_bpb',
    'compute_perplexity',
    'MetricsTracker',
    'count_parameters',
    
    # Logging
    'setup_logging',
    'WandBLogger',
    'ConsoleLogger',
    'ExperimentLogger',
]