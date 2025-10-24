"""Training module for Candide framework."""

from .base import TrainingConfig, TrainingState, CallbackProtocol
from .trainer import Trainer
from .callbacks import (
    Callback,
    WandBCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    ProgressCallback,
    MetricsSaverCallback,
    SparsityTrackerCallback
)


__all__ = [
    # Config and state
    'TrainingConfig',
    'TrainingState',
    'CallbackProtocol',
    
    # Trainer
    'Trainer',
    
    # Callbacks
    'Callback',
    'WandBCallback',
    'EarlyStoppingCallback',
    'CheckpointCallback',
    'ProgressCallback',
    'MetricsSaverCallback',
    'SparsityTrackerCallback',
]