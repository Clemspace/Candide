"""Test fixtures for trainer tests - Updated for new API."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from pathlib import Path

from ramanujan.training.trainer.base import TrainingConfig
from ramanujan.training.optimizers import create_optimizer_from_config
from ramanujan.training.losses import create_loss_from_config
from ramanujan.training.schedulers import create_scheduler_from_config


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, vocab_size=100, d_model=32, num_classes=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        x = x.mean(dim=1)
        return self.fc(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def simple_optimizer(simple_model):
    """Create optimizer for simple model."""
    return create_optimizer_from_config(
        {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.01},
        simple_model.parameters()
    )


@pytest.fixture
def simple_loss():
    """Create loss function."""
    return create_loss_from_config({
        'name': 'cross_entropy',
        'vocab_size': 100,
    })


@pytest.fixture
def simple_scheduler(simple_optimizer):
    """Create scheduler."""
    return create_scheduler_from_config(
        {'name': 'constant'},
        simple_optimizer
    )


@pytest.fixture
def dummy_dataloader():
    """Create a dummy dataloader."""
    # Create synthetic data - classification task
    input_ids = torch.randint(0, 100, (50, 16))  # vocab tokens
    targets = torch.randint(0, 10, (50,))  # class labels (0-9)
    dataset = TensorDataset(input_ids, targets)
    
    def collate_fn(batch):
        # batch is list of tuples: [(input_tensor, target), ...]
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return {'input_ids': inputs, 'labels': labels}
    
    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def training_config(temp_output_dir):
    """Create training configuration."""
    return TrainingConfig(
        output_dir=temp_output_dir,
        experiment_name=Path(temp_output_dir).name,
        max_steps=10,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.01,
        max_grad_norm=1.0,
        log_every=2,
        eval_every=5,
        save_every=5,
        mixed_precision=False,
        device='cpu',
    )