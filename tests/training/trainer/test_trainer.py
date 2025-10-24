"""Comprehensive tests for trainer module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile
import shutil

from ramanujan.training.trainer import (
    Trainer,
    TrainingConfig,
    TrainingState,
    Callback,
    WandBCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    ProgressCallback
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_model():
    """Simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Embedding + output layer for sequence modeling
            self.embed = nn.Embedding(100, 32)
            self.fc = nn.Linear(32, 10)  # vocab_size = 10
        
        def forward(self, input_ids, **kwargs):
            # Input: (batch, seq) with token indices
            # Output: (batch, seq, vocab)
            x = self.embed(input_ids)  # (batch, seq, 32)
            logits = self.fc(x)  # (batch, seq, 10)
            return {'logits': logits}
    
    return SimpleModel()


@pytest.fixture
def dummy_dataloader():
    """Dummy dataloader for testing."""
    # Create synthetic data
    input_ids = torch.randint(0, 100, (100, 10))
    labels = torch.randint(0, 10, (100, 10))
    
    dataset = TensorDataset(input_ids, labels)
    
    # Create dataloader that returns dicts
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        return {
            'input_ids': torch.stack(inputs),
            'labels': torch.stack(targets)
        }
    
    return DataLoader(dataset, batch_size=4, collate_fn=collate_fn)


@pytest.fixture
def temp_output_dir():
    """Temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def training_config(temp_output_dir):
    """Basic training config."""
    return TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=10,
        batch_size=4,
        learning_rate=1e-3,
        eval_every=5,
        save_every=5,
        log_every=2,
        mixed_precision=False,  # Disable for testing
        use_wandb=False
    )


# ============================================================================
# TEST CONFIG
# ============================================================================

def test_training_config_creation():
    """Test creating training config."""
    config = TrainingConfig(
        output_dir='runs/test',
        max_steps=1000,
        batch_size=32
    )
    
    assert config.output_dir == 'runs/test'
    assert config.max_steps == 1000
    assert config.batch_size == 32
    assert config.learning_rate == 1e-3  # Default


def test_training_config_to_dict():
    """Test config serialization."""
    config = TrainingConfig(
        output_dir='runs/test',
        max_steps=1000
    )
    
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict['output_dir'] == 'runs/test'
    assert config_dict['max_steps'] == 1000


def test_training_config_from_dict():
    """Test config deserialization."""
    config_dict = {
        'output_dir': 'runs/test',
        'max_steps': 1000,
        'batch_size': 32
    }
    
    config = TrainingConfig.from_dict(config_dict)
    
    assert config.output_dir == 'runs/test'
    assert config.max_steps == 1000
    assert config.batch_size == 32


def test_training_config_validation():
    """Test config validation."""
    with pytest.raises(ValueError, match="max_steps or max_epochs"):
        TrainingConfig(output_dir='runs/test', max_steps=0, max_epochs=None)


# ============================================================================
# TEST TRAINING STATE
# ============================================================================

def test_training_state_creation():
    """Test creating training state."""
    state = TrainingState()
    
    assert state.step == 0
    assert state.epoch == 0
    assert state.global_step == 0
    assert state.best_metric is None


def test_training_state_serialization():
    """Test state save/load."""
    state = TrainingState(
        step=100,
        epoch=5,
        global_step=100,
        best_metric=2.5,
        best_step=90
    )
    
    state_dict = state.to_dict()
    loaded_state = TrainingState.from_dict(state_dict)
    
    assert loaded_state.step == 100
    assert loaded_state.epoch == 5
    assert loaded_state.best_metric == 2.5


# ============================================================================
# TEST TRAINER INITIALIZATION
# ============================================================================

def test_trainer_creation(simple_model, dummy_dataloader, training_config):
    """Test creating trainer."""
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert trainer.loss_fn is not None


def test_trainer_with_validation_data(simple_model, dummy_dataloader, training_config):
    """Test trainer with validation dataloader."""
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader
    )
    
    assert trainer.val_dataloader is not None


def test_trainer_with_callbacks(simple_model, dummy_dataloader, training_config):
    """Test trainer with callbacks."""
    callback = ProgressCallback()
    
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader,
        callbacks=[callback]
    )
    
    assert len(trainer.callbacks) == 1


# ============================================================================
# TEST TRAINING
# ============================================================================

def test_trainer_train_steps(simple_model, dummy_dataloader, training_config):
    """Test training for fixed number of steps."""
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    initial_step = trainer.state.global_step
    
    trainer.train()
    
    # Should have completed training
    assert trainer.state.global_step == training_config.max_steps
    assert trainer.state.global_step > initial_step


def test_trainer_train_with_validation(simple_model, dummy_dataloader, training_config):
    """Test training with validation."""
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader
    )
    
    trainer.train()
    
    # Should have run validation
    assert len(trainer.state.metrics_history) > 0


def test_trainer_gradient_accumulation(simple_model, dummy_dataloader, temp_output_dir):
    """Test gradient accumulation."""
    config = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=10,
        batch_size=4,
        gradient_accumulation_steps=2,
        mixed_precision=False
    )
    
    trainer = Trainer(
        model=simple_model,
        config=config,
        train_dataloader=dummy_dataloader
    )
    
    trainer.train()
    
    # Should have completed training
    assert trainer.state.global_step == config.max_steps


def test_trainer_mixed_precision(simple_model, dummy_dataloader, temp_output_dir):
    """Test mixed precision training."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    config = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=5,
        batch_size=4,
        mixed_precision=True,
        device='cuda'
    )
    
    trainer = Trainer(
        model=simple_model,
        config=config,
        train_dataloader=dummy_dataloader
    )
    
    trainer.train()
    
    assert trainer.scaler is not None


# ============================================================================
# TEST CHECKPOINTING
# ============================================================================

def test_trainer_save_checkpoint(simple_model, dummy_dataloader, training_config):
    """Test saving checkpoint."""
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    # Train a bit
    for _ in range(5):
        batch = next(iter(dummy_dataloader))
        trainer._train_step(batch)
    
    # Save checkpoint
    checkpoint_path = Path(training_config.output_dir) / 'test_checkpoint.pt'
    trainer.save_checkpoint(str(checkpoint_path))
    
    assert checkpoint_path.exists()


def test_trainer_load_checkpoint(simple_model, dummy_dataloader, training_config):
    """Test loading checkpoint."""
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    # Train and save
    for _ in range(5):
        batch = next(iter(dummy_dataloader))
        trainer._train_step(batch)
    
    saved_step = trainer.state.global_step
    checkpoint_path = Path(training_config.output_dir) / 'test_checkpoint.pt'
    trainer.save_checkpoint(str(checkpoint_path))
    
    # Create new trainer and load
    trainer2 = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    trainer2.load_checkpoint(str(checkpoint_path))
    
    assert trainer2.state.global_step == saved_step


def test_trainer_resume_training(simple_model, dummy_dataloader, temp_output_dir):
    """Test resuming training from checkpoint."""
    # First training run
    config1 = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=5,
        batch_size=4,
        save_every=5,
        mixed_precision=False
    )
    
    trainer1 = Trainer(
        model=simple_model,
        config=config1,
        train_dataloader=dummy_dataloader
    )
    trainer1.train()
    
    checkpoint_path = Path(temp_output_dir) / 'checkpoints' / f'checkpoint_step_{config1.max_steps}.pt'
    assert checkpoint_path.exists()
    
    # Resume training
    config2 = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=10,
        batch_size=4,
        resume_from=str(checkpoint_path),
        mixed_precision=False
    )
    
    trainer2 = Trainer(
        model=simple_model,
        config=config2,
        train_dataloader=dummy_dataloader
    )
    
    assert trainer2.state.global_step == 5
    
    trainer2.train()
    
    assert trainer2.state.global_step == 10


# ============================================================================
# TEST VALIDATION
# ============================================================================

def test_trainer_validate(simple_model, dummy_dataloader, training_config):
    """Test validation."""
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader
    )
    
    val_metrics = trainer.validate()
    
    assert 'val_loss' in val_metrics
    assert isinstance(val_metrics['val_loss'], float)


def test_trainer_validate_with_eval_steps(simple_model, dummy_dataloader, temp_output_dir):
    """Test validation with limited steps."""
    config = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=10,
        eval_steps=2,  # Only evaluate on 2 batches
        mixed_precision=False
    )
    
    trainer = Trainer(
        model=simple_model,
        config=config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader
    )
    
    val_metrics = trainer.validate()
    
    assert 'val_loss' in val_metrics


# ============================================================================
# TEST CALLBACKS
# ============================================================================

def test_callback_interface():
    """Test callback interface."""
    callback = Callback()
    
    # Should have all required methods
    assert hasattr(callback, 'on_train_begin')
    assert hasattr(callback, 'on_train_end')
    assert hasattr(callback, 'on_step_begin')
    assert hasattr(callback, 'on_step_end')


def test_progress_callback(simple_model, dummy_dataloader, training_config, capsys):
    """Test progress callback."""
    callback = ProgressCallback()
    
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader,
        callbacks=[callback]
    )
    
    trainer.train()
    
    captured = capsys.readouterr()
    assert "Training started" in captured.out
    assert "Training completed" in captured.out


def test_checkpoint_callback(simple_model, dummy_dataloader, training_config):
    """Test checkpoint callback."""
    callback = CheckpointCallback(save_every=5, keep_last_n=2)
    
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader,
        callbacks=[callback]
    )
    
    trainer.train()
    
    checkpoint_dir = Path(training_config.output_dir) / 'checkpoints'
    checkpoints = list(checkpoint_dir.glob('checkpoint_step_*.pt'))
    
    assert len(checkpoints) > 0


def test_early_stopping_callback(simple_model, dummy_dataloader, temp_output_dir):
    """Test early stopping callback."""
    config = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=100,  # Many steps, but should stop early
        eval_every=2,
        mixed_precision=False
    )
    
    callback = EarlyStoppingCallback(
        monitor='val_loss',
        patience=2,
        mode='min',
        min_delta=0.01  # Require 0.01 improvement to count as progress
    )
    
    trainer = Trainer(
        model=simple_model,
        config=config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader,
        callbacks=[callback]
    )
    
    trainer.train()
    
    # Should have stopped before max_steps
    assert trainer.state.global_step < config.max_steps


# ============================================================================
# TEST METRICS
# ============================================================================

def test_trainer_metrics_tracking(simple_model, dummy_dataloader, training_config):
    """Test metrics are tracked."""
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    trainer.train()
    
    assert len(trainer.state.metrics_history) > 0
    assert 'loss' in trainer.state.metrics_history[0]


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_trainer_empty_validation(simple_model, dummy_dataloader, training_config):
    """Test trainer without validation dataloader."""
    trainer = Trainer(
        model=simple_model,
        config=training_config,
        train_dataloader=dummy_dataloader,
        val_dataloader=None
    )
    
    val_metrics = trainer.validate()
    
    assert val_metrics == {}


def test_trainer_with_invalid_config():
    """Test trainer with invalid config."""
    with pytest.raises(ValueError):
        TrainingConfig(
            output_dir='runs/test',
            max_steps=0,
            max_epochs=None
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])