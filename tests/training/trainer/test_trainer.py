"""Tests for Trainer - Updated for new component injection API."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from ramanujan.training.trainer import Trainer
from ramanujan.training.trainer.base import TrainingConfig, TrainingState
from ramanujan.training.trainer.callbacks import (
    ProgressCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
)


# ============================================================================
# CONFIG TESTS
# ============================================================================

def test_training_config_creation():
    """Test creating training configuration."""
    config = TrainingConfig(
        output_dir="/tmp/test",
        max_steps=1000,
        batch_size=32,
    )
    assert config.output_dir == "/tmp/test"
    assert config.max_steps == 1000
    assert config.batch_size == 32


def test_training_config_to_dict():
    """Test config serialization."""
    config = TrainingConfig(output_dir="/tmp/test", max_steps=100)
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict['output_dir'] == "/tmp/test"


def test_training_config_from_dict():
    """Test config deserialization."""
    config_dict = {'output_dir': '/tmp/test', 'max_steps': 100}
    config = TrainingConfig.from_dict(config_dict)
    assert config.output_dir == "/tmp/test"


def test_training_config_validation():
    """Test config validation."""
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(output_dir="/tmp/test", max_steps=-1)


# ============================================================================
# STATE TESTS
# ============================================================================

def test_training_state_creation():
    """Test creating training state."""
    state = TrainingState()
    assert state.step == 0
    assert state.global_step == 0


def test_training_state_serialization():
    """Test state save/load."""
    state = TrainingState()
    state.step = 100
    state_dict = state.to_dict()
    
    new_state = TrainingState.from_dict(state_dict)
    assert new_state.step == 100


# ============================================================================
# TRAINER TESTS
# ============================================================================

def test_trainer_creation(simple_model, dummy_dataloader, training_config,
                          simple_optimizer, simple_loss):
    """Test creating trainer."""
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    assert trainer.model is simple_model
    assert trainer.optimizer is simple_optimizer
    assert trainer.loss_fn is simple_loss


def test_trainer_with_validation_data(simple_model, dummy_dataloader, training_config,
                                      simple_optimizer, simple_loss):
    """Test trainer with validation dataloader."""
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader
    )
    
    assert trainer.val_dataloader is dummy_dataloader


def test_trainer_with_callbacks(simple_model, dummy_dataloader, training_config,
                                simple_optimizer, simple_loss):
    """Test trainer with callbacks."""
    callback = ProgressCallback()
    
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader,
        callbacks=[callback]
    )
    
    assert len(trainer.callbacks) == 1


def test_trainer_train_steps(simple_model, dummy_dataloader, training_config,
                             simple_optimizer, simple_loss):
    """Test training for fixed number of steps."""
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    trainer.train()
    assert trainer.state.global_step == training_config.max_steps


def test_trainer_train_with_validation(simple_model, dummy_dataloader, training_config,
                                       simple_optimizer, simple_loss):
    """Test training with validation."""
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader
    )
    
    trainer.train()
    assert trainer.state.global_step == training_config.max_steps


def test_trainer_gradient_accumulation(simple_model, dummy_dataloader, temp_output_dir):
    """Test gradient accumulation."""
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.losses import create_loss_from_config
    
    config = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=10,
        batch_size=4,
        gradient_accumulation_steps=2,
        mixed_precision=False
    )
    
    opt = create_optimizer_from_config({'name': 'adamw', 'lr': 0.001}, simple_model.parameters())
    loss = create_loss_from_config({'name': 'cross_entropy', 'vocab_size': 100})
    
    trainer = Trainer(
        model=simple_model,
        optimizer=opt,
        loss_fn=loss,
        config=config,
        train_dataloader=dummy_dataloader
    )
    
    trainer.train()
    assert trainer.state.global_step == 10


def test_trainer_mixed_precision(simple_model, dummy_dataloader, temp_output_dir):
    """Test mixed precision training."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.losses import create_loss_from_config
    
    config = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=5,
        batch_size=4,
        mixed_precision=True,
        device='cuda'
    )
    
    opt = create_optimizer_from_config({'name': 'adamw', 'lr': 0.001}, simple_model.parameters())
    loss = create_loss_from_config({'name': 'cross_entropy', 'vocab_size': 100})
    
    trainer = Trainer(
        model=simple_model,
        optimizer=opt,
        loss_fn=loss,
        config=config,
        train_dataloader=dummy_dataloader
    )
    
    assert trainer.scaler is not None


def test_trainer_save_checkpoint(simple_model, dummy_dataloader, training_config,
                                 simple_optimizer, simple_loss):
    """Test saving checkpoint."""
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    trainer.train()
    
    checkpoint_path = f"{training_config.output_dir}/test_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)
    
    assert Path(checkpoint_path).exists()


def test_trainer_load_checkpoint(simple_model, dummy_dataloader, training_config,
                                 simple_optimizer, simple_loss):
    """Test loading checkpoint."""
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.losses import create_loss_from_config
    
    trainer1 = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    trainer1.train()
    checkpoint_path = f"{training_config.output_dir}/test_checkpoint.pt"
    trainer1.save_checkpoint(checkpoint_path)
    
    # New trainer
    model2 = type(simple_model)()
    opt2 = create_optimizer_from_config({'name': 'adamw', 'lr': 0.001}, model2.parameters())
    loss2 = create_loss_from_config({'name': 'cross_entropy', 'vocab_size': 100})
    
    trainer2 = Trainer(
        model=model2,
        optimizer=opt2,
        loss_fn=loss2,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    trainer2.load_checkpoint(checkpoint_path)
    assert trainer2.state.global_step == trainer1.state.global_step


def test_trainer_resume_training(simple_model, dummy_dataloader, temp_output_dir):
    """Test resuming training from checkpoint."""
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.losses import create_loss_from_config
    
    config1 = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=5,
        batch_size=4,
        save_every=5,
        mixed_precision=False
    )
    
    opt1 = create_optimizer_from_config({'name': 'adamw', 'lr': 0.001}, simple_model.parameters())
    loss1 = create_loss_from_config({'name': 'cross_entropy', 'vocab_size': 100})
    
    trainer1 = Trainer(
        model=simple_model,
        optimizer=opt1,
        loss_fn=loss1,
        config=config1,
        train_dataloader=dummy_dataloader
    )
    
    trainer1.train()
    checkpoint_path = f"{temp_output_dir}/checkpoint.pt"
    trainer1.save_checkpoint(checkpoint_path)
    
    # Resume
    config2 = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=10,
        batch_size=4,
        resume_from=checkpoint_path,
        mixed_precision=False
    )
    
    model2 = type(simple_model)()
    opt2 = create_optimizer_from_config({'name': 'adamw', 'lr': 0.001}, model2.parameters())
    loss2 = create_loss_from_config({'name': 'cross_entropy', 'vocab_size': 100})
    
    trainer2 = Trainer(
        model=model2,
        optimizer=opt2,
        loss_fn=loss2,
        config=config2,
        train_dataloader=dummy_dataloader
    )
    
    assert trainer2.state.global_step == 5


def test_trainer_validate(simple_model, dummy_dataloader, training_config,
                         simple_optimizer, simple_loss):
    """Test validation."""
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader
    )
    
    metrics = trainer.validate()
    assert 'val_loss' in metrics
    assert 'val_perplexity' in metrics


def test_trainer_validate_with_eval_steps(simple_model, dummy_dataloader, temp_output_dir):
    """Test validation with limited steps."""
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.losses import create_loss_from_config
    
    config = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=10,
        eval_steps=2,
        mixed_precision=False
    )
    
    opt = create_optimizer_from_config({'name': 'adamw', 'lr': 0.001}, simple_model.parameters())
    loss = create_loss_from_config({'name': 'cross_entropy', 'vocab_size': 100})
    
    trainer = Trainer(
        model=simple_model,
        optimizer=opt,
        loss_fn=loss,
        config=config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader
    )
    
    metrics = trainer.validate()
    assert 'val_loss' in metrics


# ============================================================================
# CALLBACK TESTS
# ============================================================================

def test_callback_interface():
    """Test callback interface."""
    from ramanujan.training.trainer.callbacks import Callback
    
    callback = Callback()
    assert hasattr(callback, 'on_train_begin')
    assert hasattr(callback, 'on_step_end')


def test_progress_callback(simple_model, dummy_dataloader, training_config,
                          simple_optimizer, simple_loss, capsys):
    """Test progress callback."""
    callback = ProgressCallback()
    
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader,
        callbacks=[callback]
    )
    
    trainer.train()
    captured = capsys.readouterr()
    assert "Training Started" in captured.out or "Training" in captured.out


def test_checkpoint_callback(simple_model, dummy_dataloader, training_config,
                            simple_optimizer, simple_loss):
    """Test checkpoint callback."""
    callback = CheckpointCallback(save_every=5, keep_last_n=2)
    
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader,
        callbacks=[callback]
    )
    
    trainer.train()
    
    checkpoint_dir = Path(training_config.output_dir) / 'checkpoints'
    assert checkpoint_dir.exists()


def test_early_stopping_callback(simple_model, dummy_dataloader, temp_output_dir):
    """Test early stopping callback."""
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.losses import create_loss_from_config
    
    config = TrainingConfig(
        output_dir=temp_output_dir,
        max_steps=100,
        eval_every=2,
        mixed_precision=False
    )
    
    callback = EarlyStoppingCallback(
        monitor='val_loss',
        patience=2,
        mode='min',
        min_delta=0.01
    )
    
    opt = create_optimizer_from_config({'name': 'adamw', 'lr': 0.001}, simple_model.parameters())
    loss = create_loss_from_config({'name': 'cross_entropy', 'vocab_size': 100})
    
    trainer = Trainer(
        model=simple_model,
        optimizer=opt,
        loss_fn=loss,
        config=config,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader,
        callbacks=[callback]
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass
    
    assert trainer.state.global_step < 100


def test_trainer_metrics_tracking(simple_model, dummy_dataloader, training_config,
                                  simple_optimizer, simple_loss):
    """Test metrics are tracked."""
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader
    )
    
    trainer.train()
    assert len(trainer.state.metrics_history) > 0


def test_trainer_empty_validation(simple_model, dummy_dataloader, training_config,
                                  simple_optimizer, simple_loss):
    """Test trainer without validation dataloader."""
    trainer = Trainer(
        model=simple_model,
        optimizer=simple_optimizer,
        loss_fn=simple_loss,
        config=training_config,
        train_dataloader=dummy_dataloader,
        val_dataloader=None
    )
    
    metrics = trainer.validate()
    assert metrics == {}


def test_trainer_with_invalid_config():
    """Test trainer with invalid config."""
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(max_steps=-1)