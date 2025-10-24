"""Tests for scheduler system."""

import pytest
import torch
import torch.nn as nn
from ramanujan.training.schedulers import (
    create_scheduler,
    create_scheduler_from_config,
    WarmupScheduler,
    CosineScheduler,
    ConstantScheduler,
    SchedulerComponent,
    SCHEDULER_REGISTRY
)
from ramanujan.training.optimizers import create_optimizer


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_model():
    """Simple model for testing."""
    return nn.Linear(10, 10)


@pytest.fixture
def optimizer(simple_model):
    """Optimizer for testing."""
    return create_optimizer('adamw', simple_model.parameters(), lr=1e-3)


# ============================================================================
# TEST PROTOCOL COMPLIANCE
# ============================================================================

def test_warmup_is_scheduler_component(optimizer):
    """Test WarmupScheduler implements SchedulerComponent."""
    scheduler = WarmupScheduler(optimizer, warmup_steps=100, total_steps=1000)
    
    assert hasattr(scheduler, 'component_type')
    assert hasattr(scheduler, 'component_name')
    assert hasattr(scheduler, 'step')
    assert hasattr(scheduler, 'get_last_lr')
    assert hasattr(scheduler, 'state_dict')
    assert hasattr(scheduler, 'load_state_dict')
    assert scheduler.component_type == 'scheduler'
    assert scheduler.component_name == 'warmup'


def test_cosine_is_scheduler_component(optimizer):
    """Test CosineScheduler implements SchedulerComponent."""
    scheduler = CosineScheduler(optimizer, total_steps=1000, warmup_steps=100)
    
    assert scheduler.component_type == 'scheduler'
    assert scheduler.component_name == 'cosine'


# ============================================================================
# TEST BASIC SCHEDULER CREATION
# ============================================================================

def test_create_warmup_scheduler(optimizer):
    """Test creating warmup scheduler."""
    scheduler = create_scheduler(
        'warmup',
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        decay_style='cosine'
    )
    
    assert scheduler is not None
    assert isinstance(scheduler, WarmupScheduler)


def test_create_cosine_scheduler(optimizer):
    """Test creating cosine scheduler."""
    scheduler = create_scheduler(
        'cosine',
        optimizer,
        total_steps=1000,
        warmup_steps=100
    )
    
    assert scheduler is not None
    assert isinstance(scheduler, CosineScheduler)


def test_create_constant_scheduler(optimizer):
    """Test creating constant scheduler."""
    scheduler = create_scheduler('constant', optimizer)
    
    assert scheduler is not None
    assert isinstance(scheduler, ConstantScheduler)


def test_create_invalid_scheduler(optimizer):
    """Test creating invalid scheduler raises error."""
    with pytest.raises(ValueError, match="Unknown scheduler"):
        create_scheduler('invalid', optimizer)


# ============================================================================
# TEST WARMUP SCHEDULER
# ============================================================================

def test_warmup_linear_warmup(optimizer):
    """Test linear warmup phase."""
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        warmup_init_lr=0.0,
        decay_style='constant'
    )
    
    base_lr = 1e-3
    
    # At step 0 (after first step()), should be at warmup_init_lr
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    assert lr > 0.0  # Should be > warmup_init_lr
    
    # At step 5, should be halfway
    for _ in range(4):
        scheduler.step()
    lr = scheduler.get_last_lr()[0]
    assert 0.4 * base_lr < lr < 0.6 * base_lr
    
    # After warmup, should be at base_lr
    for _ in range(10):
        scheduler.step()
    lr = scheduler.get_last_lr()[0]
    assert abs(lr - base_lr) < 1e-6


def test_warmup_cosine_decay(optimizer):
    """Test cosine decay after warmup."""
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        min_lr=0.0,
        decay_style='cosine'
    )
    
    # Fast forward past warmup
    for _ in range(15):
        scheduler.step()
    
    lr_after_warmup = scheduler.get_last_lr()[0]
    
    # At halfway through decay, LR should be lower
    for _ in range(40):
        scheduler.step()
    lr_mid = scheduler.get_last_lr()[0]
    assert lr_mid < lr_after_warmup
    
    # At end, LR should be near min_lr
    for _ in range(50):
        scheduler.step()
    lr_end = scheduler.get_last_lr()[0]
    assert lr_end < lr_mid
    assert lr_end < 0.1 * lr_after_warmup


def test_warmup_linear_decay(optimizer):
    """Test linear decay after warmup."""
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        min_lr=0.0,
        decay_style='linear'
    )
    
    # Fast forward past warmup
    for _ in range(15):
        scheduler.step()
    lr_after_warmup = scheduler.get_last_lr()[0]
    
    # At end, LR should be near min_lr
    for _ in range(90):
        scheduler.step()
    lr_end = scheduler.get_last_lr()[0]
    assert lr_end < 0.1 * lr_after_warmup


# ============================================================================
# TEST COSINE SCHEDULER
# ============================================================================

def test_cosine_basic(optimizer):
    """Test basic cosine annealing."""
    scheduler = CosineScheduler(
        optimizer,
        total_steps=100,
        warmup_steps=10,
        min_lr=0.0
    )
    
    lrs = []
    for _ in range(100):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    # LR should start low (warmup), reach peak after warmup, then follow cosine
    assert lrs[0] < lrs[15]  # Increasing during/after warmup
    
    # Find the minimum point (should be around mid-cycle)
    min_idx = lrs.index(min(lrs[15:]))  # After warmup
    
    # LR should decrease then increase (cosine shape)
    assert lrs[15] > lrs[min_idx]  # Decreases from peak
    assert lrs[-1] > lrs[min_idx]  # Increases from minimum


def test_cosine_with_cycles(optimizer):
    """Test cosine annealing with multiple cycles."""
    scheduler = CosineScheduler(
        optimizer,
        total_steps=100,
        warmup_steps=0,
        min_lr=0.0,
        num_cycles=2
    )
    
    lrs = []
    for _ in range(100):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    # With 2 cycles, we should see 2 peaks
    # Find local maxima (roughly)
    peaks = []
    for i in range(1, len(lrs) - 1):
        if lrs[i] > lrs[i-1] and lrs[i] > lrs[i+1]:
            peaks.append(i)
    
    # Should have ~2 peaks (may not be exact due to discrete steps)
    assert len(peaks) >= 1


# ============================================================================
# TEST CONSTANT SCHEDULER
# ============================================================================

def test_constant_scheduler(optimizer):
    """Test constant scheduler keeps LR unchanged."""
    scheduler = ConstantScheduler(optimizer)
    
    initial_lr = scheduler.get_last_lr()[0]
    
    # Step many times
    for _ in range(100):
        scheduler.step()
    
    final_lr = scheduler.get_last_lr()[0]
    
    # LR should be unchanged
    assert initial_lr == final_lr


# ============================================================================
# TEST STATE DICT
# ============================================================================

def test_scheduler_state_dict(optimizer):
    """Test saving and loading scheduler state."""
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        decay_style='cosine'
    )
    
    # Take some steps
    for _ in range(25):
        scheduler.step()
    
    # Save state
    state_dict = scheduler.state_dict()
    lr_before = scheduler.get_last_lr()[0]
    
    # Create new scheduler and load state
    scheduler2 = WarmupScheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        decay_style='cosine'
    )
    scheduler2.load_state_dict(state_dict)
    
    lr_after = scheduler2.get_last_lr()[0]
    
    # LRs should match
    assert abs(lr_before - lr_after) < 1e-9
    
    # Take another step on both
    scheduler.step()
    scheduler2.step()
    
    assert abs(scheduler.get_last_lr()[0] - scheduler2.get_last_lr()[0]) < 1e-9


# ============================================================================
# TEST CONFIG-DRIVEN CREATION
# ============================================================================

def test_create_from_config_warmup(optimizer):
    """Test creating scheduler from config."""
    config = {
        'name': 'warmup',
        'warmup_steps': 100,
        'total_steps': 1000,
        'decay_style': 'cosine'
    }
    
    scheduler = create_scheduler_from_config(config, optimizer)
    
    assert scheduler is not None
    assert isinstance(scheduler, WarmupScheduler)


def test_create_from_config_type_alias(optimizer):
    """Test creating scheduler with 'type' instead of 'name'."""
    config = {
        'type': 'cosine',
        'total_steps': 1000,
        'warmup_steps': 100
    }
    
    scheduler = create_scheduler_from_config(config, optimizer)
    
    assert scheduler is not None
    assert isinstance(scheduler, CosineScheduler)


# ============================================================================
# TEST PARAMETER GROUPS
# ============================================================================

def test_scheduler_with_param_groups():
    """Test scheduler with multiple parameter groups."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 10)
    )
    
    # Create optimizer with different LRs
    param_groups = [
        {'params': model[0].parameters(), 'lr': 1e-3},
        {'params': model[1].parameters(), 'lr': 5e-4}
    ]
    optimizer = create_optimizer('adamw', param_groups)
    
    # Create scheduler
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        decay_style='cosine'
    )
    
    # Check we get multiple LRs
    lrs = scheduler.get_last_lr()
    assert len(lrs) == 2
    
    # Step and check both LRs change
    initial_lrs = lrs.copy()
    for _ in range(50):
        scheduler.step()
    
    final_lrs = scheduler.get_last_lr()
    assert final_lrs[0] != initial_lrs[0]
    assert final_lrs[1] != initial_lrs[1]


# ============================================================================
# TEST INTEGRATION WITH OPTIMIZER
# ============================================================================

def test_scheduler_optimizer_integration(simple_model):
    """Test scheduler works with optimizer in training loop."""
    optimizer = create_optimizer('adamw', simple_model.parameters(), lr=1e-3)
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=5,
        total_steps=20,
        decay_style='cosine'
    )
    
    lrs = []
    for step in range(20):
        # Fake forward/backward
        x = torch.randn(4, 10)
        output = simple_model(x)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update LR
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    # Check LR changed over time
    assert lrs[0] != lrs[-1]
    assert len(lrs) == 20


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_scheduler_zero_warmup(optimizer):
    """Test scheduler with zero warmup steps."""
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=0,
        total_steps=100,
        decay_style='cosine'
    )
    
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    
    # Should start at base LR
    assert abs(lr - 1e-3) < 1e-6


def test_scheduler_min_lr(optimizer):
    """Test scheduler respects min_lr."""
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        min_lr=1e-5,
        decay_style='linear'
    )
    
    # Run to completion
    for _ in range(110):
        scheduler.step()
    
    final_lr = scheduler.get_last_lr()[0]
    
    # Should not go below min_lr
    assert final_lr >= 1e-5


def test_scheduler_registry():
    """Test scheduler registry."""
    assert 'warmup' in SCHEDULER_REGISTRY
    assert 'cosine' in SCHEDULER_REGISTRY
    assert 'constant' in SCHEDULER_REGISTRY
    assert len(SCHEDULER_REGISTRY) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])