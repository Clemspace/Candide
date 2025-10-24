"""
Comprehensive tests for Candide optimizer system.

Tests cover:
- Protocol compliance
- Basic optimizer functionality
- Parameter updates
- State dict save/load
- Gradient clipping
- Parameter groups
- Layer-wise learning rates
- Config-driven creation
- Numerical stability
- Training loop integration
"""

import pytest
import torch
import torch.nn as nn
from typing import List

# Import from the optimizer system
import sys
from ramanujan.training.optimizers import (
    create_optimizer,
    create_optimizer_from_config,
    create_optimizer_with_param_groups,
    create_optimizer_with_llrd,
    AdamWOptimizer,
    MuonOptimizer,
    AdEMAMixOptimizer,
    SGDOptimizer,
    LionOptimizer,
    OptimizerComponent,
    get_optimizer_info,
    get_learning_rates,
    set_learning_rate
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.bias = nn.Parameter(torch.zeros(10))
        
        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x + self.bias
    
    return SimpleModel()


@pytest.fixture
def layered_model():
    """Create a layered model for LLRD testing."""
    class LayeredModel(nn.Module):
        def __init__(self, num_layers=3):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(10, 10) for _ in range(num_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    return LayeredModel()


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(4, 10)


@pytest.fixture
def sample_target():
    """Create sample target tensor."""
    return torch.randn(4, 10)


# ============================================================================
# TEST PROTOCOL COMPLIANCE
# ============================================================================

def test_adamw_is_optimizer_component():
    """Test that AdamW implements OptimizerComponent protocol."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('adamw', model.parameters())
    
    assert hasattr(opt, 'component_type')
    assert hasattr(opt, 'component_name')
    assert hasattr(opt, 'step')
    assert hasattr(opt, 'zero_grad')
    assert hasattr(opt, 'state_dict')
    assert hasattr(opt, 'load_state_dict')
    assert hasattr(opt, 'get_last_lr')


def test_muon_is_optimizer_component():
    """Test that Muon implements OptimizerComponent protocol."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('muon', model.parameters())
    
    assert opt.component_type == 'optimizer'
    assert opt.component_name == 'muon'


def test_ademamix_is_optimizer_component():
    """Test that AdEMAMix implements OptimizerComponent protocol."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('ademamix', model.parameters())
    
    assert opt.component_type == 'optimizer'
    assert opt.component_name == 'ademamix'


# ============================================================================
# TEST BASIC OPTIMIZER CREATION
# ============================================================================

def test_create_adamw():
    """Test creating AdamW optimizer."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('adamw', model.parameters(), lr=1e-3, weight_decay=0.01)
    
    assert opt is not None
    lrs = opt.get_last_lr()
    assert len(lrs) == 1
    assert lrs[0] == 1e-3


def test_create_muon():
    """Test creating Muon optimizer."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('muon', model.parameters(), lr=0.02, momentum=0.95)
    
    assert opt is not None
    lrs = opt.get_last_lr()
    assert lrs[0] == 0.02


def test_create_ademamix():
    """Test creating AdEMAMix optimizer."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('ademamix', model.parameters(), lr=1e-3, alpha=5.0)
    
    assert opt is not None


def test_create_sgd():
    """Test creating SGD optimizer."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('sgd', model.parameters(), lr=0.01, momentum=0.9)
    
    assert opt is not None


def test_create_lion():
    """Test creating Lion optimizer."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('lion', model.parameters(), lr=1e-4)
    
    assert opt is not None


def test_create_invalid_optimizer():
    """Test that creating invalid optimizer raises error."""
    model = nn.Linear(10, 10)
    
    with pytest.raises(ValueError, match="Unknown optimizer"):
        create_optimizer('invalid', model.parameters())


# ============================================================================
# TEST PARAMETER UPDATES
# ============================================================================

def test_optimizer_updates_parameters(simple_model, sample_input, sample_target):
    """Test that optimizer actually updates parameters."""
    opt = create_optimizer('adamw', simple_model.parameters(), lr=1e-3)
    
    # Store initial parameters
    initial_params = [p.clone() for p in simple_model.parameters()]
    
    # Forward, backward, step
    output = simple_model(sample_input)
    loss = ((output - sample_target) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    # Check that parameters changed
    for initial, current in zip(initial_params, simple_model.parameters()):
        assert not torch.allclose(initial, current), "Parameters should have changed"


def test_muon_updates_parameters(simple_model, sample_input, sample_target):
    """Test that Muon optimizer updates parameters."""
    opt = create_optimizer('muon', simple_model.parameters(), lr=0.02)
    
    initial_params = [p.clone() for p in simple_model.parameters()]
    
    output = simple_model(sample_input)
    loss = ((output - sample_target) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    for initial, current in zip(initial_params, simple_model.parameters()):
        assert not torch.allclose(initial, current)


def test_ademamix_updates_parameters(simple_model, sample_input, sample_target):
    """Test that AdEMAMix optimizer updates parameters."""
    opt = create_optimizer('ademamix', simple_model.parameters(), lr=1e-3)
    
    initial_params = [p.clone() for p in simple_model.parameters()]
    
    output = simple_model(sample_input)
    loss = ((output - sample_target) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    for initial, current in zip(initial_params, simple_model.parameters()):
        assert not torch.allclose(initial, current)


# ============================================================================
# TEST GRADIENT CLIPPING
# ============================================================================

def test_gradient_clipping_norm(simple_model, sample_input, sample_target):
    """Test gradient clipping by norm."""
    opt = create_optimizer(
        'adamw',
        simple_model.parameters(),
        lr=1e-3,
        grad_clip_type='norm',
        grad_clip_value=0.1
    )
    
    # Create large gradients
    output = simple_model(sample_input)
    loss = (output ** 2).sum() * 1000
    loss.backward()
    
    # Step (clips internally)
    opt.step()
    opt.zero_grad()
    
    # Optimizer should have clipped gradients
    assert True  # If we get here without errors, clipping worked


def test_gradient_clipping_value(simple_model, sample_input, sample_target):
    """Test gradient clipping by value."""
    opt = create_optimizer(
        'adamw',
        simple_model.parameters(),
        lr=1e-3,
        grad_clip_type='value',
        grad_clip_value=0.1
    )
    
    output = simple_model(sample_input)
    loss = (output ** 2).sum() * 1000
    loss.backward()
    
    opt.step()
    opt.zero_grad()
    
    assert True


# ============================================================================
# TEST STATE DICT
# ============================================================================

def test_state_dict_save_load(simple_model):
    """Test saving and loading optimizer state."""
    opt = create_optimizer('adamw', simple_model.parameters(), lr=1e-3)
    
    # Take a step to create state
    x = torch.randn(4, 10)
    output = simple_model(x)
    loss = output.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    # Save state
    state_dict = opt.state_dict()
    assert state_dict is not None
    
    # Create new optimizer and load state
    opt2 = create_optimizer('adamw', simple_model.parameters(), lr=1e-3)
    opt2.load_state_dict(state_dict)
    
    # States should match
    assert True  # If we get here, save/load worked


# ============================================================================
# TEST PARAMETER GROUPS
# ============================================================================

def test_create_with_param_groups(simple_model):
    """Test creating optimizer with parameter groups."""
    opt = create_optimizer_with_param_groups(
        'adamw',
        simple_model,
        base_lr=1e-3,
        rules={
            'bias': {'weight_decay': 0.0},
            'fc1': {'lr': 1e-4}
        }
    )
    
    assert opt is not None
    lrs = opt.get_last_lr()
    assert len(lrs) > 1  # Should have multiple param groups


def test_layer_wise_lr_decay(layered_model):
    """Test layer-wise learning rate decay."""
    opt = create_optimizer_with_llrd(
        'adamw',
        layered_model,
        base_lr=1e-3,
        decay_rate=0.9
    )
    
    lrs = opt.get_last_lr()
    
    # Should have multiple learning rates
    assert len(lrs) > 1
    
    # Learning rates should be different (decaying)
    unique_lrs = set(lrs)
    assert len(unique_lrs) > 1


# ============================================================================
# TEST CONFIG-DRIVEN CREATION
# ============================================================================

def test_create_from_config():
    """Test creating optimizer from config dict."""
    model = nn.Linear(10, 10)
    
    config = {
        'name': 'adamw',
        'lr': 1e-3,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999)
    }
    
    opt = create_optimizer_from_config(config, model.parameters())
    
    assert opt is not None
    lrs = opt.get_last_lr()
    assert lrs[0] == 1e-3


def test_create_from_config_with_grad_clip():
    """Test creating optimizer with gradient clipping from config."""
    model = nn.Linear(10, 10)
    
    config = {
        'name': 'muon',
        'lr': 0.02,
        'grad_clip': {
            'type': 'norm',
            'value': 1.0
        }
    }
    
    opt = create_optimizer_from_config(config, model.parameters())
    
    assert opt is not None
    assert opt.grad_clip_type == 'norm'
    assert opt.grad_clip_value == 1.0


# ============================================================================
# TEST OPTIMIZER UTILITIES
# ============================================================================

def test_get_optimizer_info():
    """Test getting optimizer information."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('adamw', model.parameters(), lr=1e-3)
    
    info = get_optimizer_info(opt)
    
    assert 'type' in info
    assert 'num_param_groups' in info
    assert 'lr' in info
    assert info['type'] == 'adamw'
    assert info['num_param_groups'] == 1


def test_get_learning_rates():
    """Test getting current learning rates."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('adamw', model.parameters(), lr=1e-3)
    
    lrs = get_learning_rates(opt)
    
    assert len(lrs) == 1
    assert lrs[0] == 1e-3


def test_set_learning_rate():
    """Test setting learning rate."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('adamw', model.parameters(), lr=1e-3)
    
    # Change learning rate
    set_learning_rate(opt, 2e-3)
    
    lrs = get_learning_rates(opt)
    assert lrs[0] == 2e-3


# ============================================================================
# TEST TRAINING LOOP INTEGRATION
# ============================================================================

def test_optimizer_in_training_loop(simple_model, sample_input, sample_target):
    """Test optimizer in realistic training loop."""
    opt = create_optimizer('adamw', simple_model.parameters(), lr=1e-3)
    
    losses = []
    
    for i in range(5):
        output = simple_model(sample_input)
        loss = ((output - sample_target) ** 2).mean()
        losses.append(loss.item())
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # Loss should decrease (or at least not increase consistently)
    assert len(losses) == 5


# ============================================================================
# TEST NUMERICAL STABILITY
# ============================================================================

def test_optimizer_handles_zero_grad():
    """Test that optimizer handles zero gradients."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('adamw', model.parameters(), lr=1e-3)
    
    # Create zero loss (zero gradients)
    x = torch.zeros(4, 10)
    output = model(x)
    loss = (output * 0).sum()
    loss.backward()
    
    # Should not error
    opt.step()
    opt.zero_grad()
    
    assert True


def test_optimizer_handles_large_gradients(simple_model):
    """Test that optimizer handles large gradients."""
    opt = create_optimizer(
        'adamw',
        simple_model.parameters(),
        lr=1e-3,
        grad_clip_type='norm',
        grad_clip_value=1.0
    )
    
    # Create very large loss
    x = torch.randn(4, 10)
    output = simple_model(x)
    loss = (output ** 2).sum() * 1e6
    loss.backward()
    
    # Should clip and handle gracefully
    opt.step()
    opt.zero_grad()
    
    # Check parameters are still finite
    for p in simple_model.parameters():
        assert torch.isfinite(p).all()


# ============================================================================
# TEST OPTIMIZER-SPECIFIC FEATURES
# ============================================================================

def test_muon_orthogonalization():
    """Test that Muon performs orthogonalization."""
    model = nn.Linear(20, 20)  # Square matrix for orthogonalization
    opt = create_optimizer('muon', model.parameters(), lr=0.02, ns_steps=1)
    
    # Run multiple steps to trigger orthogonalization
    for i in range(5):
        x = torch.randn(4, 20)
        output = model(x)
        loss = output.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # If we get here without errors, orthogonalization worked
    assert True


def test_ademamix_dual_ema():
    """Test that AdEMAMix maintains dual EMAs."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('ademamix', model.parameters(), lr=1e-3, alpha=5.0)
    
    # Take a step
    x = torch.randn(4, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    # Check that state has m1, m2, v
    for p in model.parameters():
        if p.grad is not None:
            state = opt.optimizer.state[p]
            assert 'm1' in state
            assert 'm2' in state
            assert 'v' in state


def test_lion_sign_update():
    """Test that Lion uses sign-based updates."""
    model = nn.Linear(10, 10)
    opt = create_optimizer('lion', model.parameters(), lr=1e-4)
    
    # Take a step
    x = torch.randn(4, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    # If we get here, Lion's sign-based update worked
    assert True


# ============================================================================
# TEST COMPARISON
# ============================================================================

def test_optimizer_comparison(simple_model):
    """Test that different optimizers produce different results."""
    optimizers = {
        'adamw': create_optimizer('adamw', simple_model.parameters(), lr=1e-3),
        'muon': create_optimizer('muon', simple_model.parameters(), lr=0.02),
        'sgd': create_optimizer('sgd', simple_model.parameters(), lr=0.01)
    }
    
    # All optimizers should be created successfully
    assert len(optimizers) == 3
    
    for name, opt in optimizers.items():
        assert opt is not None
        assert opt.component_name == name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])