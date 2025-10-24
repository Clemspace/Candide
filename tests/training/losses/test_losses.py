"""
Tests for Loss System

Run with: pytest tests/training/test_losses.py -v
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

# Import the loss system
from ramanujan.training.losses import (
    LossComponent,
    CrossEntropyLoss,
    
    SemanticEntropyProbe,
    KLDivergenceLoss,
    CompositeLoss,
    LossSpec,
    create_loss,
    create_loss_from_config
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_length():
    return 10

@pytest.fixture
def vocab_size():
    return 100

@pytest.fixture
def sample_logits(batch_size, seq_length, vocab_size):
    """Generate random logits."""
    return torch.randn(batch_size, seq_length, vocab_size, requires_grad=True)

@pytest.fixture
def sample_targets(batch_size, seq_length, vocab_size):
    """Generate random target token IDs."""
    return torch.randint(0, vocab_size, (batch_size, seq_length))


# ============================================================================
# Test Protocol Compliance
# ============================================================================

def test_cross_entropy_is_loss_component():
    """Verify CrossEntropyLoss implements LossComponent protocol."""
    loss_fn = CrossEntropyLoss()
    
    # Check required attributes
    assert hasattr(loss_fn, 'loss_name')
    assert hasattr(loss_fn, 'component_type')
    assert hasattr(loss_fn, 'component_name')
    assert hasattr(loss_fn, 'compute')
    
    # Check values
    assert loss_fn.component_type == 'loss'
    assert loss_fn.component_name == 'cross_entropy'
    assert loss_fn.loss_name == 'cross_entropy'


def test_semantic_entropy_is_loss_component():
    """Verify SemanticEntropyProbe implements LossComponent protocol."""
    loss_fn = SemanticEntropyProbe()
    
    assert loss_fn.component_type == 'loss'
    assert loss_fn.component_name == 'semantic_entropy'
    assert hasattr(loss_fn, 'compute')


# ============================================================================
# Test CrossEntropyLoss
# ============================================================================

def test_cross_entropy_basic(sample_logits, sample_targets):
    """Test basic cross-entropy computation."""
    loss_fn = CrossEntropyLoss()
    
    loss, metrics = loss_fn.compute(sample_logits, sample_targets)
    
    # Check loss is scalar
    assert loss.dim() == 0
    assert loss.requires_grad
    
    # Check metrics
    assert 'accuracy' in metrics
    assert 'perplexity' in metrics
    assert 'ce_loss' in metrics
    
    # Check metric ranges
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert metrics['perplexity'] > 0


def test_cross_entropy_with_label_smoothing(sample_logits, sample_targets):
    """Test cross-entropy with label smoothing."""
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    
    loss_smooth, metrics_smooth = loss_fn.compute(sample_logits, sample_targets)
    
    # Compare with no smoothing
    loss_fn_no_smooth = CrossEntropyLoss(label_smoothing=0.0)
    loss_no_smooth, metrics_no_smooth = loss_fn_no_smooth.compute(sample_logits, sample_targets)
    
    # Losses should be different
    assert not torch.isclose(loss_smooth, loss_no_smooth)


def test_cross_entropy_with_ignore_index(batch_size, seq_length, vocab_size):
    """Test cross-entropy with padding tokens."""
    logits = torch.randn(batch_size, seq_length, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Set some tokens to ignore index
    ignore_idx = -100
    targets[:, -2:] = ignore_idx  # Last 2 tokens are padding
    
    loss_fn = CrossEntropyLoss(ignore_index=ignore_idx)
    loss, metrics = loss_fn.compute(logits, targets)
    
    # Loss should still be valid
    assert loss.requires_grad
    assert 0.0 <= metrics['accuracy'] <= 1.0


def test_cross_entropy_backward(sample_logits, sample_targets):
    """Test that cross-entropy loss supports backpropagation."""
    sample_logits.requires_grad = True
    loss_fn = CrossEntropyLoss()
    
    loss, _ = loss_fn.compute(sample_logits, sample_targets)
    loss.backward()
    
    # Check gradients exist
    assert sample_logits.grad is not None
    assert sample_logits.grad.shape == sample_logits.shape


# ============================================================================
# Test SemanticEntropyLoss
# ============================================================================

def test_semantic_entropy_basic(sample_logits, sample_targets):
    """Test basic semantic entropy computation."""
    loss_fn = SemanticEntropyProbe(num_samples=5, temperature=1.0)
    
    loss, metrics = loss_fn.compute(sample_logits, sample_targets)
    
    # Check loss
    assert loss.dim() == 0
    assert loss.requires_grad
    
    # Check metrics
    assert 'semantic_entropy' in metrics
    assert 'ce_loss' in metrics
    assert 'total_loss' in metrics
    
    # All should be positive
    assert metrics['semantic_entropy'] >= 0
    assert metrics['ce_loss'] >= 0


def test_semantic_entropy_temperature_effect(sample_logits, sample_targets):
    """Test that temperature affects semantic entropy."""
    loss_fn_low_temp = SemanticEntropyProbe(temperature=0.5)
    loss_fn_high_temp = SemanticEntropyProbe(temperature=2.0)
    
    loss_low, metrics_low = loss_fn_low_temp.compute(sample_logits, sample_targets)
    loss_high, metrics_high = loss_fn_high_temp.compute(sample_logits, sample_targets)
    
    # Different temperatures should give different results
    assert not torch.isclose(loss_low, loss_high)


# ============================================================================
# Test KLDivergenceLoss
# ============================================================================

def test_kl_divergence_basic(sample_logits, vocab_size):
    """Test basic KL divergence computation."""
    loss_fn = KLDivergenceLoss(temperature=1.0)
    
    # Create student and teacher logits
    student_logits = sample_logits
    teacher_logits = torch.randn_like(sample_logits)
    
    dummy_targets = torch.randint(0, vocab_size, (sample_logits.size(0), sample_logits.size(1)))
    loss, metrics = loss_fn.compute(student_logits, dummy_targets, reference_logits=teacher_logits)
    
    # Check loss
    assert loss.dim() == 0
    assert loss.requires_grad
    
    # Check metrics
    assert 'kl_loss' in metrics
    assert metrics['kl_loss'] >= 0  # KL divergence is non-negative


def test_kl_divergence_same_distributions(sample_logits):
    """Test KL divergence is zero for identical distributions."""
    loss_fn = KLDivergenceLoss(temperature=1.0)
    
    # Same logits for student and teacher
    vocab_size = sample_logits.size(-1)
    dummy_targets = torch.randint(0, vocab_size, (sample_logits.size(0), sample_logits.size(1)))
    loss, metrics = loss_fn.compute(sample_logits, dummy_targets, reference_logits=sample_logits)
    
    # Should be very close to zero
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


# ============================================================================
# Test CompositeLoss
# ============================================================================

def test_composite_loss_basic(sample_logits, sample_targets):
    """Test basic composite loss with multiple components."""
    composite = CompositeLoss([
        LossSpec('cross_entropy', weight=1.0, config={'label_smoothing': 0.0}),
        LossSpec('semantic_entropy', weight=0.1, config={'num_samples': 3})
    ])
    
    loss, metrics = composite.compute(sample_logits, sample_targets)
    
    # Check loss
    assert loss.dim() == 0
    assert loss.requires_grad
    
    # Check metrics from both losses are present
    assert 'cross_entropy/accuracy' in metrics
    assert 'cross_entropy/ce_loss' in metrics
    assert 'semantic_entropy/semantic_entropy' in metrics
    assert 'total_loss' in metrics


def test_composite_loss_weights(sample_logits, sample_targets):
    """Test that loss weights affect the final loss value."""
    # Composite with high weight on CE
    composite_high_ce = CompositeLoss([
        LossSpec('cross_entropy', weight=1.0, config={}),
        LossSpec('semantic_entropy', weight=0.01, config={})
    ])
    
    # Composite with high weight on semantic entropy
    composite_high_se = CompositeLoss([
        LossSpec('cross_entropy', weight=0.01, config={}),
        LossSpec('semantic_entropy', weight=1.0, config={})
    ])
    
    loss_high_ce, _ = composite_high_ce.compute(sample_logits, sample_targets)
    loss_high_se, _ = composite_high_se.compute(sample_logits, sample_targets)
    
    # Losses should be different
    assert not torch.isclose(loss_high_ce, loss_high_se)


def test_composite_loss_from_config(sample_logits, sample_targets):
    """Test creating composite loss from config dict."""
    config = {
        'losses': [
            {'name': 'cross_entropy', 'weight': 1.0, 'config': {'label_smoothing': 0.1}},
            {'name': 'kl_divergence', 'weight': 0.5, 'config': {'temperature': 2.0}}
        ]
    }
    
    composite = CompositeLoss.from_config(config)
    
    # For composite with KL divergence, we need to provide both:
    # - targets: class indices for cross-entropy
    # - reference_logits: teacher logits for KL divergence
    
    # Create teacher logits (slightly different from student)
    teacher_logits = sample_logits.detach() + torch.randn_like(sample_logits) * 0.1
    
    # Pass both as kwargs
    loss, metrics = composite.compute(
        sample_logits, 
        sample_targets,
        reference_logits=teacher_logits
    )
    
    # Should have metrics from both losses
    assert 'cross_entropy/ce_loss' in metrics
    assert 'kl_divergence/kl_loss' in metrics
    assert 'total_loss' in metrics
    
    # Total loss should be weighted sum
    expected_total = metrics['cross_entropy/ce_loss'] * 1.0 + metrics['kl_divergence/kl_loss'] * 0.5
    assert torch.isclose(torch.tensor(metrics['total_loss']), torch.tensor(expected_total), rtol=1e-4)


# ============================================================================
# Test Factory Functions
# ============================================================================

def test_create_loss_basic():
    """Test creating loss from name and config."""
    loss_fn = create_loss('cross_entropy', {'label_smoothing': 0.1})
    
    assert isinstance(loss_fn, CrossEntropyLoss)
    assert loss_fn.label_smoothing == 0.1


def test_create_loss_default_config():
    """Test creating loss with default config."""
    loss_fn = create_loss('semantic_entropy')
    
    assert isinstance(loss_fn, SemanticEntropyProbe)
    assert loss_fn.num_samples == 5  # Default value


def test_create_loss_from_config_simple(sample_logits, sample_targets):
    """Test create_loss_from_config for simple loss."""
    config = {
        'type': 'cross_entropy',
        'config': {'label_smoothing': 0.1, 'ignore_index': -100}
    }
    
    loss_fn = create_loss_from_config(config)
    loss, metrics = loss_fn.compute(sample_logits, sample_targets)
    
    assert loss.requires_grad
    assert 'accuracy' in metrics


def test_create_loss_from_config_composite(sample_logits, sample_targets):
    """Test create_loss_from_config for composite loss."""
    config = {
        'type': 'composite',
        'losses': [
            {'name': 'cross_entropy', 'weight': 1.0, 'config': {}},
            {'name': 'semantic_entropy', 'weight': 0.1, 'config': {'num_samples': 3}}
        ]
    }
    
    loss_fn = create_loss_from_config(config)
    loss, metrics = loss_fn.compute(sample_logits, sample_targets)
    
    assert isinstance(loss_fn, CompositeLoss)
    assert 'total_loss' in metrics


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_empty_batch():
    """Test losses handle empty batches gracefully."""
    empty_logits = torch.randn(0, 10, 100)
    empty_targets = torch.randint(0, 100, (0, 10))
    
    loss_fn = CrossEntropyLoss()
    
    # Should not crash
    try:
        loss, metrics = loss_fn.compute(empty_logits, empty_targets)
    except Exception as e:
        pytest.fail(f"Loss should handle empty batch: {e}")


def test_single_example():
    """Test losses work with single example."""
    logits = torch.randn(1, 5, 50, requires_grad=True)
    targets = torch.randint(0, 50, (1, 5))
    
    loss_fn = CrossEntropyLoss()
    loss, metrics = loss_fn.compute(logits, targets)
    
    assert loss.requires_grad
    assert 0.0 <= metrics['accuracy'] <= 1.0


def test_numerical_stability():
    """Test losses are numerically stable with extreme values."""
    # Very large logits (could cause overflow in softmax)
    large_logits = torch.randn(2, 5, 20) * 1000
    targets = torch.randint(0, 20, (2, 5))
    
    loss_fn = CrossEntropyLoss()
    
    try:
        loss, metrics = loss_fn.compute(large_logits, targets)
        assert torch.isfinite(loss)
    except Exception as e:
        pytest.fail(f"Loss should be numerically stable: {e}")


# ============================================================================
# Integration Tests
# ============================================================================

def test_loss_in_training_loop(sample_logits, sample_targets):
    """Test loss works in a realistic training loop."""
    # Simple model
    model = nn.Linear(sample_logits.size(-1), sample_logits.size(-1))
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = CrossEntropyLoss()
    
    # Forward pass
    sample_logits.requires_grad = True
    loss, metrics = loss_fn.compute(sample_logits, sample_targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Should complete without error
    assert loss.item() > 0


def test_composite_loss_gradient_flow(sample_logits, sample_targets):
    """Test gradients flow through composite loss."""
    sample_logits.requires_grad = True
    
    composite = CompositeLoss([
        LossSpec('cross_entropy', weight=1.0, config={}),
        LossSpec('semantic_entropy', weight=0.1, config={})
    ])
    
    loss, _ = composite.compute(sample_logits, sample_targets)
    loss.backward()
    
    # Gradients should exist
    assert sample_logits.grad is not None
    assert torch.isfinite(sample_logits.grad).all()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])