"""Tests for feedforward networks."""

import pytest
import torch
from torch import nn

from ramanujan.models.components.feedforward import (
    SwiGLUFeedForward,
    GELUFeedForward,
    get_feedforward,
)


# ============================================================================
# TEST SWIGLU FEEDFORWARD
# ============================================================================

def test_swiglu_creation():
    """Test creating SwiGLU feedforward."""
    ffn = SwiGLUFeedForward(d_model=768)
    
    assert ffn.d_model == 768
    # SwiGLU uses ~8/3 * d_model rounded to 256
    assert ffn.d_ff > 0


def test_swiglu_custom_d_ff():
    """Test SwiGLU with custom d_ff."""
    ffn = SwiGLUFeedForward(d_model=768, d_ff=2048)
    
    assert ffn.d_ff == 2048


def test_swiglu_forward():
    """Test SwiGLU forward pass."""
    ffn = SwiGLUFeedForward(d_model=64, d_ff=256)
    x = torch.randn(2, 10, 64)
    
    out = ffn(x)
    
    assert out.shape == x.shape


def test_swiglu_no_bias():
    """Test SwiGLU without bias (LLaMA-style)."""
    ffn = SwiGLUFeedForward(d_model=64, bias=False)
    
    assert ffn.w1.bias is None
    assert ffn.w2.bias is None
    assert ffn.w3.bias is None


def test_swiglu_with_bias():
    """Test SwiGLU with bias."""
    ffn = SwiGLUFeedForward(d_model=64, bias=True)
    
    assert ffn.w1.bias is not None
    assert ffn.w2.bias is not None
    assert ffn.w3.bias is not None


def test_swiglu_dropout():
    """Test SwiGLU with dropout."""
    ffn = SwiGLUFeedForward(d_model=64, dropout=0.1)
    
    assert ffn.dropout_layer is not None
    
    # In eval mode, dropout should be inactive
    ffn.eval()
    x = torch.randn(2, 10, 64)
    out1 = ffn(x)
    out2 = ffn(x)
    assert torch.allclose(out1, out2)


def test_swiglu_gating_mechanism():
    """Test SwiGLU gating mechanism works."""
    ffn = SwiGLUFeedForward(d_model=64, d_ff=256)
    x = torch.randn(2, 10, 64)
    
    out = ffn(x)
    
    # Output should be different from input
    assert not torch.allclose(out, x, atol=1e-3)
    
    # Should not be all zeros
    assert not torch.all(out == 0)


def test_swiglu_parameter_count():
    """Test SwiGLU has correct number of parameters."""
    d_model = 768
    d_ff = 2048
    
    ffn = SwiGLUFeedForward(d_model=d_model, d_ff=d_ff, bias=False)
    
    # w1, w2, w3 without bias
    expected_params = (
        d_model * d_ff +  # w1
        d_model * d_ff +  # w2
        d_ff * d_model    # w3
    )
    
    actual_params = sum(p.numel() for p in ffn.parameters())
    assert actual_params == expected_params


# ============================================================================
# TEST GELU FEEDFORWARD
# ============================================================================

def test_gelu_creation():
    """Test creating GELU feedforward."""
    ffn = GELUFeedForward(d_model=768)
    
    assert ffn.d_model == 768
    assert ffn.d_ff == 768 * 4  # Default 4x


def test_gelu_custom_d_ff():
    """Test GELU with custom d_ff."""
    ffn = GELUFeedForward(d_model=768, d_ff=2048)
    
    assert ffn.d_ff == 2048


def test_gelu_forward():
    """Test GELU forward pass."""
    ffn = GELUFeedForward(d_model=64)
    x = torch.randn(2, 10, 64)
    
    out = ffn(x)
    
    assert out.shape == x.shape


def test_gelu_activation_types():
    """Test GELU with different activation functions."""
    # Standard GELU
    ffn1 = GELUFeedForward(d_model=64, activation='gelu')
    assert isinstance(ffn1.activation, nn.GELU)
    
    # GELU with tanh approximation
    ffn2 = GELUFeedForward(d_model=64, activation='gelu_new')
    assert isinstance(ffn2.activation, nn.GELU)
    
    # ReLU
    ffn3 = GELUFeedForward(d_model=64, activation='relu')
    assert isinstance(ffn3.activation, nn.ReLU)


def test_gelu_invalid_activation():
    """Test GELU raises error for invalid activation."""
    with pytest.raises(ValueError, match="Unknown activation"):
        GELUFeedForward(d_model=64, activation='invalid')


def test_gelu_with_bias():
    """Test GELU with bias."""
    ffn = GELUFeedForward(d_model=64, bias=True)
    
    assert ffn.fc1.bias is not None
    assert ffn.fc2.bias is not None


def test_gelu_no_bias():
    """Test GELU without bias."""
    ffn = GELUFeedForward(d_model=64, bias=False)
    
    assert ffn.fc1.bias is None
    assert ffn.fc2.bias is None


def test_gelu_dropout():
    """Test GELU with dropout."""
    ffn = GELUFeedForward(d_model=64, dropout=0.1)
    
    assert ffn.dropout_layer is not None


def test_gelu_parameter_count():
    """Test GELU has correct number of parameters."""
    d_model = 768
    d_ff = 3072
    
    ffn = GELUFeedForward(d_model=d_model, d_ff=d_ff, bias=False)
    
    # fc1, fc2 without bias
    expected_params = (
        d_model * d_ff +  # fc1
        d_ff * d_model    # fc2
    )
    
    actual_params = sum(p.numel() for p in ffn.parameters())
    assert actual_params == expected_params


# ============================================================================
# TEST COMPARISON: SWIGLU VS GELU
# ============================================================================

def test_swiglu_vs_gelu_param_count():
    """Test SwiGLU has more parameters than GELU (due to gating)."""
    d_model = 768
    d_ff = 2048
    
    swiglu = SwiGLUFeedForward(d_model=d_model, d_ff=d_ff, bias=False)
    gelu = GELUFeedForward(d_model=d_model, d_ff=d_ff, bias=False)
    
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    gelu_params = sum(p.numel() for p in gelu.parameters())
    
    # SwiGLU has 3 weight matrices, GELU has 2
    assert swiglu_params > gelu_params


def test_swiglu_vs_gelu_different_outputs():
    """Test SwiGLU and GELU produce different outputs."""
    x = torch.randn(2, 10, 64)
    
    swiglu = SwiGLUFeedForward(d_model=64, d_ff=256)
    gelu = GELUFeedForward(d_model=64, d_ff=256)
    
    out_swiglu = swiglu(x)
    out_gelu = gelu(x)
    
    # Should be different (different architectures)
    assert not torch.allclose(out_swiglu, out_gelu, atol=1e-2)


# ============================================================================
# TEST PRUNING
# ============================================================================

def test_feedforward_pruning_mask():
    """Test feedforward with pruning mask."""
    ffn = SwiGLUFeedForward(d_model=64, d_ff=256)
    
    # Set pruning mask
    mask = torch.ones(2, 10, 256) * 0.5
    ffn.set_pruning_mask(mask)
    
    assert ffn.pruning_mask.shape == mask.shape


def test_feedforward_apply_pruning():
    """Test applying pruning mask."""
    ffn = SwiGLUFeedForward(d_model=64, d_ff=256)
    x = torch.randn(2, 10, 64)
    
    # Without pruning
    out_no_prune = ffn(x)
    
    # With pruning (50% mask)
    mask = torch.ones(2, 10, 256) * 0.5
    ffn.set_pruning_mask(mask)
    out_with_prune = ffn(x)
    
    # Should be different
    assert not torch.allclose(out_no_prune, out_with_prune)


# ============================================================================
# TEST FACTORY FUNCTION
# ============================================================================

def test_get_feedforward_swiglu():
    """Test factory creates SwiGLU."""
    ffn = get_feedforward('swiglu', d_model=768)
    
    assert isinstance(ffn, SwiGLUFeedForward)


def test_get_feedforward_gelu():
    """Test factory creates GELU."""
    ffn = get_feedforward('gelu', d_model=768)
    
    assert isinstance(ffn, GELUFeedForward)


def test_get_feedforward_relu():
    """Test factory creates ReLU variant."""
    ffn = get_feedforward('relu', d_model=768)
    
    assert isinstance(ffn, GELUFeedForward)
    assert isinstance(ffn.activation, nn.ReLU)


def test_get_feedforward_case_insensitive():
    """Test factory is case insensitive."""
    ffn1 = get_feedforward('SWIGLU', d_model=768)
    ffn2 = get_feedforward('Gelu', d_model=768)
    
    assert isinstance(ffn1, SwiGLUFeedForward)
    assert isinstance(ffn2, GELUFeedForward)


def test_get_feedforward_with_kwargs():
    """Test factory passes kwargs correctly."""
    ffn = get_feedforward('swiglu', d_model=768, d_ff=2048, dropout=0.1, bias=False)
    
    assert ffn.d_ff == 2048
    assert ffn.dropout_layer is not None
    assert ffn.w1.bias is None


def test_get_feedforward_invalid():
    """Test factory raises error for invalid type."""
    with pytest.raises(ValueError, match="Unknown ffn_type"):
        get_feedforward('invalid', d_model=768)


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_feedforward_different_batch_sizes():
    """Test feedforward with different batch sizes."""
    ffn = SwiGLUFeedForward(d_model=64)
    
    for batch_size in [1, 2, 8, 32]:
        x = torch.randn(batch_size, 10, 64)
        out = ffn(x)
        assert out.shape == (batch_size, 10, 64)


def test_feedforward_different_seq_lengths():
    """Test feedforward with different sequence lengths."""
    ffn = SwiGLUFeedForward(d_model=64)
    
    for seq_len in [1, 5, 10, 100, 512]:
        x = torch.randn(2, seq_len, 64)
        out = ffn(x)
        assert out.shape == (2, seq_len, 64)


def test_feedforward_gradient_flow():
    """Test gradients flow through feedforward."""
    ffn = SwiGLUFeedForward(d_model=64)
    x = torch.randn(2, 10, 64, requires_grad=True)
    
    out = ffn(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert ffn.w1.weight.grad is not None
    assert ffn.w2.weight.grad is not None
    assert ffn.w3.weight.grad is not None


def test_feedforward_no_nan_or_inf():
    """Test feedforward doesn't produce NaN/Inf."""
    ffn = SwiGLUFeedForward(d_model=64)
    
    # Normal input
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    
    # Large input
    x = torch.randn(2, 10, 64) * 100
    out = ffn(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_feedforward_eval_mode():
    """Test feedforward in eval mode."""
    ffn = SwiGLUFeedForward(d_model=64, dropout=0.1)
    x = torch.randn(2, 10, 64)
    
    # Train mode
    ffn.train()
    out_train = ffn(x)
    
    # Eval mode
    ffn.eval()
    out_eval1 = ffn(x)
    out_eval2 = ffn(x)
    
    # In eval mode, should be deterministic
    assert torch.allclose(out_eval1, out_eval2)


def test_feedforward_reset_parameters():
    """Test feedforward parameter reset."""
    ffn = SwiGLUFeedForward(d_model=64, bias=True)
    
    # Modify weights
    with torch.no_grad():
        ffn.w1.weight.fill_(1.0)
        ffn.w1.bias.fill_(1.0)
    
    # Reset
    ffn.reset_parameters()
    
    # Should be reinitialized
    assert not torch.all(ffn.w1.weight == 1.0)
    assert torch.all(ffn.w1.bias == 0.0)  # Bias reset to zero


def test_feedforward_device_movement():
    """Test feedforward can move between devices."""
    ffn = SwiGLUFeedForward(d_model=64)
    
    # CPU
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert out.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])