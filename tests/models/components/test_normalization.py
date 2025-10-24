"""Tests for normalization layers."""

import pytest
import torch
from torch import nn

from ramanujan.models.components.normalization import (
    RMSNorm,
    LayerNorm,
    get_normalization,
)


# ============================================================================
# TEST RMSNORM
# ============================================================================

def test_rmsnorm_creation():
    """Test creating RMSNorm."""
    norm = RMSNorm(d_model=768)
    
    assert norm.d_model == 768
    assert norm.eps == 1e-6  # Default
    assert norm.elementwise_affine is True


def test_rmsnorm_forward():
    """Test RMSNorm forward pass."""
    norm = RMSNorm(d_model=64)
    x = torch.randn(2, 10, 64)
    
    out = norm(x)
    
    assert out.shape == x.shape
    # Check normalization
    rms = (out.pow(2).mean(dim=-1, keepdim=True) + norm.eps).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rmsnorm_no_affine():
    """Test RMSNorm without learnable parameters."""
    norm = RMSNorm(d_model=64, elementwise_affine=False)
    
    assert norm.weight is None
    assert sum(p.numel() for p in norm.parameters()) == 0


def test_rmsnorm_with_affine():
    """Test RMSNorm with learnable scale."""
    norm = RMSNorm(d_model=64, elementwise_affine=True)
    
    assert norm.weight is not None
    assert norm.weight.shape == (64,)
    assert sum(p.numel() for p in norm.parameters()) == 64


def test_rmsnorm_reset_parameters():
    """Test RMSNorm parameter initialization."""
    norm = RMSNorm(d_model=64)
    
    # Modify weights
    with torch.no_grad():
        norm.weight.fill_(2.0)
    
    assert torch.all(norm.weight == 2.0)
    
    # Reset
    norm.reset_parameters()
    
    assert torch.all(norm.weight == 1.0)


def test_rmsnorm_custom_eps():
    """Test RMSNorm with custom eps."""
    norm = RMSNorm(d_model=64, eps=1e-8)
    
    assert norm.eps == 1e-8


def test_rmsnorm_different_input_shapes():
    """Test RMSNorm with different input shapes."""
    norm = RMSNorm(d_model=64)
    
    # 2D
    x = torch.randn(10, 64)
    out = norm(x)
    assert out.shape == (10, 64)
    
    # 3D
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64)
    
    # 4D
    x = torch.randn(2, 3, 10, 64)
    out = norm(x)
    assert out.shape == (2, 3, 10, 64)


def test_rmsnorm_numerical_stability():
    """Test RMSNorm doesn't produce NaN/Inf."""
    norm = RMSNorm(d_model=64)
    
    # Very small values
    x = torch.randn(2, 10, 64) * 1e-10
    out = norm(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    
    # Very large values
    x = torch.randn(2, 10, 64) * 1e10
    out = norm(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ============================================================================
# TEST LAYERNORM
# ============================================================================

def test_layernorm_creation():
    """Test creating LayerNorm."""
    norm = LayerNorm(d_model=768)
    
    assert norm.d_model == 768
    assert norm.eps == 1e-5  # Default
    assert norm.elementwise_affine is True
    assert norm.use_bias is True


def test_layernorm_forward():
    """Test LayerNorm forward pass."""
    norm = LayerNorm(d_model=64)
    x = torch.randn(2, 10, 64)
    
    out = norm(x)
    
    assert out.shape == x.shape
    # Check normalization (mean ≈ 0, std ≈ 1)
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-2)  # Slightly larger tolerance for std


def test_layernorm_no_affine():
    """Test LayerNorm without learnable parameters."""
    norm = LayerNorm(d_model=64, elementwise_affine=False)
    
    assert norm.weight is None
    assert norm.bias is None
    assert sum(p.numel() for p in norm.parameters()) == 0


def test_layernorm_no_bias():
    """Test LayerNorm without bias."""
    norm = LayerNorm(d_model=64, bias=False)
    
    assert norm.weight is not None
    assert norm.bias is None
    assert sum(p.numel() for p in norm.parameters()) == 64


def test_layernorm_with_affine():
    """Test LayerNorm with learnable parameters."""
    norm = LayerNorm(d_model=64)
    
    assert norm.weight is not None
    assert norm.bias is not None
    assert norm.weight.shape == (64,)
    assert norm.bias.shape == (64,)
    assert sum(p.numel() for p in norm.parameters()) == 128


def test_layernorm_reset_parameters():
    """Test LayerNorm parameter initialization."""
    norm = LayerNorm(d_model=64)
    
    # Modify parameters
    with torch.no_grad():
        norm.weight.fill_(2.0)
        norm.bias.fill_(1.0)
    
    assert torch.all(norm.weight == 2.0)
    assert torch.all(norm.bias == 1.0)
    
    # Reset
    norm.reset_parameters()
    
    assert torch.all(norm.weight == 1.0)
    assert torch.all(norm.bias == 0.0)


def test_layernorm_matches_pytorch():
    """Test LayerNorm produces same results as PyTorch."""
    d_model = 64
    
    # Our implementation
    our_norm = LayerNorm(d_model=d_model)
    
    # PyTorch implementation
    torch_norm = nn.LayerNorm(d_model)
    
    # Copy weights to ensure same initialization
    with torch.no_grad():
        torch_norm.weight.copy_(our_norm.weight)
        torch_norm.bias.copy_(our_norm.bias)
    
    # Test
    x = torch.randn(2, 10, d_model)
    our_out = our_norm(x)
    torch_out = torch_norm(x)
    
    assert torch.allclose(our_out, torch_out, atol=1e-6)


# ============================================================================
# TEST COMPARISON RMSNORM VS LAYERNORM
# ============================================================================

def test_rmsnorm_vs_layernorm_speed():
    """Test that RMSNorm is faster than LayerNorm (conceptually)."""
    # RMSNorm should have fewer parameters
    rms = RMSNorm(d_model=768)
    layer = LayerNorm(d_model=768)
    
    rms_params = sum(p.numel() for p in rms.parameters())
    layer_params = sum(p.numel() for p in layer.parameters())
    
    assert rms_params < layer_params  # RMSNorm has no bias


def test_rmsnorm_vs_layernorm_different_outputs():
    """Test that RMSNorm and LayerNorm produce different outputs."""
    rms = RMSNorm(d_model=64, elementwise_affine=False)
    layer = LayerNorm(d_model=64, elementwise_affine=False)
    
    x = torch.randn(2, 10, 64)
    
    rms_out = rms(x)
    layer_out = layer(x)
    
    # Should be different (RMSNorm doesn't center mean)
    assert not torch.allclose(rms_out, layer_out, atol=1e-3)


# ============================================================================
# TEST GET_NORMALIZATION FACTORY
# ============================================================================

def test_get_normalization_rms():
    """Test factory creates RMSNorm."""
    norm = get_normalization('rms', d_model=768)
    
    assert isinstance(norm, RMSNorm)
    assert norm.d_model == 768


def test_get_normalization_rmsnorm():
    """Test factory with 'rmsnorm' string."""
    norm = get_normalization('rmsnorm', d_model=768)
    
    assert isinstance(norm, RMSNorm)


def test_get_normalization_layer():
    """Test factory creates LayerNorm."""
    norm = get_normalization('layer', d_model=768)
    
    assert isinstance(norm, LayerNorm)
    assert norm.d_model == 768


def test_get_normalization_layernorm():
    """Test factory with 'layernorm' string."""
    norm = get_normalization('layernorm', d_model=768)
    
    assert isinstance(norm, LayerNorm)


def test_get_normalization_case_insensitive():
    """Test factory is case insensitive."""
    norm1 = get_normalization('RMS', d_model=768)
    norm2 = get_normalization('LAYER', d_model=768)
    
    assert isinstance(norm1, RMSNorm)
    assert isinstance(norm2, LayerNorm)


def test_get_normalization_with_kwargs():
    """Test factory passes kwargs correctly."""
    norm = get_normalization('rms', d_model=768, eps=1e-8, elementwise_affine=False)
    
    assert norm.eps == 1e-8
    assert norm.elementwise_affine is False


def test_get_normalization_invalid():
    """Test factory raises error for invalid type."""
    with pytest.raises(ValueError, match="Unknown norm_type"):
        get_normalization('invalid', d_model=768)


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_normalization_with_zeros():
    """Test normalization with zero input."""
    rms = RMSNorm(d_model=64)
    layer = LayerNorm(d_model=64)
    
    x = torch.zeros(2, 10, 64)
    
    # Should not crash (thanks to eps)
    rms_out = rms(x)
    layer_out = layer(x)
    
    assert not torch.isnan(rms_out).any()
    assert not torch.isnan(layer_out).any()


def test_normalization_gradient_flow():
    """Test gradients flow through normalization."""
    norm = RMSNorm(d_model=64)
    x = torch.randn(2, 10, 64, requires_grad=True)
    
    out = norm(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_normalization_device_movement():
    """Test normalization can move between devices."""
    norm = RMSNorm(d_model=64)
    
    # CPU
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.device.type == 'cpu'
    
    # GPU (if available)
    if torch.cuda.is_available():
        norm = norm.cuda()
        x = x.cuda()
        out = norm(x)
        assert out.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])