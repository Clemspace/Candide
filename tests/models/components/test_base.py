"""Tests for model component base classes."""

import pytest
import torch
from torch import nn

from ramanujan.models.components.base import (
    BaseNormalization,
    BaseAttention,
    BaseFeedForward,
    BaseEmbedding,
    BaseTransformerBlock,
)


# ============================================================================
# TEST FIXTURES - Dummy implementations
# ============================================================================

class DummyNorm(BaseNormalization):
    """Dummy normalization for testing."""
    
    def forward(self, x):
        return x / (x.std(dim=-1, keepdim=True) + self.eps)
    
    def reset_parameters(self):
        pass


class DummyAttention(BaseAttention):
    """Dummy attention for testing."""
    
    def forward(self, x, mask=None, position_ids=None, past_key_value=None, use_cache=False, **kwargs):
        # Simple identity attention
        output = x
        cache = (x, x) if use_cache else None
        return output, cache


class DummyFeedForward(BaseFeedForward):
    """Dummy feedforward for testing."""
    
    def __init__(self, d_model, d_ff=None, **kwargs):
        super().__init__(d_model, d_ff, **kwargs)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        return self.linear(x)


class DummyEmbedding(BaseEmbedding):
    """Dummy embedding for testing."""
    
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size=vocab_size, d_model=d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x, **kwargs):
        return self.embed(x)


class DummyTransformerBlock(BaseTransformerBlock):
    """Dummy transformer block for testing."""
    
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(d_model, n_heads, **kwargs)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None, position_ids=None, past_key_value=None, use_cache=False, **kwargs):
        output = self.linear(x)
        cache = (x, x) if use_cache else None
        return output, cache


# ============================================================================
# TEST BASE NORMALIZATION
# ============================================================================

def test_base_normalization_creation():
    """Test creating base normalization."""
    norm = DummyNorm(d_model=768, eps=1e-5)
    
    assert norm.d_model == 768
    assert norm.eps == 1e-5


def test_base_normalization_forward():
    """Test normalization forward pass."""
    norm = DummyNorm(d_model=64)
    x = torch.randn(2, 10, 64)
    
    out = norm(x)
    
    assert out.shape == x.shape
    # Should normalize variance (std should be closer to 1 after normalization)
    out_std = out.std(dim=-1).mean()
    assert out_std > 0.9 and out_std < 1.1  # Normalized around 1.0


def test_base_normalization_abstract():
    """Test that BaseNormalization is abstract."""
    # Cannot instantiate abstract class
    with pytest.raises(TypeError):
        BaseNormalization(d_model=768)


# ============================================================================
# TEST BASE ATTENTION
# ============================================================================

def test_base_attention_creation():
    """Test creating base attention."""
    attn = DummyAttention(d_model=768, n_heads=12)
    
    assert attn.d_model == 768
    assert attn.n_heads == 12
    assert attn.head_dim == 64  # 768 / 12


def test_base_attention_forward():
    """Test attention forward pass."""
    attn = DummyAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    output, cache = attn(x, use_cache=False)
    
    assert output.shape == x.shape
    assert cache is None


def test_base_attention_with_cache():
    """Test attention with cache."""
    attn = DummyAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    output, cache = attn(x, use_cache=True)
    
    assert output.shape == x.shape
    assert cache is not None
    assert len(cache) == 2  # (key, value)


def test_base_attention_head_dim_validation():
    """Test that d_model must be divisible by n_heads."""
    with pytest.raises(AssertionError, match="d_model must be divisible by n_heads"):
        DummyAttention(d_model=768, n_heads=11)


def test_base_attention_pruning_mask():
    """Test pruning mask functionality."""
    attn = DummyAttention(d_model=768, n_heads=12)
    
    # Initially no pruning
    assert attn.pruning_mask.numel() == 1
    
    # Set pruning mask
    mask = torch.ones(768, 768) * 0.5
    attn.set_pruning_mask(mask)
    
    assert attn.pruning_mask.shape == (768, 768)
    assert attn.pruning_mask[0, 0] == 0.5


def test_base_attention_apply_pruning():
    """Test applying pruning mask to weights."""
    attn = DummyAttention(d_model=64, n_heads=4)
    
    # No pruning initially
    weight = torch.ones(64, 64)
    masked = attn._apply_pruning_mask(weight)
    assert torch.equal(masked, weight)
    
    # With pruning mask
    mask = torch.ones(64, 64) * 0.5
    attn.set_pruning_mask(mask)
    masked = attn._apply_pruning_mask(weight)
    assert torch.equal(masked, weight * 0.5)


def test_base_attention_abstract():
    """Test that BaseAttention is abstract."""
    with pytest.raises(TypeError):
        BaseAttention(d_model=768, n_heads=12)


# ============================================================================
# TEST BASE FEEDFORWARD
# ============================================================================

def test_base_feedforward_creation():
    """Test creating base feedforward."""
    ffn = DummyFeedForward(d_model=768)
    
    assert ffn.d_model == 768
    assert ffn.d_ff == 768 * 4  # Default


def test_base_feedforward_custom_d_ff():
    """Test feedforward with custom d_ff."""
    ffn = DummyFeedForward(d_model=768, d_ff=2048)
    
    assert ffn.d_ff == 2048


def test_base_feedforward_forward():
    """Test feedforward forward pass."""
    ffn = DummyFeedForward(d_model=64)
    x = torch.randn(2, 10, 64)
    
    out = ffn(x)
    
    assert out.shape == x.shape


def test_base_feedforward_pruning():
    """Test feedforward pruning mask."""
    ffn = DummyFeedForward(d_model=64)
    
    # Set pruning mask
    mask = torch.ones(64, 64) * 0.5
    ffn.set_pruning_mask(mask)
    
    assert ffn.pruning_mask.shape == (64, 64)


def test_base_feedforward_abstract():
    """Test that BaseFeedForward is abstract."""
    with pytest.raises(TypeError):
        BaseFeedForward(d_model=768)


# ============================================================================
# TEST BASE EMBEDDING
# ============================================================================

def test_base_embedding_creation():
    """Test creating base embedding."""
    embed = DummyEmbedding(vocab_size=50000, d_model=768)
    
    assert embed.vocab_size == 50000
    assert embed.d_model == 768


def test_base_embedding_forward():
    """Test embedding forward pass."""
    embed = DummyEmbedding(vocab_size=100, d_model=64)
    x = torch.randint(0, 100, (2, 10))
    
    out = embed(x)
    
    assert out.shape == (2, 10, 64)


def test_base_embedding_abstract():
    """Test that BaseEmbedding is abstract."""
    with pytest.raises(TypeError):
        BaseEmbedding(vocab_size=1000, d_model=768)


# ============================================================================
# TEST BASE TRANSFORMER BLOCK
# ============================================================================

def test_base_transformer_block_creation():
    """Test creating base transformer block."""
    block = DummyTransformerBlock(d_model=768, n_heads=12)
    
    assert block.d_model == 768
    assert block.n_heads == 12
    assert block.d_ff == 768 * 4
    assert block.norm_first is True  # Default pre-norm


def test_base_transformer_block_post_norm():
    """Test transformer block with post-norm."""
    block = DummyTransformerBlock(d_model=768, n_heads=12, norm_first=False)
    
    assert block.norm_first is False


def test_base_transformer_block_forward():
    """Test transformer block forward pass."""
    block = DummyTransformerBlock(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    
    output, cache = block(x, use_cache=False)
    
    assert output.shape == x.shape
    assert cache is None


def test_base_transformer_block_with_cache():
    """Test transformer block with cache."""
    block = DummyTransformerBlock(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    
    output, cache = block(x, use_cache=True)
    
    assert output.shape == x.shape
    assert cache is not None


def test_base_transformer_block_abstract():
    """Test that BaseTransformerBlock is abstract."""
    with pytest.raises(TypeError):
        BaseTransformerBlock(d_model=768, n_heads=12)


# ============================================================================
# TEST INHERITANCE AND COMPOSITION
# ============================================================================

def test_all_inherit_from_module():
    """Test that all base classes inherit from nn.Module."""
    assert issubclass(BaseNormalization, nn.Module)
    assert issubclass(BaseAttention, nn.Module)
    assert issubclass(BaseFeedForward, nn.Module)
    assert issubclass(BaseEmbedding, nn.Module)
    assert issubclass(BaseTransformerBlock, nn.Module)


def test_dummy_implementations_work():
    """Test that dummy implementations can be composed."""
    # Create components
    norm = DummyNorm(d_model=64)
    attn = DummyAttention(d_model=64, n_heads=4)
    ffn = DummyFeedForward(d_model=64)
    embed = DummyEmbedding(vocab_size=100, d_model=64)
    
    # Simple forward pass composition
    tokens = torch.randint(0, 100, (2, 10))
    x = embed(tokens)
    x = norm(x)
    x, _ = attn(x)
    x = ffn(x)
    
    assert x.shape == (2, 10, 64)


# ============================================================================
# TEST PARAMETER INITIALIZATION
# ============================================================================

def test_base_classes_have_parameters():
    """Test that implementations have parameters."""
    ffn = DummyFeedForward(d_model=64)
    embed = DummyEmbedding(vocab_size=100, d_model=64)
    
    # Should have trainable parameters
    assert sum(p.numel() for p in ffn.parameters()) > 0
    assert sum(p.numel() for p in embed.parameters()) > 0


def test_pruning_mask_not_trainable():
    """Test that pruning masks are not trainable parameters."""
    attn = DummyAttention(d_model=64, n_heads=4)
    
    # Set pruning mask
    mask = torch.ones(64, 64)
    attn.set_pruning_mask(mask)
    
    # Pruning mask should not be in parameters
    param_names = [name for name, _ in attn.named_parameters()]
    assert 'pruning_mask' not in param_names
    
    # But should be in buffers
    buffer_names = [name for name, _ in attn.named_buffers()]
    assert 'pruning_mask' in buffer_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])