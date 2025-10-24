"""Tests for attention mechanisms."""

import pytest
import torch
from torch import nn

from ramanujan.models.components.attention import (
    MultiHeadAttention,
    GroupedQueryAttention,
    get_attention,
)
from ramanujan.models.components.embeddings import RotaryEmbedding


# ============================================================================
# TEST MULTI-HEAD ATTENTION (MHA)
# ============================================================================

def test_mha_creation():
    """Test creating multi-head attention."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    
    assert attn.d_model == 768
    assert attn.n_heads == 12
    assert attn.head_dim == 64  # 768 / 12


def test_mha_head_dim_validation():
    """Test MHA validates d_model divisible by n_heads."""
    with pytest.raises(AssertionError, match="d_model must be divisible by n_heads"):
        MultiHeadAttention(d_model=768, n_heads=11)


def test_mha_forward():
    """Test MHA forward pass."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    out, cache = attn(x, use_cache=False)
    
    assert out.shape == x.shape
    assert cache is None


def test_mha_with_cache():
    """Test MHA with KV caching."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    out, cache = attn(x, use_cache=True)
    
    assert out.shape == x.shape
    assert cache is not None
    assert len(cache) == 2  # (key, value)
    
    # Cache shapes: (batch, n_heads, seq_len, head_dim)
    k_cache, v_cache = cache
    assert k_cache.shape == (2, 12, 10, 64)
    assert v_cache.shape == (2, 12, 10, 64)


def test_mha_with_past_cache():
    """Test MHA using past cache for generation."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    
    # First forward pass
    x1 = torch.randn(2, 10, 768)
    out1, cache1 = attn(x1, use_cache=True)
    
    # Second forward pass with cache
    x2 = torch.randn(2, 5, 768)
    out2, cache2 = attn(x2, past_key_value=cache1, use_cache=True)
    
    assert out2.shape == (2, 5, 768)
    
    # New cache should have accumulated length
    k_cache, v_cache = cache2
    assert k_cache.shape[2] == 15  # 10 + 5


def test_mha_with_mask():
    """Test MHA with attention mask."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    # Causal mask (lower triangular)
    mask = torch.tril(torch.ones(2, 10, 10))
    
    out, _ = attn(x, mask=mask, use_cache=False)
    
    assert out.shape == x.shape


def test_mha_with_rope():
    """Test MHA with RoPE."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)  # head_dim
    attn = MultiHeadAttention(d_model=768, n_heads=12, rope=rope)
    
    x = torch.randn(2, 10, 768)
    out, _ = attn(x, use_cache=False)
    
    assert out.shape == x.shape


def test_mha_dropout():
    """Test MHA with dropout."""
    attn = MultiHeadAttention(d_model=768, n_heads=12, dropout=0.1)
    
    assert attn.attn_dropout is not None
    assert attn.resid_dropout is not None


def test_mha_no_bias():
    """Test MHA without bias."""
    attn = MultiHeadAttention(d_model=768, n_heads=12, bias=False)
    
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None
    assert attn.out_proj.bias is None


def test_mha_with_bias():
    """Test MHA with bias."""
    attn = MultiHeadAttention(d_model=768, n_heads=12, bias=True)
    
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.out_proj.bias is not None


# ============================================================================
# TEST GROUPED QUERY ATTENTION (GQA)
# ============================================================================

def test_gqa_creation():
    """Test creating grouped query attention."""
    attn = GroupedQueryAttention(d_model=2048, n_heads=32, n_kv_heads=8)
    
    assert attn.d_model == 2048
    assert attn.n_heads == 32
    assert attn.n_kv_heads == 8
    assert attn.n_groups == 4  # 32 / 8


def test_gqa_default_kv_heads():
    """Test GQA default n_kv_heads."""
    attn = GroupedQueryAttention(d_model=2048, n_heads=32)
    
    # Default should be n_heads // 4 or 1
    assert attn.n_kv_heads == 8  # 32 // 4


def test_gqa_kv_heads_validation():
    """Test GQA validates n_heads divisible by n_kv_heads."""
    with pytest.raises(AssertionError, match="n_heads must be divisible by n_kv_heads"):
        GroupedQueryAttention(d_model=2048, n_heads=32, n_kv_heads=7)


def test_gqa_forward():
    """Test GQA forward pass."""
    attn = GroupedQueryAttention(d_model=2048, n_heads=32, n_kv_heads=8)
    x = torch.randn(2, 10, 2048)
    
    out, cache = attn(x, use_cache=False)
    
    assert out.shape == x.shape
    assert cache is None


def test_gqa_with_cache():
    """Test GQA with KV caching."""
    attn = GroupedQueryAttention(d_model=2048, n_heads=32, n_kv_heads=8)
    x = torch.randn(2, 10, 2048)
    
    out, cache = attn(x, use_cache=True)
    
    assert out.shape == x.shape
    assert cache is not None
    
    # Cache should have n_kv_heads, not n_heads
    k_cache, v_cache = cache
    assert k_cache.shape == (2, 8, 10, 64)  # n_kv_heads
    assert v_cache.shape == (2, 8, 10, 64)


def test_gqa_cache_size_reduction():
    """Test GQA reduces cache size compared to MHA."""
    # MHA
    mha = MultiHeadAttention(d_model=2048, n_heads=32)
    x = torch.randn(2, 10, 2048)
    _, mha_cache = mha(x, use_cache=True)
    
    # GQA
    gqa = GroupedQueryAttention(d_model=2048, n_heads=32, n_kv_heads=8)
    _, gqa_cache = gqa(x, use_cache=True)
    
    # GQA cache should be 4x smaller
    mha_k, mha_v = mha_cache
    gqa_k, gqa_v = gqa_cache
    
    assert gqa_k.shape[1] == mha_k.shape[1] // 4  # n_kv_heads dimension


def test_gqa_repeat_kv():
    """Test GQA repeats KV heads correctly."""
    attn = GroupedQueryAttention(d_model=2048, n_heads=32, n_kv_heads=8)
    
    # Test internal _repeat_kv method
    kv = torch.randn(2, 8, 10, 64)  # (batch, n_kv_heads, seq, head_dim)
    kv_repeated = attn._repeat_kv(kv)
    
    assert kv_repeated.shape == (2, 32, 10, 64)  # n_heads


def test_gqa_mqa_equivalence():
    """Test GQA with n_kv_heads=1 is Multi-Query Attention."""
    attn = GroupedQueryAttention(d_model=2048, n_heads=32, n_kv_heads=1)
    
    assert attn.n_kv_heads == 1
    assert attn.n_groups == 32
    
    x = torch.randn(2, 10, 2048)
    out, cache = attn(x, use_cache=True)
    
    # Cache should have only 1 KV head
    k_cache, v_cache = cache
    assert k_cache.shape[1] == 1


def test_gqa_with_rope():
    """Test GQA with RoPE."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    attn = GroupedQueryAttention(d_model=2048, n_heads=32, n_kv_heads=8, rope=rope)
    
    x = torch.randn(2, 10, 2048)
    out, _ = attn(x, use_cache=False)
    
    assert out.shape == x.shape


# ============================================================================
# TEST ATTENTION WITH ROPE
# ============================================================================

def test_attention_rope_integration():
    """Test attention with RoPE integration."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    attn = MultiHeadAttention(d_model=768, n_heads=12, rope=rope)
    
    x = torch.randn(2, 10, 768)
    position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
    
    out, _ = attn(x, position_ids=position_ids, use_cache=False)
    
    assert out.shape == x.shape


def test_attention_rope_with_cache():
    """Test attention with RoPE and caching."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    attn = MultiHeadAttention(d_model=768, n_heads=12, rope=rope)
    
    # First forward
    x1 = torch.randn(2, 10, 768)
    pos1 = torch.arange(10).unsqueeze(0).expand(2, -1)
    out1, cache1 = attn(x1, position_ids=pos1, use_cache=True)
    
    # Second forward with cache
    x2 = torch.randn(2, 5, 768)
    pos2 = torch.arange(10, 15).unsqueeze(0).expand(2, -1)
    out2, cache2 = attn(x2, position_ids=pos2, past_key_value=cache1, use_cache=True)
    
    assert out2.shape == (2, 5, 768)


# ============================================================================
# TEST ATTENTION MASKING
# ============================================================================

def test_attention_causal_mask():
    """Test attention with causal mask."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    # Causal mask: each position can only attend to previous positions
    mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
    
    out, _ = attn(x, mask=mask, use_cache=False)
    
    assert out.shape == x.shape


def test_attention_padding_mask():
    """Test attention with padding mask."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    # Padding mask: mask out last 3 positions
    mask = torch.ones(2, 10, 10)
    mask[:, :, 7:] = 0  # Mask positions 7, 8, 9
    
    out, _ = attn(x, mask=mask, use_cache=False)
    
    assert out.shape == x.shape


def test_attention_mask_shape_handling():
    """Test attention handles different mask shapes."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    # 3D mask (batch, seq, seq)
    mask3d = torch.tril(torch.ones(2, 10, 10))
    out3d, _ = attn(x, mask=mask3d, use_cache=False)
    
    # 4D mask (batch, 1, seq, seq)
    mask4d = torch.tril(torch.ones(2, 1, 10, 10))
    out4d, _ = attn(x, mask=mask4d, use_cache=False)
    
    assert out3d.shape == x.shape
    assert out4d.shape == x.shape


# ============================================================================
# TEST PRUNING
# ============================================================================

def test_attention_pruning_mask():
    """Test attention with pruning mask."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    
    # Set pruning mask
    mask = torch.ones(768) * 0.5
    attn.set_pruning_mask(mask)
    
    assert attn.pruning_mask.numel() > 1


def test_attention_apply_pruning():
    """Test applying pruning mask."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    # Without pruning
    out_no_prune, _ = attn(x, use_cache=False)
    
    # With pruning
    mask = torch.ones(768) * 0.5
    attn.set_pruning_mask(mask)
    out_with_prune, _ = attn(x, use_cache=False)
    
    # Should be different
    assert not torch.allclose(out_no_prune, out_with_prune)


# ============================================================================
# TEST FACTORY FUNCTION
# ============================================================================

def test_get_attention_mha():
    """Test factory creates MHA."""
    attn = get_attention('mha', d_model=768, n_heads=12)
    
    assert isinstance(attn, MultiHeadAttention)


def test_get_attention_gqa():
    """Test factory creates GQA."""
    attn = get_attention('gqa', d_model=2048, n_heads=32, n_kv_heads=8)
    
    assert isinstance(attn, GroupedQueryAttention)
    assert attn.n_kv_heads == 8


def test_get_attention_mqa():
    """Test factory creates MQA (GQA with n_kv_heads=1)."""
    attn = get_attention('mqa', d_model=2048, n_heads=32)
    
    assert isinstance(attn, GroupedQueryAttention)
    assert attn.n_kv_heads == 1


def test_get_attention_case_insensitive():
    """Test factory is case insensitive."""
    attn1 = get_attention('MHA', d_model=768, n_heads=12)
    attn2 = get_attention('Gqa', d_model=2048, n_heads=32, n_kv_heads=8)
    
    assert isinstance(attn1, MultiHeadAttention)
    assert isinstance(attn2, GroupedQueryAttention)


def test_get_attention_with_rope():
    """Test factory with RoPE."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    attn = get_attention('mha', d_model=768, n_heads=12, rope=rope)
    
    assert attn.rope is rope


def test_get_attention_invalid():
    """Test factory raises error for invalid type."""
    with pytest.raises(ValueError, match="Unknown attention_type"):
        get_attention('invalid', d_model=768, n_heads=12)


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_attention_different_batch_sizes():
    """Test attention with different batch sizes."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    
    for batch_size in [1, 2, 8, 32]:
        x = torch.randn(batch_size, 10, 768)
        out, _ = attn(x, use_cache=False)
        assert out.shape == (batch_size, 10, 768)


def test_attention_different_seq_lengths():
    """Test attention with different sequence lengths."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    
    for seq_len in [1, 5, 10, 50, 128]:
        x = torch.randn(2, seq_len, 768)
        out, _ = attn(x, use_cache=False)
        assert out.shape == (2, seq_len, 768)


def test_attention_gradient_flow():
    """Test gradients flow through attention."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768, requires_grad=True)
    
    out, _ = attn(x, use_cache=False)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert attn.q_proj.weight.grad is not None
    assert attn.k_proj.weight.grad is not None
    assert attn.v_proj.weight.grad is not None
    assert attn.out_proj.weight.grad is not None


def test_attention_no_nan_or_inf():
    """Test attention doesn't produce NaN/Inf."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    
    x = torch.randn(2, 10, 768)
    out, _ = attn(x, use_cache=False)
    
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_attention_eval_mode():
    """Test attention in eval mode."""
    attn = MultiHeadAttention(d_model=768, n_heads=12, dropout=0.1)
    x = torch.randn(2, 10, 768)
    
    # Eval mode should be deterministic
    attn.eval()
    out1, _ = attn(x, use_cache=False)
    out2, _ = attn(x, use_cache=False)
    
    assert torch.allclose(out1, out2)


def test_attention_self_attention():
    """Test attention as self-attention (Q=K=V from same input)."""
    attn = MultiHeadAttention(d_model=768, n_heads=12)
    x = torch.randn(2, 10, 768)
    
    # Self-attention: Q, K, V all from x
    out, _ = attn(x, use_cache=False)
    
    assert out.shape == x.shape


def test_attention_parameter_count():
    """Test attention has correct number of parameters."""
    d_model = 768
    
    mha = MultiHeadAttention(d_model=d_model, n_heads=12, bias=False)
    
    # Q, K, V, O projections without bias
    expected_params = 4 * (d_model * d_model)
    
    actual_params = sum(p.numel() for p in mha.parameters())
    assert actual_params == expected_params


def test_gqa_parameter_reduction():
    """Test GQA has fewer parameters than MHA."""
    d_model = 2048
    n_heads = 32
    
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, bias=False)
    gqa = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=8, bias=False)
    
    mha_params = sum(p.numel() for p in mha.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())
    
    # GQA should have fewer parameters (smaller K, V projections)
    assert gqa_params < mha_params


def test_attention_reset_parameters():
    """Test attention parameter reset."""
    attn = MultiHeadAttention(d_model=768, n_heads=12, bias=True)
    
    # Modify weights
    with torch.no_grad():
        attn.q_proj.weight.fill_(1.0)
        attn.q_proj.bias.fill_(1.0)
    
    # Reset
    attn.reset_parameters()
    
    # Should be reinitialized
    assert not torch.all(attn.q_proj.weight == 1.0)
    assert torch.all(attn.q_proj.bias == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])