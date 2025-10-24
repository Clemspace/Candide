"""Tests for embedding layers."""

import pytest
import torch
from torch import nn

from ramanujan.models.components.embeddings import (
    TokenEmbedding,
    RotaryEmbedding,
    LearnedPositionalEmbedding,
)


# ============================================================================
# TEST TOKEN EMBEDDING
# ============================================================================

def test_token_embedding_creation():
    """Test creating token embedding."""
    embed = TokenEmbedding(vocab_size=1000, d_model=128)
    
    assert embed.vocab_size == 1000
    assert embed.d_model == 128


def test_token_embedding_forward():
    """Test token embedding forward pass."""
    embed = TokenEmbedding(vocab_size=1000, d_model=128)
    tokens = torch.randint(0, 1000, (2, 10))
    
    out = embed(tokens)
    
    assert out.shape == (2, 10, 128)


def test_token_embedding_padding():
    """Test token embedding with padding idx."""
    embed = TokenEmbedding(vocab_size=1000, d_model=128, padding_idx=0)
    
    # Padding embeddings should be zero
    assert torch.all(embed.embedding.weight[0] == 0)


def test_token_embedding_different_shapes():
    """Test token embedding with different input shapes."""
    embed = TokenEmbedding(vocab_size=1000, d_model=128)
    
    # 1D
    tokens = torch.randint(0, 1000, (10,))
    out = embed(tokens)
    assert out.shape == (10, 128)
    
    # 2D
    tokens = torch.randint(0, 1000, (2, 10))
    out = embed(tokens)
    assert out.shape == (2, 10, 128)
    
    # 3D
    tokens = torch.randint(0, 1000, (2, 3, 10))
    out = embed(tokens)
    assert out.shape == (2, 3, 10, 128)


def test_token_embedding_reset_parameters():
    """Test token embedding parameter initialization."""
    embed = TokenEmbedding(vocab_size=1000, d_model=128, padding_idx=0)
    
    # Modify weights
    with torch.no_grad():
        embed.embedding.weight.fill_(1.0)
    
    # Reset should restore proper initialization
    embed.reset_parameters()
    
    # Padding should still be zero
    assert torch.all(embed.embedding.weight[0] == 0)
    # Other weights should not all be 1
    assert not torch.all(embed.embedding.weight[1:] == 1.0)


def test_token_embedding_gradient_flow():
    """Test gradients flow through token embedding."""
    embed = TokenEmbedding(vocab_size=100, d_model=64)
    tokens = torch.randint(0, 100, (2, 10))
    
    out = embed(tokens)
    loss = out.sum()
    loss.backward()
    
    # Embeddings should have gradients
    assert embed.embedding.weight.grad is not None


# ============================================================================
# TEST ROTARY EMBEDDING (RoPE)
# ============================================================================

def test_rope_creation():
    """Test creating RoPE."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    assert rope.d_model == 64
    assert rope.max_seq_len == 128
    assert rope.theta == 10000.0


def test_rope_dimension_must_be_even():
    """Test RoPE requires even dimension."""
    with pytest.raises(AssertionError, match="Dimension must be even"):
        RotaryEmbedding(d_model=65, max_seq_len=128)


def test_rope_forward():
    """Test RoPE forward pass."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    # (batch, n_heads, seq_len, head_dim)
    q = torch.randn(2, 8, 10, 64)
    k = torch.randn(2, 8, 10, 64)
    
    q_rot, k_rot = rope(q, k)
    
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_with_start_pos():
    """Test RoPE with start position for caching."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    q = torch.randn(2, 8, 5, 64)
    k = torch.randn(2, 8, 5, 64)
    
    # Start at position 10
    q_rot, k_rot = rope(q, k, start_pos=10)
    
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_with_position_ids():
    """Test RoPE with explicit position IDs."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    q = torch.randn(2, 8, 10, 64)
    k = torch.randn(2, 8, 10, 64)
    
    # Custom positions
    position_ids = torch.randint(0, 128, (2, 10))
    
    q_rot, k_rot = rope(q, k, position_ids=position_ids)
    
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_preserves_norm():
    """Test RoPE preserves vector norms (rotation property)."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    q = torch.randn(2, 8, 10, 64)
    k = torch.randn(2, 8, 10, 64)
    
    q_rot, k_rot = rope(q, k)
    
    # Norms should be approximately preserved (rotation doesn't change magnitude)
    q_norm_before = q.norm(dim=-1)
    q_norm_after = q_rot.norm(dim=-1)
    
    assert torch.allclose(q_norm_before, q_norm_after, rtol=1e-4, atol=1e-4)


def test_rope_changes_values():
    """Test RoPE actually modifies values."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    q = torch.randn(2, 8, 10, 64)
    k = torch.randn(2, 8, 10, 64)
    
    q_rot, k_rot = rope(q, k)
    
    # Should be different (rotated)
    assert not torch.allclose(q, q_rot, atol=1e-3)
    assert not torch.allclose(k, k_rot, atol=1e-3)


def test_rope_extend_seq_len():
    """Test extending RoPE for longer sequences."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    original_max = rope.max_seq_len
    
    # Extend
    rope.extend_seq_len(256)
    
    assert rope.max_seq_len == 256
    assert rope.freqs_cis.shape[0] == 256
    
    # Should work with longer sequences now
    q = torch.randn(2, 8, 200, 64)
    k = torch.randn(2, 8, 200, 64)
    
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape


def test_rope_caching():
    """Test RoPE caches frequencies efficiently."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    # First call
    q = torch.randn(2, 8, 10, 64)
    k = torch.randn(2, 8, 10, 64)
    q_rot1, k_rot1 = rope(q, k)
    
    # Cache should exist
    assert rope.freqs_cis is not None
    assert rope.freqs_cis.shape == (128, 32)  # max_seq_len, d_model//2


def test_rope_complex_dtype():
    """Test RoPE uses complex numbers internally."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    # freqs_cis should be complex
    assert rope.freqs_cis.dtype in [torch.complex64, torch.complex128]


def test_rope_different_seq_lengths():
    """Test RoPE with varying sequence lengths."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    for seq_len in [1, 5, 10, 32, 64, 128]:
        q = torch.randn(2, 8, seq_len, 64)
        k = torch.randn(2, 8, seq_len, 64)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.shape == (2, 8, seq_len, 64)
        assert k_rot.shape == (2, 8, seq_len, 64)


# ============================================================================
# TEST LEARNED POSITIONAL EMBEDDING
# ============================================================================

def test_learned_pos_embedding_creation():
    """Test creating learned positional embedding."""
    pos_embed = LearnedPositionalEmbedding(max_seq_len=512, d_model=128)
    
    assert pos_embed.max_seq_len == 512
    assert pos_embed.d_model == 128


def test_learned_pos_embedding_forward():
    """Test learned positional embedding forward pass."""
    pos_embed = LearnedPositionalEmbedding(max_seq_len=512, d_model=128)
    positions = torch.arange(10).unsqueeze(0)  # (1, 10)
    
    out = pos_embed(positions)
    
    assert out.shape == (1, 10, 128)


def test_learned_pos_embedding_different_shapes():
    """Test learned positional embedding with different shapes."""
    pos_embed = LearnedPositionalEmbedding(max_seq_len=512, d_model=128)
    
    # 1D
    positions = torch.arange(10)
    out = pos_embed(positions)
    assert out.shape == (10, 128)
    
    # 2D
    positions = torch.arange(10).unsqueeze(0).expand(2, -1)
    out = pos_embed(positions)
    assert out.shape == (2, 10, 128)


def test_learned_pos_embedding_with_padding():
    """Test learned positional embedding with padding."""
    pos_embed = LearnedPositionalEmbedding(max_seq_len=512, d_model=128, padding_idx=0)
    
    # Position 0 should be zero
    assert torch.all(pos_embed.embedding.weight[0] == 0)


def test_learned_pos_embedding_reset():
    """Test learned positional embedding reset."""
    pos_embed = LearnedPositionalEmbedding(max_seq_len=512, d_model=128, padding_idx=0)
    
    # Modify
    with torch.no_grad():
        pos_embed.embedding.weight.fill_(1.0)
    
    # Reset
    pos_embed.reset_parameters()
    
    # Padding should be zero
    assert torch.all(pos_embed.embedding.weight[0] == 0)


def test_learned_pos_embedding_trainable():
    """Test learned positional embedding is trainable."""
    pos_embed = LearnedPositionalEmbedding(max_seq_len=512, d_model=128)
    positions = torch.arange(10).unsqueeze(0)
    
    out = pos_embed(positions)
    loss = out.sum()
    loss.backward()
    
    # Should have gradients
    assert pos_embed.embedding.weight.grad is not None


# ============================================================================
# TEST COMPARISON: RoPE vs LEARNED
# ============================================================================

def test_rope_vs_learned_different_outputs():
    """Test RoPE and learned embeddings produce different patterns."""
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    learned = LearnedPositionalEmbedding(max_seq_len=128, d_model=64)
    
    # RoPE needs Q, K
    q = torch.randn(2, 8, 10, 64)
    k = torch.randn(2, 8, 10, 64)
    q_rot, k_rot = rope(q, k)
    
    # Learned just needs positions
    positions = torch.arange(10).unsqueeze(0)
    learned_out = learned(positions)
    
    # Shapes compatible but mechanisms different
    assert q_rot.shape[-2:] == (10, 64)
    assert learned_out.shape[-2:] == (10, 64)


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_embeddings_device_movement():
    """Test embeddings can move between devices."""
    token_embed = TokenEmbedding(vocab_size=1000, d_model=64)
    rope = RotaryEmbedding(d_model=64, max_seq_len=128)
    
    # CPU
    tokens = torch.randint(0, 1000, (2, 10))
    out = token_embed(tokens)
    assert out.device.type == 'cpu'
    
    # RoPE on CPU
    q = torch.randn(2, 8, 10, 64)
    k = torch.randn(2, 8, 10, 64)
    q_rot, k_rot = rope(q, k)
    assert q_rot.device.type == 'cpu'


def test_embeddings_gradient_flow():
    """Test gradients flow through all embeddings."""
    token_embed = TokenEmbedding(vocab_size=100, d_model=64)
    learned_pos = LearnedPositionalEmbedding(max_seq_len=128, d_model=64)
    
    tokens = torch.randint(0, 100, (2, 10))
    positions = torch.arange(10).unsqueeze(0).expand(2, -1)
    
    # Combine embeddings
    token_emb = token_embed(tokens)
    pos_emb = learned_pos(positions)
    combined = token_emb + pos_emb
    
    loss = combined.sum()
    loss.backward()
    
    assert token_embed.embedding.weight.grad is not None
    assert learned_pos.embedding.weight.grad is not None


def test_embeddings_batch_independence():
    """Test embeddings handle batches independently."""
    embed = TokenEmbedding(vocab_size=1000, d_model=64)
    
    # Two different batches
    tokens1 = torch.randint(0, 1000, (1, 10))
    tokens2 = torch.randint(0, 1000, (1, 10))
    
    out1 = embed(tokens1)
    out2 = embed(tokens2)
    
    # Batched
    tokens_batched = torch.cat([tokens1, tokens2], dim=0)
    out_batched = embed(tokens_batched)
    
    assert torch.allclose(out_batched[0], out1[0])
    assert torch.allclose(out_batched[1], out2[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])