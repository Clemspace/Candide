"""Tests for the anGPT transformer block."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ramanujan.models.blocks.angpt import ANGPTBlock
from ramanujan.models.components import RotaryEmbedding, RMSNorm, LayerNorm


def _unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def _causal(seq_len, past=0):
    full = seq_len + past
    return torch.tril(torch.ones(seq_len, full), diagonal=past).view(1, 1, seq_len, full)


def _stack_norm(mode, n_blocks=8, d_model=256):
    """Mean residual-stream norm after a stack of freshly initialised blocks."""
    torch.manual_seed(0)
    blocks = nn.ModuleList([
        ANGPTBlock(d_model=d_model, n_heads=4, d_ff=4 * d_model, residual_mode=mode)
        for _ in range(n_blocks)
    ])
    x = _unit(2, 16, d_model)
    mask = _causal(16)
    with torch.no_grad():
        for block in blocks:
            x, _ = block(x, mask=mask)
    return x.norm(dim=-1).mean().item()


def test_block_forward_shape():
    block = ANGPTBlock(d_model=128, n_heads=4, d_ff=512, n_kv_heads=2)
    out, cache = block(_unit(2, 10, 128), mask=_causal(10))
    assert out.shape == (2, 10, 128)
    assert cache is None


def test_block_has_no_norm_layers():
    block = ANGPTBlock(d_model=64, n_heads=4)
    assert not any(isinstance(m, (RMSNorm, LayerNorm, nn.LayerNorm)) for m in block.modules())


def test_block_alpha_initial_value():
    block = ANGPTBlock(d_model=64, n_heads=4, alpha_init=0.05, alpha_scale=0.01)
    assert torch.allclose(block.residual_attn.alpha(), torch.full((64,), 0.05))
    assert torch.allclose(block.residual_ffn.alpha(), torch.full((64,), 0.05))


def test_block_keeps_stream_near_unit_norm():
    norm = _stack_norm('approx', n_blocks=1)
    assert norm == pytest.approx(1.0, abs=0.05)


def test_deep_stack_approx_stays_near_unit():
    """No 'curse of depth': the stream norm does not drift over 8 blocks."""
    assert _stack_norm('approx', n_blocks=8) == pytest.approx(1.0, abs=0.2)


def test_deep_stack_exact_is_unit():
    assert _stack_norm('exact', n_blocks=8) == pytest.approx(1.0, abs=1e-4)


def test_deep_stack_paper_literal_decays():
    """The printed nu(alpha) formula collapses the stream with depth."""
    assert _stack_norm('paper_literal', n_blocks=8) < 0.5


def test_block_cache_passthrough_with_rope():
    torch.manual_seed(0)
    rope = RotaryEmbedding(d_model=16, max_seq_len=64)
    block = ANGPTBlock(d_model=64, n_heads=4, n_kv_heads=2, rope=rope).eval()
    x = _unit(1, 6, 64)
    full, _ = block(x, mask=_causal(6))
    prefix, cache = block(x[:, :5], mask=_causal(5), use_cache=True)
    step, new_cache = block(
        x[:, 5:], mask=_causal(1, past=5), position_ids=torch.tensor([[5]]),
        past_key_value=cache, use_cache=True,
    )
    assert new_cache[0].shape[2] == 6
    assert torch.allclose(full[:, 5:], step, atol=1e-5)


def test_block_gradients_reach_all_parameters():
    block = ANGPTBlock(d_model=64, n_heads=4, d_ff=256)
    out, _ = block(_unit(2, 6, 64), mask=_causal(6))
    (out * torch.randn_like(out)).sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, name
        assert p.grad.abs().sum() > 0, name


def test_block_invalid_residual_mode():
    with pytest.raises(ValueError):
        ANGPTBlock(d_model=64, n_heads=4, residual_mode='bogus')
