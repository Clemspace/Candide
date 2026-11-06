"""Tests for the qk_norm option (GPT+ baseline) in standard attention."""

import pytest
import torch

from ramanujan.models.components.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    get_attention,
)
from ramanujan.models.blocks.transformer import TransformerBlock
from ramanujan.models.architectures.config import TransformerConfig
from ramanujan.models.architectures.transformer import RamanujanTransformer


def _causal(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)


@pytest.mark.parametrize("cls,kwargs", [
    (MultiHeadAttention, {}),
    (GroupedQueryAttention, {'n_kv_heads': 2}),
])
def test_qk_norm_off_by_default(cls, kwargs):
    attn = cls(d_model=64, n_heads=4, **kwargs)
    assert attn.qk_norm is False
    assert attn.q_norm is None and attn.k_norm is None
    assert not any('q_norm' in n or 'k_norm' in n for n, _ in attn.named_parameters())


@pytest.mark.parametrize("cls,kwargs", [
    (MultiHeadAttention, {}),
    (GroupedQueryAttention, {'n_kv_heads': 2}),
])
def test_qk_norm_adds_head_dim_scales(cls, kwargs):
    attn = cls(d_model=64, n_heads=4, qk_norm=True, **kwargs)
    assert attn.q_norm.weight.shape == (16,)
    assert attn.k_norm.weight.shape == (16,)


@pytest.mark.parametrize("attention_type", ['mha', 'gqa', 'mqa'])
def test_factory_forwards_qk_norm(attention_type):
    attn = get_attention(attention_type, d_model=64, n_heads=4, n_kv_heads=2, qk_norm=True)
    assert attn.qk_norm is True


def test_qk_norm_makes_attention_invariant_to_qk_weight_scale():
    """With QK-norm, rescaling W_q and W_k does not change the output."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, qk_norm=True)
    x = torch.randn(2, 6, 64)
    ref, _ = attn(x, mask=_causal(6))
    with torch.no_grad():
        attn.q_proj.weight.mul_(10.0)
        attn.k_proj.weight.mul_(0.5) # scale up: keeps eps negligible vs. variance
    out, _ = attn(x, mask=_causal(6))
    assert torch.allclose(ref, out, atol=1e-5)


def test_without_qk_norm_weight_scale_changes_output():
    torch.manual_seed(0)
    attn = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2)
    x = torch.randn(2, 6, 64)
    ref, _ = attn(x, mask=_causal(6))
    with torch.no_grad():
        attn.q_proj.weight.mul_(10.0)
    out, _ = attn(x, mask=_causal(6))
    assert not torch.allclose(ref, out, atol=1e-3)


def test_block_forwards_qk_norm():
    block = TransformerBlock(d_model=64, n_heads=4, attention_type='gqa', n_kv_heads=2, qk_norm=True)
    assert block.attention.qk_norm is True
    out, _ = block(torch.randn(2, 6, 64), mask=_causal(6))
    assert out.shape == (2, 6, 64)


def test_model_with_qk_norm():
    cfg = TransformerConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                            attention_type='gqa', n_kv_heads=2, qk_norm=True)
    model = RamanujanTransformer(cfg)
    assert all(b.attention.qk_norm for b in model.blocks)
    logits = model(torch.randint(0, 100, (2, 8)))
    assert logits.shape == (2, 8, 100)
    # QK-norm scales start at 1 (not re-initialised by _init_weights)
    assert torch.allclose(model.blocks[0].attention.q_norm.weight, torch.ones(16))


def test_config_roundtrip_keeps_qk_norm():
    cfg = TransformerConfig(vocab_size=100, d_model=64, n_heads=4, qk_norm=True)
    assert TransformerConfig.from_dict(cfg.to_dict()).qk_norm is True
