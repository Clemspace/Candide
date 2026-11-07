"""MHA/GQA on the fused SDPA kernel must match the explicit softmax path."""

import math

import pytest
import torch
import torch.nn.functional as F

from ramanujan.models.components import RotaryEmbedding
from ramanujan.models.components.attention import GroupedQueryAttention, MultiHeadAttention


def _causal(seq_len, past=0):
    full = seq_len + past
    return torch.tril(torch.ones(seq_len, full), diagonal=past).view(1, 1, seq_len, full)


def _reference(attn, x, mask=None, position_ids=None):
    """The pre-2026-11-07 forward: explicit scores, masked_fill, softmax, matmul."""
    b, s, _ = x.shape
    n_kv = getattr(attn, "n_kv_heads", attn.n_heads)
    q = attn.q_proj(x).view(b, s, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(b, s, n_kv, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(b, s, n_kv, attn.head_dim).transpose(1, 2)
    if attn.qk_norm:
        q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None:
        q, k = attn.rope(q, k, position_ids=position_ids)
    if n_kv != attn.n_heads:
        k, v = attn._repeat_kv(k), attn._repeat_kv(v)
    scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    out = torch.matmul(F.softmax(scores, dim=-1), v)
    out = out.transpose(1, 2).contiguous().view(b, s, attn.d_model)
    return attn.out_proj(out)


def _make(kind, qk_norm=False, rope=False):
    torch.manual_seed(0)
    r = RotaryEmbedding(d_model=16, max_seq_len=64) if rope else None
    if kind == "mha":
        return MultiHeadAttention(d_model=64, n_heads=4, rope=r, qk_norm=qk_norm).eval()
    return GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, rope=r, qk_norm=qk_norm).eval()


@pytest.mark.parametrize("kind", ["mha", "gqa"])
@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("rope", [False, True])
def test_matches_explicit_path_causal(kind, qk_norm, rope):
    attn = _make(kind, qk_norm, rope)
    x = torch.randn(2, 10, 64)
    out, _ = attn(x, mask=_causal(10))
    assert torch.allclose(out, _reference(attn, x, _causal(10)), atol=1e-5)


@pytest.mark.parametrize("kind", ["mha", "gqa"])
def test_matches_explicit_path_without_mask(kind):
    attn = _make(kind)
    x = torch.randn(2, 7, 64)
    out, _ = attn(x)
    assert torch.allclose(out, _reference(attn, x), atol=1e-5)


@pytest.mark.parametrize("kind", ["mha", "gqa"])
def test_mask_formats_agree(kind):
    """Float 0/1 4-D, boolean, and 3-D masks give the same output."""
    attn = _make(kind)
    x = torch.randn(2, 6, 64)
    m = _causal(6)
    a, _ = attn(x, mask=m)
    b, _ = attn(x, mask=m.bool())
    c, _ = attn(x, mask=m[0].expand(2, 6, 6))
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)


def test_padding_mask_matches_explicit_path():
    attn = _make("gqa")
    x = torch.randn(2, 6, 64)
    m = _causal(6).repeat(2, 1, 1, 1)
    m[1, :, :, 4:] = 0          # second sequence: last two keys are padding
    out, _ = attn(x, mask=m)
    assert torch.allclose(out, _reference(attn, x, m), atol=1e-5)


def test_incremental_decoding_matches_full():
    attn = _make("gqa", qk_norm=True, rope=True)
    x = torch.randn(1, 6, 64)
    full, _ = attn(x, mask=_causal(6))
    _, cache = attn(x[:, :5], mask=_causal(5), use_cache=True)
    step, _ = attn(x[:, 5:], mask=_causal(1, past=5), position_ids=torch.tensor([[5]]),
                   past_key_value=cache, use_cache=True)
    assert torch.allclose(full[:, 5:], step, atol=1e-5)


def test_attention_dropout_active_only_in_training():
    torch.manual_seed(0)
    attn = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout=0.5)
    x = torch.randn(2, 8, 64)
    attn.eval()
    e1, _ = attn(x, mask=_causal(8))
    e2, _ = attn(x, mask=_causal(8))
    assert torch.equal(e1, e2)
    attn.train()
    t1, _ = attn(x, mask=_causal(8))
    t2, _ = attn(x, mask=_causal(8))
    assert not torch.allclose(t1, t2)


def test_gradients_match_explicit_path():
    attn = _make("gqa", qk_norm=True)
    x = torch.randn(2, 6, 64)
    out, _ = attn(x, mask=_causal(6))
    out.pow(2).sum().backward()
    g_sdpa = {n: p.grad.clone() for n, p in attn.named_parameters()}
    attn.zero_grad()
    _reference(attn, x, _causal(6)).pow(2).sum().backward()
    for n, p in attn.named_parameters():
        assert torch.allclose(g_sdpa[n], p.grad, atol=1e-4), n
