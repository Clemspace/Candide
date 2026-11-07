"""GeGLU feedforward (ModernBERT MLP)."""

import pytest
import torch
import torch.nn.functional as F

from ramanujan.models.components.feedforward import GeGLUFeedForward, get_feedforward


def test_factory_returns_geglu():
    assert isinstance(get_feedforward("geglu", d_model=32, d_ff=48), GeGLUFeedForward)
    ffn = get_feedforward("geglu_tanh", d_model=32, d_ff=48)
    assert isinstance(ffn, GeGLUFeedForward) and ffn.approximate == "tanh"


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_forward_formula(approximate):
    torch.manual_seed(0)
    ffn = GeGLUFeedForward(d_model=32, d_ff=48, approximate=approximate).eval()
    x = torch.randn(2, 5, 32)
    expected = ffn.w3(F.gelu(ffn.w1(x), approximate=approximate) * ffn.w2(x))
    assert torch.allclose(ffn(x), expected, atol=1e-6)
    assert ffn(x).shape == (2, 5, 32)


def test_matches_modernbert_fused_layout():
    """HF ModernBertMLP: input, gate = Wi(x).chunk(2); Wo(gelu(input) * gate)."""
    torch.manual_seed(0)
    ffn = GeGLUFeedForward(d_model=32, d_ff=48).eval()
    wi = torch.cat([ffn.w1.weight, ffn.w2.weight], dim=0)  # (2*d_ff, d_model)
    x = torch.randn(3, 7, 32)
    inp, gate = F.linear(x, wi).chunk(2, dim=-1)
    ref = F.linear(F.gelu(inp) * gate, ffn.w3.weight)
    assert torch.allclose(ffn(x), ref, atol=1e-6)


def test_bias_flag_and_param_count():
    no_bias = GeGLUFeedForward(d_model=32, d_ff=48, bias=False)
    assert all(m.bias is None for m in (no_bias.w1, no_bias.w2, no_bias.w3))
    assert sum(p.numel() for p in no_bias.parameters()) == 3 * 32 * 48
    with_bias = GeGLUFeedForward(d_model=32, d_ff=48, bias=True)
    assert sum(p.numel() for p in with_bias.parameters()) == 3 * 32 * 48 + 2 * 48 + 32


def test_gradients_flow():
    ffn = GeGLUFeedForward(d_model=16, d_ff=24)
    ffn(torch.randn(2, 3, 16)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in ffn.parameters())
