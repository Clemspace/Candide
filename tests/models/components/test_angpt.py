"""Tests for anGPT attention and feedforward components."""

import math

import pytest
import torch
import torch.nn.functional as F

from ramanujan.models.components import RotaryEmbedding
from ramanujan.models.components.angpt import ANGPTAttention, ANGPTSwiGLU
from ramanujan.models.components.normalized import swiglu_act_norm_factor


def _unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def _causal(seq_len, past=0):
    full = seq_len + past
    return torch.tril(torch.ones(seq_len, full), diagonal=past).view(1, 1, seq_len, full)


# ============================================================================
# ATTENTION
# ============================================================================

@pytest.mark.parametrize("n_kv_heads", [None, 2, 1])
def test_attention_output_shape_and_unit_norm(n_kv_heads):
    torch.manual_seed(0)
    attn = ANGPTAttention(d_model=128, n_heads=4, n_kv_heads=n_kv_heads)
    x = _unit(2, 10, 128)
    out, cache = attn(x, mask=_causal(10))
    assert out.shape == (2, 10, 128)
    assert cache is None
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 10), atol=1e-5)


def test_attention_weights_have_unit_rows_at_init():
    attn = ANGPTAttention(d_model=128, n_heads=4, n_kv_heads=2)
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
        assert torch.allclose(proj.weight.norm(dim=-1), torch.ones(proj.weight.shape[0]), atol=1e-5)


def test_attention_norm_factors():
    attn = ANGPTAttention(d_model=768, n_heads=12, n_kv_heads=4)
    assert attn.nu_qkv == pytest.approx(math.sqrt(768 / 64))
    assert attn.nu_p == pytest.approx(math.sqrt(64 / 768))


def test_attention_rejects_bias():
    with pytest.raises(ValueError):
        ANGPTAttention(d_model=64, n_heads=4, bias=True)


def test_attention_rejects_bad_kv_heads():
    with pytest.raises(ValueError):
        ANGPTAttention(d_model=64, n_heads=4, n_kv_heads=3)


def test_attention_is_causal_with_causal_mask():
    torch.manual_seed(0)
    attn = ANGPTAttention(d_model=64, n_heads=4)
    x = _unit(1, 8, 64)
    out1, _ = attn(x, mask=_causal(8))
    x2 = x.clone()
    x2[:, 5:] = _unit(1, 3, 64)
    out2, _ = attn(x2, mask=_causal(8))
    assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-5)
    assert not torch.allclose(out1[:, 5:], out2[:, 5:], atol=1e-3)


def test_attention_mask_formats_agree():
    """Float 0/1 (1,1,s,s), bool, and 3-D masks give the same result."""
    torch.manual_seed(0)
    attn = ANGPTAttention(d_model=64, n_heads=4)
    x = _unit(2, 6, 64)
    m = _causal(6)
    out_float, _ = attn(x, mask=m)
    out_bool, _ = attn(x, mask=m.bool())
    out_3d, _ = attn(x, mask=m[0].expand(2, 6, 6))
    assert torch.allclose(out_float, out_bool, atol=1e-6)
    assert torch.allclose(out_float, out_3d, atol=1e-6)


def test_attention_output_invariant_to_constant_factors():
    """With exact output normalization, nu_qkv and nu_p do not change the output."""
    torch.manual_seed(0)
    attn = ANGPTAttention(d_model=64, n_heads=4)
    x = _unit(2, 6, 64)
    ref, _ = attn(x, mask=_causal(6))
    attn.nu_qkv *= 7.0
    attn.nu_p *= 0.1
    out, _ = attn(x, mask=_causal(6))
    assert torch.allclose(ref, out, atol=1e-5)


def test_attention_temperature_matters():
    torch.manual_seed(0)
    a1 = ANGPTAttention(d_model=64, n_heads=4, learnable_qk_scale=False, qk_scale_init=1.0)
    a2 = ANGPTAttention(d_model=64, n_heads=4, learnable_qk_scale=False, qk_scale_init=30.0)
    a2.load_state_dict(a1.state_dict())
    x = _unit(1, 8, 64)
    o1, _ = a1(x, mask=_causal(8))
    o2, _ = a2(x, mask=_causal(8))
    assert not torch.allclose(o1, o2, atol=1e-3)


def test_attention_learnable_temperature_init():
    attn = ANGPTAttention(d_model=64, n_heads=4)
    assert torch.allclose(attn.qk_scale(), torch.full((4,), math.sqrt(16)))


def test_attention_cache_shapes_gqa():
    attn = ANGPTAttention(d_model=128, n_heads=8, n_kv_heads=2)
    x = _unit(2, 5, 128)
    _, (k, v) = attn(x, mask=_causal(5), use_cache=True)
    assert k.shape == (2, 2, 5, 16)
    assert v.shape == (2, 2, 5, 16)
    # Cached keys are unit norm along head_dim (normalized, RoPE-free here)
    assert torch.allclose(k.norm(dim=-1), torch.ones(2, 2, 5), atol=1e-5)


def test_attention_incremental_decoding_matches_full():
    torch.manual_seed(0)
    rope = RotaryEmbedding(d_model=16, max_seq_len=64)
    attn = ANGPTAttention(d_model=64, n_heads=4, n_kv_heads=2, rope=rope).eval()
    x = _unit(1, 6, 64)
    full, _ = attn(x, mask=_causal(6))

    prefix, cache = attn(x[:, :5], mask=_causal(5), use_cache=True)
    step, _ = attn(
        x[:, 5:],
        mask=_causal(1, past=5),
        position_ids=torch.tensor([[5]]),
        past_key_value=cache,
        use_cache=True,
    )
    assert torch.allclose(full[:, :5], prefix, atol=1e-5)
    assert torch.allclose(full[:, 5:], step, atol=1e-5)


def test_attention_gradients_reach_all_parameters():
    attn = ANGPTAttention(d_model=64, n_heads=4, n_kv_heads=2)
    out, _ = attn(_unit(2, 6, 64), mask=_causal(6))
    (out * torch.randn_like(out)).sum().backward()
    for name, p in attn.named_parameters():
        assert p.grad is not None, name
        assert p.grad.abs().sum() > 0, name


# ============================================================================
# FEEDFORWARD
# ============================================================================

def test_swiglu_output_shape_and_unit_norm():
    torch.manual_seed(0)
    ffn = ANGPTSwiGLU(d_model=128, d_ff=512)
    out = ffn(_unit(2, 10, 128))
    assert out.shape == (2, 10, 128)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 10), atol=1e-5)


def test_swiglu_default_dff_is_4x():
    ffn = ANGPTSwiGLU(d_model=64)
    assert ffn.d_ff == 256


def test_swiglu_norm_factors():
    ffn = ANGPTSwiGLU(d_model=768, d_ff=3072)
    assert ffn.nu_uz == pytest.approx(0.5)
    assert ffn.nu_d == pytest.approx(2.0)
    assert ffn.nu_act == pytest.approx(swiglu_act_norm_factor(3072))


def test_swiglu_act_override():
    ffn = ANGPTSwiGLU(d_model=64, d_ff=256, act_norm_factor=3.74)
    assert ffn.nu_act == pytest.approx(3.74)


def test_swiglu_hidden_is_approximately_unit_at_init():
    torch.manual_seed(0)
    ffn = ANGPTSwiGLU(d_model=256, d_ff=1024)
    x = _unit(512, 256)
    z, u = ffn.w1(x) * ffn.nu_uz, ffn.w2(x) * ffn.nu_uz
    hidden = u * F.silu(z) * ffn.nu_act
    assert hidden.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=0.15)


def test_swiglu_output_invariant_to_post_activation_factors():
    torch.manual_seed(0)
    ffn = ANGPTSwiGLU(d_model=64, d_ff=256)
    x = _unit(2, 4, 64)
    ref = ffn(x)
    ffn.nu_act *= 9.0
    ffn.nu_d *= 0.3
    assert torch.allclose(ref, ffn(x), atol=1e-5)


def test_swiglu_gate_scale_matters():
    """nu_uz sets the SiLU input scale, the one non-homogeneous step."""
    torch.manual_seed(0)
    ffn = ANGPTSwiGLU(d_model=64, d_ff=256)
    x = _unit(2, 4, 64)
    ref = ffn(x)
    ffn.nu_uz *= 20.0
    assert not torch.allclose(ref, ffn(x), atol=1e-3)


def test_swiglu_gate_input_stays_bounded_with_paper_scaling():
    """|z_i| <= nu_uz for unit input and rows in the unit ball: SiLU stays near-linear."""
    torch.manual_seed(0)
    ffn = ANGPTSwiGLU(d_model=64, d_ff=256)
    z = ffn.w1(_unit(512, 64)) * ffn.nu_uz * ffn.gate_scale
    assert z.abs().max().item() <= ffn.nu_uz + 1e-5


@pytest.mark.parametrize("gate_scale", [1.0, 8.0])
def test_swiglu_gate_scale_keeps_hidden_unit(gate_scale):
    """nu_act is recomputed for the gate scale, so the hidden stays ~unit."""
    torch.manual_seed(0)
    ffn = ANGPTSwiGLU(d_model=256, d_ff=1024, gate_scale=gate_scale)
    x = _unit(512, 256)
    z = ffn.w1(x) * ffn.nu_uz * ffn.gate_scale
    u = ffn.w2(x) * ffn.nu_uz
    hidden = u * F.silu(z) * ffn.nu_act
    assert hidden.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=0.15)


def test_swiglu_rejects_nonpositive_gate_scale():
    with pytest.raises(ValueError):
        ANGPTSwiGLU(d_model=64, gate_scale=0.0)


def test_swiglu_rejects_bias():
    with pytest.raises(ValueError):
        ANGPTSwiGLU(d_model=64, bias=True)


def test_swiglu_pruning_mask():
    ffn = ANGPTSwiGLU(d_model=64, d_ff=256)
    mask = torch.zeros(256)
    mask[:128] = 1.0
    ffn.set_pruning_mask(mask)
    out = ffn(_unit(2, 4, 64))
    assert out.shape == (2, 4, 64)
    assert torch.isfinite(out).all()
