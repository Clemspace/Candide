"""Tests for approximately-normalized primitives (anGPT)."""

import math

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from ramanujan.models.components.normalized import (
    LERPResidual,
    ScaledParameter,
    bounded_weights,
    lerp_norm_factor,
    linear_norm_factor,
    max_row_norm,
    normalize_rows_,
    project_rows_to_unit_ball,
    swiglu_act_norm_factor,
)


def _unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def _unit_rows(d_out, d_in):
    w = torch.randn(d_out, d_in)
    return normalize_rows_(w)


# ============================================================================
# NORMALIZATION FACTORS
# ============================================================================

def test_linear_norm_factor_value():
    assert linear_norm_factor(768, 64) == pytest.approx(math.sqrt(12))
    assert linear_norm_factor(64, 64) == pytest.approx(1.0)


@pytest.mark.parametrize("d_in,d_out", [(256, 64), (256, 1024), (512, 512)])
def test_linear_norm_factor_normalizes_empirically(d_in, d_out):
    torch.manual_seed(0)
    w = _unit_rows(d_out, d_in)
    x = _unit(2048, d_in)
    y = (x @ w.T) * linear_norm_factor(d_in, d_out)
    assert y.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=0.05)


def test_swiglu_act_norm_factor_deterministic():
    assert swiglu_act_norm_factor(1024) == swiglu_act_norm_factor(1024)


@pytest.mark.parametrize("d_model,d_ff", [(256, 1024), (384, 1024)])
def test_swiglu_act_norm_factor_normalizes_empirically(d_model, d_ff):
    """Under the actual pipeline scaling, nu_act brings u*SiLU(z) to unit norm."""
    torch.manual_seed(0)
    w_gate, w_up = _unit_rows(d_ff, d_model), _unit_rows(d_ff, d_model)
    x = _unit(1024, d_model)
    nu_uz = linear_norm_factor(d_model, d_ff)
    z, u = (x @ w_gate.T) * nu_uz, (x @ w_up.T) * nu_uz
    hidden = u * F.silu(z) * swiglu_act_norm_factor(d_ff)
    assert hidden.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=0.1)


def test_lerp_norm_factor_zero_alpha_is_identity():
    assert lerp_norm_factor(torch.zeros(16)).item() == pytest.approx(1.0)


def test_lerp_norm_factor_sqrt_normalizes_independent_units():
    torch.manual_seed(0)
    d = 512
    h, x = _unit(4096, d), _unit(4096, d)
    alpha = torch.rand(d) * 0.5
    out = (h + alpha * (x - h)) * lerp_norm_factor(alpha, mode='sqrt')
    assert out.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=0.02)


def test_lerp_norm_factor_paper_literal_shrinks():
    """The printed nu(alpha) = 1 - 2a + 2a^2 is E||.||^2; multiplying by it shrinks."""
    torch.manual_seed(0)
    d = 512
    h, x = _unit(2048, d), _unit(2048, d)
    alpha = torch.full((d,), 0.05)
    out = (h + alpha * (x - h)) * lerp_norm_factor(alpha, mode='paper_literal')
    assert out.norm(dim=-1).mean().item() < 0.9


def test_lerp_norm_factor_invalid_mode():
    with pytest.raises(ValueError):
        lerp_norm_factor(torch.zeros(4), mode='nope')


# ============================================================================
# SCALED PARAMETER
# ============================================================================

def test_scaled_parameter_initial_value():
    p = ScaledParameter(8, init=0.05, scale=0.01)
    assert torch.allclose(p(), torch.full((8,), 0.05))
    assert torch.allclose(p.raw, torch.full((8,), 0.01))
    assert p.lr_multiplier == pytest.approx(5.0)


def test_scaled_parameter_effective_lr_with_adam():
    """First Adam step moves raw by ~lr, so the effective value moves by ~lr * init/scale."""
    p = ScaledParameter(4, init=0.05, scale=0.01)
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    before = p().detach().clone()
    p().sum().backward()
    opt.step()
    delta = (before - p().detach()).abs()
    assert torch.allclose(delta, torch.full((4,), 1e-3 * 5.0), rtol=1e-3)


def test_scaled_parameter_reset():
    p = ScaledParameter(3, init=2.0, scale=0.5)
    with torch.no_grad():
        p.raw.fill_(7.0)
    p.reset_parameters()
    assert torch.allclose(p(), torch.full((3,), 2.0))


def test_scaled_parameter_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        ScaledParameter(3, init=1.0, scale=0.0)


# ============================================================================
# LERP RESIDUAL
# ============================================================================

def test_lerp_residual_approx_keeps_unit_norm():
    torch.manual_seed(0)
    res = LERPResidual(512, alpha_init=0.2, mode='approx')
    h, x = _unit(4, 32, 512), _unit(4, 32, 512)
    out = res(h, x)
    assert out.shape == h.shape
    assert out.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=0.03)


def test_lerp_residual_exact_is_unit():
    res = LERPResidual(64, mode='exact')
    out = res(_unit(2, 8, 64), _unit(2, 8, 64))
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_lerp_residual_moves_toward_input():
    """With alpha = 1 the output points along the sub-layer output."""
    res = LERPResidual(64, alpha_init=1.0, alpha_scale=1.0, mode='exact')
    h, x = _unit(2, 64), _unit(2, 64)
    assert torch.allclose(res(h, x), x, atol=1e-5)


def test_lerp_residual_gradient_reaches_alpha():
    res = LERPResidual(32)
    res(_unit(2, 4, 32), _unit(2, 4, 32)).sum().backward()
    assert res.alpha.raw.grad is not None
    assert res.alpha.raw.grad.abs().sum() > 0


def test_lerp_residual_invalid_mode():
    with pytest.raises(ValueError):
        LERPResidual(16, mode='wrong')


# ============================================================================
# WEIGHT CONSTRAINTS
# ============================================================================

def test_normalize_rows_gives_unit_rows():
    w = torch.randn(10, 20) * 3
    normalize_rows_(w)
    assert torch.allclose(w.norm(dim=-1), torch.ones(10), atol=1e-6)


def test_normalize_rows_keeps_zero_row():
    w = torch.randn(3, 5)
    w[1].zero_()
    normalize_rows_(w)
    assert torch.all(w[1] == 0)


def test_project_rows_only_shrinks_rows_outside_ball():
    w = torch.zeros(2, 4)
    w[0] = torch.tensor([2.0, 0.0, 0.0, 0.0])   # norm 2 -> projected to 1
    w[1] = torch.tensor([0.3, 0.4, 0.0, 0.0])   # norm 0.5 -> untouched
    original_small = w[1].clone()
    project_rows_to_unit_ball(w)
    assert w[0].norm().item() == pytest.approx(1.0)
    assert torch.equal(w[1], original_small)


def test_bounded_weights_deduplicates_tied():
    emb = nn.Embedding(10, 8)
    head = nn.Linear(8, 10, bias=False)
    head.weight = emb.weight
    model = nn.ModuleDict({'emb': emb, 'head': head, 'other': nn.Linear(8, 8)})
    weights = list(bounded_weights(model))
    assert len(weights) == 2


def test_max_row_norm():
    lin = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[3.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0.5, 0]]))
    assert max_row_norm(lin) == pytest.approx(3.0)
