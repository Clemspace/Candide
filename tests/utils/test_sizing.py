"""Model sizing: data-constrained allocation and shape choice."""

import math

import numpy as np
import pytest

from ramanujan.utils import sizing as S

U = 1.748e9   # unique tokens of the 18c corpus
V = 32132


def test_unique_data_counts_fully():
    assert S.effective_data(1e9, 2e9) == pytest.approx(1e9)
    # one epoch exactly: no repetition
    assert S.effective_data(U, U) == pytest.approx(U)


def test_repetition_is_discounted():
    four = S.effective_data(4 * U, U)
    assert 3.5 * U < four < 4 * U                      # ~4 epochs nearly as good as fresh data
    many = S.effective_data(1000 * U, U)
    assert many == pytest.approx(U * (1 + S.R_D_STAR), rel=1e-3)   # saturates at 1 + R*_D


def test_matches_paper_equation_17():
    N, D = 2e8, 5e9
    u_d = U
    d_eff = u_d + 15.4 * u_d * (1 - math.exp(-(D / u_d - 1) / 15.4))
    u_n = 0.051 * u_d
    n_eff = u_n + 5.3 * u_n * (1 - math.exp(-(N / u_n - 1) / 5.3))
    paper = 521 / n_eff ** 0.353 + 1488 / d_eff ** 0.353 + 1.87   # Eq. 17 prints 0.35
    assert S.predicted_loss(N, D, U) == pytest.approx(paper, abs=0.01)


def test_unconstrained_allocation_is_about_20_tokens_per_param():
    a = S.allocate(1e19, unique_tokens=1e15)          # effectively unlimited data
    assert 15 < a.tokens / a.params < 25
    assert a.compute == pytest.approx(6 * a.params * a.tokens, rel=1e-6)


def test_more_compute_more_params_and_epochs():
    a = S.allocate(2e18, U)
    b = S.allocate(8e18, U)
    assert b.params > a.params and b.epochs > a.epochs


def test_tolerance_picks_smaller_model():
    a = S.allocate(7e18, U)
    b = S.allocate(7e18, U, tolerance=0.02)
    assert b.params < a.params and b.loss <= a.loss + 0.02 + 1e-9


def test_optimum_is_flat():
    a = S.allocate(7e18, U)
    assert 0 < a.loss_half_params - a.loss < 0.1
    assert 0 < a.loss_double_params - a.loss < 0.1


def test_count_params_matches_pilot_model():
    # pilot GPT+: 768 x 8, 12 heads (4 KV), d_ff 3072, untied, QK-norm -> 118.6M logged
    n = S.count_params(768, 8, V, 12, 4, 3072, tie=False)
    assert n == 118_574_848


def test_choose_shape_constraints():
    s = S.choose_shape(118.6e6, V)
    assert s.d_model % 192 == 0 and s.n_heads == s.d_model // 64
    assert s.n_heads % s.n_kv_heads == 0 and s.n_heads // s.n_kv_heads == 3
    assert s.d_ff % 64 == 0 and s.tie_word_embeddings
    assert abs(s.params - 118.6e6) / 118.6e6 < 0.05
    assert 12 <= s.aspect <= 96


def test_choose_shape_prefers_target_aspect():
    deep = S.choose_shape(2e8, V, target_aspect=20)
    wide = S.choose_shape(2e8, V, target_aspect=80)
    assert deep.aspect < wide.aspect and deep.n_layers > wide.n_layers


def test_size_model_end_to_end():
    f = S.flops_per_sec_from_run(118.6e6, 40_000)
    r = S.size_model(U, gpu_hours=72, flops_per_sec=f, vocab_size=V)
    assert r.max_steps * r.tokens_per_step <= 72 * 3600 * f / (6 * r.shape.params) + r.tokens_per_step
    assert r.est_hours == pytest.approx(72, rel=0.01)
    assert set(r.shape.model_section(V)) >= {"d_model", "n_layers", "n_heads", "n_kv_heads",
                                             "d_ff", "tie_word_embeddings", "vocab_size"}


def test_count_params_matches_real_model():
    torch = pytest.importorskip("torch")
    from ramanujan.models.architectures.config import TransformerConfig
    from ramanujan.models.architectures.transformer import RamanujanTransformer
    s = S.choose_shape(3e7, 1000)
    cfg = TransformerConfig(vocab_size=1000, d_model=s.d_model, n_layers=s.n_layers,
                            n_heads=s.n_heads, n_kv_heads=s.n_kv_heads, d_ff=s.d_ff,
                            attention_type="gqa", ffn_type="swiglu", norm_type="rms",
                            qk_norm=True, bias=False, tie_word_embeddings=True)
    model = RamanujanTransformer(cfg)
    assert sum(p.numel() for p in model.parameters()) == s.params
