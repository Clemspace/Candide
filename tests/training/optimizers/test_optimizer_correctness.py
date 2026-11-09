"""
Correctness tests: each optimizer implements the update it claims, checked
against a hand-written reference or PyTorch's own implementation.
(test_optimizers.py covers the factory and wrapper interface.)
"""

import math

import pytest
import torch
from torch import nn

from ramanujan.training.optimizers import create_optimizer
from ramanujan.training.optimizers.ademamix import (
    AdEMAMixOptimizer, alpha_schedule, beta3_schedule, half_life,
)
from ramanujan.training.optimizers.muon import (
    MuonOptimizer, muon_param_groups, muon_scale, zeropower_via_newtonschulz5,
)
from ramanujan.training.optimizers.standard import AdamWOptimizer, LionOptimizer, SGDOptimizer


def _param(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return nn.Parameter(torch.randn(*shape, generator=g))


def _grads(shape, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape, generator=g) for _ in range(n)]


def _run(opt, p, grads):
    for g in grads:
        p.grad = g.clone()
        opt.step()
    return p.detach().clone()


# ── AdamW ─────────────────────────────────────────────────────────────────────

def test_adamw_matches_reference_formula():
    p = _param((4, 3))
    p0 = p.detach().clone()
    grads = _grads((4, 3), 3)
    lr, b1, b2, eps, wd = 1e-2, 0.9, 0.999, 1e-8, 0.1
    out = _run(AdamWOptimizer([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd), p, grads)

    theta, m, v = p0.clone(), torch.zeros_like(p0), torch.zeros_like(p0)
    for t, g in enumerate(grads, 1):
        theta = theta * (1 - lr * wd)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta - lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
    assert torch.allclose(out, theta, atol=1e-5)


def test_adamw_kernel_choice_on_cpu():
    opt = AdamWOptimizer([_param((3, 3))])
    assert not opt.defaults["fused"]             # CPU tensors: no fused kernel


def test_adamw_accepts_generator_groups():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    opt = AdamWOptimizer([{"params": model[0].parameters(), "lr": 1e-3},
                          {"params": model[1].parameters()}], lr=1e-2)
    assert [len(g["params"]) for g in opt.param_groups] == [2, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_adamw_fused_on_cuda():
    opt = AdamWOptimizer([nn.Parameter(torch.randn(3, 3, device="cuda"))])
    assert opt.defaults["fused"]


# ── SGD ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nesterov", [False, True])
def test_sgd_matches_torch(nesterov):
    p1, p2 = _param((5,)), _param((5,))
    grads = _grads((5,), 4)
    a = _run(SGDOptimizer([p1], lr=0.1, momentum=0.9, weight_decay=0.01, nesterov=nesterov), p1, grads)
    b = _run(torch.optim.SGD([p2], lr=0.1, momentum=0.9, weight_decay=0.01, nesterov=nesterov), p2, grads)
    assert torch.allclose(a, b)


# ── Lion ──────────────────────────────────────────────────────────────────────

def test_lion_matches_reference_formula():
    p = _param((6,))
    p0 = p.detach().clone()
    grads = _grads((6,), 3)
    lr, b1, b2, wd = 1e-3, 0.9, 0.99, 0.5
    out = _run(LionOptimizer([p], lr=lr, betas=(b1, b2), weight_decay=wd), p, grads)

    theta, m = p0.clone(), torch.zeros_like(p0)
    for g in grads:
        theta = theta * (1 - lr * wd) - lr * torch.sign(b1 * m + (1 - b1) * g)
        m = b2 * m + (1 - b2) * g
    assert torch.allclose(out, theta, atol=1e-5)


def test_lion_moves_every_coordinate_by_lr():
    p = _param((50,))
    p0 = p.detach().clone()
    out = _run(LionOptimizer([p], lr=1e-3), p, _grads((50,), 1))
    assert torch.allclose((out - p0).abs(), torch.full((50,), 1e-3), atol=1e-5)


# ── AdEMAMix ──────────────────────────────────────────────────────────────────

def test_ademamix_matches_paper_update():
    p = _param((4, 4))
    p0 = p.detach().clone()
    grads = _grads((4, 4), 4)
    lr, b1, b2, b3, alpha, eps, wd = 1e-2, 0.9, 0.999, 0.9999, 5.0, 1e-8, 0.1
    out = _run(AdEMAMixOptimizer([p], lr=lr, beta1=b1, beta2=b2, beta3=b3, alpha=alpha,
                                 eps=eps, weight_decay=wd), p, grads)

    theta, m1, m2, v = p0.clone(), torch.zeros_like(p0), torch.zeros_like(p0), torch.zeros_like(p0)
    for t, g in enumerate(grads, 1):
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b3 * m2 + (1 - b3) * g                      # slow EMA, no bias correction
        v = b2 * v + (1 - b2) * g * g
        upd = (m1 / (1 - b1 ** t) + alpha * m2) / ((v / (1 - b2 ** t)).sqrt() + eps)
        theta = theta * (1 - lr * wd) - lr * upd
    assert torch.allclose(out, theta, atol=1e-5)


def test_ademamix_with_alpha_zero_is_adamw():
    p1, p2 = _param((3, 5)), _param((3, 5))
    grads = _grads((3, 5), 5)
    a = _run(AdEMAMixOptimizer([p1], lr=1e-2, alpha=0.0, weight_decay=0.1), p1, grads)
    b = _run(torch.optim.AdamW([p2], lr=1e-2, betas=(0.9, 0.999), weight_decay=0.1), p2, grads)
    assert torch.allclose(a, b, atol=1e-5)


def test_ademamix_state_keys():
    p = _param((2, 2))
    opt = AdEMAMixOptimizer([p])
    _run(opt, p, _grads((2, 2), 1))
    assert {"m1", "m2", "v", "step"} <= set(opt.state[p])


def test_ademamix_warmup_schedules():
    assert alpha_schedule(0, 8.0, 100) == 0.0
    assert alpha_schedule(50, 8.0, 100) == pytest.approx(4.0)
    assert alpha_schedule(100, 8.0, 100) == 8.0 and alpha_schedule(7, 8.0, None) == 8.0
    assert beta3_schedule(0, 0.9999, 0.9, 1000) == pytest.approx(0.9)
    assert beta3_schedule(1000, 0.9999, 0.9, 1000) == 0.9999
    mid = beta3_schedule(500, 0.9999, 0.9, 1000)
    # halfway in half-life, not in beta
    assert half_life(mid) == pytest.approx((half_life(0.9) + half_life(0.9999)) / 2, rel=1e-6)
    betas = [beta3_schedule(t, 0.9999, 0.9, 1000) for t in range(0, 1001, 50)]
    assert all(a <= b for a, b in zip(betas, betas[1:]))


# ── Muon: Newton-Schulz ───────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (96, 48)])
def test_newton_schulz_orthogonalizes(shape):
    G = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    X = zeropower_via_newtonschulz5(G, steps=5, dtype=torch.float32)
    assert X.shape == G.shape
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.3 and s.max() < 1.3              # quintic: roughly [0.5, 1.2]
    assert s.max() / s.min() < 2.5                       # input condition number ~6


def test_newton_schulz_is_scale_invariant():
    G = torch.randn(16, 24, generator=torch.Generator().manual_seed(1))
    a = zeropower_via_newtonschulz5(G, 5, torch.float32)
    b = zeropower_via_newtonschulz5(1000 * G, 5, torch.float32)
    assert torch.allclose(a, b, atol=1e-4)


def test_newton_schulz_zero_steps_is_normalized_input():
    G = torch.randn(8, 4)
    X = zeropower_via_newtonschulz5(G, steps=0, dtype=torch.float32)
    assert torch.allclose(X, G / G.norm(), atol=1e-5)


def test_muon_scale_modes():
    assert muon_scale((3072, 768), "spectral") == pytest.approx(2.0)
    assert muon_scale((768, 3072), "spectral") == 1.0
    assert muon_scale((768, 3072), "match_rms") == pytest.approx(0.2 * math.sqrt(3072))
    with pytest.raises(ValueError):
        muon_scale((4, 4), "other")


# ── Muon: optimizer ───────────────────────────────────────────────────────────

def test_muon_first_step_is_scaled_orthogonalized_gradient():
    """M1 = (1-b) G and the Nesterov update (1-b^2) G orthogonalize to NS(G)."""
    p = _param((32, 16))
    p0 = p.detach().clone()
    g = _grads((32, 16), 1)[0]
    lr = 0.02
    out = _run(MuonOptimizer([p], lr=lr, ns_dtype=torch.float32), p, [g])
    expected = p0 - lr * muon_scale((32, 16), "spectral") * zeropower_via_newtonschulz5(g, 5, torch.float32)
    assert torch.allclose(out, expected, atol=1e-5)


def test_muon_step_size_ignores_gradient_scale():
    p1, p2 = _param((16, 16)), _param((16, 16))
    g = _grads((16, 16), 1)[0]
    a = _run(MuonOptimizer([p1], lr=0.02, ns_dtype=torch.float32), p1, [g])
    b = _run(MuonOptimizer([p2], lr=0.02, ns_dtype=torch.float32), p2, [100 * g])
    assert torch.allclose(a, b, atol=1e-4)


def test_muon_match_rms_gives_adamw_like_update_rms():
    p = _param((256, 64))
    p0 = p.detach().clone()
    lr = 1e-3
    out = _run(MuonOptimizer([p], lr=lr, adjust_lr="match_rms", ns_dtype=torch.float32),
               p, _grads((256, 64), 1))
    rms = ((out - p0) / lr).pow(2).mean().sqrt().item()
    assert 0.1 < rms < 0.3                                # target 0.2


def test_muon_momentum_accumulates():
    """Two identical gradients: second step equals the first (same direction, same size)."""
    p = _param((8, 8))
    g = _grads((8, 8), 1)[0]
    opt = MuonOptimizer([p], lr=0.01, ns_dtype=torch.float32)
    x0 = p.detach().clone()
    x1 = _run(opt, p, [g])
    x2 = _run(opt, p, [g])
    assert torch.allclose(x2 - x1, x1 - x0, atol=1e-5)
    buf = opt.state[p]["momentum_buffer"]
    assert torch.allclose(buf, (1 - 0.95 ** 2) * g, atol=1e-5)


def test_muon_weight_decay_is_decoupled():
    p = _param((8, 8))
    p0 = p.detach().clone()
    out = _run(MuonOptimizer([p], lr=0.1, weight_decay=0.5), p, [torch.zeros(8, 8)])
    assert torch.allclose(out, p0 * (1 - 0.1 * 0.5), atol=1e-5)


def test_muon_routes_vectors_to_adamw():
    w, b = _param((8, 4)), _param((8,))
    opt = MuonOptimizer([w, b], lr=0.02)
    w.grad, b.grad = torch.randn(8, 4), torch.randn(8)
    opt.step()
    assert "momentum_buffer" in opt.state[w] and "exp_avg" not in opt.state[w]
    assert "exp_avg" in opt.state[b] and "momentum_buffer" not in opt.state[b]


def test_muon_adamw_part_matches_adamw():
    """use_muon=False params follow AdamW at lr * adamw_lr_ratio."""
    p1, p2 = _param((6, 6)), _param((6, 6))
    grads = _grads((6, 6), 3)
    a = _run(MuonOptimizer([{"params": [p1], "use_muon": False}], lr=0.02, adamw_lr_ratio=0.5,
                           adamw_betas=(0.9, 0.95), adamw_weight_decay=0.1), p1, grads)
    b = _run(torch.optim.AdamW([p2], lr=0.01, betas=(0.9, 0.95), weight_decay=0.1), p2, grads)
    assert torch.allclose(a, b, atol=1e-5)


def test_muon_adamw_lr_follows_schedule():
    """A schedule that rescales group['lr'] rescales the AdamW part too."""
    p = _param((4,))
    opt = MuonOptimizer([p], lr=0.02, adjust_lr="match_rms")    # ratio 1.0
    for group in opt.param_groups:
        group["lr"] = 0.0
    p0 = p.detach().clone()
    out = _run(opt, p, _grads((4,), 1))
    assert torch.equal(out, p0)


def test_muon_param_groups_keeps_embeddings_and_head_on_adamw():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 16)
            self.proj = nn.Linear(16, 16, bias=True)
            self.norm = nn.LayerNorm(16)
            self.lm_head = nn.Linear(16, 50, bias=False)
    m = Tiny()
    groups = muon_param_groups(m)
    muon = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
    adam = {id(p) for g in groups if not g["use_muon"] for p in g["params"]}
    assert muon == {id(m.proj.weight)}
    assert {id(m.emb.weight), id(m.lm_head.weight), id(m.proj.bias),
            id(m.norm.weight), id(m.norm.bias)} == adam


def test_muon_param_groups_tied_embedding_counted_once():
    emb = nn.Embedding(20, 8)
    head = nn.Linear(8, 20, bias=False)
    head.weight = emb.weight
    m = nn.ModuleDict({"emb": emb, "lm_head": head, "mid": nn.Linear(8, 8, bias=False)})
    groups = muon_param_groups(m)
    total = sum(len(g["params"]) for g in groups)
    assert total == 2


def test_muon_state_dict_roundtrip():
    p = _param((8, 8))
    opt = MuonOptimizer([p], lr=0.02)
    _run(opt, p, _grads((8, 8), 2))
    opt2 = MuonOptimizer([p], lr=0.02)
    opt2.load_state_dict(opt.state_dict())
    assert torch.equal(opt2.state[p]["momentum_buffer"], opt.state[p]["momentum_buffer"])


def test_muon_old_arguments_are_accepted_but_unknown_ones_raise():
    MuonOptimizer([_param((4, 4))], backend="newtonschulz5", dampening=0.0, foreach=True)
    with pytest.raises(TypeError):
        MuonOptimizer([_param((4, 4))], lrr=0.1)


# ── All optimizers: they train ────────────────────────────────────────────────

@pytest.mark.parametrize("name,lr", [("adamw", 3e-2), ("sgd", 5e-2), ("lion", 3e-2),
                                     ("ademamix", 3e-2), ("muon", 0.2)])
def test_optimizers_reduce_a_regression_loss(name, lr):
    torch.manual_seed(0)
    X = torch.randn(256, 16)
    W_true = torch.randn(16, 8)
    Y = X @ W_true
    model = nn.Linear(16, 8)
    params = muon_param_groups(model) if name == "muon" else model.parameters()
    kwargs = {"adamw_lr_ratio": 1.0} if name == "muon" else {}
    opt = create_optimizer(name, params, lr=lr, **kwargs)
    first = None
    for _ in range(200):
        loss = ((model(X) - Y) ** 2).mean()
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first, (name, first, loss.item())
