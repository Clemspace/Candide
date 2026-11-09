"""LR schedules: cosine (pilot) and warmup-stable-decay."""

import math

import pytest

from ramanujan.training.schedules import cooldown_factor, lr_multiplier


def test_cosine_matches_pilot_formula():
    max_steps, warm, ratio = 1526, 153, 0.1
    for step in [0, 50, 153, 600, 1525]:
        if step < warm:
            ref = step / warm
        else:
            p = (step - warm) / (max_steps - warm)
            ref = ratio + 0.5 * (1 - ratio) * (1 + math.cos(math.pi * p))
        assert lr_multiplier(step, max_steps, warm, "cosine", min_ratio=ratio) == pytest.approx(ref)


def test_wsd_phases():
    kw = dict(max_steps=1000, warmup_steps=100, schedule="wsd", cooldown_frac=0.2)
    assert lr_multiplier(50, **kw) == pytest.approx(0.5)
    assert lr_multiplier(100, **kw) == 1.0
    assert lr_multiplier(799, **kw) == 1.0
    assert lr_multiplier(800, **kw) == pytest.approx(1.0)
    assert lr_multiplier(1000, **kw) == pytest.approx(0.0)
    mids = [lr_multiplier(s, **kw) for s in range(800, 1001)]
    assert all(a >= b for a, b in zip(mids, mids[1:]))          # monotone cooldown


def test_one_minus_sqrt_shape():
    assert cooldown_factor(0.25, "1-sqrt") == pytest.approx(0.5)
    assert cooldown_factor(0.25, "linear") == pytest.approx(0.75)
    kw = dict(max_steps=1000, warmup_steps=0, schedule="wsd", cooldown_frac=0.2)
    assert lr_multiplier(850, **kw) == pytest.approx(0.5)        # t = 0.25


def test_wsd_final_ratio_and_branch_start():
    kw = dict(max_steps=1000, warmup_steps=0, schedule="wsd")
    assert lr_multiplier(1000, final_ratio=0.1, **kw) == pytest.approx(0.1)
    # cooldown branched off step 400, running to 500
    assert lr_multiplier(399, cooldown_start=400, max_steps=500, schedule="wsd") == 1.0
    assert lr_multiplier(500, cooldown_start=400, max_steps=500, schedule="wsd") == pytest.approx(0.0)


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        lr_multiplier(10, 100, schedule="step")
    with pytest.raises(ValueError):
        cooldown_factor(0.5, "cosine")
