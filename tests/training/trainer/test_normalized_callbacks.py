"""Tests for WeightNormBoundCallback."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ramanujan.models.architectures.angpt import ANGPTConfig, ANGPTTransformer
from ramanujan.training.trainer.normalized_callbacks import (
    WeightNormBoundCallback,
    unwrap_model,
)


class _Wrapper(nn.Module):
    """Stands in for DDP (exposes .module)."""

    def __init__(self, module):
        super().__init__()
        self.module = module


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = ANGPTConfig(vocab_size=50, d_model=32, n_layers=1, n_heads=4, d_ff=128, max_seq_len=32)
    return ANGPTTransformer(cfg)


def _inflate(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(3.0)


def test_unwrap_model_handles_ddp_and_compile(model):
    assert unwrap_model(model) is model
    assert unwrap_model(_Wrapper(model)) is model
    compiled_like = SimpleNamespace(_orig_mod=_Wrapper(model))
    assert unwrap_model(compiled_like) is model


def test_on_step_end_projects(model):
    cb = WeightNormBoundCallback()
    trainer = SimpleNamespace(model=_Wrapper(model))
    _inflate(model)
    assert model.max_weight_row_norm() > 1.5
    cb.on_step_end(trainer, step=1, metrics={})
    assert model.max_weight_row_norm() <= 1.0 + 1e-5


def test_on_train_begin_projects(model):
    cb = WeightNormBoundCallback()
    _inflate(model)
    cb.on_train_begin(SimpleNamespace(model=model))
    assert model.max_weight_row_norm() <= 1.0 + 1e-5


def test_logging_records_pre_projection_norm(model):
    cb = WeightNormBoundCallback(log_every=1)
    _inflate(model)
    cb.on_step_end(SimpleNamespace(model=model), step=1, metrics={})
    assert cb.last_max_row_norm is not None
    assert cb.last_max_row_norm > 1.5


def test_logging_disabled_by_default(model):
    cb = WeightNormBoundCallback()
    cb.on_step_end(SimpleNamespace(model=model), step=1, metrics={})
    assert cb.last_max_row_norm is None


def test_rejects_model_without_projection():
    cb = WeightNormBoundCallback()
    with pytest.raises(AttributeError):
        cb.on_step_end(SimpleNamespace(model=nn.Linear(4, 4)), step=1, metrics={})


def test_projection_after_real_optimizer_steps(model):
    """Large-LR Adam steps push rows out; the callback pulls them back every step."""
    cb = WeightNormBoundCallback()
    trainer = SimpleNamespace(model=model)
    opt = torch.optim.Adam(model.parameters(), lr=0.5)
    x = torch.randint(0, 50, (2, 8))
    for step in range(3):
        opt.zero_grad()
        model(x).pow(2).mean().backward()
        opt.step()
        cb.on_step_end(trainer, step=step, metrics={})
        assert model.max_weight_row_norm() <= 1.0 + 1e-5
