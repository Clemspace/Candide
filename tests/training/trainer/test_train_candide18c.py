"""Tests for the pilot-arm paths in ramanujan/training/train_candide18c.py.

Covers model/optimizer selection per arch and a 3-step end-to-end run of
train() on an in-memory loader (no shards, no wandb). Runs on GPU if one is
available, since the script picks its device itself.
"""

import importlib
import math

import pytest
import torch
from torch.utils.data import DataLoader

script = importlib.import_module("ramanujan.training.train_candide18c")

VOCAB = 64
SEQ = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_cfg(arch, **overrides):
    kwargs = dict(
        vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=128, max_seq_len=SEQ, dropout=0.0, arch=arch,
    )
    kwargs.update(overrides)
    return script.ModelConfig(**kwargs)


# ============================================================================
# MODEL SELECTION
# ============================================================================

def test_setup_model_gpt_with_qk_norm():
    from ramanujan.models.architectures import RamanujanTransformer
    model = script.setup_model(_model_cfg("gpt", qk_norm=True))
    assert isinstance(model, RamanujanTransformer)
    assert all(b.attention.qk_norm for b in model.blocks)


def test_setup_model_gpt_without_qk_norm_is_unchanged_baseline():
    model = script.setup_model(_model_cfg("gpt"))
    assert not any(b.attention.qk_norm for b in model.blocks)


def test_setup_model_angpt():
    from ramanujan.models.architectures.angpt import ANGPTTransformer
    model = script.setup_model(_model_cfg("angpt", gate_scale=2.0, residual_mode="exact"))
    assert isinstance(model, ANGPTTransformer)
    assert model.blocks[0].ffn.gate_scale == 2.0
    assert model.blocks[0].residual_attn.mode == "exact"


def test_setup_model_unknown_arch():
    with pytest.raises(ValueError):
        script.setup_model(_model_cfg("bogus"))


# ============================================================================
# OPTIMIZER
# ============================================================================

def test_optimizer_angpt_has_no_weight_decay():
    model = script.setup_model(_model_cfg("angpt")).to(DEVICE)
    opt = script.setup_optimizer(model, script.TrainingConfig(weight_decay=0.1), arch="angpt")
    assert all(g["weight_decay"] == 0.0 for g in opt.param_groups)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in model.parameters())


def test_optimizer_gpt_excludes_norm_scales_from_decay():
    model = script.setup_model(_model_cfg("gpt", qk_norm=True)).to(DEVICE)
    opt = script.setup_optimizer(model, script.TrainingConfig(weight_decay=0.1), arch="gpt")
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    assert opt.param_groups[0]["weight_decay"] == 0.1
    for name, p in model.named_parameters():
        if "norm" in name:   # RMSNorm and QK-norm scales
            assert id(p) not in decay_ids, name


# ============================================================================
# END-TO-END (3 optimizer steps)
# ============================================================================

class _FakeShardLoader:
    """Stands in for ShardLoader: shifted labels, like the real shards."""

    def __init__(self, seq_len, batch_size, seed, **kwargs):
        g = torch.Generator().manual_seed(seed)
        tokens = torch.randint(0, VOCAB, (32, seq_len + 1), generator=g)
        self.items = [{"input_ids": t[:-1], "labels": t[1:]} for t in tokens]
        self.batch_size = batch_size

    def get_dataloaders(self):
        loader = DataLoader(self.items, batch_size=self.batch_size)
        return loader, loader


def _full_cfg(arch, tmp_path, **model_overrides):
    cfg = script.FullConfig()
    cfg.model = _model_cfg(arch, **model_overrides)
    cfg.training = script.TrainingConfig(
        max_steps=3, batch_size=2, gradient_accumulation_steps=2,
        learning_rate=1e-3, min_lr=1e-4, warmup_steps=0 if arch == "angpt" else 1,
        weight_decay=0.0 if arch == "angpt" else 0.1,
        eval_every=2, eval_steps=2, save_every=1000, log_every=1, seed=1,
    )
    cfg.data = script.DataConfig(sequence_length=SEQ, num_workers=0, pin_memory=False)
    cfg.logging = script.LoggingConfig(use_wandb=False, output_dir=str(tmp_path / arch))
    return cfg


@pytest.mark.parametrize("arch,overrides", [("gpt", {"qk_norm": True}), ("angpt", {})])
def test_train_runs_and_saves_final_checkpoint(arch, overrides, tmp_path, monkeypatch):
    monkeypatch.setattr(script, "ShardLoader", _FakeShardLoader)
    cfg = _full_cfg(arch, tmp_path, **overrides)
    script.train(cfg)
    ckpt = tmp_path / arch / "checkpoints" / "final.pt"
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert state["step"] == 3


def test_train_angpt_keeps_rows_bounded(tmp_path, monkeypatch):
    """Projection runs after every optimizer step inside train()."""
    monkeypatch.setattr(script, "ShardLoader", _FakeShardLoader)
    cfg = _full_cfg("angpt", tmp_path)
    cfg.training.learning_rate = 0.5   # large steps push rows out of the ball
    cfg.training.min_lr = 0.05

    built = {}
    real_setup = script.setup_model

    def capture(model_cfg):
        built["model"] = real_setup(model_cfg)
        return built["model"]

    monkeypatch.setattr(script, "setup_model", capture)
    script.train(cfg)
    assert built["model"].max_weight_row_norm() <= 1.0 + 1e-4


# ============================================================================
# DATA OPTIONS AND EVALUATION
# ============================================================================

def test_loader_options_reach_shard_loader(tmp_path, monkeypatch):
    seen = {}

    class _Capture(_FakeShardLoader):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(script, "ShardLoader", _Capture)
    cfg = _full_cfg("gpt", tmp_path, qk_norm=True)
    cfg.data.train_order, cfg.data.val_windows, cfg.data.val_seed = "global", 32, 7
    script.train(cfg)
    assert seen["train_order"] == "global"
    assert seen["val_windows"] == 32
    assert seen["val_seed"] == 7


def test_data_defaults_are_global_order_and_fixed_val():
    d = script.DataConfig()
    assert d.train_order == "global"
    assert d.val_windows and d.val_windows > 0


def test_evaluate_reads_whole_loader_and_weights_by_tokens():
    """n_steps=None reads every batch; ignored labels do not count as tokens."""

    class _Const(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, VOCAB)      # uniform prediction: loss = log(VOCAB)

    x = torch.randint(0, VOCAB, (5, SEQ))
    labels = x.clone()
    labels[0, :] = -100                               # one fully ignored row
    batches = [{"input_ids": x[i:i + 2], "labels": labels[i:i + 2]} for i in range(0, 5, 2)]
    out = script.evaluate(_Const(), batches, torch.device("cpu"), None, torch.autocast("cpu", enabled=False))
    assert out["val/loss"] == pytest.approx(math.log(VOCAB), rel=1e-5)
    assert out["val/eval_tokens"] == 4 * SEQ
