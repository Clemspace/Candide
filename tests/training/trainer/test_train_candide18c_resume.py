"""train_candide18c: WSD schedule, exact resume, cooldown data mix, extra validation.

End-to-end runs use real shards (tiny, random tokens) and the real ShardLoader,
so the resume test covers the data position as well as weights, optimizer and
RNG. On CPU a resumed run must match an uninterrupted one to float precision;
on GPU, kernels are not bit-deterministic, so the tolerance is looser.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from ramanujan.data.loaders.shard_loader import HEADER_SIZE, MAGIC

script = importlib.import_module("ramanujan.training.train_candide18c")

VOCAB = 64
SEQ = 16
CUDA = torch.cuda.is_available()


def _write_shard(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(HEADER_SIZE // 4, dtype=np.int32)
    header[0], header[1] = MAGIC, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def _shards(root: Path, seed: int, n_tokens: int = 2000, with_val: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    _write_shard(root / "train" / "shard_00000.bin", rng.integers(0, VOCAB, n_tokens))
    if with_val:
        _write_shard(root / "val" / "shard_00000.bin", rng.integers(0, VOCAB, 600))
    return root


def _cfg(tmp_path, out: str, max_steps=6, schedule="cosine", dropout=0.1, **data):
    cfg = script.FullConfig()
    cfg.model = script.ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                                   d_ff=128, max_seq_len=SEQ, dropout=dropout, arch="gpt", qk_norm=True)
    cfg.training = script.TrainingConfig(
        max_steps=max_steps, batch_size=2, gradient_accumulation_steps=2, learning_rate=1e-3,
        min_lr=1e-4, warmup_steps=1, weight_decay=0.1, eval_every=1000, save_every=3,
        log_every=1, seed=1, schedule=schedule, cooldown_frac=0.5)
    cfg.data = script.DataConfig(shards_dir=str(tmp_path / "main"), sequence_length=SEQ,
                                 num_workers=0, pin_memory=False, val_windows=8, **data)
    cfg.logging = script.LoggingConfig(use_wandb=False, output_dir=str(tmp_path / out))
    return cfg


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def _assert_same(a, b):
    tol = dict(rtol=1e-3, atol=1e-4) if CUDA else dict(rtol=0, atol=1e-6)
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], **tol, msg=k)


# ============================================================================
# SCHEDULE AND PHASES (pure)
# ============================================================================

def test_cooldown_start_matches_lr_schedule():
    t = script.TrainingConfig(max_steps=100, warmup_steps=10, schedule="wsd", cooldown_frac=0.2,
                              learning_rate=1e-3, min_lr=1e-4)
    cs = script.cooldown_start_step(t)
    assert cs == 80
    assert script.lr_factor(cs - 1, t) == 1.0 and script.lr_factor(cs, t) == 1.0
    assert script.lr_factor(cs + 1, t) < 1.0
    assert script.lr_factor(100, t) == pytest.approx(0.0)
    t.cooldown_start = 50
    assert script.cooldown_start_step(t) == 50 and script.lr_factor(51, t) < 1.0


def test_cosine_is_the_pilot_schedule():
    """lr_factor(cosine) = the formula the pilot runs used."""
    import math
    t = script.TrainingConfig(max_steps=100, warmup_steps=10, learning_rate=1e-3, min_lr=1e-4)
    assert script.cooldown_start_step(t) is None
    for step in (0, 5, 10, 37, 99):
        if step < 10:
            ref = step / 10
        else:
            p = (step - 10) / 90
            ref = 0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * p))
        assert script.lr_factor(step, t) == pytest.approx(ref)


def test_phase_at():
    cfg = script.FullConfig()
    cfg.training = script.TrainingConfig(max_steps=10, warmup_steps=0, schedule="wsd", cooldown_frac=0.3)
    assert script.phase_at(6, cfg) == 0                     # no cooldown mix: always phase 0
    cfg.data.mix, cfg.data.cooldown_mix = {"main": 1.0}, {"main": 0.5, "enc": 0.5}
    assert [script.phase_at(s, cfg) for s in (0, 6, 7, 9)] == [0, 0, 1, 1]


def test_mixture_passes():
    cfg = script.FullConfig()
    cfg.training = script.TrainingConfig(max_steps=100, schedule="wsd", cooldown_frac=0.2,
                                         batch_size=2, gradient_accumulation_steps=2)
    cfg.data.sequence_length = 8
    cfg.data.mix = {"main": 0.9, "enc": 0.1}
    cfg.data.cooldown_mix = {"main": 0.5, "enc": 0.5}
    passes = script.mixture_passes(cfg, {"main": 360, "enc": 20}, tokens_per_step=4 * 8, cooldown_start=80)
    assert passes[0] == pytest.approx({"main": 0.9 * 80 * 4 / 360, "enc": 0.1 * 80 * 4 / 20})
    assert passes[1] == pytest.approx({"main": 0.5 * 20 * 4 / 360, "enc": 0.5 * 20 * 4 / 20})


def test_latest_checkpoint(tmp_path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    assert script.latest_checkpoint(str(tmp_path)) is None
    for name in ("step_900.pt", "step_1000.pt", "final.pt", "pre_cooldown.pt", "step_50.pt.tmp"):
        (ck / name).touch()
    assert script.latest_checkpoint(str(tmp_path)).name == "step_1000.pt"


def test_prune_checkpoints(tmp_path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for n in (1000, 2000, 3000, 4000, 5000):
        (ck / f"step_{n}.pt").touch()
    for name in ("final.pt", "pre_cooldown.pt"):
        (ck / name).touch()
    assert script.prune_checkpoints(str(tmp_path), None) == []          # default: keep all
    deleted = script.prune_checkpoints(str(tmp_path), keep_last=2, keep_steps=[2000])
    assert sorted(f.name for f in deleted) == ["step_1000.pt", "step_3000.pt"]
    assert sorted(f.name for f in ck.iterdir()) == ["final.pt", "pre_cooldown.pt", "step_2000.pt",
                                                     "step_4000.pt", "step_5000.pt"]


def test_retention_during_training(tmp_path):
    _shards(tmp_path / "main", 0)
    cfg = _cfg(tmp_path, "r", max_steps=6)
    cfg.training.save_every, cfg.training.keep_last, cfg.training.keep_steps = 1, 2, [2]
    script.train(cfg)
    names = sorted(f.name for f in (tmp_path / "r" / "checkpoints").iterdir())
    assert names == ["final.pt", "step_2.pt", "step_5.pt", "step_6.pt"]


def test_non_finite_loss_stops_before_any_checkpoint(tmp_path, monkeypatch):
    """A diverged run must raise at the first logged non-finite loss and write no checkpoint."""
    _shards(tmp_path / "main", 0)
    cfg = _cfg(tmp_path, "nan", max_steps=6, dropout=0.0)      # log_every=1, save_every=3
    monkeypatch.setattr(script, "lr_factor", lambda step, t: float("nan"))   # NaN weights after step 1
    with pytest.raises(FloatingPointError):
        script.train(cfg)
    ckpts = tmp_path / "nan" / "checkpoints"
    assert not ckpts.exists() or not any(ckpts.iterdir())


def test_muon_groups_cover_every_parameter_once():
    cfg = script.FullConfig()
    model = script.setup_model(script.ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4,
                                                  n_kv_heads=2, d_ff=128, max_seq_len=SEQ, dropout=0.0,
                                                  arch="gpt", qk_norm=True, tie_word_embeddings=True))
    opt = script.setup_optimizer(model, script.TrainingConfig(optimizer="muon", weight_decay=0.1,
                                                              learning_rate=1e-3))
    from ramanujan.training.optimizers.muon import MuonOptimizer
    assert isinstance(opt, MuonOptimizer)
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len({id(p) for p in model.parameters()})
    emb = model.token_embedding.embedding.weight
    for g in opt.param_groups:
        ps = {id(p) for p in g["params"]}
        if g["use_muon"]:
            assert id(emb) not in ps and all(p.ndim == 2 for p in g["params"])
            assert g["weight_decay"] == 0.1
        elif id(emb) in ps:
            assert g["adamw_weight_decay"] == 0.1                # embedding decayed, as with AdamW
        else:
            assert all(p.ndim < 2 for p in g["params"]) and g["adamw_weight_decay"] == 0.0


def test_unknown_optimizer():
    model = script.setup_model(script.ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=1, n_heads=4,
                                                  n_kv_heads=2, d_ff=128, max_seq_len=SEQ))
    with pytest.raises(ValueError, match="Unknown optimizer"):
        script.setup_optimizer(model, script.TrainingConfig(optimizer="sgd"))


def test_muon_resume_matches_uninterrupted_run(tmp_path):
    _shards(tmp_path / "main", 0)
    full = _cfg(tmp_path, "full", schedule="wsd")
    full.training.optimizer = "muon"
    script.train(full)
    half = _cfg(tmp_path, "resumed", schedule="wsd")
    half.training.optimizer = "muon"
    script.train(half, resume=str(tmp_path / "full" / "checkpoints" / "step_3.pt"))
    _assert_same(_weights(tmp_path / "full" / "checkpoints" / "final.pt"),
                 _weights(tmp_path / "resumed" / "checkpoints" / "final.pt"))


@pytest.mark.skipif(not CUDA, reason="torch.compile run on GPU only")
def test_compiled_run_saves_plain_weights(tmp_path):
    _shards(tmp_path / "main", 0)
    cfg = _cfg(tmp_path, "c", max_steps=3)
    cfg.training.compile = True
    script.train(cfg)
    keys = _weights(tmp_path / "c" / "checkpoints" / "final.pt").keys()
    assert not any(k.startswith("_orig_mod.") for k in keys)


def test_config_errors(tmp_path):
    _shards(tmp_path / "main", 0)
    cfg = _cfg(tmp_path, "x", mix={"main": 1.0}, cooldown_mix={"main": 0.5, "enc": 0.5})
    with pytest.raises(ValueError, match="wsd"):                      # cosine + cooldown mix
        script.train(cfg)
    cfg.training.schedule = "wsd"
    with pytest.raises(ValueError, match="unknown sources"):          # enc not declared
        script.train(cfg)


# ============================================================================
# END-TO-END
# ============================================================================

@pytest.mark.parametrize("schedule", ["cosine", "wsd"])
def test_resume_matches_uninterrupted_run(tmp_path, schedule):
    _shards(tmp_path / "main", 0)
    full = _cfg(tmp_path, "full", schedule=schedule)
    script.train(full)
    half = _cfg(tmp_path, "resumed", schedule=schedule)
    script.train(half, resume=str(tmp_path / "full" / "checkpoints" / "step_3.pt"))
    _assert_same(_weights(tmp_path / "full" / "checkpoints" / "final.pt"),
                 _weights(tmp_path / "resumed" / "checkpoints" / "final.pt"))


def test_cooldown_switches_mix_and_resume_inside_cooldown(tmp_path):
    _shards(tmp_path / "main", 0)
    _shards(tmp_path / "enc", 1, n_tokens=600, with_val=True)
    data = dict(sources={"enc": str(tmp_path / "enc")}, mix={"main": 0.9, "enc": 0.1},
                cooldown_mix={"main": 0.5, "enc": 0.5}, extra_val={"enc": str(tmp_path / "enc")})
    full = _cfg(tmp_path, "full", max_steps=8, schedule="wsd", **data)     # cooldown from step 4
    full.training.save_every = 5
    script.train(full)
    ck = tmp_path / "full" / "checkpoints"
    pre = torch.load(ck / "pre_cooldown.pt", map_location="cpu", weights_only=False)
    assert pre["step"] == 4 and pre["phase"] == 0
    step5 = torch.load(ck / "step_5.pt", map_location="cpu", weights_only=False)
    assert step5["phase"] == 1

    # resume one step into the cooldown: phase 1 data, resumed at its first micro-batches
    resumed = _cfg(tmp_path, "resumed", max_steps=8, schedule="wsd", **data)
    resumed.training.save_every = 5
    script.train(resumed, resume=str(ck / "step_5.pt"))
    _assert_same(_weights(ck / "final.pt"), _weights(tmp_path / "resumed" / "checkpoints" / "final.pt"))
    assert not (tmp_path / "resumed" / "checkpoints" / "pre_cooldown.pt").exists()


def test_branch_cooldown_from_stable_checkpoint(tmp_path):
    """A cooldown branched off step 3 of a longer stable run, into its own directory."""
    _shards(tmp_path / "main", 0)
    stable = _cfg(tmp_path, "stable", max_steps=6, schedule="wsd")
    stable.training.cooldown_start = 6                    # never cools down within 6 steps
    script.train(stable)
    branch = _cfg(tmp_path, "branch", max_steps=5, schedule="wsd")
    branch.training.cooldown_start = 3
    script.train(branch, resume=str(tmp_path / "stable" / "checkpoints" / "step_3.pt"))
    final = torch.load(tmp_path / "branch" / "checkpoints" / "final.pt", map_location="cpu",
                       weights_only=False)
    assert final["step"] == 5 and final["cooldown_start"] == 3


def test_a_branch_gets_its_own_wandb_run(tmp_path):
    """Resuming inside the output directory continues the W&B run; a checkpoint from elsewhere starts a new one."""
    out = tmp_path / "runs" / "branch"
    own = out / "checkpoints" / "step_3.pt"
    other = tmp_path / "runs" / "stable" / "checkpoints" / "step_3.pt"
    ckpt = {"wandb_id": "abc123"}
    assert script.resume_wandb_id(ckpt, str(own), str(out)) == "abc123"
    assert script.resume_wandb_id(ckpt, str(other), str(out)) is None
    assert script.resume_wandb_id(ckpt, None, str(out)) is None
    assert script.resume_wandb_id({}, str(own), str(out)) is None
    # a sibling directory whose name merely starts the same way is not the same run
    assert script.resume_wandb_id(ckpt, str(tmp_path / "runs" / "branch2" / "checkpoints" / "s.pt"), str(out)) is None


def test_extra_val_metrics_reach_final_summary(tmp_path, monkeypatch):
    _shards(tmp_path / "main", 0)
    _shards(tmp_path / "enc", 1, n_tokens=600)
    logged = {}
    monkeypatch.setattr(script, "log_metrics", lambda run, step, m: logged.update(m))
    cfg = _cfg(tmp_path, "x", max_steps=2, extra_val={"enc": str(tmp_path / "enc")})
    script.train(cfg)
    assert {"val/loss", "val_enc/loss", "final/loss", "final_enc/loss"} <= set(logged)
    assert logged["final/loss"] != logged["final_enc/loss"]
