"""Schedule, retention and config tests for train_encoder18c.py (no shards, no GPU)."""

import importlib
import math
from pathlib import Path

import pytest

script = importlib.import_module("ramanujan.training.train_encoder18c")
decoder = importlib.import_module("ramanujan.training.train_candide18c")

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def tcfg(**kw):
    base = dict(max_steps=1000, warmup_steps=100, learning_rate=1e-3, min_lr=1e-4)
    base.update(kw)
    return script.EncoderTrainingConfig(**base)


def old_cosine(step, cfg):
    """The schedule of train_encoder18c.py before the shared schedule was used."""
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    ratio = cfg.min_lr / cfg.learning_rate
    return ratio + 0.5 * (1 - ratio) * (1 + math.cos(math.pi * progress))


def test_cosine_is_unchanged():
    cfg = tcfg()
    for step in range(0, 1001, 7):
        assert script.get_lr(step, cfg) == pytest.approx(old_cosine(step, cfg), abs=1e-12)


def test_wsd_phases():
    cfg = tcfg(schedule="wsd", cooldown_frac=0.2)
    assert script.cooldown_start_step(cfg) == 800
    assert script.get_lr(50, cfg) == pytest.approx(0.5)
    assert all(script.get_lr(s, cfg) == 1.0 for s in (100, 400, 799))
    cool = [script.get_lr(s, cfg) for s in range(800, 1000)]
    assert cool[0] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(cool, cool[1:]))
    assert cool[-1] < 0.05


def test_wsd_cooldown_start_overrides_fraction():
    cfg = tcfg(schedule="wsd", cooldown_frac=0.2, cooldown_start=500)
    assert script.cooldown_start_step(cfg) == 500
    assert script.get_lr(499, cfg) == 1.0
    assert script.get_lr(600, cfg) < 1.0


def test_cosine_has_no_cooldown_start():
    assert script.cooldown_start_step(tcfg()) is None


def test_wsd_matches_the_decoder_schedule():
    kw = dict(max_steps=1000, warmup_steps=100, learning_rate=1e-3, min_lr=1e-4,
              schedule="wsd", cooldown_frac=0.3)
    enc, dec = script.EncoderTrainingConfig(**kw), decoder.TrainingConfig(**kw)
    for step in range(0, 1000, 13):
        assert script.get_lr(step, enc) == decoder.lr_factor(step, dec)


def test_keep_steps_must_be_multiples_of_save_every():
    script.check_retention(tcfg(save_every=500, keep_steps=[500, 1000, 5000]))
    with pytest.raises(ValueError, match="keep_steps"):
        script.check_retention(tcfg(save_every=500, keep_steps=[750]))


def test_prune_keeps_last_and_listed(tmp_path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for n in (1000, 2000, 3000, 4000, 5000):
        (ck / f"step_{n}.pt").write_bytes(b"")
    for name in ("pre_cooldown.pt", "final.pt"):
        (ck / name).write_bytes(b"")
    deleted = script.prune_checkpoints(str(tmp_path), keep_last=2, keep_steps=[1000])
    assert sorted(f.name for f in deleted) == ["step_2000.pt", "step_3000.pt"]
    assert sorted(f.name for f in ck.iterdir()) == [
        "final.pt", "pre_cooldown.pt", "step_1000.pt", "step_4000.pt", "step_5000.pt"]


def test_prune_is_off_by_default(tmp_path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "step_1000.pt").write_bytes(b"")
    assert script.prune_checkpoints(str(tmp_path), keep_last=None) == []
    assert (ck / "step_1000.pt").exists()


def test_v2g_config_loads_and_is_consistent():
    cfg = script.load_config(str(CONFIGS / "full" / "encoder_v2g.yaml"))
    assert cfg.model.vocab_size == 32132 and cfg.model.mask_token_id == 132
    assert cfg.training.schedule == "wsd" and cfg.training.compile
    assert "v2g" in cfg.data.shards_dir and "v2g" in cfg.data.tokenizer_path
    script.check_retention(cfg.training)
    tokens_per_step = (cfg.training.batch_size * cfg.training.gradient_accumulation_steps
                       * cfg.data.sequence_length)
    assert tokens_per_step == 65536


def test_v2g_model_section_matches_the_modernbert_config():
    new = script.load_config(str(CONFIGS / "full" / "encoder_v2g.yaml")).model
    old = script.load_config(str(CONFIGS / "encoder_modernbert18c.yaml")).model
    differ = {k for k in new.__dataclass_fields__ if getattr(new, k) != getattr(old, k)}
    assert differ == {"vocab_size", "mask_token_id", "qk_norm"}
    assert new.qk_norm and not old.qk_norm


def test_v2g_second_launch_schedule():
    cfg = script.load_config(str(CONFIGS / "full" / "encoder_v2g.yaml")).training
    assert cfg.learning_rate == 1e-3 and cfg.max_steps == 52000
    assert script.cooldown_start_step(cfg) == 31200
    assert cfg.batch_size * cfg.gradient_accumulation_steps == 64


def test_sources_main_is_shards_dir():
    data = script.DataConfig(shards_dir="/a", sources={"enc": "/b"}, mix={"main": 0.9, "enc": 0.1})
    assert script.resolve_sources(data) == {"main": "/a", "enc": "/b"}
    assert script.resolve_sources(script.DataConfig(shards_dir="/a")) == {"main": "/a"}


def test_sources_reject_unknown_mix_name_and_redefined_main():
    with pytest.raises(ValueError, match="unknown sources"):
        script.resolve_sources(script.DataConfig(shards_dir="/a", mix={"main": 1.0, "enc": 0.1}))
    with pytest.raises(ValueError, match="main"):
        script.resolve_sources(script.DataConfig(shards_dir="/a", sources={"main": "/b"}))


def test_v2g_config_mixes_in_the_encyclopedie():
    cfg = script.load_config(str(CONFIGS / "full" / "encoder_v2g.yaml"))
    assert set(cfg.data.mix) == {"main", "enc"} and sum(cfg.data.mix.values()) == pytest.approx(1.0)
    assert set(script.resolve_sources(cfg.data)) == {"main", "enc"}
    assert cfg.data.extra_val == {"enc": cfg.data.sources["enc"]}
