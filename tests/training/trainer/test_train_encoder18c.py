"""Tests for ramanujan/training/train_encoder18c.py (no shards, no GPU needed)."""

import contextlib
import importlib
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from ramanujan.data.loaders.mlm_collator import MLMCollator

script = importlib.import_module("ramanujan.training.train_encoder18c")

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
V = 64
NULL = contextlib.nullcontext()


def tiny_model_cfg(**kw):
    """The ModernBERT YAML's model section, shrunk."""
    raw = yaml.safe_load((CONFIGS / "encoder_modernbert18c.yaml").read_text())["model"]
    raw.update(vocab_size=V, mask_token_id=V - 1, d_model=32, n_layers=4, n_heads=4,
               d_ff=48, max_seq_len=32, local_attention=4)
    raw.update(kw)
    return script.EncoderModelConfig(**raw)


def collator(p=0.3):
    return MLMCollator(vocab_size=V, mask_token_id=V - 1, mlm_probability=p)


def val_loader(n_batches=3, batch=4, seq=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(2, V - 1, (batch, seq), generator=g)}
            for _ in range(n_batches)]


# ── Configs ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["encoder18c.yaml", "encoder_modernbert18c.yaml"])
def test_yaml_keys_all_known(name):
    path = CONFIGS / name
    if not path.exists():
        pytest.skip(f"{name} not in this checkout")
    raw = yaml.safe_load(path.read_text())
    assert script.unknown_keys(raw) == []
    script.load_config(str(path))


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  d_model: 64\n  local_windw: 3\n")
    with pytest.raises(ValueError, match="model.local_windw"):
        script.load_config(str(p))


def test_modernbert_yaml_builds_layer_pattern():
    from ramanujan.models.blocks.encoder import EncoderBlock
    model = script.setup_model(tiny_model_cfg())
    assert model.layer_is_global == [True, False, False, True]
    assert all(isinstance(b, EncoderBlock) for b in model.blocks)
    assert model.mlm_head.decoder.weight is model.token_embedding.embedding.weight


def test_every_model_field_reaches_the_encoder_config():
    cfg = tiny_model_cfg()
    model = script.setup_model(cfg)
    built = model.config.to_dict()
    for k, v in asdict(cfg).items():
        assert built[k] == v, k


# ── Loss and evaluation ───────────────────────────────────────────────────────

def test_sparse_loss_equals_dense_loss():
    torch.manual_seed(0)
    model = script.setup_model(tiny_model_cfg()).eval()
    torch.manual_seed(1)
    batch = collator()(val_loader()[0]["input_ids"])
    dense = script.mlm_loss(model, batch, "cpu", False, NULL)
    sparse = script.mlm_loss(model, batch, "cpu", True, NULL)
    assert torch.allclose(dense[0], sparse[0], rtol=1e-5)
    assert int(dense[1]) == int(sparse[1]) == int((batch["labels"] != -100).sum())
    assert int(dense[2]) == int(sparse[2])


def test_evaluate_is_token_weighted_and_deterministic():
    torch.manual_seed(0)
    model = script.setup_model(tiny_model_cfg()).eval()
    loader, col = val_loader(), collator()

    a = script.evaluate(model, loader, col, "cpu", None, NULL, sparse=True, mask_seed=3)
    b = script.evaluate(model, loader, col, "cpu", None, NULL, sparse=False, mask_seed=3)
    assert a["val/loss"] == pytest.approx(b["val/loss"], rel=1e-5)
    assert a["val/masked_tokens"] == b["val/masked_tokens"] > 0

    # manual token-weighted mean with the same masks
    total, n = 0.0, 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        for batch in loader:
            s, k, _ = script.mlm_loss(model, col(batch["input_ids"]), "cpu", False, NULL)
            total += s.item()
            n += int(k)
    assert a["val/loss"] == pytest.approx(total / n, rel=1e-5)
    assert a["val/perplexity"] == pytest.approx(math.exp(total / n), rel=1e-5)


def test_evaluate_leaves_training_rng_untouched():
    model = script.setup_model(tiny_model_cfg()).eval()
    torch.manual_seed(123)
    expected = torch.rand(5)
    torch.manual_seed(123)
    script.evaluate(model, val_loader(), collator(), "cpu", None, NULL)
    assert torch.equal(torch.rand(5), expected)
    assert model.training  # evaluate() restores train mode


def test_evaluate_respects_n_steps():
    model = script.setup_model(tiny_model_cfg())
    one = script.evaluate(model, val_loader(), collator(), "cpu", 1, NULL)
    full = script.evaluate(model, val_loader(), collator(), "cpu", None, NULL)
    assert 0 < one["val/masked_tokens"] < full["val/masked_tokens"]


# ── Tokenizer checks ──────────────────────────────────────────────────────────

def _tokenizer(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    vocab = {"<|pad|>": 0, "<|unk|>": 1, "<|bos|>": 2, "<|eos|>": 3, "a": 4, "[MASK]": 5}
    tok = tokenizers.Tokenizer(WordLevel(vocab, unk_token="<|unk|>"))
    path = tmp_path / "tok.json"
    tok.save(str(path))
    return path


def test_resolve_special_ids(tmp_path):
    cfg = script.FullEncoderConfig()
    cfg.data.tokenizer_path = str(_tokenizer(tmp_path))
    cfg.model.vocab_size, cfg.model.mask_token_id = 6, 5
    assert script.resolve_special_ids(cfg) == [2, 3]


def test_resolve_special_ids_rejects_wrong_mask_id(tmp_path):
    cfg = script.FullEncoderConfig()
    cfg.data.tokenizer_path = str(_tokenizer(tmp_path))
    cfg.model.vocab_size, cfg.model.mask_token_id = 6, 4
    with pytest.raises(ValueError, match="not a mask token"):
        script.resolve_special_ids(cfg)


def test_resolve_special_ids_rejects_small_vocab(tmp_path):
    cfg = script.FullEncoderConfig()
    cfg.data.tokenizer_path = str(_tokenizer(tmp_path))
    cfg.model.vocab_size, cfg.model.mask_token_id = 5, 5
    with pytest.raises(ValueError, match="vocab_size"):
        script.resolve_special_ids(cfg)


def test_missing_tokenizer_only_warns():
    cfg = script.FullEncoderConfig()
    cfg.data.tokenizer_path = "/nonexistent/tokenizer.json"
    assert script.resolve_special_ids(cfg) == []
