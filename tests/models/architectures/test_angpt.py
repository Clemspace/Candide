"""Tests for ANGPTConfig and ANGPTTransformer."""

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ramanujan.models.architectures.angpt import ANGPTConfig, ANGPTTransformer
from ramanujan.models.architectures.config import TransformerConfig
from ramanujan.models.components import RMSNorm, LayerNorm


VOCAB = 100


@pytest.fixture
def config():
    return ANGPTConfig(
        vocab_size=VOCAB,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        attention_type='gqa',
        d_ff=256,
        max_seq_len=64,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return ANGPTTransformer(config)


def _tokens(batch=2, seq=12):
    return torch.randint(0, VOCAB, (batch, seq))


# ============================================================================
# CONFIG
# ============================================================================

def test_config_defaults(config):
    assert config.residual_mode == 'approx'
    assert config.alpha_init == 0.05
    assert config.logit_scale_scale == pytest.approx(1 / math.sqrt(64))


def test_config_rejects_bias():
    with pytest.raises(ValueError):
        ANGPTConfig(vocab_size=VOCAB, d_model=64, n_heads=4, bias=True)


def test_config_rejects_learned_positions():
    with pytest.raises(ValueError):
        ANGPTConfig(vocab_size=VOCAB, d_model=64, n_heads=4, use_rope=False)


def test_config_rejects_bad_residual_mode():
    with pytest.raises(ValueError):
        ANGPTConfig(vocab_size=VOCAB, d_model=64, n_heads=4, residual_mode='x')


def test_config_roundtrip(config):
    restored = ANGPTConfig.from_dict(config.to_dict())
    assert restored == config


def test_model_rejects_plain_config():
    with pytest.raises(TypeError):
        ANGPTTransformer(TransformerConfig(vocab_size=VOCAB, d_model=64, n_heads=4))


# ============================================================================
# STRUCTURE AND INIT
# ============================================================================

def test_no_normalization_layers(model):
    assert not any(isinstance(m, (RMSNorm, LayerNorm, nn.LayerNorm)) for m in model.modules())


def test_all_weight_rows_unit_at_init(model):
    for sub in model.modules():
        if isinstance(sub, (nn.Linear, nn.Embedding)):
            norms = sub.weight.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), sub


def test_gqa_heads_wired(model):
    attn = model.blocks[0].attention
    assert attn.n_kv_heads == 2


def test_mha_config_uses_all_heads():
    cfg = ANGPTConfig(vocab_size=VOCAB, d_model=64, n_layers=1, n_heads=4, attention_type='mha')
    m = ANGPTTransformer(cfg)
    assert m.blocks[0].attention.n_kv_heads == 4


def test_pad_token_row_is_zero():
    cfg = ANGPTConfig(vocab_size=VOCAB, d_model=64, n_layers=1, n_heads=4, pad_token_id=0)
    m = ANGPTTransformer(cfg)
    assert torch.all(m.token_embedding.embedding.weight[0] == 0)


# ============================================================================
# FORWARD
# ============================================================================

def test_forward_shape(model):
    logits = model(_tokens())
    assert logits.shape == (2, 12, VOCAB)
    assert torch.isfinite(logits).all()


def test_forward_return_dict_and_cache(model, config):
    out = model(_tokens(), use_cache=True, return_dict=True)
    assert out['logits'].shape == (2, 12, VOCAB)
    assert len(out['past_key_values']) == config.n_layers


def test_initial_logits_bounded_by_logit_scale(model):
    """Unit-norm rows and an ~unit stream keep initial logits near [-s_z, s_z] = [-1, 1]."""
    logits = model(_tokens())
    assert logits.abs().max().item() < 1.5


def test_model_is_causal(model):
    x = _tokens(1, 10)
    l1 = model(x)
    x2 = x.clone()
    x2[:, 6:] = (x2[:, 6:] + 1) % VOCAB
    l2 = model(x2)
    assert torch.allclose(l1[:, :6], l2[:, :6], atol=1e-5)


def test_incremental_decoding_matches_full(model):
    model.eval()
    x = _tokens(1, 8)
    with torch.no_grad():
        full = model(x)
        _, cache = model(x[:, :7], use_cache=True)
        step, _ = model(x[:, 7:], past_key_values=cache, use_cache=True)
    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-4)


def test_generate_length(model):
    out = model.generate(_tokens(1, 4), max_new_tokens=5, top_k=10)
    assert out.shape == (1, 9)


# ============================================================================
# CONSTRAINTS
# ============================================================================

def test_project_parameters_bounds_rows(model):
    with torch.no_grad():
        model.blocks[0].ffn.w1.weight.mul_(3.0)
        model.token_embedding.embedding.weight[:5].mul_(2.0)
    assert model.max_weight_row_norm() > 1.5
    model.project_parameters()
    assert model.max_weight_row_norm() <= 1.0 + 1e-5


def test_project_parameters_leaves_short_rows(model):
    with torch.no_grad():
        model.lm_head.weight[3].mul_(0.5)
    before = model.lm_head.weight[3].clone()
    model.project_parameters()
    assert torch.equal(model.lm_head.weight[3], before)


def test_project_parameters_ignores_scale_parameters(model):
    with torch.no_grad():
        model.logit_scale.raw.fill_(5.0)
    model.project_parameters()
    assert torch.all(model.logit_scale.raw == 5.0)


def test_tied_embeddings_stay_tied():
    cfg = ANGPTConfig(vocab_size=VOCAB, d_model=64, n_layers=1, n_heads=4, tie_word_embeddings=True)
    m = ANGPTTransformer(cfg)
    assert m.lm_head.weight is m.token_embedding.embedding.weight
    with torch.no_grad():
        m.lm_head.weight.mul_(4.0)
    m.project_parameters()
    assert m.lm_head.weight is m.token_embedding.embedding.weight
    assert m.max_weight_row_norm() <= 1.0 + 1e-5


# ============================================================================
# TRAINING
# ============================================================================

def test_training_reduces_loss_without_wd_or_warmup(model):
    """Memorise one batch with Adam (wd=0, no warmup) + projection after each step."""
    torch.manual_seed(0)
    x = _tokens(4, 16)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0.0)

    def loss_fn():
        logits = model(x[:, :-1])
        return F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))

    initial = loss_fn().item()
    for _ in range(100):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        model.project_parameters()
        assert torch.isfinite(loss)
    final = loss_fn().item()

    assert final < 0.9 * initial
    assert model.max_weight_row_norm() <= 1.0 + 1e-5


def test_save_and_load_roundtrip(model):
    model.eval()
    x = _tokens()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'angpt.pt'
        model.save_pretrained(str(path))
        loaded = ANGPTTransformer.from_pretrained(str(path)).eval()
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)
