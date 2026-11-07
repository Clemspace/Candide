"""CandideEncoder: legacy path unchanged, ModernBERT layer-pattern path."""

import pytest
import torch
from torch import nn

from ramanujan.models.architectures.encoder import (
    CandideEncoder, CandideEncoderConfig, ENCODER_DEFAULTS,
)
from ramanujan.models.blocks import TransformerBlock
from ramanujan.models.blocks.encoder import EncoderBlock
from ramanujan.models.components.normalization import LayerNorm, RMSNorm

V, D, H, FF = 64, 32, 4, 48


def modern_cfg(**kw):
    args = dict(
        vocab_size=V, d_model=D, n_layers=4, n_heads=H, d_ff=FF, max_seq_len=64,
        dropout=0.0, attention_type="mha", ffn_type="geglu", norm_type="layer",
        global_attn_every_n_layers=3, local_attention=4,
        rope_theta=160000.0, local_rope_theta=10000.0,
        norm_eps=1e-5, norm_bias=False, embedding_norm=True, bias=False,
        head_bias=False, decoder_bias=True, add_pooler=False, mask_token_id=63,
    )
    args.update(kw)
    return CandideEncoderConfig(**args)


def legacy_cfg(**kw):
    args = dict(vocab_size=V, d_model=D, n_layers=2, n_heads=H, n_kv_heads=2, d_ff=FF,
                max_seq_len=64, attention_type="gqa", ffn_type="swiglu", norm_type="rms")
    args.update(kw)
    return CandideEncoderConfig(**args)


def make(cfg, seed=0):
    torch.manual_seed(seed)
    return CandideEncoder(cfg).eval()


# ── Legacy path ───────────────────────────────────────────────────────────────

def test_legacy_defaults_unchanged():
    m = make(legacy_cfg())
    assert all(isinstance(b, TransformerBlock) for b in m.blocks)
    assert m.layer_is_global is None and m.embedding_norm is None
    assert all(b.attention.rope is m.rope for b in m.blocks)
    assert isinstance(m.final_norm, RMSNorm) and m.final_norm.eps == 1e-6
    assert m.mlm_head.norm.eps == 1e-12 and m.mlm_head.norm.bias is not None
    assert m.mlm_head.dense.bias is not None and m.mlm_head.decoder.bias is not None
    assert m.pooler is not None
    assert m.mlm_head.decoder.weight is m.token_embedding.embedding.weight


def test_legacy_layer_norm_final_norm_has_bias():
    m = make(legacy_cfg(norm_type="layer"))
    assert isinstance(m.final_norm, LayerNorm) and m.final_norm.bias is not None


def test_legacy_forward():
    m = make(legacy_cfg())
    logits, pooled = m(torch.randint(0, V, (2, 10)))
    assert logits.shape == (2, 10, V) and pooled.shape == (2, D)


# ── Layer pattern ─────────────────────────────────────────────────────────────

def test_layer_pattern_and_ropes():
    m = make(modern_cfg(n_layers=7))
    assert m.layer_is_global == [True, False, False, True, False, False, True]
    assert all(isinstance(b, EncoderBlock) for b in m.blocks)
    for b, g in zip(m.blocks, m.layer_is_global):
        assert b.attention.rope is (m.rope if g else m.local_rope)
    assert m.rope.theta == 160000.0 and m.local_rope.theta == 10000.0


def test_single_rope_when_local_theta_unset():
    m = make(modern_cfg(local_rope_theta=None))
    assert m.local_rope is None
    assert all(b.attention.rope is m.rope for b in m.blocks)


def test_norms_and_biases_match_modernbert():
    m = make(modern_cfg())
    assert isinstance(m.blocks[0].norm1, nn.Identity)
    assert all(isinstance(b.norm1, LayerNorm) for b in m.blocks[1:])
    assert isinstance(m.embedding_norm, LayerNorm) and m.embedding_norm.eps == 1e-5
    assert m.mlm_head.norm.eps == 1e-5
    biases = [n for n, _ in m.named_parameters() if n.endswith("bias")]
    assert biases == ["mlm_head.decoder.bias"]


def test_param_count():
    m = make(modern_cfg())
    L = 4
    layer = 4 * D * D + 3 * D * FF + 2 * D          # q,k,v,o + w1,w2,w3 + 2 norm weights
    expected = (L * layer - D                       # layer 0 has no attention norm
                + V * D + D + D                     # tied embedding, embedding norm, final norm
                + D * D + D + V)                    # head dense, head norm, decoder bias
    assert m.get_num_params() == expected


def test_layer_masks():
    m = make(modern_cfg())
    g, l = m.get_layer_masks(10, None, "cpu")
    assert g is None and l.shape == (1, 1, 10, 10)
    g, l = m.get_layer_masks(10, torch.ones(2, 10, dtype=torch.long), "cpu")
    assert g is None
    am = torch.ones(2, 10, dtype=torch.long)
    am[1, 6:] = 0
    g, l = m.get_layer_masks(10, am, "cpu")
    assert g.shape == (2, 1, 1, 10) and l.shape == (2, 1, 10, 10)
    assert l[1, 0, 9, 9] and not l[1, 0, 9, 8]   # padded query: itself only


def test_forward_shapes_and_sparse_prediction():
    m = make(modern_cfg())
    ids = torch.randint(0, V, (2, 12))
    logits, pooled = m(ids)
    assert logits.shape == (2, 12, V) and pooled is None
    masked = torch.rand(2, 12) < 0.3
    masked[0, 0] = True
    sparse, _ = m(ids, masked_positions=masked)
    assert sparse.shape == (int(masked.sum()), V)
    assert torch.allclose(sparse, logits[masked], atol=1e-5)


def test_padding_does_not_leak_and_no_nan():
    """Valid positions ignore pad tokens; a pad run longer than the window gives no NaN."""
    m = make(modern_cfg())
    ids = torch.randint(0, V, (1, 16))
    am = torch.ones(1, 16, dtype=torch.long)
    am[:, 6:] = 0                                   # 10 pad positions > local_attention // 2
    out1, _ = m(ids, attention_mask=am)
    ids2 = ids.clone()
    ids2[:, 6:] = torch.randint(0, V, (1, 10))
    out2, _ = m(ids2, attention_mask=am)
    assert torch.isfinite(out1).all()
    assert torch.allclose(out1[:, :6], out2[:, :6], atol=1e-5)


def test_global_layer_sees_whole_sequence():
    """Layer 0 is global, so position 0 depends on a token beyond the local window."""
    m = make(modern_cfg(n_layers=1))
    ids = torch.randint(0, V - 1, (1, 20))
    out1, _ = m(ids)
    ids[0, 19] = V - 1
    out2, _ = m(ids)
    assert not torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)


def test_gradients_reach_all_trained_params():
    m = make(modern_cfg()).train()
    logits, _ = m(torch.randint(0, V, (2, 12)))
    logits.sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert missing == []


def test_megatron_init():
    torch.manual_seed(0)
    m = CandideEncoder(modern_cfg(d_model=128, d_ff=256, n_layers=8, init_scheme="megatron"))
    q = m.blocks[1].attention.q_proj.weight
    o = m.blocks[1].attention.out_proj.weight
    w3 = m.blocks[1].ffn.w3.weight
    assert q.abs().max() <= 2 * 0.02 + 1e-6
    assert abs(q.std().item() - 0.02 * 0.88) < 0.002      # truncated at 2 sigma: std ~0.88 sigma
    out_std = 0.02 / 4.0                                   # sqrt(2 * 8) = 4
    assert o.abs().max() <= 2 * out_std + 1e-6 and w3.abs().max() <= 2 * out_std + 1e-6


# ── Config ────────────────────────────────────────────────────────────────────

def test_config_roundtrip_and_save_load(tmp_path):
    cfg = modern_cfg()
    d = cfg.to_dict()
    assert all(k in d for k in ENCODER_DEFAULTS)
    assert CandideEncoderConfig.from_dict(d).to_dict() == d
    m = make(cfg)
    path = tmp_path / "enc.pt"
    m.save_pretrained(str(path))
    m2 = CandideEncoder.from_pretrained(str(path)).eval()
    ids = torch.randint(0, V, (2, 9))
    assert torch.allclose(m(ids)[0], m2(ids)[0])
    assert m2.layer_is_global == m.layer_is_global


def test_from_dict_does_not_mutate_input():
    d = modern_cfg().to_dict()
    snapshot = dict(d)
    CandideEncoderConfig.from_dict(d)
    assert d == snapshot


@pytest.mark.parametrize("bad", [
    dict(attention_type="alternating"),
    dict(global_attn_every_n_layers=0),
    dict(local_attention=1),
    dict(init_scheme="xavier"),
])
def test_invalid_configs(bad):
    with pytest.raises(ValueError):
        modern_cfg(**bad)
