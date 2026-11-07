"""EncoderBlock, local_window_mask and make_norm (ModernBERT layout)."""

import pytest
import torch
from torch import nn

from ramanujan.models.blocks.encoder import EncoderBlock, local_window_mask, make_norm
from ramanujan.models.components import RotaryEmbedding
from ramanujan.models.components.normalization import LayerNorm, RMSNorm


def test_window_mask_is_total_window():
    """window=4 -> |i-j| <= 2, as ModernBERT's local_attention."""
    m = local_window_mask(10, 4)[0, 0]
    i, j = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    assert torch.equal(m, (i - j).abs() <= 2)
    assert m.dtype == torch.bool and m[5].sum() == 5
    assert local_window_mask(10, 4).shape == (1, 1, 10, 10)


def test_make_norm():
    ln = make_norm("layer", 16, eps=1e-5, bias=False)
    assert isinstance(ln, LayerNorm) and ln.eps == 1e-5 and ln.bias is None
    assert isinstance(make_norm("rms", 16, eps=1e-6, bias=False), RMSNorm)
    with pytest.raises(ValueError):
        make_norm("batch", 16)


def _block(**kw):
    torch.manual_seed(0)
    args = dict(d_model=32, n_heads=4, d_ff=48, attention_type="mha", ffn_type="geglu",
                norm_type="layer", norm_eps=1e-5, norm_bias=False)
    args.update(kw)
    return EncoderBlock(**args).eval()


def test_shapes_and_no_bias():
    block = _block()
    out, cache = block(torch.randn(2, 9, 32))
    assert out.shape == (2, 9, 32) and cache is None
    assert not any(n.endswith("bias") for n, _ in block.named_parameters())


def test_attn_norm_can_be_removed():
    assert isinstance(_block(attn_norm=False).norm1, nn.Identity)
    assert isinstance(_block(attn_norm=True).norm1, LayerNorm)


def test_local_mask_limits_receptive_field():
    """With a window of 4, position 0 ignores tokens 3+ and sees tokens 1-2."""
    block = _block(rope=RotaryEmbedding(d_model=8, max_seq_len=16))
    x = torch.randn(1, 12, 32)
    mask = local_window_mask(12, 4)
    base, _ = block(x, mask=mask)

    far = x.clone()
    far[:, 3:] += torch.randn_like(far[:, 3:])
    out_far, _ = block(far, mask=mask)
    assert torch.allclose(out_far[:, 0], base[:, 0], atol=1e-6)

    near = x.clone()
    near[:, 2] += torch.randn_like(near[:, 2])
    out_near, _ = block(near, mask=mask)
    assert not torch.allclose(out_near[:, 0], base[:, 0], atol=1e-4)


def test_gqa_variant():
    block = _block(attention_type="gqa", n_kv_heads=2)
    assert block(torch.randn(2, 5, 32))[0].shape == (2, 5, 32)
