"""
Bidirectional encoder block with configurable normalization (ModernBERT layout).

    x = x + attention(norm1(x), mask)
    x = x + ffn(norm2(x))

Differences from TransformerBlock, all needed to match ModernBERT:
  - norm eps and norm bias are configurable (ModernBERT: LayerNorm, eps 1e-5,
    no bias);
  - the attention pre-norm can be removed (ModernBERT layer 0, whose input is
    already normalized by the embedding norm).

Parameter names (attention.*, ffn.*, norm1, norm2) are those of
TransformerBlock, so checkpoint tooling treats both blocks alike.
"""

from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from ..components import get_attention, get_feedforward
from ..components.normalization import LayerNorm, RMSNorm


def make_norm(norm_type: str, d_model: int, eps: float = 1e-6, bias: bool = True) -> nn.Module:
    """RMSNorm or LayerNorm with explicit eps; `bias` only applies to LayerNorm."""
    norm_type = norm_type.lower()
    if norm_type in ("rms", "rmsnorm"):
        return RMSNorm(d_model, eps=eps)
    if norm_type in ("layer", "layernorm"):
        return LayerNorm(d_model, eps=eps, bias=bias)
    raise ValueError(f"Unknown norm_type: {norm_type}. Available: 'rms', 'layer'")


def local_window_mask(seq_len: int, window: int, device=None) -> Tensor:
    """
    Boolean sliding-window mask, True = attend, shape (1, 1, seq_len, seq_len).

    `window` is the TOTAL window, as in ModernBERT's `local_attention`: query i
    attends to keys j with |i - j| <= window // 2 (window=128 -> 64 positions on
    each side plus itself).
    """
    pos = torch.arange(seq_len, device=device)
    band = (pos[:, None] - pos[None, :]).abs() <= window // 2
    return band.view(1, 1, seq_len, seq_len)


class EncoderBlock(nn.Module):
    """
    Pre-norm bidirectional transformer block.

    Args:
        d_model, n_heads, d_ff: dimensions
        attention_type: 'mha' (ModernBERT), 'gqa' or 'mqa'
        n_kv_heads: KV heads for GQA
        ffn_type: any get_feedforward type ('geglu' for ModernBERT)
        dropout: attention-probability and residual dropout
        norm_type, norm_eps, norm_bias: normalization layers
        attn_norm: if False, norm1 is the identity (ModernBERT layer 0)
        bias: bias in attention and feedforward projections
        rope: RotaryEmbedding (global and local layers may use different ones)
        qk_norm: RMSNorm on q and k over head_dim
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attention_type: str = "mha",
        n_kv_heads: Optional[int] = None,
        ffn_type: str = "geglu",
        dropout: float = 0.0,
        norm_type: str = "layer",
        norm_eps: float = 1e-5,
        norm_bias: bool = False,
        attn_norm: bool = True,
        bias: bool = False,
        rope: Optional[nn.Module] = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.norm_first = True

        self.attention = get_attention(
            attention_type=attention_type,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            bias=bias,
            rope=rope,
            qk_norm=qk_norm,
        )
        self.ffn = get_feedforward(
            ffn_type=ffn_type, d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias,
        )
        self.norm1 = make_norm(norm_type, d_model, norm_eps, norm_bias) if attn_norm else nn.Identity()
        self.norm2 = make_norm(norm_type, d_model, norm_eps, norm_bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        attn_out, _ = self.attention(self.norm1(x), mask=mask, position_ids=position_ids)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        if self.dropout is not None:
            ffn_out = self.dropout(ffn_out)
        return x + ffn_out, None
