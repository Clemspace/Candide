"""anGPT transformer block.

    h_a <- ANGPTAttention(h)                 (unit norm)
    h   <- (h + alpha_a (h_a - h)) * nu(alpha_a)
    h_m <- ANGPTSwiGLU(h)                    (unit norm)
    h   <- (h + alpha_m (h_m - h)) * nu(alpha_m)

No normalization layers. The learnable LERP rates alpha replace the
per-element affine weight of the pre-norm, and the residual stream stays
approximately on the unit sphere instead of growing with depth.

Reference: Franke et al., arXiv:2505.22014, Table 1 and Section 4.3.
"""

from typing import Optional, Tuple

from torch import nn, Tensor

from ..components.base import BaseTransformerBlock
from ..components.angpt import ANGPTAttention, ANGPTSwiGLU
from ..components.normalized import LERPResidual


class ANGPTBlock(BaseTransformerBlock):
    """
    Approximately normalized transformer block.

    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        d_ff: FFN hidden dimension (default: 4 * d_model)
        n_kv_heads: Number of KV heads (default: n_heads)
        rope: Optional shared RotaryEmbedding
        dropout: Attention dropout (anGPT uses 0)
        alpha_init: Effective initial LERP rate
        alpha_scale: Raw stored LERP rate (effective LR multiplier = init / scale)
        residual_mode: 'approx' (anGPT), 'exact' (nGPT-style retraction)
            or 'paper_literal' (ablation of the printed nu(alpha) formula)
        qk_scale_init: Initial attention temperature (default: sqrt(head_dim))
        learnable_qk_scale: Learn one temperature per head
        act_norm_factor: Override for the SwiGLU activation constant
        gate_scale: Multiplier on the SiLU input (1.0 = paper; see ANGPTSwiGLU)

    Example:
        >>> block = ANGPTBlock(d_model=768, n_heads=12, n_kv_heads=4, d_ff=3072)
        >>> x = torch.nn.functional.normalize(torch.randn(2, 10, 768), dim=-1)
        >>> out, _ = block(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
        rope: Optional[nn.Module] = None,
        dropout: float = 0.0,
        alpha_init: float = 0.05,
        alpha_scale: float = 0.01,
        residual_mode: str = 'approx',
        qk_scale_init: Optional[float] = None,
        learnable_qk_scale: bool = True,
        act_norm_factor: Optional[float] = None,
        gate_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            norm_first=False,  # no norm layers; kept for interface parity
            **kwargs
        )

        self.attention = ANGPTAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            rope=rope,
            qk_scale_init=qk_scale_init,
            learnable_qk_scale=learnable_qk_scale,
        )
        self.ffn = ANGPTSwiGLU(
            d_model=d_model,
            d_ff=self.d_ff,
            act_norm_factor=act_norm_factor,
            gate_scale=gate_scale,
        )

        self.residual_attn = LERPResidual(
            d_model, alpha_init=alpha_init, alpha_scale=alpha_scale, mode=residual_mode
        )
        self.residual_ffn = LERPResidual(
            d_model, alpha_init=alpha_init, alpha_scale=alpha_scale, mode=residual_mode
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            x: Residual stream (batch, seq_len, d_model), approximately unit norm
            mask: Attention mask (nonzero = attend)
            position_ids: Position indices
            past_key_value: Cached (k, v)
            use_cache: Return updated cache

        Returns:
            (updated residual stream, cache or None)
        """
        h_a, past_key_value = self.attention(
            x,
            mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = self.residual_attn(x, h_a)

        h_m = self.ffn(x)
        x = self.residual_ffn(x, h_m)

        return x, past_key_value

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}'
