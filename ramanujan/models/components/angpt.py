"""anGPT attention and feedforward components.

Reference: Franke et al., "Learning in Compact Spaces with Approximately
Normalized Transformers", arXiv:2505.22014, Table 1.

Both components expect an (approximately) unit-norm input along d_model and
return an exactly unit-norm output along d_model. They carry no
normalization layers and no biases; all linear weights are initialised with
unit-norm rows and must be kept in the unit ball during training (see
``ANGPTTransformer.project_parameters`` and ``WeightNormBoundCallback``).

Attention (per Table 1, extended to GQA):
    q, k, v <- W_{q,k,v} h * nu_qkv          nu_qkv = sqrt(d_model / d_head)
    q, k    <- normalize(q), normalize(k)    (along d_head, before RoPE;
                                              RoPE is a rotation, so order
                                              does not change the norm)
    A       <- softmax(q k^T * g)            g: per-head temperature
    h_a     <- W_p (A v) * nu_p              nu_p = sqrt(d_head / d_model)
    h_a     <- normalize(h_a)

SwiGLU:
    u, z    <- W_up h * nu_uz, W_gate h * nu_uz     nu_uz = sqrt(d_model / d_ff)
    h_m     <- u * SiLU(z) * nu_act
    h_m     <- W_down h_m * nu_d                    nu_d = sqrt(d_ff / d_model)
    h_m     <- normalize(h_m)

Because of the final exact normalization, nu_p, nu_act and nu_d do not
change the forward pass (normalize is invariant to positive rescaling); and
q, k are normalized, v only reaches the output through W_p and the final
normalize, so nu_qkv is inert too. The constants are kept for fidelity to
the paper and so that the exact output norms can later be dropped. The
factors that do shape the computation are the SiLU input scale
(nu_uz * gate_scale) and the attention temperature g.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base import BaseAttention, BaseFeedForward
from .normalized import (
    ScaledParameter,
    linear_norm_factor,
    normalize_rows_,
    swiglu_act_norm_factor,
)


def _to_bool_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
    """
    Convert a Candide attention mask (nonzero = attend) to a boolean
    SDPA mask (True = attend), broadcastable to (batch, heads, q, k).
    """
    if mask is None:
        return None
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dtype != torch.bool:
        mask = mask != 0
    return mask


class ANGPTAttention(BaseAttention):
    """
    Approximately normalized grouped-query attention.

    ``n_kv_heads=None`` gives standard multi-head attention; ``n_kv_heads=1``
    gives multi-query attention. The nu factors depend on the per-head
    output dimension, not on the number of heads, so GQA needs no change.

    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (default: n_heads)
        dropout: Attention-weight dropout (anGPT uses 0)
        bias: Must be False (biases break the norm bound)
        pruning_mask: Optional activation mask (Ramanujan sparsity hook)
        rope: Optional RotaryEmbedding (head_dim)
        qk_scale_init: Initial softmax temperature g (default: sqrt(head_dim))
        learnable_qk_scale: Learn one temperature per head (default: True)

    Example:
        >>> attn = ANGPTAttention(d_model=768, n_heads=12, n_kv_heads=4)
        >>> x = torch.nn.functional.normalize(torch.randn(2, 10, 768), dim=-1)
        >>> out, _ = attn(x)
        >>> out.norm(dim=-1)  # all ~1.0
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        pruning_mask: Optional[Tensor] = None,
        rope: Optional[nn.Module] = None,
        qk_scale_init: Optional[float] = None,
        learnable_qk_scale: bool = True,
        eps: float = 1e-6,
        **kwargs
    ):
        if bias:
            raise ValueError("ANGPTAttention does not support bias (it breaks the weight norm bound)")
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=False,
            pruning_mask=pruning_mask,
            **kwargs
        )

        if n_kv_heads is None:
            n_kv_heads = n_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.rope = rope
        self.eps = eps

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Constant normalization factors (Table 1)
        self.nu_qkv = linear_norm_factor(d_model, self.head_dim)
        self.nu_p = math.sqrt(self.head_dim / d_model)

        # Softmax temperature g. With unit q, k the logits are cosines in
        # [-1, 1]; sqrt(head_dim) matches the scale of standard QK-norm
        # (RMS-normalized q, k with the usual 1/sqrt(d) factor).
        if qk_scale_init is None:
            qk_scale_init = math.sqrt(self.head_dim)
        self.learnable_qk_scale = learnable_qk_scale
        if learnable_qk_scale:
            self.qk_scale = ScaledParameter(n_heads, init=qk_scale_init, scale=qk_scale_init)
        else:
            self.register_buffer(
                'qk_scale_const', torch.full((n_heads,), float(qk_scale_init)), persistent=False
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Gaussian init, then rescale every row to unit norm."""
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.normal_(proj.weight, mean=0.0, std=1.0)
            normalize_rows_(proj.weight)
        if self.learnable_qk_scale:
            self.qk_scale.reset_parameters()

    def _get_qk_scale(self) -> Tensor:
        return self.qk_scale() if self.learnable_qk_scale else self.qk_scale_const

    def _repeat_kv(self, x: Tensor) -> Tensor:
        """(batch, n_kv_heads, seq, head_dim) -> (batch, n_heads, seq, head_dim)"""
        if self.n_groups == 1:
            return x
        b, h, s, d = x.shape
        return x[:, :, None].expand(b, h, self.n_groups, s, d).reshape(b, h * self.n_groups, s, d)

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
            x: Input (batch, seq_len, d_model), approximately unit norm
            mask: Nonzero/True = attend; (seq, kv), (batch, seq, kv) or
                (batch, 1, seq, kv). None = full attention.
            position_ids: Position indices for RoPE
            past_key_value: Cached (k, v), k already normalized and rotated
            use_cache: Return updated cache

        Returns:
            (output with unit norm along d_model, cache or None)
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x) * self.nu_qkv
        k = self.k_proj(x) * self.nu_qkv
        v = self.v_proj(x) * self.nu_qkv

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm along the head dimension
        q = F.normalize(q, dim=-1, eps=self.eps)
        k = F.normalize(k, dim=-1, eps=self.eps)

        if self.rope is not None:
            q, k = self.rope(q, k, position_ids=position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        past_key_value = (k, v) if use_cache else None

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # SDPA divides logits by sqrt(head_dim); pre-multiply q so the
        # effective logits are g * cos(q, k). Avoids the `scale` kwarg
        # (torch >= 2.1) to keep torch 2.0 compatibility.
        g = self._get_qk_scale().to(q.dtype).view(1, self.n_heads, 1, 1)
        q = q * (g * math.sqrt(self.head_dim))

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=_to_bool_mask(mask),
            dropout_p=self.dropout if self.training else 0.0,
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)

        if self.pruning_mask.numel() > 1:
            output = output * self.pruning_mask

        output = self.out_proj(output) * self.nu_p
        output = F.normalize(output, dim=-1, eps=self.eps)

        return output, past_key_value

    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, '
            f'nu_qkv={self.nu_qkv:.4f}, nu_p={self.nu_p:.4f}, learnable_qk_scale={self.learnable_qk_scale}'
        )


class ANGPTSwiGLU(BaseFeedForward):
    """
    Approximately normalized SwiGLU feedforward.

    Weight naming follows ``SwiGLUFeedForward``: w1 = gate, w2 = up,
    w3 = down, so Ramanujan masks and checkpoint conversion map 1:1.

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension (default: 4 * d_model, as in the paper)
        dropout: Must be 0 in practice for anGPT; kept for interface parity
        bias: Must be False
        pruning_mask: Optional hidden-activation mask (d_ff,)
        act_norm_factor: Override for nu_act (default: Monte Carlo estimate
            under the actual SiLU input scale; the paper reports ~3.74)
        gate_scale: Extra constant multiplier on the SiLU input z only
            (default 1.0 = paper). With the paper's nu_uz and rows bounded
            in the unit ball, |z_i| <= nu_uz = sqrt(d_model / d_ff) (0.5 for
            d_ff = 4 d_model) for the whole run, so SiLU stays near its
            linear regime and the block behaves close to a bilinear GLU.
            nGPT rescales z by sqrt(d_model) for this reason; gate_scale
            lets you test that (e.g. gate_scale = sqrt(d_model)).

    Example:
        >>> ffn = ANGPTSwiGLU(d_model=768, d_ff=3072)
        >>> x = torch.nn.functional.normalize(torch.randn(2, 10, 768), dim=-1)
        >>> ffn(x).norm(dim=-1)  # all ~1.0
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        pruning_mask: Optional[Tensor] = None,
        act_norm_factor: Optional[float] = None,
        gate_scale: float = 1.0,
        eps: float = 1e-6,
        **kwargs
    ):
        if bias:
            raise ValueError("ANGPTSwiGLU does not support bias (it breaks the weight norm bound)")
        if gate_scale <= 0:
            raise ValueError(f"gate_scale must be > 0, got {gate_scale}")
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=False,
            pruning_mask=pruning_mask,
            **kwargs
        )
        self.eps = eps

        self.w1 = nn.Linear(d_model, self.d_ff, bias=False)  # gate (z)
        self.w2 = nn.Linear(d_model, self.d_ff, bias=False)  # up (u)
        self.w3 = nn.Linear(self.d_ff, d_model, bias=False)  # down

        self.nu_uz = linear_norm_factor(d_model, self.d_ff)
        self.gate_scale = float(gate_scale)
        base_std = 1.0 / math.sqrt(self.d_ff)
        self.nu_act = (
            float(act_norm_factor) if act_norm_factor is not None
            else swiglu_act_norm_factor(self.d_ff, gate_std=base_std * self.gate_scale, up_std=base_std)
        )
        self.nu_d = linear_norm_factor(self.d_ff, d_model)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Gaussian init, then rescale every row to unit norm."""
        for proj in (self.w1, self.w2, self.w3):
            nn.init.normal_(proj.weight, mean=0.0, std=1.0)
            normalize_rows_(proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model), approximately unit norm

        Returns:
            Output (batch, seq_len, d_model), unit norm along d_model
        """
        z = self.w1(x) * (self.nu_uz * self.gate_scale)
        u = self.w2(x) * self.nu_uz
        hidden = u * F.silu(z) * self.nu_act

        if self.pruning_mask.numel() > 1:
            hidden = hidden * self.pruning_mask

        if self.dropout_layer is not None:
            hidden = self.dropout_layer(hidden)

        output = self.w3(hidden) * self.nu_d
        return F.normalize(output, dim=-1, eps=self.eps)

    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, d_ff={self.d_ff}, nu_uz={self.nu_uz:.4f}, '
            f'gate_scale={self.gate_scale:.4f}, nu_act={self.nu_act:.4f}, nu_d={self.nu_d:.4f}'
        )
