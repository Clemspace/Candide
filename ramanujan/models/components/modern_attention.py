"""
modern_attention.py — Flash Attention + alternating local/global attention.

Three additions over the baseline attention.py:

1. FlashGQA — GroupedQueryAttention using F.scaled_dot_product_attention
   (Flash Attention 2 kernel on PyTorch 2.0+, ~2x throughput, no code change needed
   for CUDA — PyTorch dispatches automatically when inputs are on GPU + bf16/fp16)

2. LocalAttention — sliding window attention (attends to ±window_size tokens)
   Cheap per-layer, handles local syntactic dependencies well.

3. GlobalAttention — full bidirectional attention (same as FlashGQA, explicit class
   for clarity in alternating block config)

Update:   ramanujan/components/__init__.py to export FlashGQA, LocalAttention, GlobalAttention
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Flash GQA (replaces GroupedQueryAttention for encoder) ───────────────────

class FlashGQA(nn.Module):
    """
    Grouped Query Attention using F.scaled_dot_product_attention.

    On PyTorch 2.0+ with CUDA + bf16/fp16, this automatically uses the
    Flash Attention 2 kernel — no extra dependencies needed.

    For encoder use (bidirectional): is_causal=False always.
    For decoder use: pass is_causal=True.

    Key differences from GroupedQueryAttention:
    - No manual softmax / score computation — delegated to SDPA kernel
    - No attn_dropout module (SDPA handles dropout internally)
    - ~2x throughput on A40 for typical sequence lengths
    - Same interface: forward(x, mask, position_ids, ...) unchanged

    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key-value heads (GQA; None = same as n_heads)
        dropout: Attention dropout probability (applied inside SDPA)
        bias: Whether to use bias in projections
        rope: Optional RotaryEmbedding module
        is_causal: If True, apply causal mask (for decoder). Default: False (encoder).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        rope: Optional[nn.Module] = None,
        is_causal: bool = False,
        **kwargs,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d_model // n_heads
        self.n_groups = self.n_heads // self.n_kv_heads
        self.dropout = dropout
        self.rope = rope
        self.is_causal = is_causal
        self.scale = self.head_dim ** -0.5

        assert n_heads % self.n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else None

        self._init_weights()

    def _init_weights(self):
        # Depth-scaled initialization (ModernBERT / GPT-NeoX style)
        # std scales with 1/sqrt(2*n_layers) but we don't know n_layers here;
        # caller (TransformerBlock or encoder) applies depth scaling post-init
        std = self.d_model ** -0.5
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        """Expand KV heads to match Q heads. (batch, kv_heads, seq, dim) → (batch, q_heads, seq, dim)"""
        if self.n_groups == 1:
            return x
        b, h, s, d = x.shape
        return x[:, :, None, :, :].expand(b, h, self.n_groups, s, d).reshape(b, h * self.n_groups, s, d)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k, position_ids=position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_kv = (k, v) if use_cache else None

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Flash Attention via SDPA — dispatches to FlashAttention2 kernel on CUDA
        # mask here should be additive (0=attend, -inf=ignore) or None
        attn_mask = None
        if mask is not None:
            # Convert 1/0 mask to additive mask expected by SDPA
            # SDPA expects: True = ignore, or additive float mask
            attn_mask = mask.to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal and mask is None,  # SDPA handles causal internally
            scale=self.scale,
        )

        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        out = self.out_proj(out)

        if self.resid_dropout is not None:
            out = self.resid_dropout(out)

        return out, past_kv


# ── Local (sliding window) attention ─────────────────────────────────────────

class LocalAttention(nn.Module):
    """
    Sliding window attention — each token attends to ±window_size neighbors.

    ModernBERT uses window_size=128 for local layers. For historical French
    with long periodic sentences, 64-128 is appropriate.

    Implementation: we compute full attention but apply a sliding window mask.
    True sparse local attention (like Longformer) requires custom CUDA kernels;
    this version uses the same SDPA path with a precomputed band mask.
    For seq_len=512 this is fine — the full attention matrix is small enough.
    For seq_len > 2048 you'd want the sparse kernel.

    Args:
        d_model, n_heads, n_kv_heads, dropout, bias, rope: same as FlashGQA
        window_size: One-sided window — token i attends to [i-w, i+w]. Default: 128.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        rope: Optional[nn.Module] = None,
        window_size: int = 128,
        **kwargs,
    ):
        super().__init__()

        self.window_size = window_size
        # LocalAttention is always bidirectional (encoder only)
        # Reuse FlashGQA with is_causal=False; mask handles windowing
        self.attn = FlashGQA(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            bias=bias,
            rope=rope,
            is_causal=False,
        )
        self._window_mask_cache: dict = {}  # cache masks by seq_len

    def _get_window_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Build sliding window mask: 1 where |i-j| <= window_size, 0 elsewhere.
        Cached by seq_len to avoid recomputation.

        Returns: (1, 1, seq_len, seq_len) float mask
        """
        key = (seq_len, device)
        if key not in self._window_mask_cache:
            # Band mask: positions within window get 1, outside get 0
            i = torch.arange(seq_len, device=device).unsqueeze(1)
            j = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = (torch.abs(i - j) <= self.window_size).float()
            self._window_mask_cache[key] = mask.unsqueeze(0).unsqueeze(0)
        return self._window_mask_cache[key]

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        b, s, _ = x.shape

        window_mask = self._get_window_mask(s, x.device)

        # Combine with padding mask if provided
        if mask is not None:
            # mask: (batch, 1, 1, seq_len) padding mask (1=attend)
            combined = window_mask * mask  # zero out padded + out-of-window
        else:
            combined = window_mask.expand(b, 1, s, s)

        return self.attn(
            x,
            mask=combined,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )


# ── Global attention (explicit alias for clarity in block configs) ────────────

class GlobalAttention(FlashGQA):
    """
    Full bidirectional attention — alias of FlashGQA with is_causal=False.

    Explicit class so block configs can say attention_type='global'
    and the intent is unambiguous.
    """
    def __init__(self, *args, **kwargs):
        kwargs['is_causal'] = False
        super().__init__(*args, **kwargs)