"""
CandideEncoder — Bidirectional encoder for MLM pretraining.

Reuses the same attention, FFN and normalization components as
RamanujanTransformer (decoder), but:
  - No causal mask → full bidirectional attention
  - MLM head instead of LM head
  - Pooler for sentence-level tasks (CLS pooling)

Two block layouts:
  - legacy (global_attn_every_n_layers=None, the default): TransformerBlock,
    or AlternatingTransformerBlock with attention_type='alternating'.
    Unchanged; existing configs and checkpoints load as before.
  - layer pattern (global_attn_every_n_layers=N): EncoderBlock, layer i global
    iff i % N == 0, the others sliding-window with a total window of
    `local_attention` tokens, optionally their own RoPE theta. With the options
    of configs/encoder_modernbert18c.yaml this is the ModernBERT architecture
    (Warner et al., 2024, arXiv:2412.13663).
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from .config import TransformerConfig
from ..components import TokenEmbedding, RotaryEmbedding, LearnedPositionalEmbedding
from ..blocks import TransformerBlock
from ..blocks.encoder import EncoderBlock, local_window_mask, make_norm


# Encoder-only fields, with their defaults. Defaults reproduce the encoder as
# it was before these options existed.
ENCODER_DEFAULTS = {
    "mask_token_id": 4,
    "add_pooler": True,
    "classifier_dropout": 0.1,
    # Legacy alternating blocks (attention_type='alternating')
    "local_window_size": 128,
    # Layer pattern (ModernBERT)
    "global_attn_every_n_layers": None,  # None = legacy blocks
    "local_attention": 128,              # total sliding window (|i-j| <= 64)
    "local_rope_theta": None,            # None = rope_theta on every layer
    # Normalization
    "norm_eps": None,                    # None = 1e-6 in blocks/final norm, 1e-12 in MLM head
    "norm_bias": True,                   # LayerNorm bias (no effect on RMSNorm)
    "embedding_norm": False,             # norm after token embedding; drops layer-0 attn norm
    # MLM head
    "head_bias": True,                   # bias of the head's dense layer
    "decoder_bias": True,                # bias of the (tied) vocabulary projection
    # Init
    "init_scheme": "normal",             # 'normal' (N(0, 0.02)) or 'megatron'
}

INIT_SCHEMES = ("normal", "megatron")


class CandideEncoderConfig(TransformerConfig):
    """
    Encoder config: TransformerConfig plus the fields in ENCODER_DEFAULTS.

    Everything else (d_model, n_layers, n_heads, GQA, RoPE, FFN type...)
    inherits from TransformerConfig unchanged.
    """

    def __init__(self, **kwargs):
        encoder_kwargs = {k: kwargs.pop(k, v) for k, v in ENCODER_DEFAULTS.items()}
        super().__init__(**kwargs)
        for k, v in encoder_kwargs.items():
            setattr(self, k, v)
        self._validate_encoder()

    def _validate_encoder(self):
        if self.init_scheme not in INIT_SCHEMES:
            raise ValueError(f"Unknown init_scheme: {self.init_scheme}. Available: {INIT_SCHEMES}")
        n = self.global_attn_every_n_layers
        if n is not None:
            if n < 1:
                raise ValueError(f"global_attn_every_n_layers must be >= 1, got {n}")
            if self.attention_type == "alternating":
                raise ValueError(
                    "global_attn_every_n_layers replaces attention_type='alternating'; "
                    "set attention_type to 'mha' or 'gqa'"
                )
            if self.local_attention < 2:
                raise ValueError(f"local_attention is the total window, got {self.local_attention}")

    @property
    def uses_layer_pattern(self) -> bool:
        return self.global_attn_every_n_layers is not None

    def to_dict(self) -> dict:
        d = super().to_dict()  # dataclass fields only
        d.update({k: getattr(self, k) for k in ENCODER_DEFAULTS})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CandideEncoderConfig":
        return cls(**dict(d))


class MLMHead(nn.Module):
    """
    Masked Language Modeling head: Linear → GELU → LayerNorm → Linear (tied).

    Defaults reproduce the original head (biases, LayerNorm eps 1e-12).
    ModernBERT: dense_bias=False, norm_eps=1e-5, norm_bias=False, decoder_bias=True.

    Args:
        d_model: Hidden dimension
        vocab_size: Output vocabulary size
        embedding_weight: Optional weight tensor to tie with token embeddings
    """

    def __init__(self, d_model: int, vocab_size: int,
                 embedding_weight: Optional[Tensor] = None,
                 dense_bias: bool = True, norm_eps: float = 1e-12,
                 norm_bias: bool = True, decoder_bias: bool = True):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model, bias=dense_bias)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model, eps=norm_eps, bias=norm_bias)
        self.decoder = nn.Linear(d_model, vocab_size, bias=decoder_bias)
        if embedding_weight is not None:
            self.decoder.weight = embedding_weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        """(..., d_model) → (..., vocab_size)"""
        return self.decoder(self.norm(self.act(self.dense(hidden_states))))


class CandidePooler(nn.Module):
    """
    CLS-token pooler for sentence-level representations.

    Takes the first token hidden state and projects it. Only meaningful when
    position 0 holds a [CLS]-like token; raw shard windows start mid-text, so
    the pooler is not trained by MLM pretraining on shards.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        cls_token = hidden_states[:, 0, :]
        return self.act(self.dense(self.dropout(cls_token)))


class CandideEncoder(nn.Module):
    """
    Candide Encoder — bidirectional transformer for MLM pretraining.

        Input → Token Embedding → [Embedding norm] → N blocks (bidirectional,
        global or sliding-window) → Final Norm → MLM Head
                                               └→ Pooler (optional)

    Example (ModernBERT-base shape):
        >>> config = CandideEncoderConfig(
        ...     vocab_size=32133, d_model=768, n_layers=22, n_heads=12, d_ff=1152,
        ...     attention_type='mha', ffn_type='geglu', norm_type='layer',
        ...     global_attn_every_n_layers=3, local_attention=128,
        ...     rope_theta=160000.0, local_rope_theta=10000.0,
        ...     norm_eps=1e-5, norm_bias=False, embedding_norm=True,
        ...     head_bias=False, add_pooler=False, max_seq_len=1024,
        ... )
        >>> model = CandideEncoder(config)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers

        opt = {k: getattr(config, k, v) for k, v in ENCODER_DEFAULTS.items()}
        self.mask_token_id = opt["mask_token_id"]
        self.add_pooler = opt["add_pooler"]
        self.global_every = opt["global_attn_every_n_layers"]
        self.local_attention = opt["local_attention"]
        norm_eps = opt["norm_eps"] if opt["norm_eps"] is not None else 1e-6
        head_norm_eps = opt["norm_eps"] if opt["norm_eps"] is not None else 1e-12

        # ── Embeddings ────────────────────────────────────────────────────────
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.pad_token_id,
        )

        self.local_rope = None
        if config.use_rope:
            self.rope = RotaryEmbedding(
                d_model=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
            local_theta = opt["local_rope_theta"]
            if self.global_every is not None and local_theta is not None and local_theta != config.rope_theta:
                self.local_rope = RotaryEmbedding(
                    d_model=config.head_dim,
                    max_seq_len=config.max_seq_len,
                    theta=local_theta,
                )
            self.position_embedding = None
        else:
            self.rope = None
            self.position_embedding = LearnedPositionalEmbedding(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
                padding_idx=config.pad_token_id,
            )

        self.embedding_norm = (
            make_norm(config.norm_type, config.d_model, norm_eps, opt["norm_bias"])
            if opt["embedding_norm"] else None
        )
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # ── Blocks ────────────────────────────────────────────────────────────
        if self.global_every is not None:
            self.layer_is_global = [i % self.global_every == 0 for i in range(config.n_layers)]
            self.blocks = nn.ModuleList([
                EncoderBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    attention_type=config.attention_type,
                    n_kv_heads=config.n_kv_heads,
                    ffn_type=config.ffn_type,
                    dropout=config.dropout,
                    norm_type=config.norm_type,
                    norm_eps=norm_eps,
                    norm_bias=opt["norm_bias"],
                    attn_norm=not (i == 0 and opt["embedding_norm"]),
                    bias=config.bias,
                    rope=self.rope if (is_global or self.local_rope is None) else self.local_rope,
                    qk_norm=getattr(config, "qk_norm", False),
                )
                for i, is_global in enumerate(self.layer_is_global)
            ])
        else:
            self.layer_is_global = None
            self.blocks = self._legacy_blocks(config, opt, self.rope)

        # ── Output heads ──────────────────────────────────────────────────────
        # With default options this is get_normalization(norm_type, d_model),
        # i.e. the legacy final norm (eps 1e-6, LayerNorm with bias).
        self.final_norm = make_norm(config.norm_type, config.d_model, norm_eps, opt["norm_bias"])

        self.mlm_head = MLMHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            embedding_weight=self.token_embedding.embedding.weight,
            dense_bias=opt["head_bias"],
            norm_eps=head_norm_eps,
            norm_bias=opt["norm_bias"],
            decoder_bias=opt["decoder_bias"],
        )

        self.pooler = CandidePooler(config.d_model, opt["classifier_dropout"]) \
            if self.add_pooler else None

        # ── Init ──────────────────────────────────────────────────────────────
        self.apply(self._init_weights)
        if opt["init_scheme"] == "megatron":
            self._init_megatron()
        self.n_params = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _legacy_blocks(config, opt, rope) -> nn.ModuleList:
        """The pre-2026-11-07 encoder blocks, built exactly as before."""
        if getattr(config, "attention_type", "gqa") == "alternating":
            from ..blocks import AlternatingTransformerBlock
            return nn.ModuleList([
                AlternatingTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    layer_idx=i,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    norm_first=config.norm_first,
                    norm_type=config.norm_type,
                    ffn_type=config.ffn_type,
                    n_kv_heads=config.n_kv_heads,
                    rope=rope,
                    bias=config.bias,
                    window_size=opt["local_window_size"],
                    n_layers=config.n_layers,
                )
                for i in range(config.n_layers)
            ])
        return nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                norm_first=config.norm_first,
                norm_type=config.norm_type,
                attention_type=config.attention_type,
                ffn_type=config.ffn_type,
                n_kv_heads=config.n_kv_heads,
                rope=rope,
                bias=config.bias,
                qk_norm=getattr(config, "qk_norm", False),
            )
            for _ in range(config.n_layers)
        ])

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @torch.no_grad()
    def _init_megatron(self, std: float = 0.02, cutoff: float = 2.0):
        """
        Truncated normal (±cutoff·σ) with σ = 0.02, and σ / sqrt(2·n_layers) on
        the projections that write into the residual stream (attention out_proj,
        FFN w3). The scheme family ModernBERT uses; not claimed bit-identical.
        """
        out_std = std / math.sqrt(2.0 * self.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                s = out_std if name.endswith(("attention.out_proj", "ffn.w3")) else std
                nn.init.trunc_normal_(module.weight, std=s, a=-cutoff * s, b=cutoff * s)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=std, a=-cutoff * std, b=cutoff * std)
                if module.padding_idx is not None:
                    module.weight[module.padding_idx].zero_()

    # ── Masks ─────────────────────────────────────────────────────────────────

    def get_padding_mask(
        self, input_ids: Tensor, attention_mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        """HF-style (batch, seq_len) 1/0 mask → (batch, 1, 1, seq_len), or None."""
        if attention_mask is None:
            return None
        return attention_mask.unsqueeze(1).unsqueeze(2)

    def get_layer_masks(
        self, seq_len: int, attention_mask: Optional[Tensor], device
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Boolean (True = attend) masks for the layer-pattern path.

        Returns (global_mask, local_mask). global_mask is None when there is no
        padding, which lets SDPA pick its fastest kernel. In local_mask every
        query may attend to itself, so a padded query whose whole window is
        padding still has one finite attention row (no NaN in the residual).
        """
        pad = None
        if attention_mask is not None and not bool(attention_mask.all()):
            pad = attention_mask.bool()[:, None, None, :]  # keys
        band = local_window_mask(seq_len, self.local_attention, device)
        if pad is None:
            return None, band
        eye = torch.eye(seq_len, dtype=torch.bool, device=device).view(1, 1, seq_len, seq_len)
        return pad, (band & pad) | eye

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        return_hidden_states: bool = False,
        masked_positions: Optional[Tensor] = None,
    ):
        """
        Args:
            input_ids: (batch, seq_len) — input with [MASK] tokens already applied
            attention_mask: (batch, seq_len) — 1=attend, 0=padding. Optional.
            position_ids: (batch, seq_len). Optional.
            return_hidden_states: also return final hidden states
            masked_positions: optional (batch, seq_len) bool. If given, the MLM
                head runs only on these positions and logits are
                (n_masked, vocab_size), in row-major order of the mask
                (ModernBERT's sparse prediction).

        Returns:
            logits: (batch, seq_len, vocab_size), or (n_masked, vocab_size)
            pooled: (batch, d_model) or None
            hidden_states: (batch, seq_len, d_model), only if return_hidden_states
        """
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            hidden_states = hidden_states + self.position_embedding(position_ids)
        if self.embedding_norm is not None:
            hidden_states = self.embedding_norm(hidden_states)
        if self.emb_dropout is not None:
            hidden_states = self.emb_dropout(hidden_states)

        if self.layer_is_global is not None:
            global_mask, local_mask = self.get_layer_masks(seq_len, attention_mask, input_ids.device)
            masks = [global_mask if g else local_mask for g in self.layer_is_global]
        else:
            mask = self.get_padding_mask(input_ids, attention_mask)
            masks = [mask] * len(self.blocks)

        for block, mask in zip(self.blocks, masks):
            hidden_states, _ = block(
                hidden_states,
                mask=mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
            )

        hidden_states = self.final_norm(hidden_states)

        if masked_positions is not None:
            logits = self.mlm_head(hidden_states[masked_positions])
        else:
            logits = self.mlm_head(hidden_states)

        pooled = self.pooler(hidden_states) if self.pooler is not None else None

        if return_hidden_states:
            return logits, pooled, hidden_states
        return logits, pooled

    def get_num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.embedding.weight.numel()
            if self.position_embedding is not None:
                n -= self.position_embedding.embedding.weight.numel()
        return n

    @classmethod
    def from_pretrained(cls, path: str) -> "CandideEncoder":
        ckpt = torch.load(path, map_location="cpu")
        config = CandideEncoderConfig.from_dict(ckpt.get("model_config", ckpt.get("config")))
        model = cls(config)
        model.load_state_dict(ckpt["model"])
        return model

    def save_pretrained(self, path: str):
        torch.save({
            "model_config": self.config.to_dict(),
            "model": self.state_dict(),
        }, path)
