"""
CandideEncoder — Bidirectional encoder for MLM pretraining.

Reuses the same TransformerBlock, attention, and FFN components as
RamanujanTransformer (decoder), but:
  - No causal mask → full bidirectional attention
  - MLM head instead of LM head
  - Pooler for sentence-level tasks (CLS pooling)

Drop-in addition to ramanujan/models/architectures/encoder.py
"""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List

from .config import TransformerConfig
from ..components import TokenEmbedding, RotaryEmbedding, LearnedPositionalEmbedding
from ..components import get_normalization
from ..blocks import TransformerBlock


class CandideEncoderConfig(TransformerConfig):
    """
    Encoder-specific config extending TransformerConfig.

    Adds:
      - mask_token_id: token ID used for [MASK]
      - add_pooler: whether to add a CLS pooling head
      - classifier_dropout: dropout for classification head

    Everything else (d_model, n_layers, n_heads, GQA, RoPE, SwiGLU...)
    inherits from TransformerConfig unchanged.
    """

    def __init__(self, mask_token_id: int = 4, add_pooler: bool = True,
                 classifier_dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.add_pooler = add_pooler
        self.classifier_dropout = classifier_dropout

    @classmethod
    def from_dict(cls, d: dict) -> "CandideEncoderConfig":
        encoder_keys = {"mask_token_id", "add_pooler", "classifier_dropout"}
        encoder_kwargs = {k: d.pop(k) for k in encoder_keys if k in d}
        base = super().from_dict(d)
        for k, v in encoder_kwargs.items():
            setattr(base, k, v)
        return base


class MLMHead(nn.Module):
    """
    Masked Language Modeling head.

    Transforms hidden states → vocab logits for masked positions.
    Follows BERT convention: Linear → GELU → LayerNorm → Linear (tied to embedding).

    Args:
        d_model: Hidden dimension
        vocab_size: Output vocabulary size
        embedding_weight: Optional weight tensor to tie with token embeddings
    """

    def __init__(self, d_model: int, vocab_size: int,
                 embedding_weight: Optional[Tensor] = None):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)

        # Tie decoder weights to token embeddings if provided
        # (standard BERT practice — reduces params, improves alignment)
        if embedding_weight is not None:
            self.decoder.weight = embedding_weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model) — all positions or masked only

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.norm(x)
        return self.decoder(x)


class CandidePooler(nn.Module):
    """
    CLS-token pooler for sentence-level representations.

    Takes the first token ([CLS]) hidden state and projects it.
    Used for classification and similarity tasks.

    Args:
        d_model: Hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            pooled: (batch, d_model) — CLS token representation
        """
        cls_token = hidden_states[:, 0, :]   # First token = [CLS]
        return self.act(self.dense(self.dropout(cls_token)))


class CandideEncoder(nn.Module):
    """
    Candide Encoder — bidirectional transformer for MLM pretraining.

    Architecture:
        [CLS] + Input → Token Embedding → [Position Encoding] →
        N × TransformerBlock (BIDIRECTIONAL) → Final Norm →
        MLM Head → Masked token logits
        └→ Pooler → CLS representation (for downstream tasks)

    Key difference from RamanujanTransformer (decoder):
        - No causal mask: every position attends to every other position
        - MLM objective: predict [MASK] tokens from context
        - Pooler: CLS token for sentence-level tasks

    Reuses (zero duplication):
        - TransformerBlock (same attention, FFN, norms)
        - TokenEmbedding, RotaryEmbedding
        - TransformerConfig base

    Args:
        config: CandideEncoderConfig (or TransformerConfig — encoder_* fields optional)

    Example:
        >>> config = CandideEncoderConfig(
        ...     vocab_size=32132, d_model=768, n_layers=12,
        ...     n_heads=12, n_kv_heads=4, d_ff=3072,
        ...     mask_token_id=4, add_pooler=True
        ... )
        >>> model = CandideEncoder(config)
        >>> input_ids = torch.randint(0, 32132, (2, 128))
        >>> logits, pooled = model(input_ids)
        >>> logits.shape   # (2, 128, 32132)
        torch.Size([2, 128, 32132])
        >>> pooled.shape   # (2, 768)
        torch.Size([2, 768])
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers

        # Retrieve encoder-specific fields with safe defaults
        self.mask_token_id = getattr(config, "mask_token_id", 4)
        self.add_pooler = getattr(config, "add_pooler", True)
        classifier_dropout = getattr(config, "classifier_dropout", 0.1)

        # ── Embeddings ────────────────────────────────────────────────────────
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.pad_token_id,
        )

        if config.use_rope:
            self.rope = RotaryEmbedding(
                d_model=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
            self.position_embedding = None
        else:
            self.rope = None
            self.position_embedding = LearnedPositionalEmbedding(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
                padding_idx=config.pad_token_id,
            )

        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # ── Transformer blocks (bidirectional — NO causal mask) ───────────────
        from ..blocks import AlternatingTransformerBlock

        use_alternating = getattr(config, 'attention_type', 'gqa') == 'alternating'
        window_size = getattr(config, 'local_window_size', 128)

        if use_alternating:
            self.blocks = nn.ModuleList([
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
                    rope=self.rope,
                    bias=config.bias,
                    window_size=window_size,
                    n_layers=config.n_layers,
                )
                for i in range(config.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
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
                    rope=self.rope,
                    bias=config.bias,
                )
                for _ in range(config.n_layers)
            ])

        # ── Output heads ──────────────────────────────────────────────────────
        self.final_norm = get_normalization(config.norm_type, config.d_model)

        # MLM head (tied to token embedding weights)
        self.mlm_head = MLMHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            embedding_weight=self.token_embedding.embedding.weight,
        )

        # Optional CLS pooler
        self.pooler = CandidePooler(config.d_model, classifier_dropout) \
            if self.add_pooler else None

        # Weight init
        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_padding_mask(
        self, input_ids: Tensor, attention_mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        """
        Convert HF-style attention mask (1=attend, 0=pad) to additive mask.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) with 1s and 0s, or None

        Returns:
            mask: (batch, 1, 1, seq_len) broadcastable over heads, or None
        """
        if attention_mask is None:
            return None
        # (batch, 1, 1, seq_len): 1 where we attend, 0 where padding
        return attention_mask.unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass — bidirectional, no causal mask.

        Args:
            input_ids: (batch, seq_len) — input with [MASK] tokens already applied
            attention_mask: (batch, seq_len) — 1=attend, 0=padding. Optional.
            position_ids: (batch, seq_len) — custom positions. Optional.
            return_hidden_states: if True, also return final hidden states

        Returns:
            logits: (batch, seq_len, vocab_size) — MLM prediction logits
            pooled: (batch, d_model) — CLS pooled representation, or None
            hidden_states: (batch, seq_len, d_model) — only if return_hidden_states
        """
        batch_size, seq_len = input_ids.shape

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        hidden_states = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            hidden_states = hidden_states + self.position_embedding(position_ids)
        if self.emb_dropout is not None:
            hidden_states = self.emb_dropout(hidden_states)

        # Padding mask (no causal mask — bidirectional attention)
        mask = self.get_padding_mask(input_ids, attention_mask)

        # Transformer blocks — mask=None means full attention if no padding mask
        for block in self.blocks:
            hidden_states, _ = block(
                hidden_states,
                mask=mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
            )

        hidden_states = self.final_norm(hidden_states)

        # MLM logits (all positions — loss computed externally on masked positions)
        logits = self.mlm_head(hidden_states)

        # CLS pooling
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
        config = CandideEncoderConfig.from_dict(ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["model"])
        return model

    def save_pretrained(self, path: str):
        torch.save({
            "config": self.config.to_dict(),
            "model": self.state_dict(),
        }, path)