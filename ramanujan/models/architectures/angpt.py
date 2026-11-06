"""anGPT - Approximately normalized decoder-only transformer.

    Input -> Token Embedding (unit-norm rows) ->
    N x ANGPTBlock -> LM Head (unit-norm rows) * s_z -> logits

Differences from ``RamanujanTransformer``:
- no RMSNorm / LayerNorm anywhere (blocks, final norm removed)
- residual stream updated by LERP with learnable rates and a constant
  normalization factor, so it stays near the unit sphere at every depth
- every Linear / Embedding row is kept in the unit ball: call
  ``project_parameters()`` after each optimizer step
  (``WeightNormBoundCallback`` does this inside the Trainer)
- learnable per-vocabulary logit scale s_z replaces the final norm
- trained with Adam, weight_decay = 0, no LR warmup

Reference: Franke et al., "Learning in Compact Spaces with Approximately
Normalized Transformers", arXiv:2505.22014.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from .config import TransformerConfig
from .transformer import RamanujanTransformer
from ..components import TokenEmbedding, RotaryEmbedding
from ..components.normalized import (
    ScaledParameter,
    bounded_weights,
    max_row_norm,
    normalize_rows_,
    project_rows_to_unit_ball,
)
from ..blocks.angpt import ANGPTBlock


@dataclass
class ANGPTConfig(TransformerConfig):
    """
    TransformerConfig plus anGPT-specific options.

    Inherited fields that anGPT ignores: ``norm_first``, ``norm_type``,
    ``ffn_type`` (always SwiGLU). ``bias`` must be False and ``use_rope``
    must be True (additive learned positions would break the unit-norm
    residual stream).

    anGPT fields:
        alpha_init: Effective initial LERP rate (nGPT: 0.05)
        alpha_scale: Raw stored LERP rate; effective LR multiplier is
            alpha_init / alpha_scale (paper Table F.2: 0.01)
        residual_mode: 'approx' | 'exact' | 'paper_literal'
        qk_scale_init: Initial attention temperature (None = sqrt(head_dim))
        learnable_qk_scale: Learn one temperature per head
        logit_scale_init: Effective initial s_z
        logit_scale_scale: Raw stored s_z (None = 1/sqrt(d_model), as nGPT)
        act_norm_factor: Override for the SwiGLU activation constant
        gate_scale: Multiplier on the SiLU input (1.0 = paper; nGPT-like
            alternative: sqrt(d_model)). See ANGPTSwiGLU.
    """

    alpha_init: float = 0.05
    alpha_scale: float = 0.01
    residual_mode: str = 'approx'
    qk_scale_init: Optional[float] = None
    learnable_qk_scale: bool = True
    logit_scale_init: float = 1.0
    logit_scale_scale: Optional[float] = None
    act_norm_factor: Optional[float] = None
    gate_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.bias:
            raise ValueError("anGPT does not support bias=True")
        if not self.use_rope:
            raise ValueError("anGPT requires use_rope=True")
        if self.residual_mode not in ('approx', 'exact', 'paper_literal'):
            raise ValueError(
                f"Unknown residual_mode: {self.residual_mode}. "
                f"Available: 'approx', 'exact', 'paper_literal'"
            )
        if self.logit_scale_scale is None:
            self.logit_scale_scale = 1.0 / math.sqrt(self.d_model)


class ANGPTTransformer(nn.Module):
    """
    Approximately normalized transformer language model.

    Same call signature and outputs as ``RamanujanTransformer`` (logits,
    optional KV cache, ``return_dict``), so it plugs into the existing
    Trainer, losses and ``generate``.

    Args:
        config: ANGPTConfig

    Example:
        >>> config = ANGPTConfig(vocab_size=1000, d_model=128, n_layers=2, n_heads=4)
        >>> model = ANGPTTransformer(config)
        >>> logits = model(torch.randint(0, 1000, (2, 16)))
        >>> model.project_parameters()  # after every optimizer step
    """

    def __init__(self, config: ANGPTConfig):
        super().__init__()
        if not isinstance(config, ANGPTConfig):
            raise TypeError(f"ANGPTTransformer needs an ANGPTConfig, got {type(config).__name__}")

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers

        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.pad_token_id,
        )

        self.rope = RotaryEmbedding(
            d_model=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.position_embedding = None  # interface parity with RamanujanTransformer

        n_kv_heads = config.n_kv_heads if config.attention_type not in ('mha', 'multi_head') else None

        self.blocks = nn.ModuleList([
            ANGPTBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                n_kv_heads=n_kv_heads,
                rope=self.rope,
                dropout=config.dropout,
                alpha_init=config.alpha_init,
                alpha_scale=config.alpha_scale,
                residual_mode=config.residual_mode,
                qk_scale_init=config.qk_scale_init,
                learnable_qk_scale=config.learnable_qk_scale,
                act_norm_factor=config.act_norm_factor,
                gate_scale=config.gate_scale,
            )
            for _ in range(config.n_layers)
        ])

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.embedding.weight

        self.logit_scale = ScaledParameter(
            config.vocab_size,
            init=config.logit_scale_init,
            scale=config.logit_scale_scale,
        )

        self._init_embeddings()
        self.n_params = sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Initialisation and constraints
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_embeddings(self) -> None:
        """Unit-norm token vectors and head rows (blocks init themselves)."""
        emb = self.token_embedding.embedding.weight
        nn.init.normal_(emb, mean=0.0, std=1.0)
        normalize_rows_(emb)
        if self.config.pad_token_id is not None:
            emb[self.config.pad_token_id].zero_()
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=1.0)
            normalize_rows_(self.lm_head.weight)

    @torch.no_grad()
    def project_parameters(self) -> None:
        """
        Project every Linear / Embedding row onto the unit ball.

        Must run after each optimizer step. Idempotent and cheap relative
        to a training step. Scale parameters (alpha, s_z, qk temperature)
        are not constrained.
        """
        for w in bounded_weights(self):
            project_rows_to_unit_ball(w)

    def max_weight_row_norm(self) -> float:
        """Largest row norm over constrained weights (should be <= 1 + eps)."""
        return max_row_norm(self)

    # ------------------------------------------------------------------
    # Shared utilities, reused unchanged from RamanujanTransformer
    # ------------------------------------------------------------------

    get_position_ids = RamanujanTransformer.get_position_ids
    get_causal_mask = RamanujanTransformer.get_causal_mask
    generate = RamanujanTransformer.generate

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = False,
    ):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: Custom mask, nonzero = attend (default: causal)
            position_ids: Custom positions (default: arange after cache)
            past_key_values: Per-layer cached (k, v)
            use_cache: Return updated cache
            return_dict: Return {'logits', 'past_key_values'}

        Returns:
            Same conventions as RamanujanTransformer.forward
        """
        batch_size, seq_len = input_ids.shape

        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, past_length)

        if attention_mask is None:
            attention_mask = self.get_causal_mask(seq_len, input_ids.device, past_length)

        hidden_states = self.token_embedding(input_ids)

        new_past_key_values = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, past_kv = block(
                hidden_states,
                mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(past_kv)

        logits = self.lm_head(hidden_states) * self.logit_scale().to(hidden_states.dtype)

        if return_dict:
            output = {'logits': logits}
            if use_cache:
                output['past_key_values'] = new_past_key_values
            return output
        if use_cache:
            return logits, new_past_key_values
        return logits

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.embedding.weight.numel()
        return n_params

    @classmethod
    def from_config(cls, config: ANGPTConfig) -> 'ANGPTTransformer':
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: str) -> 'ANGPTTransformer':
        checkpoint = torch.load(path, map_location='cpu')
        config = ANGPTConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def save_pretrained(self, path: str) -> None:
        torch.save({
            'config': self.config.to_dict(),
            'model_state_dict': self.state_dict(),
        }, path)
