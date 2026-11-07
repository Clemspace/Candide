"""
mlm_collator.py — Data collator for Masked Language Modeling pretraining.

Implements two masking strategies:
  - Token MLM (BERT-style): randomly mask individual tokens
  - Span MLM (T5/RoBERTa-style): mask contiguous spans (better for encoder-decoder later)

Works directly on pre-tokenized binary shards via ShardLoader batches,
so it slots into the existing train_candide18c.py data pipeline with zero changes.

Usage:
    collator = MLMCollator(
        vocab_size=32132,
        mask_token_id=4,      # your [MASK] token id
        pad_token_id=1,
        mlm_probability=0.15,
        strategy="token",     # or "span"
    )
    # Wrap your existing DataLoader:
    for raw_batch in train_loader:
        batch = collator(raw_batch["input_ids"])
        # batch = {"input_ids": ..., "labels": ..., "attention_mask": ...}
"""

import torch
from torch import Tensor
from typing import Optional
import math


class MLMCollator:
    """
    Masked Language Modeling collator.

    Given a batch of token IDs (already sequenced), applies random masking
    and returns inputs, labels, and attention mask suitable for MLM training.

    BERT-style masking rules (applied to each selected token):
      - 80%: replace with [MASK]
      - 10%: replace with random token
      - 10%: keep original

    Labels are -100 (ignored by CrossEntropyLoss) for non-masked positions,
    and the original token ID for masked positions.

    Args:
        vocab_size: Size of vocabulary (for random token replacement)
        mask_token_id: Token ID of [MASK]
        pad_token_id: Token ID of [PAD] (never masked)
        special_token_ids: Set of token IDs that should never be masked
                           (CLS, SEP, BOS, EOS etc). Pad is added automatically.
        mlm_probability: Fraction of tokens to mask (default: 0.15)
        strategy: "token" (BERT) or "span" (T5-style, better for long deps)
        mean_span_length: Mean length of masked spans when strategy="span"
        whole_word_masking: If True, mask entire words (requires word_ids)
        seed: Optional random seed for reproducibility
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        pad_token_id: int = 1,
        special_token_ids: Optional[set] = None,
        mlm_probability: float = 0.15,
        strategy: str = "token",
        mean_span_length: int = 3,
        seed: Optional[int] = None,
    ):
        assert strategy in ("token", "span"), \
            f"strategy must be 'token' or 'span', got '{strategy}'"
        assert 0.0 < mlm_probability < 1.0, \
            f"mlm_probability must be in (0, 1), got {mlm_probability}"

        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mlm_probability = mlm_probability
        self.strategy = strategy
        self.mean_span_length = mean_span_length

        # Build set of token IDs that should never be masked
        self.special_token_ids = set(special_token_ids or [])
        self.special_token_ids.add(pad_token_id)

        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, input_ids: Tensor) -> dict:
        """
        Apply MLM masking to a batch of token sequences.

        Args:
            input_ids: (batch, seq_len) — raw token IDs from ShardLoader

        Returns:
            dict with:
                input_ids:      (batch, seq_len) — masked input
                labels:         (batch, seq_len) — original tokens at masked positions,
                                                   -100 elsewhere
                attention_mask: (batch, seq_len) — 1 everywhere (no padding in shards)
        """
        if self.strategy == "token":
            return self._token_masking(input_ids)
        else:
            return self._span_masking(input_ids)

    def _token_masking(self, input_ids: Tensor) -> dict:
        """BERT-style independent token masking."""
        batch_size, seq_len = input_ids.shape
        labels = input_ids.clone()

        # Build mask: which positions are eligible (not special tokens)
        eligible = torch.ones_like(input_ids, dtype=torch.bool)
        for sid in self.special_token_ids:
            eligible &= (input_ids != sid)

        # Sample positions to mask
        rand = torch.rand_like(input_ids, dtype=torch.float)
        masked_positions = (rand < self.mlm_probability) & eligible

        # Non-masked positions get label -100 (ignored in loss)
        labels[~masked_positions] = -100

        # Apply BERT 80/10/10 rule
        masked_input = input_ids.clone()

        # 80%: replace with [MASK]
        replace_with_mask = masked_positions & (torch.rand_like(rand) < 0.8)
        masked_input[replace_with_mask] = self.mask_token_id

        # 10%: replace with random token (from remaining 20%)
        replace_with_rand = masked_positions & ~replace_with_mask & \
                            (torch.rand_like(rand) < 0.5)
        random_tokens = torch.randint(
            0, self.vocab_size, masked_input.shape,
            device=input_ids.device
        )
        masked_input[replace_with_rand] = random_tokens[replace_with_rand]

        # 10%: keep original (remaining positions in masked_positions already unchanged)

        return {
            "input_ids": masked_input,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

    def _span_masking(self, input_ids: Tensor) -> dict:
        """
        Span masking (RoBERTa / SpanBERT style).

        Samples span lengths from a geometric distribution centered on
        mean_span_length, then masks contiguous spans until mlm_probability
        of tokens are covered. Better than token masking for learning
        long-range dependencies — useful for historical text with long sentences.
        """
        batch_size, seq_len = input_ids.shape
        labels = input_ids.clone()
        masked_input = input_ids.clone()

        for b in range(batch_size):
            seq = input_ids[b]
            mask_labels = labels[b]
            mask_input = masked_input[b]

            # Track which positions are masked
            is_masked = torch.zeros(seq_len, dtype=torch.bool)

            # Build eligible positions (not special tokens)
            eligible = torch.ones(seq_len, dtype=torch.bool)
            for sid in self.special_token_ids:
                eligible &= (seq != sid)

            n_eligible = eligible.sum().item()
            target_masked = int(n_eligible * self.mlm_probability)
            n_masked = 0

            # Geometric distribution for span lengths: p = 1/mean_span_length
            p = 1.0 / self.mean_span_length
            max_attempts = seq_len * 3  # safety valve
            attempts = 0

            while n_masked < target_masked and attempts < max_attempts:
                attempts += 1

                # Sample span start from unmasked eligible positions
                candidates = (~is_masked & eligible).nonzero(as_tuple=True)[0]
                if len(candidates) == 0:
                    break

                start_idx = candidates[torch.randint(len(candidates), (1,))].item()

                # Sample span length from geometric distribution
                span_len = math.ceil(
                    math.log(max(1e-10, torch.rand(1).item())) / math.log(1 - p)
                )
                span_len = max(1, min(span_len, self.mean_span_length * 3))

                # Apply span (clipped to seq_len, skip special tokens)
                end_idx = min(start_idx + span_len, seq_len)
                for pos in range(start_idx, end_idx):
                    if eligible[pos] and not is_masked[pos]:
                        is_masked[pos] = True
                        n_masked += 1

            # Set labels
            mask_labels[~is_masked] = -100

            # Apply 80/10/10 rule over the masked span positions
            for pos in is_masked.nonzero(as_tuple=True)[0]:
                r = torch.rand(1).item()
                if r < 0.8:
                    mask_input[pos] = self.mask_token_id
                elif r < 0.9:
                    mask_input[pos] = torch.randint(0, self.vocab_size, (1,)).item()
                # else: keep original (10%)

        return {
            "input_ids": masked_input,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

    def mask_fraction_stats(self, input_ids: Tensor) -> dict:
        """
        Diagnostic: compute actual masking statistics on a batch.
        Useful for verifying collator behavior before training.

        Returns dict with masked_fraction, n_masked, n_eligible.
        """
        batch = self(input_ids)
        labels = batch["labels"]
        n_masked = (labels != -100).sum().item()
        n_total = labels.numel()
        eligible = sum(
            (input_ids != sid).all().item()
            for sid in self.special_token_ids
        )
        return {
            "masked_fraction": n_masked / n_total,
            "n_masked": n_masked,
            "n_total": n_total,
        }