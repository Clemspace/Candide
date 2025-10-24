"""Ramanujan Transformer - Full model architecture."""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List
import math

from .config import TransformerConfig
from ..components import TokenEmbedding, RotaryEmbedding, LearnedPositionalEmbedding
from ..components import get_normalization
from ..blocks import TransformerBlock


class RamanujanTransformer(nn.Module):
    """
    Ramanujan Transformer - Full autoregressive language model.
    
    Architecture:
        Input → Token Embedding → [Position Encoding] →
        N × TransformerBlock → Final Norm → LM Head → Output
    
    Supports:
    - Causal language modeling
    - Efficient generation with KV caching
    - Various architectures (GPT, LLaMA, BERT-style)
    - Pruning hooks for Ramanujan sparsification
    
    Args:
        config: TransformerConfig with model parameters
    
    Example:
        >>> config = TransformerConfig.tiny(vocab_size=50000)
        >>> model = RamanujanTransformer(config)
        >>> tokens = torch.randint(0, 50000, (2, 10))
        >>> logits = model(tokens)
        >>> logits.shape
        torch.Size([2, 10, 50000])
        
        >>> # Generation with caching
        >>> past_key_values = None
        >>> for _ in range(10):
        ...     logits, past_key_values = model(
        ...         next_token,
        ...         past_key_values=past_key_values,
        ...         use_cache=True
        ...     )
    """
    
    def __init__(self, config: TransformerConfig):
        """Initialize Ramanujan Transformer."""
        super().__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.pad_token_id,
        )
        
        # Position encoding
        if config.use_rope:
            # RoPE: applied inside attention layers
            self.rope = RotaryEmbedding(
                d_model=config.head_dim,  # Per-head dimension!
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
            self.position_embedding = None
        else:
            # Learned positional embeddings (GPT/BERT-style)
            self.rope = None
            self.position_embedding = LearnedPositionalEmbedding(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
                padding_idx=config.pad_token_id,
            )
        
        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        
        # Transformer blocks
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
                rope=self.rope,  # Share RoPE across layers
                bias=config.bias,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final normalization
        self.final_norm = get_normalization(config.norm_type, config.d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Optionally tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        """Initialize weights following best practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def get_position_ids(
        self,
        input_ids: Tensor,
        past_key_values_length: int = 0
    ) -> Tensor:
        """
        Generate position IDs for input.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            past_key_values_length: Length of cached key-values
        
        Returns:
            Position IDs (batch, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(
            past_key_values_length,
            seq_len + past_key_values_length,
            dtype=torch.long,
            device=input_ids.device
        )
        return position_ids.unsqueeze(0).expand(batch_size, -1)
    
    def get_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        past_key_values_length: int = 0
    ) -> Tensor:
        """
        Generate causal attention mask.
        
        Args:
            seq_len: Current sequence length
            device: Device for tensor
            past_key_values_length: Length of cached key-values
        
        Returns:
            Causal mask (1, 1, seq_len, seq_len + past_length)
        """
        # Full sequence length including cache
        full_seq_len = seq_len + past_key_values_length
        
        # Create causal mask: upper triangular matrix of -inf
        mask = torch.triu(
            torch.ones(seq_len, full_seq_len, device=device),
            diagonal=past_key_values_length + 1
        )
        mask = mask.masked_fill(mask == 1, 0).masked_fill(mask == 0, 1)
        
        # Add batch and head dimensions: (1, 1, seq_len, full_seq_len)
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = False,
    ) -> Tensor | Tuple:
        """
        Forward pass through transformer.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Custom attention mask (optional)
            position_ids: Custom position IDs (optional)
            past_key_values: Cached key-values from previous steps
            use_cache: Whether to return key-value cache
            return_dict: Whether to return dictionary output
        
        Returns:
            If return_dict=False:
                logits: (batch, seq_len, vocab_size)
                or (logits, past_key_values) if use_cache=True
            If return_dict=True:
                Dictionary with 'logits' and optionally 'past_key_values'
        """
        batch_size, seq_len = input_ids.shape
        
        # Determine past length
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]  # past_key.shape[2]
        
        # Get position IDs
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, past_length)
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Add positional embeddings if not using RoPE
        if self.position_embedding is not None:
            hidden_states = hidden_states + self.position_embedding(position_ids)
        
        # Embedding dropout
        if self.emb_dropout is not None:
            hidden_states = self.emb_dropout(hidden_states)
        
        # Get attention mask
        if attention_mask is None:
            attention_mask = self.get_causal_mask(seq_len, input_ids.device, past_length)
        
        # Through transformer blocks
        new_past_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            # Get past key-value for this layer
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Forward through block
            hidden_states, past_kv = block(
                hidden_states,
                mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                new_past_key_values.append(past_kv)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Return format
        if return_dict:
            output = {'logits': logits}
            if use_cache:
                output['past_key_values'] = new_past_key_values
            return output
        else:
            if use_cache:
                return logits, new_past_key_values
            return logits
    
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (keep top k tokens)
            top_p: Nucleus sampling (keep tokens with cumulative prob > p)
            **kwargs: Additional arguments
        
        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        generated = input_ids
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                if past_key_values is None:
                    # First step: use full sequence
                    logits, past_key_values = self(
                        generated,
                        use_cache=True
                    )
                else:
                    # Subsequent steps: only use last token
                    logits, past_key_values = self(
                        generated[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                
                # Get logits for last position
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Apply nucleus (top-p) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Remove tokens with cumulative probability > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get number of parameters.
        
        Args:
            non_embedding: If True, exclude embedding parameters
        
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.token_embedding.embedding.weight.numel()
            if self.position_embedding is not None:
                n_params -= self.position_embedding.embedding.weight.numel()
        
        return n_params
    
    @classmethod
    def from_config(cls, config: TransformerConfig) -> 'RamanujanTransformer':
        """Create model from config."""
        return cls(config)
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'RamanujanTransformer':
        """Load pretrained model."""
        checkpoint = torch.load(path, map_location='cpu')
        config = TransformerConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'config': self.config.to_dict(),
            'model_state_dict': self.state_dict(),
        }, path)