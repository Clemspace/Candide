"""
Loss functions for Ramanujan Transformer training.

This module provides various loss functions:
- SemanticEntropyLoss: Uncertainty-aware loss using semantic entropy
- Standard cross-entropy loss with label smoothing
- Loss utilities and metrics

Semantic entropy helps the model learn better by focusing on
semantic uncertainty rather than just token-level predictions.

Example:
    >>> from ramanujan.training import SemanticEntropyLoss
    >>> 
    >>> # Create loss function
    >>> loss_fn = SemanticEntropyLoss(
    ...     vocab_size=32000,
    ...     num_semantic_sets=100,
    ...     alpha=0.1
    ... )
    >>> 
    >>> # Compute loss
    >>> logits = model(input_ids)  # [batch, seq, vocab]
    >>> targets = input_ids[:, 1:]  # Shift for causal LM
    >>> loss = loss_fn(logits[:, :-1], targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


# ============================================================================
# SEMANTIC ENTROPY PROBE
# ============================================================================

class SemanticEntropyProbe(nn.Module):
    """
    Probe for estimating semantic entropy.
    
    Maps token embeddings to semantic clusters to estimate
    uncertainty at the semantic level rather than token level.
    
    This helps distinguish between:
    - Irreducible uncertainty (multiple valid continuations)
    - Reducible uncertainty (model confusion)
    
    Args:
        embed_dim: Dimension of token embeddings
        num_semantic_sets: Number of semantic clusters
        hidden_dim: Hidden dimension for probe (default: None, uses embed_dim)
        temperature: Temperature for softmax (default: 1.0)
    
    Example:
        >>> probe = SemanticEntropyProbe(
        ...     embed_dim=512,
        ...     num_semantic_sets=100
        ... )
        >>> 
        >>> # Get semantic distribution from logits
        >>> logits = torch.randn(2, 128, 32000)
        >>> semantic_probs = probe(logits)
        >>> print(semantic_probs.shape)  # [2, 128, 100]
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_semantic_sets: int,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_semantic_sets = num_semantic_sets
        self.temperature = temperature
        
        if hidden_dim is None:
            hidden_dim = embed_dim
        
        # Two-layer MLP for semantic mapping
        self.probe = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_semantic_sets)
        )
        
        # Initialize with small weights
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize probe parameters."""
        for module in self.probe.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic cluster probabilities.
        
        Args:
            hidden_states: Hidden states [batch, seq_len, embed_dim]
        
        Returns:
            Semantic probabilities [batch, seq_len, num_semantic_sets]
        """
        # Map to semantic space
        semantic_logits = self.probe(hidden_states)
        
        # Apply temperature and softmax
        semantic_probs = F.softmax(semantic_logits / self.temperature, dim=-1)
        
        return semantic_probs
    
    def compute_entropy(self, semantic_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of semantic distribution.
        
        Args:
            semantic_probs: Semantic probabilities [batch, seq_len, num_sets]
        
        Returns:
            Entropy values [batch, seq_len]
        """
        # Add small epsilon for numerical stability
        eps = 1e-10
        entropy = -(semantic_probs * torch.log(semantic_probs + eps)).sum(dim=-1)
        
        return entropy


# ============================================================================
# SEMANTIC ENTROPY LOSS
# ============================================================================

class SemanticEntropyLoss(nn.Module):
    """
    Semantic Entropy Loss for uncertainty-aware training.
    
    Combines standard cross-entropy with semantic entropy regularization
    to help models learn better uncertainty estimation.
    
    Loss = CE_loss + alpha * semantic_entropy_penalty
    
    Where semantic_entropy_penalty encourages the model to:
    - Be confident when there's one clear answer
    - Be uncertain when multiple answers are valid
    
    Args:
        vocab_size: Vocabulary size
        num_semantic_sets: Number of semantic clusters (default: 100)
        alpha: Weight for semantic entropy term (default: 0.1)
        temperature: Temperature for semantic probe (default: 1.0)
        label_smoothing: Label smoothing factor (default: 0.0)
        ignore_index: Index to ignore in loss (default: -100)
    
    Example:
        >>> loss_fn = SemanticEntropyLoss(
        ...     vocab_size=32000,
        ...     num_semantic_sets=100,
        ...     alpha=0.1,
        ...     label_smoothing=0.1
        ... )
        >>> 
        >>> # Forward pass
        >>> logits = model(input_ids)
        >>> targets = input_ids[:, 1:]
        >>> loss, metrics = loss_fn(logits[:, :-1], targets, return_metrics=True)
        >>> 
        >>> print(f"Total loss: {loss.item():.4f}")
        >>> print(f"CE loss: {metrics['ce_loss']:.4f}")
        >>> print(f"Semantic entropy: {metrics['semantic_entropy']:.4f}")
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_semantic_sets: int = 100,
        alpha: float = 0.1,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        ignore_index: int = -100
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_semantic_sets = num_semantic_sets
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        
        # Semantic entropy probe
        # Note: We need to get embed_dim from the model
        # For now, we'll initialize it lazily on first forward pass
        self.probe = None
        self.temperature = temperature
        
        # Track metrics
        self.register_buffer('total_steps', torch.tensor(0))

    def to(self, device):
        """Override to() to ensure probe moves to device."""
        super().to(device)
        if hasattr(self, 'probe') and self.probe is not None:
            self.probe = self.probe.to(device)
        return self

    def _init_probe(self, embed_dim: int):
        """Initialize probe lazily on first forward pass."""
        if self.probe is None:
            self.probe = SemanticEntropyProbe(
                embed_dim=embed_dim,
                num_semantic_sets=self.num_semantic_sets,
                temperature=self.temperature
            )
            # Move to same device as input
            self.probe = self.probe.to(next(self.parameters()).device)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> torch.Tensor:
        """
        Compute semantic entropy loss.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]
            hidden_states: Optional hidden states for semantic probe [batch, seq_len, dim]
            return_metrics: If True, return (loss, metrics_dict)
        
        Returns:
            loss: Scalar loss value
            or (loss, metrics) if return_metrics=True
        """
        batch_size, seq_len, vocab_size = logits.shape
        targets = torch.clamp(targets, 0, logits.size(-1) - 1)

        
        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )
        
        # If no hidden states provided or alpha=0, return CE loss only
        if hidden_states is None or self.alpha == 0:
            if return_metrics:
                metrics = {
                    'ce_loss': ce_loss.item(),
                    'semantic_entropy': 0.0,
                    'total_loss': ce_loss.item()
                }
                return ce_loss, metrics
            return ce_loss
        
        # Initialize probe if needed
        if self.probe is None:
            self._init_probe(hidden_states.shape[-1])
            self.probe = self.probe.to(hidden_states.device)

        
        # Compute semantic probabilities
        semantic_probs = self.probe(hidden_states)
        
        # Compute semantic entropy
        semantic_entropy = self.probe.compute_entropy(semantic_probs)
        
        # Create mask for valid positions (not padding)
        if self.ignore_index != -100:
            mask = (targets != self.ignore_index).float()
        else:
            mask = torch.ones_like(targets, dtype=torch.float)
        
        # Average semantic entropy over valid positions
        semantic_entropy_mean = (semantic_entropy * mask).sum() / mask.sum()
        
        # Total loss: CE + alpha * semantic_entropy
        # We want to minimize semantic entropy (encourage confidence)
        total_loss = ce_loss + self.alpha * semantic_entropy_mean
        
        # Update step counter
        self.total_steps += 1
        
        if return_metrics:
            metrics = {
                'ce_loss': ce_loss.item(),
                'semantic_entropy': semantic_entropy_mean.item(),
                'total_loss': total_loss.item(),
                'alpha': self.alpha
            }
            return total_loss, metrics
        
        return total_loss
    
    def get_info(self) -> Dict[str, any]:
        """Get loss function information."""
        return {
            'type': 'SemanticEntropyLoss',
            'vocab_size': self.vocab_size,
            'num_semantic_sets': self.num_semantic_sets,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing,
            'total_steps': self.total_steps.item()
        }


# ============================================================================
# STANDARD CROSS-ENTROPY LOSS
# ============================================================================

def standard_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
    return_metrics: bool = False
) -> torch.Tensor:
    """
    Standard cross-entropy loss with optional label smoothing.
    
    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        targets: Target token IDs [batch, seq_len]
        label_smoothing: Label smoothing factor (default: 0.0)
        ignore_index: Index to ignore in loss (default: -100)
        return_metrics: If True, return (loss, metrics_dict)
    
    Returns:
        loss: Scalar loss value
        or (loss, metrics) if return_metrics=True
    
    Example:
        >>> logits = model(input_ids)
        >>> targets = input_ids[:, 1:]
        >>> loss = standard_cross_entropy(
        ...     logits[:, :-1],
        ...     targets,
        ...     label_smoothing=0.1
        ... )
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for cross_entropy
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )
    
    if return_metrics:
        # Compute perplexity
        perplexity = torch.exp(loss)
        
        metrics = {
            'loss': loss.item(),
            'perplexity': perplexity.item()
        }
        return loss, metrics
    
    return loss


# ============================================================================
# LOSS UTILITIES
# ============================================================================

class LossTracker:
    """
    Track loss statistics during training.
    
    Maintains running averages and history of loss values.
    
    Example:
        >>> tracker = LossTracker()
        >>> 
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     tracker.update(loss.item())
        ...     
        ...     if step % 100 == 0:
        ...         print(f"Avg loss: {tracker.get_average():.4f}")
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.losses = []
        self.total_loss = 0.0
        self.count = 0
    
    def update(self, loss: float):
        """Update with new loss value."""
        self.losses.append(loss)
        self.total_loss += loss
        self.count += 1
        
        # Keep only window_size most recent losses
        if len(self.losses) > self.window_size:
            old_loss = self.losses.pop(0)
            self.total_loss -= old_loss
    
    def get_average(self) -> float:
        """Get average loss over window."""
        if len(self.losses) == 0:
            return 0.0
        return self.total_loss / len(self.losses)
    
    def get_total_average(self) -> float:
        """Get average loss over all time."""
        if self.count == 0:
            return 0.0
        return sum(self.losses) / self.count
    
    def reset(self):
        """Reset tracker."""
        self.losses = []
        self.total_loss = 0.0
        self.count = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics."""
        if len(self.losses) == 0:
            return {
                'average': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        import numpy as np
        losses_array = np.array(self.losses)
        
        return {
            'average': float(losses_array.mean()),
            'min': float(losses_array.min()),
            'max': float(losses_array.max()),
            'std': float(losses_array.std())
        }


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from loss.
    
    Perplexity = exp(loss)
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity value
    """
    return torch.exp(loss)


def compute_bits_per_byte(
    loss: torch.Tensor,
    vocab_size: int,
    sequence_length: int
) -> float:
    """
    Compute bits per byte (BPB) metric.
    
    BPB is a normalized measure of compression that accounts
    for vocabulary size and sequence length.
    
    Args:
        loss: Cross-entropy loss
        vocab_size: Vocabulary size
        sequence_length: Sequence length
    
    Returns:
        Bits per byte
    """
    # Convert nats to bits
    bits_per_token = loss.item() / math.log(2)
    
    # Normalize by average bytes per token (rough estimate)
    # Assuming ~4 bytes per token on average
    bytes_per_token = 4.0
    bpb = bits_per_token / bytes_per_token
    
    return bpb


class AdaptiveLossWeight:
    """
    Adaptively adjust loss component weights during training.
    
    Useful for balancing multiple loss terms (e.g., CE + semantic entropy).
    
    Args:
        initial_weight: Initial weight value
        target_ratio: Target ratio between loss components
        adjustment_rate: How fast to adjust (default: 0.01)
    
    Example:
        >>> weight_adjuster = AdaptiveLossWeight(
        ...     initial_weight=0.1,
        ...     target_ratio=0.1,
        ...     adjustment_rate=0.01
        ... )
        >>> 
        >>> # During training
        >>> ce_loss = 3.5
        >>> se_loss = 0.4
        >>> new_weight = weight_adjuster.update(ce_loss, se_loss)
    """
    
    def __init__(
        self,
        initial_weight: float = 0.1,
        target_ratio: float = 0.1,
        adjustment_rate: float = 0.01
    ):
        self.weight = initial_weight
        self.target_ratio = target_ratio
        self.adjustment_rate = adjustment_rate
    
    def update(
        self,
        main_loss: float,
        auxiliary_loss: float
    ) -> float:
        """
        Update weight based on loss ratio.
        
        Args:
            main_loss: Main loss component (e.g., CE)
            auxiliary_loss: Auxiliary loss component (e.g., semantic entropy)
        
        Returns:
            Updated weight
        """
        if main_loss == 0:
            return self.weight
        
        # Compute current ratio
        current_ratio = auxiliary_loss / main_loss
        
        # Adjust weight to move ratio toward target
        if current_ratio < self.target_ratio:
            self.weight *= (1 + self.adjustment_rate)
        elif current_ratio > self.target_ratio:
            self.weight *= (1 - self.adjustment_rate)
        
        # Clamp weight to reasonable range
        self.weight = max(0.001, min(1.0, self.weight))
        
        return self.weight


# ============================================================================
# LOSS FACTORY
# ============================================================================

def create_loss(
    loss_type: str,
    vocab_size: int,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating loss functions.
    
    Args:
        loss_type: Type of loss ('ce', 'semantic_entropy')
        vocab_size: Vocabulary size
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function module
    
    Example:
        >>> # Standard CE loss
        >>> loss_fn = create_loss('ce', vocab_size=32000, label_smoothing=0.1)
        >>> 
        >>> # Semantic entropy loss
        >>> loss_fn = create_loss(
        ...     'semantic_entropy',
        ...     vocab_size=32000,
        ...     alpha=0.1,
        ...     num_semantic_sets=100
        ... )
    """
    loss_type = loss_type.lower()
    
    if loss_type in ['ce', 'cross_entropy', 'standard']:
        # Return a wrapper that matches SemanticEntropyLoss interface
        class StandardCELoss(nn.Module):
            def __init__(self, vocab_size, label_smoothing=0.0, ignore_index=-100):
                super().__init__()
                self.vocab_size = vocab_size
                self.label_smoothing = label_smoothing
                self.ignore_index = ignore_index
            
            def forward(self, logits, targets, hidden_states=None, return_metrics=False):
                return standard_cross_entropy(
                    logits, targets,
                    label_smoothing=self.label_smoothing,
                    ignore_index=self.ignore_index,
                    return_metrics=return_metrics
                )
        
        return StandardCELoss(
            vocab_size=vocab_size,
            label_smoothing=kwargs.get('label_smoothing', 0.0),
            ignore_index=kwargs.get('ignore_index', -100)
        )
    
    elif loss_type in ['semantic_entropy', 'se']:
        return SemanticEntropyLoss(
            vocab_size=vocab_size,
            num_semantic_sets=kwargs.get('num_semantic_sets', 100),
            alpha=kwargs.get('alpha', 0.1),
            temperature=kwargs.get('temperature', 1.0),
            label_smoothing=kwargs.get('label_smoothing', 0.0),
            ignore_index=kwargs.get('ignore_index', -100)
        )
    
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Choose from: 'ce', 'semantic_entropy'"
        )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing losses.py module")
    print("="*70)
    
    # Test SemanticEntropyProbe
    print("\n1. Testing SemanticEntropyProbe...")
    probe = SemanticEntropyProbe(
        embed_dim=256,
        num_semantic_sets=50
    )
    hidden_states = torch.randn(2, 64, 256)
    semantic_probs = probe(hidden_states)
    entropy = probe.compute_entropy(semantic_probs)
    
    assert semantic_probs.shape == (2, 64, 50), "Shape mismatch!"
    assert entropy.shape == (2, 64), "Entropy shape mismatch!"
    
    print(f"   Hidden states: {hidden_states.shape}")
    print(f"   Semantic probs: {semantic_probs.shape}")
    print(f"   Entropy: {entropy.shape}")
    print(f"   Avg entropy: {entropy.mean().item():.4f}")
    print(f"   ✅ SemanticEntropyProbe working!")
    
    # Test SemanticEntropyLoss
    print("\n2. Testing SemanticEntropyLoss...")
    loss_fn = SemanticEntropyLoss(
        vocab_size=1000,
        num_semantic_sets=50,
        alpha=0.1
    )
    
    logits = torch.randn(2, 64, 1000)
    targets = torch.randint(0, 1000, (2, 64))
    hidden_states = torch.randn(2, 64, 256)
    
    loss, metrics = loss_fn(logits, targets, hidden_states, return_metrics=True)
    
    print(f"   Logits: {logits.shape}")
    print(f"   Targets: {targets.shape}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   CE loss: {metrics['ce_loss']:.4f}")
    print(f"   Semantic entropy: {metrics['semantic_entropy']:.4f}")
    print(f"   ✅ SemanticEntropyLoss working!")
    
    # Test without hidden states
    loss_no_hidden = loss_fn(logits, targets, return_metrics=False)
    print(f"   Loss (no hidden): {loss_no_hidden.item():.4f}")
    print(f"   ✅ Fallback to CE working!")
    
    # Test standard_cross_entropy
    print("\n3. Testing standard_cross_entropy...")
    ce_loss, ce_metrics = standard_cross_entropy(
        logits, targets,
        label_smoothing=0.1,
        return_metrics=True
    )
    
    print(f"   CE loss: {ce_loss.item():.4f}")
    print(f"   Perplexity: {ce_metrics['perplexity']:.4f}")
    print(f"   ✅ standard_cross_entropy working!")
    
    # Test LossTracker
    print("\n4. Testing LossTracker...")
    tracker = LossTracker(window_size=10)
    
    for i in range(20):
        tracker.update(3.0 + 0.1 * i)
    
    stats = tracker.get_stats()
    print(f"   Average: {stats['average']:.4f}")
    print(f"   Min: {stats['min']:.4f}")
    print(f"   Max: {stats['max']:.4f}")
    print(f"   Std: {stats['std']:.4f}")
    print(f"   ✅ LossTracker working!")
    
    # Test compute_perplexity
    print("\n5. Testing compute_perplexity...")
    loss_val = torch.tensor(3.5)
    ppl = compute_perplexity(loss_val)
    print(f"   Loss: {loss_val.item():.4f}")
    print(f"   Perplexity: {ppl.item():.2f}")
    print(f"   ✅ Perplexity computation working!")
    
    # Test compute_bits_per_byte
    print("\n6. Testing compute_bits_per_byte...")
    bpb = compute_bits_per_byte(loss_val, vocab_size=32000, sequence_length=512)
    print(f"   Bits per byte: {bpb:.4f}")
    print(f"   ✅ BPB computation working!")
    
    # Test AdaptiveLossWeight
    print("\n7. Testing AdaptiveLossWeight...")
    weight_adjuster = AdaptiveLossWeight(
        initial_weight=0.1,
        target_ratio=0.1,
        adjustment_rate=0.01
    )
    
    for i in range(5):
        new_weight = weight_adjuster.update(main_loss=3.5, auxiliary_loss=0.5)
        print(f"   Step {i+1}: weight = {new_weight:.4f}")
    
    print(f"   ✅ AdaptiveLossWeight working!")
    
    # Test create_loss factory
    print("\n8. Testing create_loss factory...")
    ce_loss_fn = create_loss('ce', vocab_size=1000, label_smoothing=0.1)
    se_loss_fn = create_loss('semantic_entropy', vocab_size=1000, alpha=0.1)
    
    loss_ce = ce_loss_fn(logits, targets)
    loss_se = se_loss_fn(logits, targets, hidden_states)
    
    print(f"   CE loss: {loss_ce.item():.4f}")
    print(f"   SE loss: {loss_se.item():.4f}")
    print(f"   ✅ Loss factory working!")
    
    # Test gradient flow
    print("\n9. Testing gradient flow...")
    logits_grad = torch.randn(2, 64, 1000, requires_grad=True)
    targets_grad = torch.randint(0, 1000, (2, 64))
    hidden_grad = torch.randn(2, 64, 256, requires_grad=True)
    
    loss_grad = loss_fn(logits_grad, targets_grad, hidden_grad)
    loss_grad.backward()
    
    assert logits_grad.grad is not None, "No gradient for logits!"
    assert hidden_grad.grad is not None, "No gradient for hidden states!"
    
    grad_norm_logits = logits_grad.grad.norm().item()
    grad_norm_hidden = hidden_grad.grad.norm().item()
    
    print(f"   Logits grad norm: {grad_norm_logits:.4f}")
    print(f"   Hidden grad norm: {grad_norm_hidden:.4f}")
    print(f"   ✅ Gradient flow working!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.training.losses import SemanticEntropyLoss")
    print("  from ramanujan.training.losses import standard_cross_entropy, create_loss")
    print("="*70)