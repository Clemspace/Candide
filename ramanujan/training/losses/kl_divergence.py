"""KL divergence loss implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class KLDivergenceLoss:
    """
    KL divergence loss for knowledge distillation.
    
    Args:
        temperature: Temperature for softmax (default: 1.0)
        reduction: Reduction method (default: 'batchmean')
    """
    
    component_type: str = 'loss'
    component_name: str = 'kl_divergence'
    
    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = 'batchmean'
    ):
        self.temperature = temperature
        self.reduction = reduction
        self.loss_fn = nn.KLDivLoss(reduction=reduction)
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reference_logits: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute KL divergence loss.
        
        Args:
            predictions: Student logits (batch, seq, vocab)
            targets: Not used (for API consistency)
            reference_logits: Teacher logits (batch, seq, vocab) - REQUIRED
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary with kl_loss
        """
        if reference_logits is None:
            raise ValueError(
                "KLDivergenceLoss requires 'reference_logits' kwarg. "
                "Usage: loss_fn.compute(pred, targets, reference_logits=teacher)"
            )
        
        # Flatten to (batch * seq, vocab)
        vocab_size = predictions.size(-1)
        predictions_flat = predictions.view(-1, vocab_size)
        reference_flat = reference_logits.view(-1, vocab_size)
        
        # Convert to log-probs and probs
        student_log_probs = F.log_softmax(predictions_flat / self.temperature, dim=-1)
        teacher_probs = F.softmax(reference_flat / self.temperature, dim=-1)
        
        # Compute KL divergence
        loss = self.loss_fn(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        metrics = {
            'kl_loss': loss.item(),
            'temperature': self.temperature
        }
        
        return loss, metrics