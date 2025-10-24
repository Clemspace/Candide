"""Cross-entropy loss implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class CrossEntropyLoss:
    """
    Cross-entropy loss with label smoothing support.
    
    Args:
        vocab_size: Size of vocabulary
        label_smoothing: Label smoothing factor (default: 0.0)
        ignore_index: Index to ignore (default: -100)
        reduction: Reduction method (default: 'mean')
    """
    
    component_type: str = 'loss'
    component_name: str = 'cross_entropy'
    loss_name: str = 'cross_entropy'  # For backward compatibility
    
    def __init__(
        self,
        vocab_size: int = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Only pass label_smoothing if > 0 to avoid type issues
        loss_kwargs = {
            'ignore_index': ignore_index,
            'reduction': reduction
        }
        if label_smoothing > 0:
            loss_kwargs['label_smoothing'] = label_smoothing
        
        self.loss_fn = nn.CrossEntropyLoss(**loss_kwargs)
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Logits (batch, vocab) or (batch, seq, vocab)
            targets: Target indices (batch,) or (batch, seq)
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary with accuracy, perplexity, ce_loss
        """
        # Handle both 2D and 3D tensors
        if predictions.ndim == 2:
            # (batch, vocab) - simple classification
            vocab_size = predictions.shape[1]
            predictions_flat = predictions
            targets_flat = targets
        elif predictions.ndim == 3:
            # (batch, seq, vocab) - sequence modeling
            batch_size, seq_length, vocab_size = predictions.shape
            predictions_flat = predictions.view(-1, vocab_size)
            targets_flat = targets.view(-1)
        else:
            raise ValueError(f"predictions must be 2D or 3D, got shape {predictions.shape}")
        
        # Compute loss
        loss = self.loss_fn(predictions_flat, targets_flat)
        
        # Compute metrics
        with torch.no_grad():
            # Accuracy (ignoring padding)
            pred_labels = predictions_flat.argmax(dim=-1)
            mask = targets_flat != self.ignore_index
            if mask.sum() > 0:
                accuracy = (pred_labels == targets_flat)[mask].float().mean().item()
            else:
                accuracy = 0.0
            
            # Perplexity
            perplexity = torch.exp(loss).item()
        
        metrics = {
            'ce_loss': loss.item(),
            'accuracy': accuracy,
            'perplexity': perplexity
        }
        
        return loss, metrics