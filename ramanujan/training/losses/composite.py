"""Composite loss implementation."""

import torch
from typing import List, Tuple, Dict
from .base import LossSpec


class CompositeLoss:
    """
    Composite loss combining multiple loss functions.
    
    Args:
        loss_specs: List of LossSpec objects
    """
    
    component_type: str = 'loss'
    component_name: str = 'composite'
    
    def __init__(self, loss_specs: List[LossSpec]):
        from .factory import create_loss
        
        self.loss_specs = loss_specs
        self.losses = [
            (create_loss(spec.name, **spec.config), spec.weight)
            for spec in loss_specs
        ]
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            **kwargs: Additional arguments passed to individual losses
        
        Returns:
            loss: Weighted sum of losses
            metrics: Combined metrics with loss name prefixes
        """
        total_loss = 0.0
        all_metrics = {}
        
        for loss_fn, weight in self.losses:
            # Compute individual loss
            loss, metrics = loss_fn.compute(predictions, targets, **kwargs)
            
            # Accumulate weighted loss
            total_loss = total_loss + weight * loss
            
            # Add metrics with loss name prefix
            for key, value in metrics.items():
                prefixed_key = f"{loss_fn.component_name}/{key}"
                all_metrics[prefixed_key] = value
        
        # Add total loss to metrics
        all_metrics['total_loss'] = total_loss.item()
        
        return total_loss, all_metrics
    
    @classmethod
    def from_config(cls, config: Dict) -> 'CompositeLoss':
        """
        Create composite loss from config.
        
        Args:
            config: Dict with 'losses' key containing list of loss configs
        
        Returns:
            CompositeLoss instance
        """
        loss_specs = [
            LossSpec(
                name=loss_config['name'],
                weight=loss_config.get('weight', 1.0),
                config=loss_config.get('config', {})
            )
            for loss_config in config['losses']
        ]
        return cls(loss_specs)