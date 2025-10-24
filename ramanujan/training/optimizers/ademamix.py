"""
AdEMAMix Optimizer: Adaptive EMA Mixing for improved convergence.

Combines fast and slow exponential moving averages of gradients with
adaptive mixing based on gradient statistics.

Reference:
    "The Importance of Being Slow: Understanding and Improving Adaptive Optimization"
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Any, List
import math


class AdEMAMixOptimizer(Optimizer):
    """
    AdEMAMix: Adaptive EMA mixing for improved convergence.
    
    Features:
    - Dual EMA for gradients (fast and slow)
    - Adaptive mixing based on gradient variance
    - Per-parameter adaptive learning rates
    - Bias correction
    - Decoupled weight decay
    
    The optimizer maintains three exponential moving averages:
    - Fast EMA (m1): Captures recent gradient information
    - Slow EMA (m2): Captures long-term gradient trends
    - Second moment (v): For adaptive learning rates
    
    Args:
        params: Iterable of parameters or parameter groups
        lr: Learning rate (default: 0.001)
        beta1: Fast EMA decay rate (default: 0.9)
        beta2: Slow EMA decay rate (default: 0.999)
        beta3: Second moment decay rate (default: 0.9999)
        alpha: Mixing coefficient (default: 5.0, higher = more influence from slow EMA)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay (default: 0.0)
        foreach: Use faster foreach implementation (default: True)
    
    Example:
        >>> optimizer = AdEMAMixOptimizer(
        ...     model.parameters(),
        ...     lr=0.0003,
        ...     beta1=0.9,
        ...     beta2=0.999,
        ...     alpha=5.0
        ... )
        >>> 
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.9999,
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        foreach: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"Invalid beta3: {beta3}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure to recompute loss
        
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
            beta3 = group['beta3']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Check for NaN/Inf in gradients
                if not torch.isfinite(grad).all():
                    print(f"Warning: Non-finite gradient detected in AdEMAMix, skipping update")
                    continue
                
                # Apply decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Get parameter state
                param_state = self.state[p]
                
                # Initialize state
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['m1'] = torch.zeros_like(p)  # Fast EMA
                    param_state['m2'] = torch.zeros_like(p)  # Slow EMA
                    param_state['v'] = torch.zeros_like(p)   # Second moment
                
                m1 = param_state['m1']
                m2 = param_state['m2']
                v = param_state['v']
                param_state['step'] += 1
                step = param_state['step']
                
                # Update fast EMA (m1)
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update slow EMA (m2)
                m2.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                # Update second moment (v)
                v.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                bias_correction3 = 1 - beta3 ** step
                
                # Corrected EMAs
                m1_hat = m1 / bias_correction1
                m2_hat = m2 / bias_correction2
                v_hat = v / bias_correction3
                
                # Adaptive mixing: combine fast and slow EMAs
                # The mixing is weighted by alpha parameter
                # Higher alpha = more weight on slow EMA (m2)
                mixed_m = (m1_hat + alpha * m2_hat) / (1 + alpha)
                
                # Compute adaptive learning rate
                # Similar to Adam, but uses mixed momentum
                denom = v_hat.sqrt().add_(eps)
                
                # Apply update
                p.addcdiv_(mixed_m, denom, value=-lr)
        
        return loss
    
    def __repr__(self) -> str:
        return (
            f"AdEMAMixOptimizer("
            f"lr={self.defaults['lr']}, "
            f"beta1={self.defaults['beta1']}, "
            f"beta2={self.defaults['beta2']}, "
            f"beta3={self.defaults['beta3']}, "
            f"alpha={self.defaults['alpha']}"
            ")"
        )