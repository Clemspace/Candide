"""
Standard optimizers for Candide framework.

This module provides wrappers and implementations for:
- AdamW: Adam with decoupled weight decay
- SGD: Stochastic Gradient Descent with momentum
- Lion: Memory-efficient optimizer
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable
import math


# ============================================================================
# ADAMW
# ============================================================================

class AdamWOptimizer(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    
    This is a clean implementation of AdamW following the paper
    "Decoupled Weight Decay Regularization" by Loshchilov & Hutter.
    
    Args:
        params: Iterable of parameters or parameter groups
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        amsgrad: Whether to use AMSGrad variant (default: False)
        foreach: Use faster foreach implementation (default: True)
        fused: Use fused kernel if available (default: False)
    
    Example:
        >>> optimizer = AdamWOptimizer(
        ...     model.parameters(),
        ...     lr=0.001,
        ...     weight_decay=0.01
        ... )
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        foreach: bool = True,
        fused: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            fused=fused
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            lr = group['lr']
            amsgrad = group['amsgrad']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Check for NaN/Inf
                if not torch.isfinite(grad).all():
                    print(f"Warning: Non-finite gradient in AdamW, skipping")
                    continue
                
                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Get state
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                step_size = lr / bias_correction1
                
                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


# ============================================================================
# SGD
# ============================================================================

class SGDOptimizer(Optimizer):
    """
    Stochastic Gradient Descent with momentum.
    
    Simple but effective optimizer, good baseline for comparisons.
    
    Args:
        params: Iterable of parameters or parameter groups
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor (default: 0.9)
        weight_decay: Weight decay (L2 penalty) (default: 0.0)
        dampening: Dampening for momentum (default: 0.0)
        nesterov: Whether to use Nesterov momentum (default: False)
        foreach: Use faster foreach implementation (default: True)
    
    Example:
        >>> optimizer = SGDOptimizer(
        ...     model.parameters(),
        ...     lr=0.1,
        ...     momentum=0.9,
        ...     weight_decay=0.0001
        ... )
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        foreach: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            foreach=foreach
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    state = self.state[p]
                    
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                
                # Update parameters
                p.add_(grad, alpha=-lr)
        
        return loss


# ============================================================================
# LION
# ============================================================================

class LionOptimizer(Optimizer):
    """
    Lion optimizer: Memory-efficient adaptive optimizer.
    
    Lion uses sign of gradient (like SignSGD) but with momentum,
    resulting in memory usage similar to SGD but performance similar to AdamW.
    
    Reference:
        "Symbolic Discovery of Optimization Algorithms" by Chen et al.
    
    Args:
        params: Iterable of parameters or parameter groups
        lr: Learning rate (default: 0.0001, 10x smaller than AdamW)
        betas: Coefficients for computing running averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient (default: 0.0)
        foreach: Use faster foreach implementation (default: True)
    
    Example:
        >>> optimizer = LionOptimizer(
        ...     model.parameters(),
        ...     lr=0.0001,  # Note: much smaller than AdamW
        ...     weight_decay=0.1
        ... )
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        foreach: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            foreach=foreach
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Get state
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                
                # Weight decay (decoupled)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Lion update
                # update = sign(beta1 * m + (1 - beta1) * grad)
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)
                
                # Update momentum with current gradient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss
    
    def __repr__(self) -> str:
        return (
            f"LionOptimizer("
            f"lr={self.defaults['lr']}, "
            f"betas={self.defaults['betas']}, "
            f"weight_decay={self.defaults['weight_decay']}"
            ")"
        )