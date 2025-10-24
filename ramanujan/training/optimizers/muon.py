"""
Muon Optimizer: Momentum Orthogonalized by Newton-schulz.

Muon optimizer with Newton-Schulz orthogonalization for better gradient
conditioning. Particularly effective for transformer training.

Reference:
    Muon optimizer paper/implementation
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Any, List


class MuonOptimizer(Optimizer):
    """
    Muon Optimizer with Newton-Schulz orthogonalization.
    
    Features:
    - Momentum-based updates
    - Periodic Newton-Schulz orthogonalization of weight matrices
    - Decoupled weight decay (AdamW-style)
    - Numerical stability checks
    
    Args:
        params: Iterable of parameters or parameter groups
        lr: Learning rate (default: 0.02, higher than Adam)
        momentum: Momentum factor (default: 0.95)
        weight_decay: Weight decay (L2 penalty) (default: 0.0)
        ns_steps: Steps between Newton-Schulz orthogonalization (default: 5)
        dampening: Dampening for momentum (default: 0.0)
        backend: Backend for operations ('newtonschulz5' or 'newtonschulz3', default: 'newtonschulz5')
        foreach: Use faster foreach implementation (default: True)
    
    Example:
        >>> optimizer = MuonOptimizer(
        ...     model.parameters(),
        ...     lr=0.02,
        ...     momentum=0.95,
        ...     weight_decay=0.1
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
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        dampening: float = 0.0,
        backend: str = 'newtonschulz5',
        foreach: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if backend not in ['newtonschulz5', 'newtonschulz3']:
            raise ValueError(f"Invalid backend: {backend}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            dampening=dampening,
            backend=backend,
            foreach=foreach
        )
        super().__init__(params, defaults)
    
    def _newton_schulz_5(self, G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """
        Newton-Schulz iteration (5th order) for orthogonalization.
        
        More stable and faster convergence than 3rd order.
        Computes an orthogonal matrix close to G.
        
        Args:
            G: Input matrix to orthogonalize
            steps: Number of iteration steps (default: 5)
            eps: Small constant for numerical stability (default: 1e-7)
        
        Returns:
            Orthogonalized matrix with same norm as input
        """
        # Skip small matrices
        if G.numel() < 4:
            return G
        
        # Compute initial normalization
        a = G.norm()
        if a < eps:
            return G
        
        # Normalize
        X = G / a
        
        # Newton-Schulz 5th order iteration
        # X_{k+1} = (1.5625 * X_k) @ (13*I - 4.625*A + A^3)
        # where A = X_k^T @ X_k
        for _ in range(steps):
            A = X.t() @ X
            B = A @ A @ A  # A^3
            
            # Compute update
            eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            update_matrix = 13 * eye - 4.625 * A + B
            X = 1.5625 * X @ update_matrix
        
        # Restore original scale
        return X * a
    
    def _newton_schulz_3(self, G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """
        Newton-Schulz iteration (3rd order) - simpler and more stable.
        
        Args:
            G: Input matrix to orthogonalize
            steps: Number of iteration steps (default: 5)
            eps: Small constant for numerical stability (default: 1e-7)
        
        Returns:
            Orthogonalized matrix with same norm as input
        """
        # Skip small matrices
        if G.numel() < 4:
            return G
        
        # Compute initial normalization
        a = G.norm()
        if a < eps:
            return G
        
        # Normalize
        X = G / a
        
        # Newton-Schulz 3rd order iteration
        # X_{k+1} = 0.5 * X_k @ (3*I - X_k^T @ X_k)
        for _ in range(steps):
            A = X.t() @ X
            eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            X = 0.5 * X @ (3 * eye - A)
        
        # Restore original scale
        return X * a
    
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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            ns_steps = group['ns_steps']
            backend = group['backend']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Check for NaN/Inf in gradients
                if not torch.isfinite(grad).all():
                    print(f"Warning: Non-finite gradient detected in Muon, skipping update")
                    continue
                
                # Apply decoupled weight decay (AdamW-style)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Get parameter state
                param_state = self.state[p]
                
                # Initialize state
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = param_state['momentum_buffer']
                param_state['step'] += 1
                
                # Update momentum buffer
                if param_state['step'] > 1:
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                else:
                    buf.copy_(grad)
                
                # Apply parameter update
                p.add_(buf, alpha=-lr)
                
                # Periodic Newton-Schulz orthogonalization
                # Only apply to 2D parameters (weight matrices, not biases)
                if param_state['step'] % ns_steps == 0 and p.dim() == 2:
                    # Choose backend
                    if backend == 'newtonschulz5':
                        p.data.copy_(self._newton_schulz_5(p.data, steps=5))
                    else:
                        p.data.copy_(self._newton_schulz_3(p.data, steps=5))
        
        return loss
    
    def __repr__(self) -> str:
        return (
            f"MuonOptimizer("
            f"lr={self.defaults['lr']}, "
            f"momentum={self.defaults['momentum']}, "
            f"weight_decay={self.defaults['weight_decay']}, "
            f"ns_steps={self.defaults['ns_steps']}, "
            f"backend={self.defaults['backend']}"
            ")"
        )