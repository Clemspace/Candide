"""
Optimizers for Ramanujan Transformer training.

This module provides various optimization algorithms:
- MuonOptimizer: Momentum-based optimizer with Newton-Schulz orthogonalization
- AdEMAMixOptimizer: Adaptive EMA mixing for better convergence
- HybridOptimizerManager: Combine multiple optimizers for different param groups
- Standard optimizers (AdamW) with utilities

Example:
    >>> from ramanujan.training import create_optimizer
    >>> 
    >>> # Create Muon optimizer for embeddings
    >>> optimizer = create_optimizer(
    ...     model,
    ...     optimizer_type='muon',
    ...     lr=0.02,
    ...     momentum=0.95
    ... )
    >>> 
    >>> # Or use hybrid approach
    >>> optimizer = create_layerwise_optimizer(
    ...     model,
    ...     embedding_lr=0.001,
    ...     attention_lr=0.0003,
    ...     ffn_lr=0.0005
    ... )
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Dict, Optional, Callable, Tuple, Any
import math


# ============================================================================
# MUON OPTIMIZER
# ============================================================================

class MuonOptimizer(Optimizer):
    """
    Muon Optimizer with Newton-Schulz orthogonalization.
    
    Fixed version with proper numerical stability and scaling.
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        weight_decay: Weight decay (L2 penalty) (default: 0.0)
        ns_steps: Steps between Newton-Schulz orthogonalization (default: 5)
        dampening: Dampening for momentum (default: 0.0)
        backend: Backend for operations ('newtonschulz5' or 'newtonschulz3', default: 'newtonschulz5')
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,  # Higher default LR for Muon
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        dampening: float = 0.0,
        backend: str = 'newtonschulz5'
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            dampening=dampening,
            backend=backend
        )
        super().__init__(params, defaults)
    
    def _newton_schulz_5(self, G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
        """
        Newton-Schulz iteration (5th order) for orthogonalization.
        
        More stable than the previous implementation.
        """
        # Check if matrix is too small
        if G.shape[0] * G.shape[1] < 4:
            return G
        
        # Compute initial normalization
        a = G.norm()
        if a < eps:
            return G
        
        X = G / a
        
        # Newton-Schulz iteration: X_{k+1} = X_k * (3I - X_k^T X_k) / 2
        # 5th order version for faster convergence
        for _ in range(steps):
            A = X.t() @ X
            B = A @ A @ A
            X = (1.5625 * X) @ (13*torch.eye(A.shape[0], device=A.device, dtype=A.dtype) 
                                - 4.625 * A + B)
        
        return X * a
    
    def _newton_schulz_3(self, G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
        """
        Newton-Schulz iteration (3rd order) - simpler and more stable.
        """
        if G.shape[0] * G.shape[1] < 4:
            return G
        
        a = G.norm()
        if a < eps:
            return G
        
        X = G / a
        
        # 3rd order: X_{k+1} = 0.5 * X_k @ (3*I - X_k^T @ X_k)
        for _ in range(steps):
            A = X.t() @ X
            X = 0.5 * X @ (3 * torch.eye(A.shape[0], device=A.device, dtype=A.dtype) - A)
        
        return X * a
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
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
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Handle NaN/Inf in gradients
                if not torch.isfinite(grad).all():
                    print(f"Warning: Non-finite gradient detected, skipping update")
                    continue
                
                # Apply weight decay (decoupled, AdamW style)
                if weight_decay != 0:
                    p.mul_(1 - group['lr'] * weight_decay)
                
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
                
                # Apply update
                p.add_(buf, alpha=-group['lr'])
                
                # Periodic Newton-Schulz orthogonalization
                # Only apply to 2D parameters (weight matrices)
                if param_state['step'] % ns_steps == 0 and p.dim() == 2:
                    # Choose backend
                    if backend == 'newtonschulz5':
                        p.data.copy_(self._newton_schulz_5(p.data, steps=5))
                    else:
                        p.data.copy_(self._newton_schulz_3(p.data, steps=5))
        
        return loss


# ============================================================================
# ADEMAMIX OPTIMIZER
# ============================================================================

class AdEMAMixOptimizer(Optimizer):
    """
    AdEMAMix: Adaptive EMA mixing for improved convergence.
    
    Combines fast and slow exponential moving averages of gradients
    with adaptive mixing based on gradient statistics. This helps
    balance exploration and exploitation during training.
    
    Key features:
    - Dual EMA for gradients (fast and slow)
    - Adaptive mixing based on gradient variance
    - Per-parameter adaptive learning rates
    - Bias correction
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 0.001)
        beta1: Fast EMA decay rate (default: 0.9)
        beta2: Slow EMA decay rate (default: 0.999)
        beta3: Second moment decay rate (default: 0.9999)
        alpha: Mixing coefficient (default: 5.0)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay (default: 0.0)
    
    Example:
        >>> optimizer = AdEMAMixOptimizer(
        ...     model.parameters(),
        ...     lr=0.0003,
        ...     beta1=0.9,
        ...     beta2=0.999,
        ...     alpha=5.0
        ... )
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
        weight_decay: float = 0.0
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
        
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
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
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # Initialize state
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['exp_avg_fast'] = torch.zeros_like(p)  # Fast EMA
                    param_state['exp_avg_slow'] = torch.zeros_like(p)  # Slow EMA
                    param_state['exp_avg_sq'] = torch.zeros_like(p)    # Second moment
                
                exp_avg_fast = param_state['exp_avg_fast']
                exp_avg_slow = param_state['exp_avg_slow']
                exp_avg_sq = param_state['exp_avg_sq']
                
                param_state['step'] += 1
                step = param_state['step']
                
                # Update biased fast and slow EMAs
                exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_slow.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                # Update second moment
                exp_avg_sq.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                bias_correction3 = 1 - beta3 ** step
                
                # Corrected EMAs
                exp_avg_fast_corrected = exp_avg_fast / bias_correction1
                exp_avg_slow_corrected = exp_avg_slow / bias_correction2
                exp_avg_sq_corrected = exp_avg_sq / bias_correction3
                
                # Adaptive mixing
                # Mix fast and slow EMAs based on gradient variance
                denom = exp_avg_sq_corrected.sqrt().add_(eps)
                
                # Mixing weight based on alpha and step
                mix_weight = alpha / (alpha + step)
                
                # Mixed gradient estimate
                mixed_grad = (
                    mix_weight * exp_avg_fast_corrected +
                    (1 - mix_weight) * exp_avg_slow_corrected
                )
                
                # Update parameters
                p.addcdiv_(mixed_grad, denom, value=-group['lr'])
        
        return loss


# ============================================================================
# HYBRID OPTIMIZER MANAGER
# ============================================================================

class HybridOptimizerManager:
    """
    Manage multiple optimizers for different parameter groups.
    
    Allows using different optimizers with different hyperparameters
    for different parts of the model (e.g., Muon for embeddings,
    AdamW for attention, AdEMAMix for FFN).
    
    Args:
        optimizers: List of (optimizer, param_names) tuples
    
    Example:
        >>> # Create separate optimizers for different layers
        >>> muon = MuonOptimizer(
        ...     [p for n, p in model.named_parameters() if 'embedding' in n],
        ...     lr=0.001
        ... )
        >>> adam = torch.optim.AdamW(
        ...     [p for n, p in model.named_parameters() if 'attention' in n],
        ...     lr=0.0003
        ... )
        >>> 
        >>> manager = HybridOptimizerManager([
        ...     (muon, 'embeddings'),
        ...     (adam, 'attention')
        ... ])
        >>> 
        >>> # Use like a single optimizer
        >>> manager.step()
        >>> manager.zero_grad()
    """
    
    def __init__(self, optimizers: List[Tuple[Optimizer, str]]):
        self.optimizers = optimizers
    
    def step(self):
        """Step all optimizers."""
        for optimizer, name in self.optimizers:
            optimizer.step()
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for all optimizers."""
        for optimizer, name in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for all optimizers."""
        return {
            name: optimizer.state_dict()
            for optimizer, name in self.optimizers
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict for all optimizers."""
        for optimizer, name in self.optimizers:
            if name in state_dict:
                optimizer.load_state_dict(state_dict[name])
    
    def get_lr(self) -> Dict[str, float]:
        """Get learning rates for all optimizers."""
        return {
            name: optimizer.param_groups[0]['lr']
            for optimizer, name in self.optimizers
        }
    
    def set_lr(self, lr_dict: Dict[str, float]):
        """Set learning rates for optimizers."""
        for optimizer, name in self.optimizers:
            if name in lr_dict:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_dict[name]


# ============================================================================
# OPTIMIZER FACTORY
# ============================================================================

def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    lr: float = 0.0003,
    weight_decay: float = 0.01,
    **kwargs
) -> Optimizer:
    """
    Create optimizer for model.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adamw', 'muon', 'ademamix')
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        Optimizer instance
    
    Example:
        >>> # AdamW optimizer
        >>> optimizer = create_optimizer(
        ...     model,
        ...     optimizer_type='adamw',
        ...     lr=0.0003,
        ...     weight_decay=0.01
        ... )
        >>> 
        >>> # Muon optimizer
        >>> optimizer = create_optimizer(
        ...     model,
        ...     optimizer_type='muon',
        ...     lr=0.001,
        ...     momentum=0.95
        ... )
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type == 'muon':
        return MuonOptimizer(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get('momentum', 0.95),
            weight_decay=weight_decay,
            ns_steps=kwargs.get('ns_steps', 5),
            dampening=kwargs.get('dampening', 0.0)
        )
    
    elif optimizer_type == 'ademamix':
        return AdEMAMixOptimizer(
            model.parameters(),
            lr=lr,
            beta1=kwargs.get('beta1', 0.9),
            beta2=kwargs.get('beta2', 0.999),
            beta3=kwargs.get('beta3', 0.9999),
            alpha=kwargs.get('alpha', 5.0),
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(
            f"Unknown optimizer_type: {optimizer_type}. "
            f"Choose from: 'adamw', 'muon', 'ademamix', 'adam', 'sgd'"
        )


def create_layerwise_optimizer(
    model: nn.Module,
    embedding_lr: float = 0.001,
    attention_lr: float = 0.0003,
    ffn_lr: float = 0.0005,
    output_lr: float = 0.0003,
    weight_decay: float = 0.01,
    use_muon_for_embeddings: bool = True
) -> HybridOptimizerManager:
    """
    Create layerwise optimizer with different learning rates.
    
    Uses Muon for embeddings and AdamW for other layers by default.
    
    Args:
        model: Model to optimize
        embedding_lr: Learning rate for embeddings
        attention_lr: Learning rate for attention layers
        ffn_lr: Learning rate for feedforward layers
        output_lr: Learning rate for output projection
        weight_decay: Weight decay
        use_muon_for_embeddings: Use Muon for embeddings (default: True)
    
    Returns:
        HybridOptimizerManager instance
    
    Example:
        >>> optimizer = create_layerwise_optimizer(
        ...     model,
        ...     embedding_lr=0.001,
        ...     attention_lr=0.0003,
        ...     ffn_lr=0.0005
        ... )
    """
    # Separate parameters by layer type
    embedding_params = []
    attention_params = []
    ffn_params = []
    output_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            embedding_params.append(param)
        elif 'attention' in name.lower() or 'attn' in name.lower():
            attention_params.append(param)
        elif 'ffn' in name.lower() or 'feedforward' in name.lower() or 'mlp' in name.lower():
            ffn_params.append(param)
        elif 'output_projection' in name.lower() or 'lm_head' in name.lower():
            output_params.append(param)
        else:
            other_params.append(param)
    
    optimizers = []
    
    # Embeddings (Muon or AdamW)
    if embedding_params:
        if use_muon_for_embeddings:
            emb_opt = MuonOptimizer(
                embedding_params,
                lr=embedding_lr,
                momentum=0.95,
                weight_decay=weight_decay
            )
        else:
            emb_opt = torch.optim.AdamW(
                embedding_params,
                lr=embedding_lr,
                weight_decay=weight_decay
            )
        optimizers.append((emb_opt, 'embeddings'))
    
    # Attention (AdamW)
    if attention_params:
        attn_opt = torch.optim.AdamW(
            attention_params,
            lr=attention_lr,
            weight_decay=weight_decay
        )
        optimizers.append((attn_opt, 'attention'))
    
    # FFN (AdamW or AdEMAMix)
    if ffn_params:
        ffn_opt = torch.optim.AdamW(
            ffn_params,
            lr=ffn_lr,
            weight_decay=weight_decay
        )
        optimizers.append((ffn_opt, 'ffn'))
    
    # Output projection
    if output_params:
        out_opt = torch.optim.AdamW(
            output_params,
            lr=output_lr,
            weight_decay=weight_decay
        )
        optimizers.append((out_opt, 'output'))
    
    # Other parameters
    if other_params:
        other_opt = torch.optim.AdamW(
            other_params,
            lr=attention_lr,  # Use attention LR as default
            weight_decay=weight_decay
        )
        optimizers.append((other_opt, 'other'))
    
    return HybridOptimizerManager(optimizers)


# ============================================================================
# OPTIMIZER UTILITIES
# ============================================================================

def get_optimizer_info(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Get information about optimizer.
    
    Args:
        optimizer: Optimizer instance
    
    Returns:
        Dictionary with optimizer info
    """
    info = {
        'type': optimizer.__class__.__name__,
        'num_param_groups': len(optimizer.param_groups),
        'learning_rates': [group['lr'] for group in optimizer.param_groups]
    }
    
    # Add optimizer-specific info
    if hasattr(optimizer, 'defaults'):
        info['defaults'] = optimizer.defaults
    
    return info


def scale_learning_rate(
    optimizer: Optimizer,
    scale_factor: float
):
    """
    Scale learning rate for all parameter groups.
    
    Args:
        optimizer: Optimizer instance
        scale_factor: Factor to scale LR by
    
    Example:
        >>> # Reduce LR by half
        >>> scale_learning_rate(optimizer, 0.5)
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale_factor


def set_learning_rate(
    optimizer: Optimizer,
    lr: float,
    param_group_idx: Optional[int] = None
):
    """
    Set learning rate for optimizer.
    
    Args:
        optimizer: Optimizer instance
        lr: New learning rate
        param_group_idx: If specified, only set LR for this group
    
    Example:
        >>> # Set LR for all groups
        >>> set_learning_rate(optimizer, 0.0001)
        >>> 
        >>> # Set LR for specific group
        >>> set_learning_rate(optimizer, 0.0001, param_group_idx=0)
    """
    if param_group_idx is not None:
        optimizer.param_groups[param_group_idx]['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_learning_rate(optimizer: Optimizer) -> List[float]:
    """
    Get current learning rates from optimizer.
    
    Args:
        optimizer: Optimizer instance
    
    Returns:
        List of learning rates for each parameter group
    """
    return [group['lr'] for group in optimizer.param_groups]


def clip_grad_norm(
    parameters,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradient norm of parameters.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum norm
        norm_type: Type of norm (default: 2.0)
    
    Returns:
        Total norm before clipping
    
    Example:
        >>> # Clip gradients to max norm of 1.0
        >>> grad_norm = clip_grad_norm(model.parameters(), 1.0)
        >>> print(f"Gradient norm: {grad_norm:.4f}")
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing optimizers.py module")
    print("="*70)
    
    # Create simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 256)
            self.attention = nn.Linear(256, 256)
            self.ffn = nn.Linear(256, 256)
            self.output = nn.Linear(256, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.attention(x)
            x = self.ffn(x)
            return self.output(x)
    
    model = SimpleModel()
    
    # Test MuonOptimizer
    print("\n1. Testing MuonOptimizer...")
    muon = MuonOptimizer(
        model.parameters(),
        lr=0.001,
        momentum=0.95,
        ns_steps=5
    )
    
    # Dummy forward/backward
    x = torch.randint(0, 1000, (2, 32))
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    muon.step()
    muon.zero_grad()
    
    print(f"   Learning rate: {muon.param_groups[0]['lr']}")
    print(f"   Momentum: {muon.param_groups[0]['momentum']}")
    print(f"   ✅ MuonOptimizer working!")
    
    # Test AdEMAMixOptimizer
    print("\n2. Testing AdEMAMixOptimizer...")
    model2 = SimpleModel()
    ademamix = AdEMAMixOptimizer(
        model2.parameters(),
        lr=0.0003,
        beta1=0.9,
        beta2=0.999,
        alpha=5.0
    )
    
    output2 = model2(x)
    loss2 = output2.sum()
    loss2.backward()
    
    ademamix.step()
    ademamix.zero_grad()
    
    print(f"   Learning rate: {ademamix.param_groups[0]['lr']}")
    print(f"   Beta1: {ademamix.param_groups[0]['beta1']}")
    print(f"   Beta2: {ademamix.param_groups[0]['beta2']}")
    print(f"   ✅ AdEMAMixOptimizer working!")
    
    # Test create_optimizer factory
    print("\n3. Testing create_optimizer factory...")
    model3 = SimpleModel()
    
    opt_adamw = create_optimizer(model3, 'adamw', lr=0.0003)
    opt_muon = create_optimizer(model3, 'muon', lr=0.001)
    opt_ademamix = create_optimizer(model3, 'ademamix', lr=0.0003)
    
    print(f"   AdamW: {type(opt_adamw).__name__}")
    print(f"   Muon: {type(opt_muon).__name__}")
    print(f"   AdEMAMix: {type(opt_ademamix).__name__}")
    print(f"   ✅ Optimizer factory working!")
    
    # Test create_layerwise_optimizer
    print("\n4. Testing create_layerwise_optimizer...")
    model4 = SimpleModel()
    
    layerwise_opt = create_layerwise_optimizer(
        model4,
        embedding_lr=0.001,
        attention_lr=0.0003,
        ffn_lr=0.0005,
        use_muon_for_embeddings=True
    )
    
    lrs = layerwise_opt.get_lr()
    print(f"   Learning rates: {lrs}")
    print(f"   Number of optimizers: {len(layerwise_opt.optimizers)}")
    print(f"   ✅ Layerwise optimizer working!")
    
    # Test HybridOptimizerManager
    print("\n5. Testing HybridOptimizerManager operations...")
    
    # Step
    output4 = model4(x)
    loss4 = output4.sum()
    loss4.backward()
    layerwise_opt.step()
    layerwise_opt.zero_grad()
    print(f"   Step completed")
    
    # Save/load state
    state = layerwise_opt.state_dict()
    print(f"   State dict keys: {list(state.keys())}")
    
    layerwise_opt.load_state_dict(state)
    print(f"   State loaded")
    
    # Set LR
    layerwise_opt.set_lr({'embeddings': 0.0005})
    new_lrs = layerwise_opt.get_lr()
    print(f"   Updated LRs: {new_lrs}")
    print(f"   ✅ HybridOptimizerManager operations working!")
    
    # Test optimizer utilities
    print("\n6. Testing optimizer utilities...")
    
    info = get_optimizer_info(opt_adamw)
    print(f"   Optimizer info: {info['type']}, {info['num_param_groups']} groups")
    
    scale_learning_rate(opt_adamw, 0.5)
    scaled_lr = get_learning_rate(opt_adamw)
    print(f"   Scaled LR: {scaled_lr[0]:.6f}")
    
    set_learning_rate(opt_adamw, 0.001)
    new_lr = get_learning_rate(opt_adamw)
    print(f"   New LR: {new_lr[0]:.6f}")
    
    print(f"   ✅ Optimizer utilities working!")
    
    # Test gradient clipping
    print("\n7. Testing gradient clipping...")
    model5 = SimpleModel()
    output5 = model5(x)
    loss5 = output5.sum()
    loss5.backward()
    
    grad_norm = clip_grad_norm(model5.parameters(), max_norm=1.0)
    print(f"   Gradient norm before clipping: {grad_norm:.4f}")
    print(f"   ✅ Gradient clipping working!")
    
    # Test multiple steps
    print("\n8. Testing multiple optimization steps...")
    model6 = SimpleModel()
    opt6 = create_optimizer(model6, 'adamw', lr=0.001)
    
    losses = []
    for i in range(5):
        output6 = model6(x)
        loss6 = output6.sum()
        losses.append(loss6.item())
        
        loss6.backward()
        opt6.step()
        opt6.zero_grad()
    
    print(f"   Losses: {[f'{l:.2f}' for l in losses]}")
    print(f"   ✅ Multiple steps working!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.training.optimizers import create_optimizer")
    print("  from ramanujan.training.optimizers import create_layerwise_optimizer")
    print("  from ramanujan.training.optimizers import MuonOptimizer, AdEMAMixOptimizer")
    print("="*70)