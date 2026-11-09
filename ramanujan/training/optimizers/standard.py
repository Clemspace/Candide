"""
Standard optimizers for the Candide framework.

- AdamW and SGD are thin subclasses of torch.optim.AdamW / torch.optim.SGD, so
  they use PyTorch's multi-tensor ("foreach") or fused CUDA kernels instead of a
  Python loop over parameters. The math is PyTorch's, unchanged.
- Lion is implemented here (PyTorch has no Lion).

Note: the decoder/encoder pretraining scripts (train_candide18c.py,
train_encoder18c.py) already call torch.optim.AdamW directly with fused=True;
these classes serve the generic factory (create_optimizer).
"""

from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


def _all_params(params):
    """
    Materialize params (tensors, or param-group dicts) so they can be inspected
    without consuming a generator. Returns (params for the optimizer, flat tensors).
    """
    out, tensors = [], []
    for item in params:
        if isinstance(item, dict):
            group_params = item["params"]
            group_params = [group_params] if isinstance(group_params, torch.Tensor) else list(group_params)
            out.append({**item, "params": group_params})
            tensors.extend(group_params)
        else:
            out.append(item)
            tensors.append(item)
    return out, tensors


def _can_fuse(tensors) -> bool:
    """PyTorch's fused AdamW needs floating-point tensors on CUDA."""
    return len(tensors) > 0 and all(t.is_cuda and t.is_floating_point() for t in tensors)


# ============================================================================
# ADAMW
# ============================================================================

class AdamWOptimizer(torch.optim.AdamW):
    """
    AdamW (Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019).

        m <- beta1 m + (1 - beta1) g
        v <- beta2 v + (1 - beta2) g^2
        theta <- theta (1 - lr wd) - lr (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)

    Kernel choice: fused=None (default) uses the fused CUDA kernel when every
    parameter is a floating-point CUDA tensor, otherwise the foreach kernel.
    Non-finite gradients are not skipped here; use gradient clipping and, in
    fp16, a GradScaler (which skips steps with inf/nan).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        foreach: Optional[bool] = None,
        fused: Optional[bool] = None,
    ):
        params, tensors = _all_params(params)
        if fused is None:
            fused = _can_fuse(tensors)
        if fused:
            foreach = None  # PyTorch rejects fused=True together with foreach=True
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         amsgrad=amsgrad, foreach=foreach, fused=fused)


# ============================================================================
# SGD
# ============================================================================

class SGDOptimizer(torch.optim.SGD):
    """
    SGD with (optionally Nesterov) momentum and coupled L2 weight decay,
    i.e. torch.optim.SGD:

        g <- g + wd theta
        b <- momentum b + (1 - dampening) g        (b = g at the first step)
        theta <- theta - lr (g + momentum b if nesterov else b)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        foreach: Optional[bool] = None,
    ):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov, foreach=foreach)


# ============================================================================
# LION
# ============================================================================

class LionOptimizer(Optimizer):
    """
    Lion (Chen et al., "Symbolic Discovery of Optimization Algorithms", NeurIPS 2023).

        c <- beta1 m + (1 - beta1) g            interpolation used for the step
        theta <- theta (1 - lr wd) - lr sign(c)
        m <- beta2 m + (1 - beta2) g            momentum update, after the step

    Every coordinate moves by exactly lr (sign update), so Lion's LR is
    typically 3-10x smaller than AdamW's and its weight decay 3-10x larger,
    keeping lr * wd similar. One state tensor per parameter (AdamW has two).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.99),
        weight_decay: float = 0.0,
        foreach: Optional[bool] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(torch.lerp(grad, m, beta1).sign_(), alpha=-lr)   # beta1 m + (1-beta1) g
                m.lerp_(grad, 1 - beta2)
        return loss

    def __repr__(self) -> str:
        d = self.defaults
        return f"LionOptimizer(lr={d['lr']}, betas={d['betas']}, weight_decay={d['weight_decay']})"
