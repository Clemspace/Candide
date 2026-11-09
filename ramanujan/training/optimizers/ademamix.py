"""
AdEMAMix: Adam with a mixture of a fast and a slow EMA of the gradients.

Reference: Pagliardini, Ablin & Grangier, "The AdEMAMix Optimizer: Better,
Faster, Older", arXiv:2409.03137 (ICLR 2025); official code apple/ml-ademamix.

Motivation
----------
Adam's momentum (beta1 = 0.9) averages roughly the last 10 gradients. Old
gradients still carry information, but a single EMA cannot both react quickly
and remember far back. AdEMAMix keeps Adam's fast EMA and adds a slow one
(beta3 = 0.9999, a memory of ~10k steps), mixed with weight alpha.

Update (for each parameter, step t >= 1, gradient g)
----------------------------------------------------
    m1 <- beta1 m1 + (1 - beta1) g                fast EMA (Adam's momentum)
    m2 <- beta3 m2 + (1 - beta3) g                slow EMA, NOT bias-corrected
    v  <- beta2 v  + (1 - beta2) g^2              second moment
    m1_hat = m1 / (1 - beta1^t)
    v_hat  = v  / (1 - beta2^t)
    theta <- theta (1 - lr wd) - lr (m1_hat + alpha m2) / (sqrt(v_hat) + eps)

With alpha = 0 this is exactly AdamW. Note the naming: beta2 is the second
moment and beta3 the slow EMA, as in the paper.

Warmups (from the paper; needed for stability early in training, when the
slow EMA is still near zero but would otherwise get a large weight):
  - alpha_warmup = T: alpha rises linearly from 0 to alpha over T steps.
  - beta3_warmup = T: beta3 rises from beta1 to beta3 over T steps, linearly in
    the EMA's half-life (not in beta), so the memory length grows steadily.
The paper's language-model runs use T equal to the total number of steps.
"""

import math
from typing import Callable, Optional

import torch
from torch.optim import Optimizer


def half_life(beta: float) -> float:
    """Number of steps after which a weight decays by half under EMA beta (minus 1)."""
    return math.log(0.5) / math.log(beta) - 1


def beta_from_half_life(h: float) -> float:
    return 0.5 ** (1 / (h + 1))


def beta3_schedule(step: int, beta3: float, beta_start: float, warmup: Optional[int]) -> float:
    """beta3 at `step`: linear in half-life from beta_start to beta3 over `warmup` steps."""
    if not warmup or step >= warmup:
        return beta3
    a = step / warmup
    return beta_from_half_life((1 - a) * half_life(beta_start) + a * half_life(beta3))


def alpha_schedule(step: int, alpha: float, warmup: Optional[int]) -> float:
    """alpha at `step`: linear from 0 to alpha over `warmup` steps."""
    if not warmup or step >= warmup:
        return alpha
    return alpha * step / warmup


class AdEMAMixOptimizer(Optimizer):
    """
    AdEMAMix (see module docstring).

    Args:
        params: parameters or param groups
        lr: learning rate
        beta1: fast EMA (0.9)
        beta2: second-moment EMA (0.999)
        beta3: slow EMA (0.9999)
        alpha: weight of the slow EMA (paper: 5 to 8 for language models)
        eps: added to sqrt(v_hat)
        weight_decay: decoupled weight decay
        alpha_warmup, beta3_warmup: warmup lengths in steps (None = no warmup)
        foreach: accepted for compatibility, unused
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.9999,
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        alpha_warmup: Optional[int] = None,
        beta3_warmup: Optional[int] = None,
        foreach: Optional[bool] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        for name, b in (("beta1", beta1), ("beta2", beta2), ("beta3", beta3)):
            if not 0.0 <= b < 1.0:
                raise ValueError(f"Invalid {name}: {b}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, alpha=alpha, eps=eps,
                        weight_decay=weight_decay, alpha_warmup=alpha_warmup,
                        beta3_warmup=beta3_warmup)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["beta1"], group["beta2"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)   # fast EMA
                    state["m2"] = torch.zeros_like(p)   # slow EMA
                    state["v"] = torch.zeros_like(p)    # second moment
                state["step"] += 1
                t = state["step"]

                alpha = alpha_schedule(t, group["alpha"], group["alpha_warmup"])
                beta3 = beta3_schedule(t, group["beta3"], beta1, group["beta3_warmup"])

                m1, m2, v = state["m1"], state["m2"], state["v"]
                m1.lerp_(grad, 1 - beta1)
                m2.lerp_(grad, 1 - beta3)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                numer = m1 / bc1
                if alpha != 0:
                    numer.add_(m2, alpha=alpha)

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(numer, denom, value=-lr)
        return loss

    def __repr__(self) -> str:
        d = self.defaults
        return (f"AdEMAMixOptimizer(lr={d['lr']}, betas=({d['beta1']}, {d['beta2']}, {d['beta3']}), "
                f"alpha={d['alpha']}, alpha_warmup={d['alpha_warmup']}, "
                f"beta3_warmup={d['beta3_warmup']})")
