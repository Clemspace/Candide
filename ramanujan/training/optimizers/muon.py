"""
Muon: MomentUm Orthogonalized by Newton-Schulz.

Reference implementation followed here: Keller Jordan, "Muon: An optimizer for
hidden layers in neural networks" (2024), https://kellerjordan.github.io/posts/muon/
and https://github.com/KellerJordan/Muon. Scaling option "match_rms" from
Liu et al. (Moonshot AI), "Muon is Scalable for LLM Training", arXiv:2502.16982.

What Muon does, for a 2-D weight W with gradient G
---------------------------------------------------
1. Momentum (Nesterov by default):
       M <- beta * M + (1 - beta) * G
       U  = (1 - beta) * G + beta * M          (Nesterov look-ahead; U = M without)
2. Orthogonalize U: replace it by the nearest semi-orthogonal matrix
       O = U V^T  where  U = U S V^T  (SVD), i.e. all singular values set to 1.
   Computed approximately with 5 iterations of a quintic Newton-Schulz map
   (no SVD), in bfloat16. The map's coefficients (3.4445, -4.7750, 2.0315) are
   tuned for speed, not exactness: singular values land roughly in [0.5, 1.2],
   which works as well as exact orthogonalization in practice.
3. Step:  W <- W * (1 - lr * weight_decay) - lr * scale * O
   with scale = max(1, rows / cols) ** 0.5     ("spectral", Keller Jordan), or
        scale = 0.2 * max(rows, cols) ** 0.5   ("match_rms", Moonshot: the update
        then has the RMS of a typical AdamW update, so AdamW's LR and weight decay
        can be reused unchanged).

Because O has singular values ~1 whatever the size of G, the step size does
not depend on the gradient's scale: only the direction of the momentum is used,
with every direction in it given the same weight.

Which parameters use Muon
-------------------------
Only hidden-layer matrices. Embeddings, the output head, and all 1-D
parameters (norm gains, biases, scalars) must use AdamW: orthogonalizing an
embedding or a vocabulary projection is not what Muon is designed for. This
optimizer therefore contains an AdamW for those parameters:

    groups = muon_param_groups(model)        # embeddings / lm_head / 1-D -> AdamW
    opt = MuonOptimizer(groups, lr=0.02)

A param group may set use_muon=True/False explicitly; otherwise each
parameter with ndim >= 2 uses Muon and the rest AdamW.

Learning rates. The AdamW part runs at lr * adamw_lr_ratio, so a schedule that
sets group["lr"] scales both parts together. Default ratio: 1.0 with
"match_rms" (one LR for everything, as in Moonshot), 0.015 with "spectral"
(Keller Jordan's reference values: Muon 0.02, AdamW 3e-4).
"""

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

NS_COEFFS = (3.4445, -4.7750, 2.0315)
ADJUST_LR_MODES = ("spectral", "match_rms")


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """
    Approximate orthogonalization of a matrix (or a batch of matrices, last two dims).

    Returns X with the shape of G and singular values roughly in [0.5, 1.2]:
        X = a X + b (X X^T) X + c (X X^T)^2 X,   iterated `steps` times,
    starting from X = G / ||G||_F (so all singular values start in [0, 1]).
    Works on the wide orientation (rows <= cols) so X X^T is the small Gram matrix.
    steps=0 returns the Frobenius-normalized input.
    """
    assert G.ndim >= 2, "zeropower needs a matrix"
    a, b, c = NS_COEFFS
    X = G.to(dtype)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def muon_scale(shape: Sequence[int], mode: str) -> float:
    """Multiplier applied to the orthogonalized update of a (rows, cols) matrix."""
    rows, cols = shape[0], math.prod(shape[1:])
    if mode == "spectral":
        return max(1.0, rows / cols) ** 0.5
    if mode == "match_rms":
        return 0.2 * max(rows, cols) ** 0.5
    raise ValueError(f"Unknown adjust_lr mode: {mode}. Available: {ADJUST_LR_MODES}")


def muon_param_groups(model: nn.Module, head_names: Sequence[str] = ("lm_head",)) -> List[Dict]:
    """
    Split a model's parameters into a Muon group and an AdamW group.

    AdamW: weights of nn.Embedding modules (and anything tied to them), parameters
    whose name contains one of `head_names`, and every parameter with ndim < 2.
    Muon: the remaining matrices (attention and FFN projections).
    """
    adam_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            adam_ids.add(id(module.weight))
    muon, adam, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in adam_ids or p.ndim < 2 or any(h in name for h in head_names):
            adam.append(p)
        else:
            muon.append(p)
    groups = []
    if muon:
        groups.append({"params": muon, "use_muon": True})
    if adam:
        groups.append({"params": adam, "use_muon": False})
    return groups


class MuonOptimizer(Optimizer):
    """
    Muon for hidden matrices, AdamW for everything else (see module docstring).

    Args:
        params: parameters or param groups (groups may set use_muon)
        lr: Muon learning rate (0.02 with "spectral"; an AdamW-like value with "match_rms")
        momentum: Muon momentum beta (0.95)
        nesterov: Nesterov look-ahead (True, as in the reference)
        ns_steps: Newton-Schulz iterations per step (5)
        weight_decay: decoupled weight decay for Muon parameters
        adjust_lr: "spectral" or "match_rms" (see module docstring)
        adamw_lr_ratio: AdamW LR = lr * ratio; None -> 1.0 (match_rms) / 0.015 (spectral)
        adamw_betas, adamw_eps: AdamW part
        adamw_weight_decay: decoupled weight decay for the AdamW part (0.0)
        ns_dtype: dtype of the Newton-Schulz iteration (bfloat16, as in the reference)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adjust_lr: str = "spectral",
        adamw_lr_ratio: Optional[float] = None,
        adamw_betas=(0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
        ns_dtype: torch.dtype = torch.bfloat16,
        **unused,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if weight_decay < 0.0 or adamw_weight_decay < 0.0:
            raise ValueError("Invalid weight_decay")
        if adjust_lr not in ADJUST_LR_MODES:
            raise ValueError(f"Unknown adjust_lr mode: {adjust_lr}. Available: {ADJUST_LR_MODES}")
        if unused:
            # Accepted for backward compatibility with the old signature
            # (backend, dampening, warmup_steps, ...); they have no effect.
            unknown = set(unused) - {"backend", "dampening", "foreach", "warmup_steps",
                                     "scale_factor_base", "max_aspect_ratio", "eps",
                                     "ns_iterations_3", "ns_iterations_5"}
            if unknown:
                raise TypeError(f"Unexpected arguments: {sorted(unknown)}")
        if adamw_lr_ratio is None:
            adamw_lr_ratio = 1.0 if adjust_lr == "match_rms" else 0.015

        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            weight_decay=weight_decay, adjust_lr=adjust_lr, use_muon=None,
            adamw_lr_ratio=adamw_lr_ratio, adamw_betas=tuple(adamw_betas),
            adamw_eps=adamw_eps, adamw_weight_decay=adamw_weight_decay,
        )
        super().__init__(params, defaults)
        self.ns_dtype = ns_dtype

    @staticmethod
    def uses_muon(p: Tensor, group: Dict) -> bool:
        return group["use_muon"] if group["use_muon"] is not None else p.ndim >= 2

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.uses_muon(p, group):
                    self._muon_update(p, group)
                else:
                    self._adamw_update(p, group)
        return loss

    def _muon_update(self, p: Tensor, group: Dict) -> None:
        if p.ndim < 2:
            raise ValueError(f"Muon needs a matrix, got shape {tuple(p.shape)}; set use_muon=False")
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        buf = state["momentum_buffer"]
        beta = group["momentum"]
        grad = p.grad

        buf.lerp_(grad, 1 - beta)                                  # M = beta M + (1-beta) G
        update = grad.lerp(buf, beta) if group["nesterov"] else buf  # Nesterov look-ahead
        update = update.reshape(update.size(0), -1)                 # conv kernels -> matrix
        ortho = zeropower_via_newtonschulz5(update, group["ns_steps"], self.ns_dtype)
        scale = muon_scale(p.shape, group["adjust_lr"])

        lr = group["lr"]
        if group["weight_decay"] != 0:
            p.mul_(1 - lr * group["weight_decay"])
        p.add_(ortho.reshape(p.shape).to(p.dtype), alpha=-lr * scale)

    def _adamw_update(self, p: Tensor, group: Dict) -> None:
        state = self.state[p]
        if "exp_avg" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        state["step"] += 1
        t = state["step"]
        beta1, beta2 = group["adamw_betas"]
        lr = group["lr"] * group["adamw_lr_ratio"]
        grad = p.grad

        state["exp_avg"].lerp_(grad, 1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t
        denom = (state["exp_avg_sq"].sqrt() / bc2 ** 0.5).add_(group["adamw_eps"])

        if group["adamw_weight_decay"] != 0:
            p.mul_(1 - lr * group["adamw_weight_decay"])
        p.addcdiv_(state["exp_avg"], denom, value=-lr / bc1)

    def __repr__(self) -> str:
        d = self.defaults
        return (f"MuonOptimizer(lr={d['lr']}, momentum={d['momentum']}, nesterov={d['nesterov']}, "
                f"ns_steps={d['ns_steps']}, adjust_lr={d['adjust_lr']}, "
                f"adamw_lr_ratio={d['adamw_lr_ratio']})")
