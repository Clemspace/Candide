"""Primitives for approximately normalized transformers (anGPT).

Reference: Franke et al., "Learning in Compact Spaces with Approximately
Normalized Transformers", arXiv:2505.22014.

The idea: replace exact normalization layers by *constant* factors nu that
approximately normalize each operation's output, using concentration of
measure in high dimension, and bound every weight row to the unit ball so
those constants stay valid during training.

Contents
--------
- ``linear_norm_factor``       nu for a row-normalized linear map  (sqrt(d_in / d_out))
- ``swiglu_act_norm_factor``   nu for u * SiLU(z), Monte Carlo under the actual input scale
- ``lerp_norm_factor``         nu(alpha) for the LERP residual update
- ``ScaledParameter``          s_init / s_scale reparametrisation (effective LR control)
- ``LERPResidual``             h <- (h + alpha (x - h)) * nu(alpha)
- ``project_rows_to_unit_ball`` / ``normalize_rows_``  weight constraints

Conventions
-----------
``nn.Linear.weight`` has shape (d_out, d_in); y = W x, so "normalized along
the input dimension" means each *row* has L2 norm <= 1 (norm over dim=-1).
``nn.Embedding.weight`` has shape (vocab, d_model); each token vector is a
row and is bounded the same way.
"""

import math
from typing import Iterable, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ============================================================================
# NORMALIZATION FACTORS
# ============================================================================

def linear_norm_factor(d_in: int, d_out: int) -> float:
    """
    Normalizing factor for a linear map with unit-norm rows.

    For x uniform on S^{d_in - 1} and W in R^{d_out x d_in} with unit rows,
    E[||W x||^2] = d_out / d_in, hence nu = sqrt(d_in / d_out).

    Args:
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        nu such that ||nu * W x|| ~= 1
    """
    return math.sqrt(d_in / d_out)


def swiglu_act_norm_factor(
    d_ff: int,
    gate_std: Optional[float] = None,
    up_std: Optional[float] = None,
    n_samples: int = 1_000_000,
    seed: int = 0,
) -> float:
    """
    Normalizing factor for the gated activation h = u * SiLU(z).

    Assumes independent entries u_i ~ N(0, up_std^2), z_i ~ N(0, gate_std^2).
    A unit-norm input through unit-norm rows followed by
    nu_uz = sqrt(d_model / d_ff) gives std 1 / sqrt(d_ff) for both.

    Then E[||u * SiLU(z)||^2] = d_ff * up_std^2 * E[SiLU(gate_std * g)^2]
    with g ~ N(0, 1); the 1-D expectation is estimated by Monte Carlo.

    NOTE: under these assumptions nu is ~O(sqrt(d_ff)) (about 111 for
    d_ff = 3072), not the 3.74 quoted in the paper, which we could not
    reproduce from its stated assumptions. Inside an anGPT block the FFN
    output is followed by an *exact* L2 normalization, so this constant does
    not change the forward pass or the gradients; it only matters if that
    exact normalization is removed. An override is exposed on the FFN.

    Args:
        d_ff: Hidden dimension of the FFN
        gate_std: Std of the SiLU input entries (default: 1 / sqrt(d_ff))
        up_std: Std of the linear branch entries (default: 1 / sqrt(d_ff))
        n_samples: Monte Carlo sample count
        seed: RNG seed (deterministic constant across runs)

    Returns:
        nu such that ||nu * u * SiLU(z)|| ~= 1
    """
    if gate_std is None:
        gate_std = 1.0 / math.sqrt(d_ff)
    if up_std is None:
        up_std = 1.0 / math.sqrt(d_ff)
    gen = torch.Generator().manual_seed(seed)
    g = torch.randn(n_samples, generator=gen, dtype=torch.float64)
    second_moment = F.silu(gate_std * g).pow(2).mean().item()
    expected_sq_norm = d_ff * (up_std ** 2) * second_moment
    return 1.0 / math.sqrt(expected_sq_norm)


def lerp_norm_factor(alpha: Tensor, mode: str = 'sqrt') -> Tensor:
    """
    Normalizing factor for the LERP residual h + alpha * (x - h).

    For independent unit vectors h, x and per-dimension alpha:
        E[||(1 - alpha) h + alpha x||^2] = mean_i[(1 - alpha_i)^2 + alpha_i^2]
                                        = mean_i[1 - 2 alpha_i + 2 alpha_i^2]

    The paper writes nu(alpha) = 1 - 2 alpha + 2 alpha^2, which is the
    expected *squared norm*, not its inverse square root. Multiplying by it
    shrinks the stream at every sub-layer.

    Args:
        alpha: Interpolation parameter, scalar or (d_model,)
        mode: 'sqrt' (default, mathematically consistent: rsqrt of the
            expected squared norm) or 'paper_literal' (multiply by
            1 - 2 alpha + 2 alpha^2, as printed in the paper; ablation only)

    Returns:
        Scalar tensor nu(alpha)
    """
    expected_sq_norm = (1.0 - 2.0 * alpha + 2.0 * alpha.pow(2)).mean()
    if mode == 'sqrt':
        return torch.rsqrt(expected_sq_norm)
    if mode == 'paper_literal':
        return expected_sq_norm
    raise ValueError(f"Unknown lerp norm mode: {mode}. Available: 'sqrt', 'paper_literal'")


# ============================================================================
# REPARAMETRISED SCALE PARAMETERS
# ============================================================================

class ScaledParameter(nn.Module):
    """
    Learnable vector with decoupled initial value and effective learning rate.

    Stores a raw parameter initialised to ``scale`` and returns
    ``raw * (init / scale)``. At step 0 the effective value is ``init``;
    with Adam, the raw parameter moves ~lr per step, so the effective value
    moves ~lr * init / scale per step. Smaller ``scale`` -> faster learning.

    This is the s_init / s_scale trick from nGPT (Loshchilov et al., 2025)
    reused by anGPT.

    Args:
        shape: Parameter shape (int or tuple)
        init: Effective value at initialisation
        scale: Raw stored value at initialisation (controls effective LR)
    """

    def __init__(self, shape, init: float, scale: float):
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if isinstance(shape, int):
            shape = (shape,)
        self.init_value = float(init)
        self.scale = float(scale)
        self.raw = nn.Parameter(torch.full(shape, self.scale))

    def forward(self) -> Tensor:
        """Return the effective value."""
        return self.raw * (self.init_value / self.scale)

    def reset_parameters(self) -> None:
        """Reset the raw parameter so the effective value equals ``init``."""
        nn.init.constant_(self.raw, self.scale)

    @property
    def lr_multiplier(self) -> float:
        """Effective learning-rate multiplier relative to the global LR."""
        return self.init_value / self.scale

    def extra_repr(self) -> str:
        return f'shape={tuple(self.raw.shape)}, init={self.init_value}, scale={self.scale}'


# ============================================================================
# LERP RESIDUAL
# ============================================================================

class LERPResidual(nn.Module):
    """
    Residual update by linear interpolation with a learnable per-dimension rate.

        h <- (h + alpha * (x - h)) * nu(alpha)       (mode='approx', anGPT)
        h <- normalize(h + alpha * (x - h))          (mode='exact',  nGPT)
        h <- (h + alpha * (x - h)) * (1-2a+2a^2)     (mode='paper_literal')

    Args:
        d_model: Model dimension
        alpha_init: Effective initial value of alpha
        alpha_scale: Raw stored value (effective LR multiplier = init / scale)
        mode: 'approx', 'exact' or 'paper_literal'
        eps: Epsilon for exact normalization
    """

    MODES = ('approx', 'exact', 'paper_literal')

    def __init__(
        self,
        d_model: int,
        alpha_init: float = 0.05,
        alpha_scale: float = 0.01,
        mode: str = 'approx',
        eps: float = 1e-6,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}. Available: {self.MODES}")
        self.d_model = d_model
        self.mode = mode
        self.eps = eps
        self.alpha = ScaledParameter(d_model, init=alpha_init, scale=alpha_scale)

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            h: Residual stream (..., d_model), approximately unit norm
            x: Sub-layer output (..., d_model), unit norm

        Returns:
            Updated residual stream (..., d_model)
        """
        alpha = self.alpha()
        out = h + alpha * (x - h)
        if self.mode == 'exact':
            return F.normalize(out, dim=-1, eps=self.eps)
        lerp_mode = 'sqrt' if self.mode == 'approx' else 'paper_literal'
        return out * lerp_norm_factor(alpha, mode=lerp_mode).to(out.dtype)

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, mode={self.mode}'


# ============================================================================
# WEIGHT CONSTRAINTS
# ============================================================================

@torch.no_grad()
def normalize_rows_(weight: Tensor, eps: float = 1e-8) -> Tensor:
    """Rescale every row of ``weight`` to exactly unit L2 norm, in place."""
    norms = weight.norm(dim=-1, keepdim=True).clamp_min(eps)
    weight.div_(norms)
    return weight


@torch.no_grad()
def project_rows_to_unit_ball(weight: Tensor) -> Tensor:
    """
    Project every row of ``weight`` onto the unit ball, in place.

    Rows with norm <= 1 are untouched; rows with norm > 1 are rescaled to 1.
    This is the bound ||w_i|| <= 1 used by anGPT (rather than nGPT's exact
    unit norm), which lets rows shrink toward zero when useful.
    """
    norms = weight.norm(dim=-1, keepdim=True).clamp_min(1.0)
    weight.div_(norms)
    return weight


def bounded_weights(module: nn.Module) -> Iterable[Tensor]:
    """
    Yield the weights of every nn.Linear and nn.Embedding inside ``module``,
    deduplicated (tied weights are yielded once).
    """
    seen = set()
    for sub in module.modules():
        if isinstance(sub, (nn.Linear, nn.Embedding)):
            w = sub.weight
            if id(w) not in seen:
                seen.add(id(w))
                yield w


@torch.no_grad()
def max_row_norm(module: nn.Module) -> float:
    """Largest row norm over all bounded weights in ``module`` (diagnostic)."""
    values = [w.norm(dim=-1).max().item() for w in bounded_weights(module)]
    return max(values) if values else 0.0
