"""
Model sizing from corpus size and compute budget.

Given the number of unique tokens in a corpus and a compute budget, choose
the model size N (parameters) and the number of training tokens D that
minimize the data-constrained scaling law of Muennighoff et al. (2023), then
choose a transformer shape (width, depth, heads, FFN width) with N parameters.

Scaling law (Muennighoff et al., NeurIPS 2023, Eq. 17, fitted on C4):

    L(N, D) = E + A / N'^alpha + B / D'^beta

    U_D = min(U, D)                      unique tokens actually seen
    R_D = D / U_D - 1                    repetitions beyond the first epoch
    D'  = U_D + U_D R*_D (1 - exp(-R_D / R*_D))
    U_N = min(0.051 U_D, N)              compute-optimal params for U_D
    R_N = N / U_N - 1
    N'  = U_N + U_N R*_N (1 - exp(-R_N / R*_N))

    A = 521, B = 1488, E = 1.87, alpha = beta = 0.353,
    R*_N = 5.309743, R*_D = 15.387756

With unlimited unique data this is Chinchilla-style allocation (about 20
tokens per parameter). When the corpus is small, repeated epochs are worth
less than fresh tokens (about 4 epochs are nearly free; value decays to zero
around 16) and the optimum shifts toward more parameters and fewer epochs.

Caveats, stated in every report:
  - The constants were fitted on English web text (C4). Absolute losses do not
    transfer to another corpus or tokenizer; only the allocation (how to split
    compute between N and D) is used here, and that is an assumption.
  - Compute is C = 6 N D (forward + backward, attention FLOPs ignored), with
    the achieved FLOP/s measured on the target GPU, not its peak.

Shape rule (literature defaults, each one a parameter):
  - head_dim 64; KV heads = query heads / 3 (GQA), so width is a multiple of 192;
  - SwiGLU FFN width = 3 x d_model, rounded to 64 (Qwen3-0.6B ratio);
  - tied input/output embeddings (MobileLLM, SmolLM);
  - depth chosen so that d_model / n_layers is closest to 32 (deep and thin:
    SmolLM-135M has 576/30 = 19, Qwen3-0.6B 1024/28 = 37).
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

import numpy as np

# Muennighoff et al. (2023), Eq. 17 and Appendix A
A, B, E = 521.0, 1488.0, 1.87
ALPHA = BETA = 0.353
R_N_STAR = 5.309743
R_D_STAR = 15.387756
TOKENS_PER_PARAM_OPT = 1 / 0.051          # ~19.6, U_N = 0.051 U_D


# ── Scaling law ───────────────────────────────────────────────────────────────

def effective_data(D, U):
    """Repetition-discounted data D' for D training tokens from U unique tokens."""
    D = np.asarray(D, dtype=float)
    u_d = np.minimum(U, D)
    r_d = D / u_d - 1
    return u_d + u_d * R_D_STAR * (1 - np.exp(-r_d / R_D_STAR))


def effective_params(N, D, U):
    """Excess-parameter-discounted N' (parameters beyond what U_D can use count less)."""
    N = np.asarray(N, dtype=float)
    u_n = np.minimum(np.minimum(U, D) / TOKENS_PER_PARAM_OPT, N)
    r_n = N / u_n - 1
    return u_n + u_n * R_N_STAR * (1 - np.exp(-r_n / R_N_STAR))


def predicted_loss(N, D, U):
    """L(N, D) under the data-constrained law (C4 constants: use for comparisons only)."""
    return E + A / effective_params(N, D, U) ** ALPHA + B / effective_data(D, U) ** BETA


# ── Allocation ────────────────────────────────────────────────────────────────

@dataclass
class Allocation:
    params: float          # N
    tokens: float          # D
    epochs: float          # D / U
    compute: float         # 6 N D
    loss: float            # predicted, C4 scale
    loss_half_params: float
    loss_double_params: float


def allocate(compute: float, unique_tokens: float, tolerance: float = 0.0,
             n_min: float = 1e6, n_max: float = 1e12, grid: int = 4000) -> Allocation:
    """
    Split `compute` FLOPs between parameters and tokens to minimize predicted loss.

    The law is flat near its optimum. With tolerance > 0, the smallest N whose
    predicted loss is within `tolerance` nats of the minimum is returned
    (cheaper to train per token and to run).
    """
    if compute <= 0 or unique_tokens <= 0:
        raise ValueError("compute and unique_tokens must be positive")
    Ns = np.logspace(np.log10(n_min), np.log10(n_max), grid)
    Ds = compute / (6 * Ns)
    losses = predicted_loss(Ns, Ds, unique_tokens)
    best = losses.min()
    i = int(np.flatnonzero(losses <= best + tolerance)[0])
    N, D = Ns[i], Ds[i]
    return Allocation(
        params=N, tokens=D, epochs=D / unique_tokens, compute=compute, loss=float(losses[i]),
        loss_half_params=float(predicted_loss(N / 2, 2 * D, unique_tokens)),
        loss_double_params=float(predicted_loss(2 * N, D / 2, unique_tokens)),
    )


# ── Shape ─────────────────────────────────────────────────────────────────────

@dataclass
class Shape:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    tie_word_embeddings: bool
    params: int

    @property
    def aspect(self) -> float:
        return self.d_model / self.n_layers

    def model_section(self, vocab_size: int) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "params"}
        d["vocab_size"] = vocab_size
        return d


def count_params(d_model: int, n_layers: int, vocab_size: int, n_heads: int,
                 n_kv_heads: int, d_ff: int, tie: bool, qk_norm: bool = True) -> int:
    """Exact parameter count of RamanujanTransformer (GPT+: RMSNorm, GQA, SwiGLU, no biases)."""
    head_dim = d_model // n_heads
    attn = 2 * d_model * d_model + 2 * d_model * n_kv_heads * head_dim
    if qk_norm:
        attn += 2 * head_dim
    ffn = 3 * d_model * d_ff
    norms = 2 * d_model
    embed = vocab_size * d_model * (1 if tie else 2)
    return n_layers * (attn + ffn + norms) + embed + d_model  # + final norm


def choose_shape(target_params: float, vocab_size: int, head_dim: int = 64,
                 kv_ratio: int = 3, ffn_mult: float = 3.0, tie: bool = True,
                 target_aspect: float = 32.0, aspect_range=(12.0, 96.0),
                 min_layers: int = 4) -> Shape:
    """
    Shape with about `target_params` parameters: for each width (a multiple of
    head_dim * kv_ratio), the depth that best matches the target; among widths
    whose aspect ratio d/L is in range, the one closest to target_aspect (in log).
    """
    step = head_dim * kv_ratio
    candidates: List[Shape] = []
    for d in range(step, 8192 + 1, step):
        n_heads = d // head_dim
        n_kv = n_heads // kv_ratio
        d_ff = int(round(ffn_mult * d / 64)) * 64
        per_layer = count_params(d, 1, vocab_size, n_heads, n_kv, d_ff, tie) - \
            count_params(d, 0, vocab_size, n_heads, n_kv, d_ff, tie)
        base = count_params(d, 0, vocab_size, n_heads, n_kv, d_ff, tie)
        if base >= target_params:
            break
        n_layers = max(min_layers, int(round((target_params - base) / per_layer)))
        shape = Shape(d, n_layers, n_heads, n_kv, d_ff, tie,
                      count_params(d, n_layers, vocab_size, n_heads, n_kv, d_ff, tie))
        if aspect_range[0] <= shape.aspect <= aspect_range[1]:
            candidates.append(shape)
    if not candidates:
        raise ValueError(f"No shape with aspect ratio in {aspect_range} for {target_params:.3g} params")
    return min(candidates, key=lambda s: (abs(np.log(s.aspect / target_aspect)),
                                          abs(s.params - target_params)))


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class SizingReport:
    unique_tokens: float
    gpu_hours: float
    flops_per_sec: float
    allocation: Allocation
    shape: Shape
    tokens_per_step: int
    max_steps: int
    est_hours: float


def size_model(unique_tokens: float, gpu_hours: float, flops_per_sec: float,
               vocab_size: int, tokens_per_step: int = 65536, tolerance: float = 0.0,
               **shape_kwargs) -> SizingReport:
    """End to end: budget -> (N, D) -> shape -> steps, re-estimating hours for the actual N."""
    compute = gpu_hours * 3600 * flops_per_sec
    alloc = allocate(compute, unique_tokens, tolerance=tolerance)
    shape = choose_shape(alloc.params, vocab_size, **shape_kwargs)
    tokens = compute / (6 * shape.params)          # spend the budget on the actual N
    max_steps = int(tokens // tokens_per_step)
    est_hours = 6 * shape.params * max_steps * tokens_per_step / flops_per_sec / 3600
    return SizingReport(unique_tokens, gpu_hours, flops_per_sec, alloc, shape,
                        tokens_per_step, max_steps, est_hours)


def flops_per_sec_from_run(params: float, tokens_per_sec: float) -> float:
    """Achieved FLOP/s from a measured run: 6 N tokens/s."""
    return 6 * params * tokens_per_sec
