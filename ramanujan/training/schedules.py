"""
Learning-rate schedules as multipliers of the peak LR (pure functions of the step).

cosine  linear warmup, then cosine decay to min_ratio at max_steps
        (the schedule of the pilot runs).

wsd     warmup-stable-decay: linear warmup, constant peak LR, then a cooldown
        over the last `cooldown_frac` of the steps to `final_ratio`.
        Hägele et al. (NeurIPS 2024): matches a tuned cosine with a cooldown of
        about 20% of the steps; the (1-sqrt) cooldown beats linear. Any
        checkpoint of the stable phase can be cooled down into a finished model.

Cooldown shapes, t in [0, 1] = progress through the cooldown:
    linear   1 - t
    1-sqrt   1 - sqrt(t)
"""

import math

SCHEDULES = ("cosine", "wsd")
COOLDOWN_SHAPES = ("1-sqrt", "linear")


def cooldown_factor(t: float, shape: str = "1-sqrt") -> float:
    t = min(max(t, 0.0), 1.0)
    if shape == "1-sqrt":
        return 1.0 - math.sqrt(t)
    if shape == "linear":
        return 1.0 - t
    raise ValueError(f"Unknown cooldown shape: {shape}. Available: {COOLDOWN_SHAPES}")


def lr_multiplier(step: int, max_steps: int, warmup_steps: int = 0, schedule: str = "cosine",
                  min_ratio: float = 0.1, cooldown_frac: float = 0.2,
                  cooldown_shape: str = "1-sqrt", final_ratio: float = 0.0,
                  cooldown_start: int = None) -> float:
    """
    Multiplier of the peak LR at `step` (0-based).

    For wsd, `cooldown_start` overrides cooldown_frac: the cooldown then runs
    from cooldown_start to max_steps (used to branch a cooldown off a
    stable-phase checkpoint).
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))
    if schedule == "wsd":
        if cooldown_start is None:
            cooldown_start = max_steps - int(round(cooldown_frac * max_steps))
        cooldown_start = max(cooldown_start, warmup_steps)
        if step < cooldown_start:
            return 1.0
        t = (step - cooldown_start) / max(1, max_steps - cooldown_start)
        return final_ratio + (1 - final_ratio) * cooldown_factor(t, cooldown_shape)
    raise ValueError(f"Unknown schedule: {schedule}. Available: {SCHEDULES}")
