"""Callbacks for normalized architectures (anGPT, nGPT-style).

``WeightNormBoundCallback`` keeps every weight row in the unit ball by
calling ``model.project_parameters()`` after each optimizer step.

In ``Trainer.train_step``, ``on_step_end`` fires only on optimizer steps
(inside the gradient-accumulation branch) and before the next forward pass,
which is exactly where the projection must happen.

Optimizer setup that goes with it (not enforced here):
- Adam / AdamW with weight_decay = 0 for all parameters. Weight decay on the
  raw ScaledParameter values would shrink alpha, s_z and the attention
  temperature toward zero.
- No LR warmup.
"""

from typing import Dict, Optional

from torch import nn

from .callbacks import Callback


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP (``.module``) and torch.compile (``._orig_mod``) wrappers."""
    while True:
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        elif hasattr(model, 'module') and isinstance(model.module, nn.Module):
            model = model.module
        else:
            return model


class WeightNormBoundCallback(Callback):
    """
    Project model weights onto the unit ball after every optimizer step.

    Args:
        log_every: Record the max row norm every N optimizer steps
            (0 disables). Recorded *before* projection, so it shows how far
            each update pushes rows outside the ball.
        verbose: Print the recorded value

    Attributes:
        last_max_row_norm: Most recent pre-projection max row norm, or None

    Example:
        >>> trainer = Trainer(model, optimizer, loss_fn, config, loader,
        ...                   callbacks=[WeightNormBoundCallback(log_every=100)])
    """

    def __init__(self, log_every: int = 0, verbose: bool = False):
        self.log_every = log_every
        self.verbose = verbose
        self.last_max_row_norm: Optional[float] = None
        self._n_projections = 0

    def _project(self, trainer) -> None:
        model = unwrap_model(trainer.model)
        if not hasattr(model, 'project_parameters'):
            raise AttributeError(
                f"{type(model).__name__} has no project_parameters(); "
                f"WeightNormBoundCallback is only for normalized architectures"
            )
        if self.log_every and self._n_projections % self.log_every == 0:
            self.last_max_row_norm = model.max_weight_row_norm()
            if self.verbose:
                print(f"[WeightNormBound] max row norm before projection: {self.last_max_row_norm:.4f}")
        model.project_parameters()
        self._n_projections += 1

    def on_train_begin(self, trainer) -> None:
        # Guard against a checkpoint or external init that left rows outside the ball
        unwrap_model(trainer.model).project_parameters()

    def on_step_end(self, trainer, step: int, metrics: Dict[str, float]) -> None:
        self._project(trainer)
