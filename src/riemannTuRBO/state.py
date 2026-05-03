"""
Trust Region State Management for Riemannian TuRBO
==================================================

This module provides state containers for managing the trust region during
optimization using TuRBO-style success/failure counting.

TuRBO-Style Updates
-------------------
The original TuRBO approach:
- Track consecutive successes (new best found) and failures (no improvement)
- Expand TR after `success_tolerance` consecutive successes
- Shrink TR after `failure_tolerance` consecutive failures
- Restart when TR length drops below minimum

References
----------
- Eriksson et al. (2019) "Scalable Global Optimization via Local Bayesian
  Optimization" (TuRBO)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


logger = logging.getLogger("TurboState")


# =============================================================================
# Turbo State
# =============================================================================


@dataclass
class TurboState:
    """
    State container for Trust Region Bayesian Optimization.

    Manages the trust region length and tracks optimization progress using
    TuRBO-style success/failure counting.

    Parameters
    ----------
    dim : int
        Input dimensionality.
    q : int
        Batch size (number of candidates per iteration).
    length : float
        Initial trust region length (half-width).
    length_min : float
        Minimum trust region length before restart.
    length_max : float
        Maximum trust region length.
    success_tolerance : int
        Number of consecutive successes before expansion.
    failure_tolerance : int
        Computed as max(4/q, dim/q). Number of failures before shrinking.

    Attributes
    ----------
    best_value : float
        Best observed value so far.
    best_x : Optional[Tensor]
        Location of best observed value.
    restart_triggered : bool
        True when length drops below length_min.
    iteration : int
        Number of update calls.

    Examples
    --------
    >>> state = TurboState(dim=10, q=1)
    >>> # After each evaluation:
    >>> state.update(Y_new, X_next=X_new)
    >>> if state.restart_triggered:
    ...     state = TurboState(dim=10, q=1)  # Restart
    """

    dim: int
    q: int = 1

    # TR sizing
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.0

    # TuRBO-specific
    success_tolerance: int = 10
    failure_tolerance: int = field(default=0, init=False)
    success_counter: int = field(default=0, init=False)
    failure_counter: int = field(default=0, init=False)

    # State tracking
    best_value: float = field(default=-float("inf"), init=False)
    best_x: Optional[Tensor] = field(default=None, init=False)
    restart_triggered: bool = field(default=False, init=False)
    iteration: int = field(default=0, init=False)

    def __post_init__(self):
        self.failure_tolerance = min(
            math.ceil(max(4.0 / self.q, float(self.dim) / self.q)), 20
        )
        self.start_length = self.length

    def update(
        self,
        Y_next: Tensor,
        X_next: Optional[Tensor] = None,
    ) -> None:
        """
        Update state after evaluating new candidates.

        Parameters
        ----------
        Y_next : Tensor
            Objective values for the new candidates.
        X_next : Optional[Tensor]
            The evaluated points (needed for tracking best_x).
        """
        self.iteration += 1
        self._update_turbo(Y_next, X_next)

        # Check for restart
        if self.length < self.length_min:
            self.restart_triggered = True
            self.length = self.start_length
            logger.info(
                f"Restart triggered: length={self.length:.6f} < "
                f"length_min={self.length_min:.6f}"
            )

    def _update_turbo(self, Y_next: Tensor, X_next: Optional[Tensor] = None) -> None:
        """
        TuRBO-style update based on success/failure counting.

        Expand after consecutive successes, shrink after consecutive failures.
        """
        # Y_next should be (q, 1) - standard BoTorch/GPyTorch convention
        y_max = Y_next.max().item()

        # Check for improvement (with tolerance for numerical noise)
        improved = y_max > self.best_value + 1e-4 * abs(self.best_value)

        if improved:
            self.success_counter += 1
            self.failure_counter = 0
            self.best_value = y_max
            if X_next is not None:
                # argmax on (q, 1) tensor: flatten and get batch index
                best_idx = Y_next.argmax().item()
                self.best_x = X_next[best_idx].clone()

            logger.info(
                f"TuRBO: Success! best={self.best_value:.4f}, "
                f"success_count={self.success_counter}"
            )
        else:
            self.success_counter = 0
            self.failure_counter += 1

            logger.info(f"TuRBO: No improvement. failure_count={self.failure_counter}")

        # Expand on success streak
        if self.success_counter >= self.success_tolerance:
            old_length = self.length
            self.length = min(self.length * 2.0, self.length_max)
            self.success_counter = 0

            logger.info(f"TuRBO: Expanding TR: {old_length:.4f} -> {self.length:.4f}")

        # Shrink on failure streak
        elif self.failure_counter >= self.failure_tolerance:
            old_length = self.length
            self.length /= 2.0
            self.failure_counter = 0

            logger.info(f"TuRBO: Shrinking TR: {old_length:.4f} -> {self.length:.4f}")

    def __repr__(self) -> str:
        return (
            f"TurboState(dim={self.dim}, length={self.length:.4f}, "
            f"best={self.best_value:.4f}, iter={self.iteration})"
        )
