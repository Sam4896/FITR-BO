"""
Epsilon (Damping) Configuration for Riemannian Trust Regions
=============================================================

This module provides configuration and computation utilities for the epsilon
(damping/regularization) parameter used in Riemannian Trust Region methods.

The epsilon parameter controls the regularization of the metric tensor G:
    G_reg = G + ε * I

Choice of ε is critical:
- ε too large → G_reg ≈ εI → isotropic (square) trust region
- ε too small → extreme stretching in "flat" directions

References
----------
- Martens (2020) "New Insights and Perspectives on the Natural Gradient Method"
  https://www.jmlr.org/papers/volume21/17-678/17-678.pdf
- Tikhonov regularization / Levenberg-Marquardt damping
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class EpsMode(str, Enum):
    """Epsilon (damping) scheduling modes."""

    FIXED = "fixed"
    """Use a fixed epsilon value. Simple but requires manual tuning."""

    AUTO_TRACE = "auto_trace"
    """
    Automatic (scale-free) damping:
        ε = trace(G) / D + jitter = mean(eigenvalues) + jitter

    Intuition
    ---------
    This is Tikhonov / Levenberg-Marquardt style regularization where the
    damping magnitude is set to the *average* curvature scale.

    For rank-1 metrics (common for scalar objectives with G = g @ g.T):
    - If eigenvalues are {λ, 0, ..., 0}, then ε = λ/D
    - Condition number becomes (λ + ε) / ε = D + 1, bounded by dimension

    This follows Martens (2020) recommendation for scale-invariant damping.
    """


@dataclass
class EpsConfig:
    """
    Configuration for epsilon (damping/regularization) scheduling.

    Attributes
    ----------
    mode : EpsMode
        The scheduling strategy for epsilon.
    eps : float
        Fixed epsilon value (used when mode="fixed").
    jitter : float
        Small non-negative jitter added to epsilon for numerical safety.
        This is the only tunable knob in AUTO_TRACE mode.

    Examples
    --------
    >>> # Fixed epsilon
    >>> cfg = EpsConfig(mode=EpsMode.FIXED, eps=1e-3)

    >>> # Automatic epsilon (recommended)
    >>> cfg = EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12)
    """

    mode: EpsMode = EpsMode.AUTO_TRACE
    eps: float = 1.0
    jitter: float = 1e-6

    def __post_init__(self):
        if self.jitter < 0.0:
            raise ValueError("EpsConfig.jitter must be non-negative.")


def compute_eps_from_eigs(eigs: Tensor, cfg: EpsConfig) -> float:
    """
    Compute epsilon from eigenvalues (or diagonal entries) of the metric.

    Parameters
    ----------
    eigs : Tensor
        Eigenvalues or diagonal entries of G. Shape (D,) or (r,) for low-rank.
    cfg : EpsConfig
        Epsilon configuration.

    Returns
    -------
    float
        The computed epsilon value (always positive).

    Notes
    -----
    For AUTO_TRACE mode:
        ε = mean(eigenvalues) + jitter

    This ensures ε scales with the curvature magnitude, providing
    scale-invariant regularization.
    """
    eigs = torch.clamp(eigs.detach(), min=0.0)

    if cfg.mode == EpsMode.FIXED:
        eps = float(cfg.eps)
    elif cfg.mode == EpsMode.AUTO_TRACE:
        # ε = trace(G)/D + jitter = mean(eigenvalues) + jitter
        # eps = float(eigs.mean().item()) + cfg.jitter
        eps = float(eigs.mean().item()) + cfg.jitter
    else:
        raise ValueError(f"Unknown EpsMode: {cfg.mode}")

    # Numerical floor to prevent division by zero
    return float(max(eps, 1e-16))
