"""Geometry utilities for Figure 7 (mean/LogEI + pullback FIM ellipses)."""

from __future__ import annotations

import numpy as np
import torch

from neurips_viz.fig6.unit_02_geometry import (
    _model_dtype_and_bounds,
    compute_gx_at_point,
    compute_logei_grid,
)


def make_anchor_points(
    bounds: torch.Tensor,
    n_anchors: int,
    seed: int,
    mode: str = "random",
    margin: float = 0.05,
) -> list[np.ndarray]:
    """Return anchor points in random or grid mode.

    - random: sample n_anchors points uniformly.
    - grid: build n_anchors points per axis => n_anchors^2 points.
    """
    if n_anchors <= 0:
        raise ValueError("n_anchors must be positive")
    lo = bounds[0].detach().cpu().numpy().astype(np.float64)
    hi = bounds[1].detach().cpu().numpy().astype(np.float64)
    span = hi - lo
    lo_m = lo + margin * span
    hi_m = hi - margin * span
    if mode == "random":
        rng = np.random.default_rng(seed)
        pts = rng.uniform(low=lo_m, high=hi_m, size=(n_anchors, 2))
        return [pts[i] for i in range(n_anchors)]
    if mode == "grid":
        x1 = np.linspace(lo_m[0], hi_m[0], n_anchors, dtype=np.float64)
        x2 = np.linspace(lo_m[1], hi_m[1], n_anchors, dtype=np.float64)
        return [np.array([a, b], dtype=np.float64) for a in x1 for b in x2]
    raise ValueError("mode must be 'random' or 'grid'")


def build_gx_at_anchors(
    model,
    anchors: list[np.ndarray],
    bounds: torch.Tensor,
) -> list[dict]:
    """Compute pullback Fisher metric dictionaries at each anchor."""
    return [compute_gx_at_point(model=model, x_raw=pt, bounds=bounds) for pt in anchors]


def compute_mean_grid(
    model,
    x_flat: np.ndarray,
    bounds: torch.Tensor,
) -> np.ndarray:
    """Posterior mean evaluated on flattened raw grid."""
    dtype, b = _model_dtype_and_bounds(model, bounds)
    x_raw = torch.from_numpy(x_flat).to(dtype=dtype, device=b.device)
    x_model = (x_raw - b[0]) / (b[1] - b[0])
    with torch.no_grad():
        mu = model.posterior(x_model).mean.squeeze(-1)
    return mu.detach().cpu().numpy()


def compute_variance_grid(
    model,
    x_flat: np.ndarray,
    bounds: torch.Tensor,
) -> np.ndarray:
    """Posterior variance evaluated on flattened raw grid."""
    dtype, b = _model_dtype_and_bounds(model, bounds)
    x_raw = torch.from_numpy(x_flat).to(dtype=dtype, device=b.device)
    x_model = (x_raw - b[0]) / (b[1] - b[0])
    with torch.no_grad():
        var = model.posterior(x_model).variance.squeeze(-1)
    return var.detach().cpu().numpy()


# Re-export for run_fig7 compatibility (LogEI on flattened raw grid)
__all__ = [
    "make_anchor_points",
    "build_gx_at_anchors",
    "compute_mean_grid",
    "compute_variance_grid",
    "compute_logei_grid",
]
