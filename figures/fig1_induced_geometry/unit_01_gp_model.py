"""Figure 1 data and GP model setup for the NeurIPS visualization."""

from __future__ import annotations

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine


def forrester_1d(x: torch.Tensor) -> torch.Tensor:
    """Smooth 1D objective with multiple local peaks."""

    def bump(center: float, width: float, height: float) -> torch.Tensor:
        return height * torch.exp(-((x - center) ** 2) / (2.0 * width**2))

    f = bump(0.75, 0.08, 2.9) + bump(0.30, 0.10, 2.0)
    span = (x - 0.30) * (0.75 - x)
    envelope = torch.clamp(span * 8.0, 0.0, 1.0)
    phase = 2.0 * np.pi * (6.0 * x + 14.0 * x**2)
    wiggle = 0.35 * envelope * torch.sin(phase)
    return f + wiggle


def eval_objective(x_normalized: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Evaluate objective on normalized input in [0, 1]."""
    was_1d = x_normalized.dim() == 1
    if was_1d:
        x_normalized = x_normalized.unsqueeze(0)
    x_raw = unnormalize(x_normalized, bounds)
    y = forrester_1d(x_raw)
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    elif y.dim() == 0:
        y = y.unsqueeze(0).unsqueeze(-1)
    return y


def fit_gp(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    """Fit exact GP for standardized scalar outcomes."""
    model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # I-BNN with fixed hyperparameters
    # train_Yvar = torch.full_like(Y, 1e-5)
    # model = SingleTaskGP(X, Y, train_Yvar, covar_module=InfiniteWidthBNNKernel(depth=2))
    # mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_mll(mll)
    # model.eval()

    return model


def build_fig1_gp_data(
    n_train: int = 5,
    seed: int = 42,
    noise_std: float = 0.08,
) -> dict:
    """Create deterministic training set and fitted GP for Figure 1."""
    torch.manual_seed(seed)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    sobol = SobolEngine(dimension=1, scramble=True, seed=seed)
    X_train = sobol.draw(n_train).to(dtype=torch.float64)

    Y_raw = eval_objective(X_train, bounds)
    Y_raw = Y_raw + noise_std * torch.randn_like(Y_raw)
    y_mean = Y_raw.mean().item()
    y_std = Y_raw.std().item() + 1e-8
    Y_std = (Y_raw - y_mean) / y_std

    model = fit_gp(X_train, Y_std)

    return {
        "model": model,
        "bounds": bounds,
        "X_train": X_train,
        "Y_std_train": Y_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
