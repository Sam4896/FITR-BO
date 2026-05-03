"""Figure 6: training data and surrogates (GP-SE, GP-IBNN, deep BNN)."""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from poc.deep_ensemble import DeepEnsembleModel

from neurips_viz.fig3.unit_01_data import (  # noqa: F401 — re-export
    build_gp_surrogate,
    get_training_data,
    make_rotated_function,
)

__all__ = [
    "build_gp_surrogate",
    "get_training_data",
    "make_rotated_function",
    "build_ibnn_surrogate",
    "build_deep_ensemble_bnn",
]


def build_ibnn_surrogate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    depth: int = 3,
) -> SingleTaskGP:
    """Fit a GP with InfiniteWidthBNN kernel and return it in eval mode."""
    from botorch.models.kernels import InfiniteWidthBNNKernel

    train_Yvar = torch.full_like(y_train, 1e-6, dtype=y_train.dtype, device=y_train.device)
    model = SingleTaskGP(
        x_train,
        y_train,
        train_Yvar=train_Yvar,
        covar_module=InfiniteWidthBNNKernel(depth=depth),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def build_deep_ensemble_bnn(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    n_networks: int = 5,
    hidden_dims: list[int] | None = None,
    n_epochs: int = 350,
    lr: float = 1e-3,
    seed: int = 0,
) -> DeepEnsembleModel:
    """Train heteroscedastic deep ensemble compatible with BoTorch posterior + FITR."""
    # DeepEnsembleModel expects float32 weights internally
    x_fit = x_train.detach().float()
    y_fit = y_train.detach().float().view(-1, 1)
    if hidden_dims is None:
        hidden_dims = [64, 64]
    torch.manual_seed(seed)
    model = DeepEnsembleModel(
        input_dim=x_fit.shape[-1],
        n_networks=n_networks,
        hidden_dims=hidden_dims,
        n_epochs=n_epochs,
        lr=lr,
    )
    model.fit(x_fit, y_fit)
    return model
