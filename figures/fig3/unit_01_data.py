"""Figure 3 data setup: 2D objective, train data, and BoTorch GP."""

from __future__ import annotations

import math
from typing import Callable

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood


def _branin_raw(x: torch.Tensor) -> torch.Tensor:
    """Branin on [0,1]^2 mapped from its canonical domain."""
    x1 = x[..., 0] * 15.0 - 5.0
    x2 = x[..., 1] * 15.0
    a, b, c = 1.0, 5.1 / (4.0 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8.0 * math.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * torch.cos(x1) + s
    return (-y).unsqueeze(-1)


# def _forrester_like_2d_raw(x: torch.Tensor) -> torch.Tensor:
#     """Smooth 2D objective with global waviness and local structure on [0,1]^2."""
#     x1 = x[..., 0]
#     x2 = x[..., 1]

#     def bump(
#         c1: float,
#         c2: float,
#         w1: float,
#         w2: float,
#         h: float,
#     ) -> torch.Tensor:
#         return h * torch.exp(
#             -(((x1 - c1) ** 2) / (2.0 * w1**2) + ((x2 - c2) ** 2) / (2.0 * w2**2))
#         )

#     base = bump(0.78, 0.72, 0.11, 0.10, 2.2) + bump(0.28, 0.30, 0.13, 0.12, 1.6)

#     # Global oscillatory content (not envelope-limited) so the whole domain is wavy.
#     wavy_global = (
#         0.55 * torch.sin(2.0 * math.pi * (2.4 * x1 + 1.3 * x2))
#         + 0.40 * torch.cos(2.0 * math.pi * (1.2 * x1 - 2.1 * x2))
#         + 0.28 * torch.sin(2.0 * math.pi * (3.0 * x1 * x2 + 0.8 * x2))
#     )
#     gentle_trend = 0.35 * (x1 - 0.5) - 0.25 * (x2 - 0.5)

#     return (base + wavy_global + gentle_trend).unsqueeze(-1)


def _rotation_matrix_2d(angle_degrees: float) -> torch.Tensor:
    theta = math.radians(angle_degrees)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def make_rotated_function(
    angle_degrees: float = 32.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a rotated objective function on [0,1]^2."""
    if angle_degrees == 0.0:
        return _branin_raw

    rot = _rotation_matrix_2d(angle_degrees)

    def f_rot(x: torch.Tensor) -> torch.Tensor:
        x_rot = (rot @ x.double().T).T
        x_rot = (x_rot - 0.5).clamp(-0.5, 0.5) + 0.5
        return _branin_raw(x_rot.to(x.dtype))

    return f_rot


def get_training_data(
    bounds: torch.Tensor,
    n_train: int = 24,
    seed: int = 42,
    angle_degrees: float = 32.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample data, normalize inputs, and standardize outputs."""
    torch.manual_seed(seed)
    f_obj = make_rotated_function(angle_degrees=angle_degrees)

    # Draw in unit box then unnormalize to the provided domain bounds.
    unit_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    x_unit = (
        draw_sobol_samples(bounds=unit_bounds, n=n_train, q=1, seed=seed)
        .squeeze(1)
        .double()
    )
    x_train_raw = unnormalize(x_unit, bounds=bounds)
    x_train_norm = normalize(x_train_raw, bounds=bounds)

    y_raw = f_obj(x_train_raw).double()
    y_mean = y_raw.mean()
    y_std = y_raw.std().clamp_min(1e-6)
    y_train_std = (y_raw - y_mean) / y_std
    return x_train_raw, x_train_norm, y_train_std, y_mean, y_std


def build_gp_surrogate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
) -> SingleTaskGP:
    """Fit a BoTorch SingleTaskGP and return it in eval mode."""
    model = SingleTaskGP(x_train, y_train)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # from botorch.models.kernels import InfiniteWidthBNNKernel

    # train_Yvar = torch.full_like(
    #     y_train, 1e-6, dtype=y_train.dtype, device=y_train.device
    # )
    # model = SingleTaskGP(
    #     x_train, y_train, train_Yvar, covar_module=InfiniteWidthBNNKernel(depth=3)
    # )
    # mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_mll(mll)
    # model.eval()

    return model
