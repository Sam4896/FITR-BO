"""Compute det(g_t(x)) for the 2D scalar-Gaussian pullback metric."""

from __future__ import annotations

import numpy as np
import torch
from botorch.utils.transforms import normalize


def _mesh_from_bounds(bounds: torch.Tensor, grid_size: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    x1 = torch.linspace(float(bounds[0, 0]), float(bounds[1, 0]), grid_size, dtype=torch.float64)
    x2 = torch.linspace(float(bounds[0, 1]), float(bounds[1, 1]), grid_size, dtype=torch.float64)
    mesh_x1, mesh_x2 = torch.meshgrid(x1, x2, indexing="ij")
    x_flat = torch.stack([mesh_x1.reshape(-1), mesh_x2.reshape(-1)], dim=1)
    return mesh_x1.numpy(), mesh_x2.numpy(), x_flat


def compute_det_gt_2d(
    model,
    bounds: torch.Tensor,
    model_bounds: torch.Tensor,
    grid_size: int = 200,
    eps: float = 1e-14,
    return_metric: bool = False,
) -> dict:
    """Compute mu, variance Sigma, and det(g_t) on a 2D grid."""
    mesh_x1, mesh_x2, x_flat = _mesh_from_bounds(bounds=bounds, grid_size=grid_size)
    n_points = x_flat.shape[0]

    mu_vals = torch.zeros(n_points, dtype=torch.float64)
    sigma_vals = torch.zeros(n_points, dtype=torch.float64)
    grad_mu = torch.zeros(n_points, 2, dtype=torch.float64)
    grad_sigma = torch.zeros(n_points, 2, dtype=torch.float64)

    model.eval()
    for i in range(n_points):
        x_i_raw = x_flat[i].detach().clone().requires_grad_(True)
        x_i_model = normalize(x_i_raw.unsqueeze(0), bounds=model_bounds)
        posterior_i = model.posterior(x_i_model)
        mu_i = posterior_i.mean.squeeze()
        sigma_i = posterior_i.variance.squeeze().clamp_min(1e-18)

        # Gradients are with respect to raw x coordinates (chain rule through normalize).
        grad_mu_i = torch.autograd.grad(mu_i, x_i_raw, retain_graph=True)[0]
        grad_sigma_i = torch.autograd.grad(sigma_i, x_i_raw)[0]

        mu_vals[i] = mu_i.detach()
        sigma_vals[i] = sigma_i.detach()
        grad_mu[i] = grad_mu_i.detach()
        grad_sigma[i] = grad_sigma_i.detach()

    sigma_safe = sigma_vals.clamp_min(eps)
    inv_sigma = 1.0 / sigma_safe
    inv_sigma_sq = inv_sigma**2

    mu_x1, mu_x2 = grad_mu[:, 0], grad_mu[:, 1]
    sig_x1, sig_x2 = grad_sigma[:, 0], grad_sigma[:, 1]

    g11 = (mu_x1 * mu_x1) * inv_sigma + 0.5 * (sig_x1 * sig_x1) * inv_sigma_sq
    g22 = (mu_x2 * mu_x2) * inv_sigma + 0.5 * (sig_x2 * sig_x2) * inv_sigma_sq
    g12 = (mu_x1 * mu_x2) * inv_sigma + 0.5 * (sig_x1 * sig_x2) * inv_sigma_sq
    fim_trace = g11 + g22

    det_full = g11 * g22 - g12 * g12
    det_full = torch.clamp(det_full, min=0.0)
    logdet_half = 0.5 * torch.log(det_full + eps)

    det_shortcut = (
        0.5
        * (1.0 / (sigma_safe**3))
        * (mu_x1 * sig_x2 - mu_x2 * sig_x1) ** 2
    )

    out = {
        "X1": mesh_x1,
        "X2": mesh_x2,
        "X_flat": x_flat.detach().cpu().numpy(),
        "mu": mu_vals.detach().cpu().numpy().reshape(grid_size, grid_size),
        "Sigma": sigma_vals.detach().cpu().numpy().reshape(grid_size, grid_size),
        "grad_mu": grad_mu.detach().cpu().numpy().reshape(grid_size, grid_size, 2),
        "grad_Sigma": grad_sigma.detach().cpu().numpy().reshape(grid_size, grid_size, 2),
        "det_g": det_full.detach().cpu().numpy().reshape(grid_size, grid_size),
        "fim_trace": fim_trace.detach().cpu().numpy().reshape(grid_size, grid_size),
        "det_g_shortcut": det_shortcut.detach().cpu().numpy().reshape(grid_size, grid_size),
        "logdet_g_half": logdet_half.detach().cpu().numpy().reshape(grid_size, grid_size),
        "eps": eps,
    }
    if return_metric:
        out["g11"] = g11.detach().cpu().numpy().reshape(grid_size, grid_size)
        out["g12"] = g12.detach().cpu().numpy().reshape(grid_size, grid_size)
        out["g22"] = g22.detach().cpu().numpy().reshape(grid_size, grid_size)
    return out

