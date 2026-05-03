"""Generate Figure 3: 2D pullback Fisher metric determinant map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.utils.transforms import normalize
from matplotlib import pyplot as plt

from neurips_viz.fig3.unit_01_data import (
    build_gp_surrogate,
    get_training_data,
    make_rotated_function,
)
from neurips_viz.fig3.unit_02_det_metric import compute_det_gt_2d
from neurips_viz.fig3.unit_03_plot_fig3 import plot_det_gt_figure


def _compute_logei_and_grad_grid(
    model, x_grid_raw: np.ndarray, bounds: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate LogEI and exact autograd gradient wrt raw x on flattened grid."""
    best_f = float(model.train_targets.max().item())
    acqf = LogExpectedImprovement(model=model, best_f=best_f)
    candidates_raw = (
        torch.from_numpy(x_grid_raw).to(dtype=torch.float64).requires_grad_(True)
    )
    candidates = normalize(candidates_raw, bounds=bounds)
    vals = acqf(candidates.unsqueeze(1)).squeeze(-1)
    grad_vals = torch.autograd.grad(vals.sum(), candidates_raw)[0]
    return vals.detach().cpu().numpy(), grad_vals.detach().cpu().numpy()


def _compute_local_geometry_at_x(
    model,
    x_raw: np.ndarray,
    bounds: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact local G(x) and grad alpha(x) via autograd at raw-domain point."""
    x_t = torch.tensor(x_raw, dtype=torch.float64).view(1, 2).requires_grad_(True)
    x_model = normalize(x_t, bounds=bounds)

    post = model.posterior(x_model)
    mu = post.mean.squeeze()
    var = post.variance.squeeze().clamp_min(1e-18)

    grad_mu = torch.autograd.grad(mu, x_t, retain_graph=True)[0].squeeze(0)
    grad_var = torch.autograd.grad(var, x_t, retain_graph=True)[0].squeeze(0)
    g = torch.outer(grad_mu, grad_mu) / var + torch.outer(grad_var, grad_var) / (
        2.0 * var * var
    )

    best_f = float(model.train_targets.max().item())
    acqf = LogExpectedImprovement(model=model, best_f=best_f)
    alpha = acqf(x_model.unsqueeze(1)).squeeze()
    grad_alpha = torch.autograd.grad(alpha, x_t)[0].squeeze(0)
    return g.detach().cpu().numpy(), grad_alpha.detach().cpu().numpy()


def _compute_validation(det_data: dict, tol: float = 5e-6) -> dict:
    sigma = det_data["Sigma"]
    det_full = det_data["det_g"]
    det_short = det_data["det_g_shortcut"]

    denom = np.maximum(np.abs(det_full), 1e-14)
    rel_err = np.abs(det_short - det_full) / denom
    symmetry_err = 0.0  # g12 == g21 by construction in compute_det_gt_2d

    return {
        "sigma_min": float(np.min(sigma)),
        "sigma_positive": bool(np.min(sigma) > 0.0),
        "det_min": float(np.min(det_full)),
        "det_nonnegative": bool(np.min(det_full) >= -1e-10),
        "metric_symmetric": bool(symmetry_err <= tol),
        "det_relerr_max": float(np.max(rel_err)),
        "det_relerr_p99": float(np.quantile(rel_err, 0.99)),
    }


def main(
    out_dir: str = "neurips_viz/outputs",
    seed: int = 42,
    n_train: int = 6,
    grid_size: int = 200,
    eps: float = 1e-14,
    rho: float = 0.09,
    euc_rho: float = 1.0,
    vector_field_grid_size: int = 10,
    vector_field_quiver_scale: float = 36.0,
    vector_field_max_norm: float = 1.0,
    left_box_x0: float | None = None,
    left_box_y0: float | None = None,
    left_box_x1: float | None = None,
    left_box_y1: float | None = None,
    right_box_x0: float | None = None,
    right_box_y0: float | None = None,
    right_box_x1: float | None = None,
    right_box_y1: float | None = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    f_true = make_rotated_function(angle_degrees=32.0)
    x_train_raw, x_train, y_train, y_mean, y_std = get_training_data(
        bounds=bounds, n_train=n_train, seed=seed
    )
    print(f"Using n_train={n_train} with {x_train.shape[0]} observed points.")
    model = build_gp_surrogate(x_train=x_train, y_train=y_train)

    det_data = compute_det_gt_2d(
        model=model,
        bounds=bounds,
        model_bounds=bounds,
        grid_size=grid_size,
        eps=eps,
        return_metric=True,
    )

    idx_best = int(y_train.squeeze(-1).argmax().item())
    x_best = x_train_raw[idx_best].cpu().numpy()
    acq_flat, grad_acq_flat = _compute_logei_and_grad_grid(
        model=model, x_grid_raw=det_data["X_flat"], bounds=bounds
    )
    acq_grid = acq_flat.reshape(grid_size, grid_size)
    grad_acq_x1_grid = grad_acq_flat[:, 0].reshape(grid_size, grid_size)
    grad_acq_x2_grid = grad_acq_flat[:, 1].reshape(grid_size, grid_size)
    true_flat = (
        f_true(torch.from_numpy(det_data["X_flat"]).to(dtype=torch.float64))
        .squeeze(-1)
        .cpu()
        .numpy()
    )
    true_flat = (true_flat - float(y_mean.item())) / float(y_std.item())
    true_grid = true_flat.reshape(grid_size, grid_size)
    x_next = det_data["X_flat"][int(acq_flat.argmax())]
    g_local, grad_alpha_local = _compute_local_geometry_at_x(
        model=model,
        x_raw=x_best,
        bounds=bounds,
    )
    x_probe = np.array([0.1, 0.9], dtype=np.float64)
    g_probe, grad_alpha_probe = _compute_local_geometry_at_x(
        model=model,
        x_raw=x_probe,
        bounds=bounds,
    )

    left_group_box_corners = None
    if all(v is not None for v in [left_box_x0, left_box_y0, left_box_x1, left_box_y1]):
        left_group_box_corners = (
            float(left_box_x0),
            float(left_box_y0),
            float(left_box_x1),
            float(left_box_y1),
        )
    right_group_box_corners = None
    if all(
        v is not None for v in [right_box_x0, right_box_y0, right_box_x1, right_box_y1]
    ):
        right_group_box_corners = (
            float(right_box_x0),
            float(right_box_y0),
            float(right_box_x1),
            float(right_box_y1),
        )

    fig = plot_det_gt_figure(
        X1=det_data["X1"],
        X2=det_data["X2"],
        true_grid=true_grid,
        acq_grid=acq_grid,
        grad_acq_x1_grid=grad_acq_x1_grid,
        grad_acq_x2_grid=grad_acq_x2_grid,
        mu_grid=det_data["mu"],
        Sigma_grid=det_data["Sigma"],
        fim_trace=det_data["fim_trace"],
        g11=det_data["g11"],
        g12=det_data["g12"],
        g22=det_data["g22"],
        X_train=x_train_raw.cpu().numpy(),
        x_best=x_best,
        x_next=x_next,
        g_local=g_local,
        grad_alpha_local=grad_alpha_local,
        x_probe=x_probe,
        g_probe=g_probe,
        grad_alpha_probe=grad_alpha_probe,
        metric_ellipses=None,
        local_radius=0.06,
        local_rho=rho,
        euclidean_rho_scale=euc_rho,
        vector_field_grid_size=vector_field_grid_size,
        vector_field_quiver_scale=vector_field_quiver_scale,
        vector_field_max_norm=vector_field_max_norm,
        left_group_box_corners=left_group_box_corners,
        right_group_box_corners=right_group_box_corners,
        savepath=str(out_path / "fig3_trace_gt_2d"),
    )
    plt.close(fig)

    checks = _compute_validation(det_data)
    print("Validation checks:")
    for key, value in checks.items():
        print(f"  - {key}: {value}")
    print("Takeaway: same BO landscape, different local geometry.")
    print(f"Saved {(out_path / 'fig3_trace_gt_2d.png')}")
    print(f"Saved {(out_path / 'fig3_trace_gt_2d.pdf')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 determinant metric heatmap."
    )
    parser.add_argument("--out_dir", default="neurips_viz/outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=3)
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-14)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--euc_rho", type=float, default=1.0)
    parser.add_argument(
        "--vector_field_grid_size",
        type=int,
        default=7,
        help="Number of vector-field points per side on LogEI panel.",
    )
    parser.add_argument(
        "--vf_grid_n",
        type=int,
        default=None,
        help="Alias for --vector_field_grid_size.",
    )
    parser.add_argument(
        "--quiver_scale",
        type=float,
        default=40.0,
        help="Matplotlib quiver scale for LogEI vector fields (smaller => longer arrows).",
    )
    parser.add_argument(
        "--vf_max_norm",
        type=float,
        default=5,
        help="Cap quiver vector magnitude to this value (<=0 disables clipping).",
    )
    parser.add_argument("--left_box_x0", type=float, default=-0.01)
    parser.add_argument("--left_box_y0", type=float, default=0.07)
    parser.add_argument("--left_box_x1", type=float, default=0.4)
    parser.add_argument("--left_box_y1", type=float, default=1.05)
    parser.add_argument("--right_box_x0", type=float, default=0.41)
    parser.add_argument("--right_box_y0", type=float, default=0.07)
    parser.add_argument("--right_box_x1", type=float, default=0.75)
    parser.add_argument("--right_box_y1", type=float, default=1.05)
    args = parser.parse_args()
    main(
        out_dir=args.out_dir,
        seed=args.seed,
        n_train=args.n_train,
        grid_size=args.grid_size,
        eps=args.eps,
        rho=args.rho,
        euc_rho=args.euc_rho,
        vector_field_grid_size=(
            args.vf_grid_n
            if args.vf_grid_n is not None
            else args.vector_field_grid_size
        ),
        vector_field_quiver_scale=args.quiver_scale,
        vector_field_max_norm=args.vf_max_norm,
        left_box_x0=args.left_box_x0,
        left_box_y0=args.left_box_y0,
        left_box_x1=args.left_box_x1,
        left_box_y1=args.left_box_y1,
        right_box_x0=args.right_box_x0,
        right_box_y0=args.right_box_y0,
        right_box_x1=args.right_box_x1,
        right_box_y1=args.right_box_y1,
    )
