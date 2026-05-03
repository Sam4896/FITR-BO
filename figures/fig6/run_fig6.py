"""Generate Figure 6: LogEI with TuRBO, FIM ellipse, and FITR (same G as ellipse)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from neurips_viz.fig6.unit_01_data import (
    build_deep_ensemble_bnn,
    build_gp_surrogate,
    build_ibnn_surrogate,
    get_training_data,
)
from neurips_viz.fig6.unit_02_geometry import (
    compute_fitr_tr_from_same_G,
    compute_gx_at_point,
    compute_logei_grid,
    compute_turbo_tr_ard,
)
from neurips_viz.fig6.unit_03_plot_fig6 import plot_fig6


def _make_grid(
    bounds: torch.Tensor, grid_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = torch.linspace(
        float(bounds[0, 0]), float(bounds[1, 0]), grid_size, dtype=torch.float64
    )
    x2 = torch.linspace(
        float(bounds[0, 1]), float(bounds[1, 1]), grid_size, dtype=torch.float64
    )
    mesh_x1, mesh_x2 = torch.meshgrid(x1, x2, indexing="ij")
    X_flat = torch.stack([mesh_x1.reshape(-1), mesh_x2.reshape(-1)], dim=1).numpy()
    return mesh_x1.numpy(), mesh_x2.numpy(), X_flat


def main(
    out_dir: str = "neurips_viz/outputs",
    seed: int = 42,
    n_train: int = 6,
    grid_size: int = 80,
    tr_length: float = 0.72,
    rho: float = 0.14,
    fim_ellipse_scale: float = 5.0,
    bnn_epochs: int = 380,
    angle_degrees: float = 32.0,
) -> None:
    """2D domain matches Fig. 3 (`get_training_data` Sobol samples in unit box → bounds)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

    print(f"Training data (Fig. 3 pipeline): n_train={n_train}, seed={seed}")
    x_train_raw, x_train_norm, y_train, _y_mean, _y_std = get_training_data(
        bounds=bounds,
        n_train=n_train,
        seed=seed,
        angle_degrees=angle_degrees,
    )

    print("Fitting GP-SE …")
    model_gp = build_gp_surrogate(x_train=x_train_norm, y_train=y_train)

    print("Fitting GP-IBNN …")
    model_ibnn = build_ibnn_surrogate(x_train=x_train_norm, y_train=y_train)

    print("Training deep ensemble BNN …")
    model_bnn = build_deep_ensemble_bnn(
        x_train=x_train_norm,
        y_train=y_train,
        n_epochs=bnn_epochs,
        seed=seed + 17,
    )

    X1, X2, X_flat = _make_grid(bounds, grid_size)
    best_f = float(y_train.max().item())

    print("Grids: LogEI …")
    logei_gp = compute_logei_grid(model_gp, X_flat, bounds, best_f=best_f).reshape(
        grid_size, grid_size
    )
    logei_ibnn = compute_logei_grid(model_ibnn, X_flat, bounds, best_f=best_f).reshape(
        grid_size, grid_size
    )
    logei_bnn = compute_logei_grid(model_bnn, X_flat, bounds, best_f=best_f).reshape(
        grid_size, grid_size
    )

    idx_best = int(y_train.squeeze(-1).argmax().item())
    x_best = x_train_raw[idx_best].cpu().numpy()
    x_random = np.array([0.15, 0.75], dtype=np.float64)
    print(f"x_best = {x_best.tolist()}")
    print(f"x_random = {x_random.tolist()}")

    print("Geometry: G_x, TuRBO (GP-SE), FITR …")
    g_gp_best = compute_gx_at_point(model_gp, x_best, bounds)
    g_gp_random = compute_gx_at_point(model_gp, x_random, bounds)
    g_ibnn_best = compute_gx_at_point(model_ibnn, x_best, bounds)
    g_ibnn_random = compute_gx_at_point(model_ibnn, x_random, bounds)
    g_bnn_best = compute_gx_at_point(model_bnn, x_best, bounds)
    g_bnn_random = compute_gx_at_point(model_bnn, x_random, bounds)

    turbo_best = compute_turbo_tr_ard(model_gp, x_best, bounds, tr_length)
    turbo_random = compute_turbo_tr_ard(model_gp, x_random, bounds, tr_length)

    # FITR boxes: diagonal of the same pullback G as the red ellipse (compute_gx_at_point).
    fitr_gp_best = compute_fitr_tr_from_same_G(g_gp_best, x_best, bounds, tr_length)
    fitr_gp_random = compute_fitr_tr_from_same_G(
        g_gp_random, x_random, bounds, tr_length
    )
    fitr_ibnn_best = compute_fitr_tr_from_same_G(g_ibnn_best, x_best, bounds, tr_length)
    fitr_ibnn_random = compute_fitr_tr_from_same_G(
        g_ibnn_random, x_random, bounds, tr_length
    )
    fitr_bnn_best = compute_fitr_tr_from_same_G(g_bnn_best, x_best, bounds, tr_length)
    fitr_bnn_random = compute_fitr_tr_from_same_G(
        g_bnn_random, x_random, bounds, tr_length
    )

    print("Plotting …")
    fig = plot_fig6(
        X1=X1,
        X2=X2,
        logei_gp=logei_gp,
        logei_ibnn=logei_ibnn,
        logei_bnn=logei_bnn,
        turbo_best=turbo_best,
        turbo_random=turbo_random,
        fitr_gp_best=fitr_gp_best,
        fitr_gp_random=fitr_gp_random,
        fitr_ibnn_best=fitr_ibnn_best,
        fitr_ibnn_random=fitr_ibnn_random,
        fitr_bnn_best=fitr_bnn_best,
        fitr_bnn_random=fitr_bnn_random,
        g_gp_best=g_gp_best,
        g_gp_random=g_gp_random,
        g_ibnn_best=g_ibnn_best,
        g_ibnn_random=g_ibnn_random,
        g_bnn_best=g_bnn_best,
        g_bnn_random=g_bnn_random,
        x_best=x_best,
        x_random=x_random,
        x_train=x_train_raw.cpu().numpy(),
        rho=rho,
        fim_ellipse_scale=fim_ellipse_scale,
        savepath=str(out_path / "fig6_fitr_viz"),
    )
    plt.close(fig)

    metrics = {
        "bounds": bounds.tolist(),
        "angle_degrees": angle_degrees,
        "tr_length": tr_length,
        "rho_ellipse": rho,
        "fim_ellipse_scale": fim_ellipse_scale,
        "rho_ellipse_effective_plot": rho * fim_ellipse_scale,
        "fitr_diag_from_same_G_as_ellipse": True,
        "x_best": x_best.tolist(),
        "x_random": x_random.tolist(),
        "best_f_standardized": best_f,
        "turbo_lo_best": turbo_best["lo"].tolist(),
        "turbo_hi_best": turbo_best["hi"].tolist(),
        "fitr_gp_hw_best": fitr_gp_best["hw"].tolist(),
        "fitr_ibnn_hw_best": fitr_ibnn_best["hw"].tolist(),
        "fitr_bnn_hw_best": fitr_bnn_best["hw"].tolist(),
        "g_gp_best_trace": g_gp_best["g11"] + g_gp_best["g22"],
    }
    metrics_dir = out_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "fig6_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {out_path / 'fig6_fitr_viz.png'}")
    print(f"Saved {out_path / 'fig6_fitr_viz.pdf'}")
    print(f"Saved {metrics_dir / 'fig6_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Figure 6: FITR visualization."
    )
    parser.add_argument("--out_dir", default="neurips_viz/outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=6)
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--tr_length", type=float, default=0.3)
    parser.add_argument("--rho", type=float, default=0.14)
    parser.add_argument(
        "--fim_ellipse_scale",
        type=float,
        default=4.0,
        help=(
            "Multiplies rho only when drawing FIM ellipses (δᵀGδ = ρ²); "
            "does not affect TuRBO/FITR boxes. Increase if ellipses look too small."
        ),
    )
    parser.add_argument("--bnn_epochs", type=int, default=380)
    parser.add_argument("--angle_degrees", type=float, default=32.0)
    args = parser.parse_args()
    main(
        out_dir=args.out_dir,
        seed=args.seed,
        n_train=args.n_train,
        grid_size=args.grid_size,
        tr_length=args.tr_length,
        rho=args.rho,
        fim_ellipse_scale=args.fim_ellipse_scale,
        bnn_epochs=args.bnn_epochs,
        angle_degrees=args.angle_degrees,
    )
