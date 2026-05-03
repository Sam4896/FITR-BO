"""Generate Figure 7: LogEI heatmaps with pullback FIM ellipses only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from neurips_viz.fig7.unit_01_data import (
    build_deep_ensemble_bnn,
    build_gp_surrogate,
    build_ibnn_surrogate,
    get_training_data,
)
from neurips_viz.fig7.unit_02_geometry import (
    build_gx_at_anchors,
    compute_mean_grid,
    compute_variance_grid,
    compute_logei_grid,
    make_anchor_points,
)
from neurips_viz.fig7.unit_03_plot_fig7 import plot_fig7


TOP_GROUP_BOX_CORNERS: tuple[float, float, float, float] | None = (0.0, 0.0, 1.0, 0.4)
MIDDLE_GROUP_BOX_CORNERS: tuple[float, float, float, float] | None = (
    0.0,
    0.6,
    1.0,
    1.0,
)
BOTTOM_GROUP_BOX_CORNERS: tuple[float, float, float, float] | None = (
    0.0,
    0.4,
    1.0,
    0.6,
)


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
    x_flat = torch.stack([mesh_x1.reshape(-1), mesh_x2.reshape(-1)], dim=1).numpy()
    return mesh_x1.numpy(), mesh_x2.numpy(), x_flat


def main(
    out_dir: str = "neurips_viz/outputs",
    seed: int = 42,
    n_train: int = 6,
    grid_size: int = 80,
    bnn_epochs: int = 380,
    angle_degrees: float = 32.0,
    rho: float = 0.08,
    ellipse_scale_gp: float = 1.0,
    ellipse_scale_ibnn: float = 1.0,
    ellipse_scale_bnn: float = 1.0,
    heatmap_floor_pct_gp: float = 80.0,
    heatmap_floor_pct_ibnn: float = 80.0,
    heatmap_floor_pct_bnn: float = 80.0,
    eig_floor_scale: float = 1.0,
    n_ellipses: int = 4,
    anchor_seed: int | None = None,
    ellipse_mode: str = "random",
    top_group_box_corners: tuple[float, float, float, float] | None = None,
    middle_group_box_corners: tuple[float, float, float, float] | None = None,
    bottom_group_box_corners: tuple[float, float, float, float] | None = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

    x_train_raw, x_train_norm, y_train, _y_mean, _y_std = get_training_data(
        bounds=bounds,
        n_train=n_train,
        seed=seed,
        angle_degrees=angle_degrees,
    )

    model_gp = build_gp_surrogate(x_train=x_train_norm, y_train=y_train)
    model_ibnn = build_ibnn_surrogate(x_train=x_train_norm, y_train=y_train)
    model_bnn = build_deep_ensemble_bnn(
        x_train=x_train_norm,
        y_train=y_train,
        n_epochs=bnn_epochs,
        seed=seed + 17,
    )

    x1, x2, x_flat = _make_grid(bounds, grid_size)
    best_f = float(y_train.max().item())

    logei_gp = compute_logei_grid(model_gp, x_flat, bounds, best_f=best_f).reshape(
        grid_size, grid_size
    )
    logei_ibnn = compute_logei_grid(model_ibnn, x_flat, bounds, best_f=best_f).reshape(
        grid_size, grid_size
    )
    logei_bnn = compute_logei_grid(model_bnn, x_flat, bounds, best_f=best_f).reshape(
        grid_size, grid_size
    )
    mean_gp = compute_mean_grid(model_gp, x_flat, bounds).reshape(grid_size, grid_size)
    mean_ibnn = compute_mean_grid(model_ibnn, x_flat, bounds).reshape(
        grid_size, grid_size
    )
    mean_bnn = compute_mean_grid(model_bnn, x_flat, bounds).reshape(
        grid_size, grid_size
    )
    var_gp = compute_variance_grid(model_gp, x_flat, bounds).reshape(
        grid_size, grid_size
    )
    var_ibnn = compute_variance_grid(model_ibnn, x_flat, bounds).reshape(
        grid_size, grid_size
    )
    var_bnn = compute_variance_grid(model_bnn, x_flat, bounds).reshape(
        grid_size, grid_size
    )

    if anchor_seed is None:
        anchor_seed = seed
    anchors = make_anchor_points(
        bounds=bounds,
        n_anchors=n_ellipses,
        seed=anchor_seed,
        mode=ellipse_mode,
    )

    gx_gp = build_gx_at_anchors(model_gp, anchors, bounds)
    gx_ibnn = build_gx_at_anchors(model_ibnn, anchors, bounds)
    gx_bnn = build_gx_at_anchors(model_bnn, anchors, bounds)

    def _eigvals_from_g(g: dict) -> list[float]:
        mat = np.array([[g["g11"], g["g12"]], [g["g12"], g["g22"]]], dtype=np.float64)
        vals = np.linalg.eigvalsh(mat)
        return [float(vals[0]), float(vals[1])]

    eig_gp = [_eigvals_from_g(g) for g in gx_gp]
    eig_ibnn = [_eigvals_from_g(g) for g in gx_ibnn]
    eig_bnn = [_eigvals_from_g(g) for g in gx_bnn]

    print("Ellipse eigvals per anchor [lambda_min, lambda_max]:")
    for name, eigs in [("GP-SE", eig_gp), ("GP-IBNN", eig_ibnn), ("Deep BNN", eig_bnn)]:
        print(f"{name}:")
        for i, vals in enumerate(eigs):
            eps_dyn = eig_floor_scale * (vals[0] + vals[1]) / 2.0
            print(
                f"  anchor_{i}: [{vals[0]:.6e}, {vals[1]:.6e}], "
                f"eps_dyn(mean*scale)={eps_dyn:.6e}"
            )

    fig = plot_fig7(
        X1=x1,
        X2=x2,
        mean_gp=mean_gp,
        mean_ibnn=mean_ibnn,
        mean_bnn=mean_bnn,
        var_gp=var_gp,
        var_ibnn=var_ibnn,
        var_bnn=var_bnn,
        logei_gp=logei_gp,
        logei_ibnn=logei_ibnn,
        logei_bnn=logei_bnn,
        x_train=x_train_raw.cpu().numpy(),
        anchors=anchors,
        gx_gp=gx_gp,
        gx_ibnn=gx_ibnn,
        gx_bnn=gx_bnn,
        rho=rho,
        ellipse_scale_gp=ellipse_scale_gp,
        ellipse_scale_ibnn=ellipse_scale_ibnn,
        ellipse_scale_bnn=ellipse_scale_bnn,
        heatmap_floor_pct_gp=heatmap_floor_pct_gp,
        heatmap_floor_pct_ibnn=heatmap_floor_pct_ibnn,
        heatmap_floor_pct_bnn=heatmap_floor_pct_bnn,
        eig_floor_scale=eig_floor_scale,
        savepath=str(out_path / "fig7_ellipses_only"),
    )
    plt.close(fig)

    metrics = {
        "seed": seed,
        "n_train": n_train,
        "grid_size": grid_size,
        "angle_degrees": angle_degrees,
        "rho_base": rho,
        "ellipse_scale_gp": ellipse_scale_gp,
        "ellipse_scale_ibnn": ellipse_scale_ibnn,
        "ellipse_scale_bnn": ellipse_scale_bnn,
        "heatmap_floor_pct_gp": heatmap_floor_pct_gp,
        "heatmap_floor_pct_ibnn": heatmap_floor_pct_ibnn,
        "heatmap_floor_pct_bnn": heatmap_floor_pct_bnn,
        "eig_floor_scale": eig_floor_scale,
        "eigvals_gp": eig_gp,
        "eigvals_ibnn": eig_ibnn,
        "eigvals_bnn": eig_bnn,
        "n_ellipses": n_ellipses,
        "ellipse_mode": ellipse_mode,
        "anchor_seed": anchor_seed,
        "top_group_box_corners": top_group_box_corners,
        "middle_group_box_corners": middle_group_box_corners,
        "bottom_group_box_corners": bottom_group_box_corners,
        "anchors": [a.tolist() for a in anchors],
        "best_f_standardized": best_f,
    }
    metrics_dir = out_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "fig7_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {out_path / 'fig7_ellipses_only.png'}")
    print(f"Saved {out_path / 'fig7_ellipses_only.pdf'}")
    print(f"Saved {metrics_dir / 'fig7_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Figure 7 ellipses-only view."
    )
    parser.add_argument("--out_dir", default="neurips_viz/outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--bnn_epochs", type=int, default=380)
    parser.add_argument("--angle_degrees", type=float, default=32.0)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--ellipse_scale_gp", type=float, default=1.0)
    parser.add_argument("--ellipse_scale_ibnn", type=float, default=4.0)
    parser.add_argument("--ellipse_scale_bnn", type=float, default=3.0)
    parser.add_argument("--heatmap_floor_pct_gp", type=float, default=10.0)
    parser.add_argument("--heatmap_floor_pct_ibnn", type=float, default=10.0)
    parser.add_argument("--heatmap_floor_pct_bnn", type=float, default=10.0)
    parser.add_argument("--n_ellipses", type=int, default=3)
    parser.add_argument(
        "--ellipse_mode",
        choices=["random", "grid"],
        default="grid",
        help=(
            "random: plot n_ellipses anchors sampled uniformly; "
            "grid: plot n_ellipses points per axis (total n_ellipses^2)."
        ),
    )
    parser.add_argument(
        "--anchor_seed",
        type=int,
        default=None,
        help="Seed for random ellipse anchor locations (defaults to --seed).",
    )
    parser.add_argument(
        "--eig_floor_scale",
        type=float,
        default=0.1,
        help="Dynamic floor: eps = eig_floor_scale * mean(eigenvalues) per ellipse.",
    )

    args = parser.parse_args()
    main(
        out_dir=args.out_dir,
        seed=args.seed,
        n_train=args.n_train,
        grid_size=args.grid_size,
        bnn_epochs=args.bnn_epochs,
        angle_degrees=args.angle_degrees,
        rho=args.rho,
        ellipse_scale_gp=args.ellipse_scale_gp,
        ellipse_scale_ibnn=args.ellipse_scale_ibnn,
        ellipse_scale_bnn=args.ellipse_scale_bnn,
        heatmap_floor_pct_gp=args.heatmap_floor_pct_gp,
        heatmap_floor_pct_ibnn=args.heatmap_floor_pct_ibnn,
        heatmap_floor_pct_bnn=args.heatmap_floor_pct_bnn,
        eig_floor_scale=args.eig_floor_scale,
        n_ellipses=args.n_ellipses,
        anchor_seed=args.anchor_seed,
        ellipse_mode=args.ellipse_mode,
        top_group_box_corners=TOP_GROUP_BOX_CORNERS,
        middle_group_box_corners=MIDDLE_GROUP_BOX_CORNERS,
        bottom_group_box_corners=BOTTOM_GROUP_BOX_CORNERS,
    )
