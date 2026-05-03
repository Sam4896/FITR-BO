"""Plotting helpers for the Figure 3 three-panel geometry visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FormatStrFormatter

_GEOM_COLOR = "orangered"
_DET_CMAP = "cividis"


def _legend_ellipse(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return mpatches.Ellipse(
        (xdescent + 0.5 * width, ydescent + 0.5 * height),
        width=0.9 * width,
        height=0.55 * height,
        fill=False,
        edgecolor=orig_handle.get_edgecolor(),
        linewidth=orig_handle.get_linewidth(),
    )


def _legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return mpatches.FancyArrowPatch(
        (xdescent + 0.05 * width, ydescent + 0.5 * height),
        (xdescent + 0.95 * width, ydescent + 0.5 * height),
        arrowstyle="->",
        mutation_scale=fontsize * 0.9,
        linewidth=orig_handle.get_linewidth(),
        color=orig_handle.get_edgecolor(),
    )


def compute_metric_ellipse(
    g_matrix: np.ndarray,
    center: np.ndarray,
    rho: float,
    n_points: int = 100,
) -> np.ndarray:
    """Return points on delta^T g delta = rho^2 in Euclidean coordinates."""
    evals, evecs = np.linalg.eigh(g_matrix)
    evals = np.clip(evals, 1e-14, None)
    axes = rho / np.sqrt(evals)

    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    ellipse = evecs @ np.diag(axes) @ circle
    return center[:, None] + ellipse


def _build_anchor_indices(grid_size: int, n_anchor_axis: int) -> list[tuple[int, int]]:
    ids = np.linspace(1, grid_size - 2, n_anchor_axis, dtype=int)
    return [(i, j) for i in ids for j in ids]


def build_metric_ellipse_overlays(
    x1: np.ndarray,
    x2: np.ndarray,
    g11: np.ndarray,
    g12: np.ndarray,
    g22: np.ndarray,
    rho: float = 0.05,
    n_anchor_axis: int = 10,
    n_points: int = 70,
) -> list[np.ndarray]:
    """Construct sparse ellipse overlays across the grid."""
    ellipses: list[np.ndarray] = []
    for i, j in _build_anchor_indices(
        grid_size=x1.shape[0], n_anchor_axis=n_anchor_axis
    ):
        g_mat = np.array(
            [[g11[i, j], g12[i, j]], [g12[i, j], g22[i, j]]], dtype=np.float64
        )
        center = np.array([x1[i, j], x2[i, j]], dtype=np.float64)
        ellipses.append(
            compute_metric_ellipse(g_mat, center=center, rho=rho, n_points=n_points)
        )
    return ellipses


def _panel_common(
    ax: plt.Axes,
    X_train: np.ndarray | None,
    x_best: np.ndarray | None,
    x_next: np.ndarray | None,
) -> None:
    if X_train is not None:
        ax.scatter(X_train[:, 0], X_train[:, 1], c="black", s=12, zorder=8)
    if x_best is not None:
        ax.scatter(
            [x_best[0]],
            [x_best[1]],
            marker="*",
            s=40,
            c="gold",
            edgecolors="black",
            linewidths=0.7,
            zorder=9,
        )
    if x_next is not None:
        ax.scatter(
            [x_next[0]],
            [x_next[1]],
            marker="D",
            s=15,
            c="crimson",
            edgecolors="black",
            linewidths=0.6,
            zorder=9,
        )


def _build_metric_matrix(
    g11: np.ndarray, g12: np.ndarray, g22: np.ndarray, i: int, j: int
) -> np.ndarray:
    return np.array([[g11[i, j], g12[i, j]], [g12[i, j], g22[i, j]]], dtype=np.float64)


def _cbar_ticks(cbar, vmin: float, vmax: float, n: int = 4) -> None:
    ticks = np.linspace(vmin, vmax, max(2, n))
    span = vmax - vmin
    decimals = max(1, int(np.ceil(-np.log10(span / n)))) if span > 0 else 1
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def _nearest_grid_index(
    X1: np.ndarray, X2: np.ndarray, x: np.ndarray
) -> tuple[int, int]:
    d2 = (X1 - x[0]) ** 2 + (X2 - x[1]) ** 2
    return np.unravel_index(np.argmin(d2), d2.shape)


def _vector_field_indices(grid_size: int, n_side: int) -> np.ndarray:
    """Choose evenly spaced interior indices for vector-field sampling."""
    n_side = int(np.clip(n_side, 2, max(2, grid_size - 2)))
    return np.linspace(1, grid_size - 2, n_side, dtype=int)


def _finite_diff_grad_at_point(
    X1: np.ndarray, X2: np.ndarray, field: np.ndarray, x: np.ndarray
) -> np.ndarray:
    i_c, j_c = _nearest_grid_index(X1=X1, X2=X2, x=x)
    dx = float(X1[1, 0] - X1[0, 0]) if X1.shape[0] > 1 else 1e-3
    dy = float(X2[0, 1] - X2[0, 0]) if X2.shape[1] > 1 else 1e-3
    i_l, i_u = max(i_c - 1, 0), min(i_c + 1, X1.shape[0] - 1)
    j_l, j_u = max(j_c - 1, 0), min(j_c + 1, X1.shape[1] - 1)
    d_dx = (field[i_u, j_c] - field[i_l, j_c]) / max((i_u - i_l) * dx, 1e-12)
    d_dy = (field[i_c, j_u] - field[i_c, j_l]) / max((j_u - j_l) * dy, 1e-12)
    return np.array([d_dx, d_dy], dtype=np.float64)


def _clip_vector_field_norm(
    u: np.ndarray, v: np.ndarray, max_norm: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Clip vector magnitudes to max_norm while preserving directions."""
    if max_norm is None or max_norm <= 0:
        return u, v
    norm = np.sqrt(u * u + v * v)
    scale = np.minimum(1.0, max_norm / np.maximum(norm, 1e-12))
    return u * scale, v * scale


def _draw_local_gradient_panel(
    ax: plt.Axes,
    X1: np.ndarray,
    X2: np.ndarray,
    acq_grid: np.ndarray,
    g11: np.ndarray,
    g12: np.ndarray,
    g22: np.ndarray,
    x_center: np.ndarray,
    local_radius: float,
    local_rho: float,
    euclidean_rho_scale: float,
    arrow_scale: float = 1.0,
    fit_arrows_in_box: bool = False,
    grad_alpha: np.ndarray | None = None,
    g_local: np.ndarray | None = None,
    title: str = "Gradient",
    show_ylabel: bool = True,
) -> tuple[tuple[float, float, float, float], tuple[float, float], tuple[float, float]]:
    i_c, j_c = _nearest_grid_index(X1=X1, X2=X2, x=x_center)
    if g_local is not None:
        g_c = np.asarray(g_local, dtype=np.float64)
    else:
        g_c = _build_metric_matrix(g11=g11, g12=g12, g22=g22, i=i_c, j=j_c)

    metric_ellipse = compute_metric_ellipse(
        g_c, center=x_center, rho=local_rho, n_points=180
    )
    ax.plot(metric_ellipse[0], metric_ellipse[1], color="red", linewidth=1.0)

    zoom_half = 3.5 * local_radius
    x_lo = max(float(X1.min()), x_center[0] - zoom_half)
    x_hi = min(float(X1.max()), x_center[0] + zoom_half)
    y_lo = max(float(X2.min()), x_center[1] - zoom_half)
    y_hi = min(float(X2.max()), x_center[1] + zoom_half)

    if grad_alpha is None:
        grad_alpha = _finite_diff_grad_at_point(
            X1=X1, X2=X2, field=acq_grid, x=x_center
        )
    else:
        grad_alpha = np.asarray(grad_alpha, dtype=np.float64)

    if np.linalg.norm(grad_alpha) > 1e-12:
        euc_dir = grad_alpha / max(np.linalg.norm(grad_alpha), 1e-12)
        delta_euc = (euclidean_rho_scale * local_rho) * euc_dir
        ngd_raw = np.linalg.solve(g_c + 1e-10 * np.eye(2), grad_alpha)
        ngd_fisher_norm = np.sqrt(max(float(ngd_raw.T @ g_c @ ngd_raw), 1e-16))
        delta_ngd = local_rho * (ngd_raw / ngd_fisher_norm)
        delta_euc = arrow_scale * delta_euc
        delta_ngd = arrow_scale * delta_ngd

        if fit_arrows_in_box:

            def _max_feasible_scale(delta: np.ndarray) -> float:
                s = np.inf
                if delta[0] > 0:
                    s = min(s, (x_hi - x_center[0]) / delta[0])
                elif delta[0] < 0:
                    s = min(s, (x_lo - x_center[0]) / delta[0])
                if delta[1] > 0:
                    s = min(s, (y_hi - x_center[1]) / delta[1])
                elif delta[1] < 0:
                    s = min(s, (y_lo - x_center[1]) / delta[1])
                return float(s if np.isfinite(s) else 1.0)

            fit_scale = 0.92 * min(
                _max_feasible_scale(delta_euc), _max_feasible_scale(delta_ngd)
            )
            fit_scale = float(np.clip(fit_scale, 0.0, 1.0))
            delta_euc = fit_scale * delta_euc
            delta_ngd = fit_scale * delta_ngd

        ax.arrow(
            x_center[0],
            x_center[1],
            delta_euc[0],
            delta_euc[1],
            width=0.0009,
            head_width=0.010,
            head_length=0.012,
            color="darkorange",
            length_includes_head=True,
            zorder=11,
        )
        ax.arrow(
            x_center[0],
            x_center[1],
            delta_ngd[0],
            delta_ngd[1],
            width=0.0009,
            head_width=0.010,
            head_length=0.012,
            color="purple",
            length_includes_head=True,
            zorder=11,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    # ax.set_title(title, pad=0.5)
    ax.set_xlabel(r"$x_1$", labelpad=0.3)
    ax.set_ylabel(r"$x_2$" if show_ylabel else "", labelpad=0.2)
    if not show_ylabel:
        ax.set_yticklabels([])
    ax.grid(alpha=0.12)
    return (x_lo, x_hi, y_lo, y_hi), (x_center[0], x_center[1]), (x_hi, y_hi)


def plot_det_gt_figure(
    X1: np.ndarray,
    X2: np.ndarray,
    true_grid: np.ndarray,
    acq_grid: np.ndarray,
    grad_acq_x1_grid: np.ndarray | None,
    grad_acq_x2_grid: np.ndarray | None,
    mu_grid: np.ndarray,
    Sigma_grid: np.ndarray,
    fim_trace: np.ndarray,
    g11: np.ndarray,
    g12: np.ndarray,
    g22: np.ndarray,
    X_train: np.ndarray | None = None,
    x_best: np.ndarray | None = None,
    x_next: np.ndarray | None = None,
    g_local: np.ndarray | None = None,
    grad_alpha_local: np.ndarray | None = None,
    x_probe: np.ndarray | None = None,
    g_probe: np.ndarray | None = None,
    grad_alpha_probe: np.ndarray | None = None,
    metric_ellipses: list[np.ndarray] | None = None,
    local_radius: float = 0.06,
    local_rho: float = 0.09,
    euclidean_rho_scale: float = 1.0,
    vector_field_grid_size: int = 10,
    vector_field_quiver_scale: float = 36.0,
    vector_field_max_norm: float | None = 1.0,
    left_group_box_corners: tuple[float, float, float, float] | None = None,
    right_group_box_corners: tuple[float, float, float, float] | None = None,
    savepath: str | None = None,
) -> plt.Figure:
    """Create 2x4 equal-sized panel figure for Figure 3."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
            "text.usetex": True,
            "mathtext.fontset": "cm",
            "pgf.rcfonts": False,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    fig = plt.figure(figsize=(6.6, 2), constrained_layout=True)

    # Hard-coded panel geometry for direct manual control.
    # Each panel position is [x0, y0, width, height] in figure coordinates.
    panel_w = 0.13
    panel_h = 0.33
    gap_x = 0.06
    gap_y = 0.12
    left0 = 0.05
    bottom0 = 0.20
    top0 = bottom0 + panel_h + gap_y
    x_cols = [left0 + i * (panel_w + gap_x) for i in range(4)]

    pos_true = [x_cols[0], top0, panel_w, panel_h]
    pos_mu = [x_cols[1] - 0.0125, top0, panel_w, panel_h]
    pos_fim = [x_cols[2] + 0.015, top0, panel_w, panel_h]
    pos_grad_best = [x_cols[3] + 0.045, top0 + 0.08, panel_w - 0.055, panel_h - 0.08]

    pos_var = [x_cols[0], bottom0, panel_w, panel_h]
    pos_acq = [x_cols[1] - 0.0125, bottom0, panel_w, panel_h]
    pos_grad_probe = [x_cols[2] + 0.02, bottom0, panel_w - 0.055, panel_h - 0.08]
    pos_grad_norm = [x_cols[3] - 0.04, bottom0, panel_w, panel_h]

    def _group_bbox(
        positions: list[list[float]], pad: float = 0.03
    ) -> tuple[float, float, float, float]:
        x0 = min(p[0] for p in positions) - pad
        y0 = min(p[1] for p in positions) - pad
        x1 = max(p[0] + p[2] for p in positions) + pad
        y1 = max(p[1] + p[3] for p in positions) + pad
        return x0, y0, x1 - x0, y1 - y0

    left_group_positions = [pos_true, pos_mu, pos_var, pos_acq]
    right_group_positions = [pos_fim, pos_grad_best, pos_grad_probe, pos_grad_norm]
    left_box_auto = _group_bbox(left_group_positions, pad=0.03)
    right_box_auto = _group_bbox(right_group_positions, pad=0.03)

    if left_group_box_corners is not None:
        lx0, ly0, lx1, ly1 = left_group_box_corners
        left_box = (lx0, ly0, lx1 - lx0, ly1 - ly0)
    else:
        left_box = left_box_auto
    if right_group_box_corners is not None:
        rx0, ry0, rx1, ry1 = right_group_box_corners
        right_box = (rx0, ry0, rx1 - rx0, ry1 - ry0)
    else:
        right_box = right_box_auto

    fig.add_artist(
        mpatches.Rectangle(
            (left_box[0], left_box[1]),
            left_box[2],
            left_box[3],
            transform=fig.transFigure,
            facecolor="lightgray",
            edgecolor="gray",
            linewidth=0.9,
            alpha=0.20,
            zorder=-5,
            clip_on=False,
        )
    )
    fig.add_artist(
        mpatches.Rectangle(
            (right_box[0], right_box[1]),
            right_box[2],
            right_box[3],
            transform=fig.transFigure,
            facecolor="lightgray",
            edgecolor="gray",
            linewidth=0.9,
            alpha=0.20,
            zorder=-5,
            clip_on=False,
        )
    )
    fig.text(
        left_box[0] + 0.5 * left_box[2],
        left_box[1] + left_box[3] + 0.005,
        "Conventional BO view",
        ha="center",
        va="bottom",
        fontsize=7,
        color="black",
    )
    fig.text(
        right_box[0] + 0.5 * right_box[2],
        right_box[1] + right_box[3] + 0.005,
        "Gradient analysis view",
        ha="center",
        va="bottom",
        fontsize=7,
        color="black",
    )

    ax_true = fig.add_axes(pos_true)
    ax_mu = fig.add_axes(pos_mu)
    ax_b = fig.add_axes(pos_fim)  # FIM trace
    ax_c_best = fig.add_axes(pos_grad_best)  # local at x_best

    ax_var = fig.add_axes(pos_var)
    ax_a = fig.add_axes(pos_acq)  # LogEI
    ax_c_probe = fig.add_axes(pos_grad_probe)  # local at probe
    ax_d = fig.add_axes(pos_grad_norm)  # ||grad acq||

    ticks_percentile_lower = 0.1
    ticks_percentile_upper = 99.9

    # Top-left: true function over input domain
    true_vmin = np.percentile(true_grid, ticks_percentile_lower)
    true_vmax = np.percentile(true_grid, ticks_percentile_upper)
    true_cf = ax_true.contourf(
        X1,
        X2,
        true_grid,
        levels=np.linspace(true_vmin, true_vmax, 55),
        cmap="viridis",
        extend="both",
    )
    cbar_true = fig.colorbar(true_cf, ax=ax_true)
    _cbar_ticks(cbar_true, true_vmin, true_vmax)
    _panel_common(ax_true, X_train=X_train, x_best=x_best, x_next=x_next)
    ax_true.set_title(r"$f_\mathrm{true}(\mathbf{x})$", pad=0.5)
    ax_true.set_xlim(float(X1.min()), float(X1.max()))
    ax_true.set_ylim(float(X2.min()), float(X2.max()))
    ax_true.set_aspect("auto")
    # ax_true.set_aspect("equal", adjustable="box")
    ax_true.set_xlabel("")
    ax_true.set_ylabel(r"$x_2$")
    ax_true.set_xticklabels([])

    # Left column: GP mean and variance over input domain
    mu_vmin = np.percentile(mu_grid, ticks_percentile_lower)
    mu_vmax = np.percentile(mu_grid, ticks_percentile_upper)
    mu_cf = ax_mu.contourf(
        X1,
        X2,
        mu_grid,
        levels=np.linspace(mu_vmin, mu_vmax, 55),
        cmap="coolwarm",
        extend="both",
    )
    cbar_mu = fig.colorbar(mu_cf, ax=ax_mu)
    _cbar_ticks(cbar_mu, mu_vmin, mu_vmax)
    _panel_common(ax_mu, X_train=X_train, x_best=x_best, x_next=x_next)
    ax_mu.set_title(r"GP mean $\mu$", pad=0.5)
    ax_mu.set_xlim(float(X1.min()), float(X1.max()))
    ax_mu.set_ylim(float(X2.min()), float(X2.max()))
    ax_mu.set_aspect("auto")
    # ax_mu.set_aspect("equal", adjustable="box")
    ax_mu.set_xlabel("")
    ax_mu.set_ylabel("")
    ax_mu.set_yticklabels([])
    ax_mu.set_xticklabels([])

    var_vmin = np.percentile(Sigma_grid, ticks_percentile_lower)
    var_vmax = np.percentile(Sigma_grid, ticks_percentile_upper)
    var_cf = ax_var.contourf(
        X1,
        X2,
        Sigma_grid,
        levels=np.linspace(var_vmin, var_vmax, 55),
        cmap="plasma",
        extend="both",
    )
    cbar_var = fig.colorbar(var_cf, ax=ax_var)
    _cbar_ticks(cbar_var, var_vmin, var_vmax)
    _panel_common(ax_var, X_train=X_train, x_best=x_best, x_next=x_next)
    ax_var.set_title(
        r"GP var $\Sigma$",
        pad=0.5,
    )
    ax_var.set_xlim(float(X1.min()), float(X1.max()))
    ax_var.set_ylim(float(X2.min()), float(X2.max()))
    ax_var.set_aspect("auto")
    # ax_var.set_aspect("equal", adjustable="box")
    ax_var.set_xlabel(r"$x_1$", labelpad=0.3)
    ax_var.set_ylabel(r"$x_2$")
    ax_var.set_xticks([0, 0.5, 1])
    ax_var.set_xticklabels(["0", "0.5", "1"])

    # Panel (a): conventional BO view (acquisition heatmap)
    acq_vmin = np.percentile(acq_grid, ticks_percentile_lower)
    acq_vmax = np.percentile(acq_grid, 80)
    acq_cf = ax_a.contourf(
        X1,
        X2,
        acq_grid,
        levels=np.linspace(acq_vmin, acq_vmax, 60),
        cmap="magma",
        extend="both",
    )
    cbar_acq = fig.colorbar(acq_cf, ax=ax_a)
    _cbar_ticks(cbar_acq, acq_vmin, acq_vmax)
    _panel_common(ax_a, X_train=X_train, x_best=x_best, x_next=x_next)
    ax_a.set_title(r"LogEI $\alpha(\mathbf{x})$", pad=0.5)
    ax_a.set_xlabel(r"$x_1$", labelpad=0.3)
    ax_a.set_ylabel("")
    ax_a.set_aspect("auto")
    # ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlim(float(X1.min()), float(X1.max()))
    ax_a.set_ylim(float(X2.min()), float(X2.max()))
    ax_a.set_yticklabels([])
    ax_a.set_xticks([0, 0.5, 1])
    ax_a.set_xticklabels(["0", "0.5", "1"])
    # Overlay Euclidean and natural gradients on the LogEI heatmap.
    det_g = g11 * g22 - g12 * g12
    det_g = np.clip(det_g, 1e-12, None)
    inv11 = g22 / det_g
    inv12 = -g12 / det_g
    inv22 = g11 / det_g

    if grad_acq_x1_grid is None or grad_acq_x2_grid is None:
        dx = float(X1[1, 0] - X1[0, 0]) if X1.shape[0] > 1 else 1e-3
        dy = float(X2[0, 1] - X2[0, 0]) if X2.shape[1] > 1 else 1e-3
        d_acq_dx, d_acq_dy = np.gradient(acq_grid, dx, dy)
    else:
        d_acq_dx = np.asarray(grad_acq_x1_grid, dtype=np.float64)
        d_acq_dy = np.asarray(grad_acq_x2_grid, dtype=np.float64)
    nat_dx = inv11 * d_acq_dx + inv12 * d_acq_dy
    nat_dy = inv12 * d_acq_dx + inv22 * d_acq_dy

    ids = _vector_field_indices(X1.shape[0], vector_field_grid_size)
    Xq = X1[np.ix_(ids, ids)]
    Yq = X2[np.ix_(ids, ids)]
    U_grad, V_grad = d_acq_dx[np.ix_(ids, ids)], d_acq_dy[np.ix_(ids, ids)]
    U_nat, V_nat = nat_dx[np.ix_(ids, ids)], nat_dy[np.ix_(ids, ids)]
    U_grad, V_grad = _clip_vector_field_norm(U_grad, V_grad, vector_field_max_norm)
    U_nat, V_nat = _clip_vector_field_norm(U_nat, V_nat, vector_field_max_norm)
    ax_a.quiver(
        Xq,
        Yq,
        U_grad,
        V_grad,
        color="darkorange",
        angles="xy",
        scale_units="xy",
        scale=max(float(vector_field_quiver_scale), 1e-6),
        width=0.01,
        alpha=0.95,
        zorder=8,
    )
    ax_a.quiver(
        Xq,
        Yq,
        U_nat,
        V_nat,
        color="purple",
        angles="xy",
        scale_units="xy",
        scale=max(float(vector_field_quiver_scale), 1e-6),
        width=0.01,
        alpha=0.95,
        zorder=8,
    )

    # Panel (b): sqrt(FIM trace) + dashed zoom boxes for both local views
    fim_trace_sqrt = np.sqrt(np.clip(fim_trace, a_min=0.0, a_max=None))
    vmin = np.percentile(fim_trace_sqrt, 2.5)
    vmax = np.percentile(fim_trace_sqrt, 90.0)
    heat = ax_b.contourf(
        X1,
        X2,
        fim_trace_sqrt,
        levels=np.linspace(vmin, vmax, 80),
        cmap=_DET_CMAP,
        extend="both",
    )
    cbar = fig.colorbar(heat, ax=ax_b)
    _cbar_ticks(cbar, vmin, vmax)
    _panel_common(ax_b, X_train=X_train, x_best=x_best, x_next=x_next)

    if x_probe is None:
        x_probe = np.array([0.1, 0.9], dtype=np.float64)
    else:
        x_probe = np.asarray(x_probe, dtype=np.float64)

    if x_best is not None:
        x_best_arr = np.asarray(x_best, dtype=np.float64)
    elif x_next is not None:
        x_best_arr = np.asarray(x_next, dtype=np.float64)
    else:
        x_best_arr = np.array([0.5, 0.5], dtype=np.float64)

    def _zoom_box(
        center: np.ndarray, scale: float = 3.5
    ) -> tuple[float, float, float, float]:
        zoom_half = scale * local_radius
        x_lo = max(float(X1.min()), center[0] - zoom_half)
        x_hi = min(float(X1.max()), center[0] + zoom_half)
        y_lo = max(float(X2.min()), center[1] - zoom_half)
        y_hi = min(float(X2.max()), center[1] + zoom_half)
        return (x_lo, x_hi, y_lo, y_hi)

    # Keep zoom box scales consistent across panels for direct comparison.
    box_best = _zoom_box(x_best_arr, scale=3.5)
    box_probe = _zoom_box(x_probe, scale=3.5)
    for (x_lo, x_hi, y_lo, y_hi), color in [
        (box_best, _GEOM_COLOR),
        (box_probe, "magenta"),
    ]:
        ax_b.add_patch(
            mpatches.Rectangle(
                (x_lo, y_lo),
                x_hi - x_lo,
                y_hi - y_lo,
                linewidth=0.8,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                alpha=0.9,
                zorder=10,
                clip_on=False,
            )
        )
    ax_b.set_xlim(float(X1.min()), float(X1.max()))
    ax_b.set_ylim(float(X2.min()), float(X2.max()))
    ax_b.set_xlabel(r"$x_1$", labelpad=0.3)
    ax_b.set_ylabel(r"$x_2$", labelpad=0.35)
    ax_b.set_yticks([0.0, 0.5, 1.0])
    ax_b.set_yticklabels(["0.0", "0.5", "1.0"])
    ax_b.set_title(r"$\sqrt{\mathrm{tr}(g_t(\mathbf{x}))}$", pad=0.5)
    ax_b.set_aspect("auto")

    _draw_local_gradient_panel(
        ax=ax_c_best,
        X1=X1,
        X2=X2,
        acq_grid=acq_grid,
        g11=g11,
        g12=g12,
        g22=g22,
        x_center=x_best_arr,
        local_radius=local_radius,
        local_rho=local_rho,
        euclidean_rho_scale=euclidean_rho_scale,
        arrow_scale=1.0,
        fit_arrows_in_box=False,
        grad_alpha=grad_alpha_local,
        g_local=g_local,
        # title=r"Gradient $x_{\mathrm{best}}$",
        show_ylabel=True,
    )
    ax_c_best.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    _draw_local_gradient_panel(
        ax=ax_c_probe,
        X1=X1,
        X2=X2,
        acq_grid=acq_grid,
        g11=g11,
        g12=g12,
        g22=g22,
        x_center=x_probe,
        local_radius=local_radius,
        local_rho=local_rho,
        euclidean_rho_scale=euclidean_rho_scale,
        arrow_scale=0.65,
        fit_arrows_in_box=True,
        grad_alpha=grad_alpha_probe,
        g_local=g_probe,
        # title=r"Gradient $(0.1,0.9)$",
        show_ylabel=True,
    )

    # Dashed connectors from FIM zoom boxes to corresponding local panels.
    (x_lo_best, x_hi_best, y_lo_best, y_hi_best) = box_best
    (x_lo_probe, x_hi_probe, y_lo_probe, y_hi_probe) = box_probe
    conn_best_top = ConnectionPatch(
        xyA=(x_hi_best, y_hi_best),
        coordsA=ax_b.transData,
        xyB=(0.0, 1.0),
        coordsB=ax_c_best.transAxes,
        linestyle="--",
        linewidth=0.8,
        color=_GEOM_COLOR,
        alpha=0.9,
        zorder=8,
    )
    conn_best_bot = ConnectionPatch(
        xyA=(x_hi_best, y_lo_best),
        coordsA=ax_b.transData,
        xyB=(0.0, 0.0),
        coordsB=ax_c_best.transAxes,
        linestyle="--",
        linewidth=0.8,
        color=_GEOM_COLOR,
        alpha=0.9,
        zorder=8,
    )
    conn_probe_top = ConnectionPatch(
        xyA=(x_lo_probe, y_lo_probe),
        coordsA=ax_b.transData,
        xyB=(0.0, 1.0),
        coordsB=ax_c_probe.transAxes,
        linestyle="--",
        linewidth=0.8,
        color="magenta",
        alpha=0.9,
        zorder=8,
    )
    conn_probe_bot = ConnectionPatch(
        xyA=(x_hi_probe, y_lo_probe),
        coordsA=ax_b.transData,
        xyB=(1.0, 1.0),
        coordsB=ax_c_probe.transAxes,
        linestyle="--",
        linewidth=0.8,
        color="magenta",
        alpha=0.9,
        zorder=8,
    )
    fig.add_artist(conn_best_top)
    fig.add_artist(conn_best_bot)
    fig.add_artist(conn_probe_top)
    fig.add_artist(conn_probe_bot)

    # Panel (d): full-domain gradient and natural-gradient vector fields.
    if grad_acq_x1_grid is None or grad_acq_x2_grid is None:
        dx = float(X1[1, 0] - X1[0, 0]) if X1.shape[0] > 1 else 1e-3
        dy = float(X2[0, 1] - X2[0, 0]) if X2.shape[1] > 1 else 1e-3
        d_acq_dx, d_acq_dy = np.gradient(acq_grid, dx, dy)
    else:
        d_acq_dx = np.asarray(grad_acq_x1_grid, dtype=np.float64)
        d_acq_dy = np.asarray(grad_acq_x2_grid, dtype=np.float64)
    grad_norm = np.sqrt(d_acq_dx**2 + d_acq_dy**2)

    # Uncomment to plot the natural gradient norm.
    # # eta = G^{-1} grad(alpha), then plot ||eta||_2.
    # det_g = g11 * g22 - g12 * g12
    # det_g = np.clip(det_g, 1e-12, None)
    # inv11 = g22 / det_g
    # inv12 = -g12 / det_g
    # inv22 = g11 / det_g
    # nat_dx = inv11 * d_acq_dx + inv12 * d_acq_dy
    # nat_dy = inv12 * d_acq_dx + inv22 * d_acq_dy
    # grad_norm = np.sqrt(nat_dx**2 + nat_dy**2)

    # Use robust clipping to avoid saturation by a few extreme spikes.
    grad_clip_lo = 2.5
    grad_clip_hi = 80.0
    grad_vmin = np.percentile(grad_norm, grad_clip_lo)
    grad_vmax = np.percentile(grad_norm, grad_clip_hi)
    grad_norm_plot = np.clip(grad_norm, grad_vmin, grad_vmax)
    grad_cf = ax_d.contourf(
        X1,
        X2,
        grad_norm_plot,
        levels=np.linspace(grad_vmin, grad_vmax, 60),
        cmap="magma",
        extend="both",
    )
    cbar_grad = fig.colorbar(grad_cf, ax=ax_d)
    _cbar_ticks(cbar_grad, grad_vmin, grad_vmax)
    _panel_common(ax_d, X_train=X_train, x_best=x_best, x_next=x_next)
    ax_d.set_xlim(float(X1.min()), float(X1.max()))
    ax_d.set_ylim(float(X2.min()), float(X2.max()))
    ax_d.set_aspect("auto")
    ax_d.set_xlabel(r"$x_1$", labelpad=0.3)
    ax_d.set_ylabel(r"$x_2$", labelpad=0.2)
    ax_d.set_yticks([0.0, 0.5, 1.0])
    ax_d.set_yticklabels(["0.0", "0.5", "1.0"])
    ax_d.set_xticks([0, 0.5, 1])
    ax_d.set_xticklabels(["0", "0.5", "1"])
    ax_d.set_title(r"$\|\nabla_x \alpha(\mathbf{x})\|_2$", pad=0.5)
    # ax_d.set_title(r"$\|G_x^{-1}\nabla_x \alpha(\mathbf{x})\|_2$", pad=0.5)
    for (x_lo, x_hi, y_lo, y_hi), color in [
        (box_best, _GEOM_COLOR),
        (box_probe, "magenta"),
    ]:
        ax_d.add_patch(
            mpatches.Rectangle(
                (x_lo, y_lo),
                x_hi - x_lo,
                y_hi - y_lo,
                linewidth=0.6,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                alpha=0.9,
                zorder=10,
                clip_on=False,
            )
        )

    # Dashed connectors from grad-norm boxes to corresponding local panels.
    conn_best_top_d = ConnectionPatch(
        xyA=(x_lo_best, y_hi_best),
        coordsA=ax_d.transData,
        xyB=(0.0, 0.0),
        coordsB=ax_c_best.transAxes,
        linestyle="--",
        linewidth=0.8,
        color=_GEOM_COLOR,
        alpha=0.9,
        zorder=8,
    )
    conn_best_bot_d = ConnectionPatch(
        xyA=(x_hi_best, y_hi_best),
        coordsA=ax_d.transData,
        xyB=(1.0, 0.0),
        coordsB=ax_c_best.transAxes,
        linestyle="--",
        linewidth=0.8,
        color=_GEOM_COLOR,
        alpha=0.9,
        zorder=8,
    )
    conn_probe_top_d = ConnectionPatch(
        xyA=(x_lo_probe, y_hi_probe),
        coordsA=ax_d.transData,
        xyB=(1.0, 1.0),
        coordsB=ax_c_probe.transAxes,
        linestyle="--",
        linewidth=0.8,
        color="magenta",
        alpha=0.9,
        zorder=8,
    )
    conn_probe_bot_d = ConnectionPatch(
        xyA=(x_lo_probe, y_lo_probe),
        coordsA=ax_d.transData,
        xyB=(1.0, 0.0),
        coordsB=ax_c_probe.transAxes,
        linestyle="--",
        linewidth=0.8,
        color="magenta",
        alpha=0.9,
        zorder=8,
    )
    fig.add_artist(conn_best_top_d)
    fig.add_artist(conn_best_bot_d)
    fig.add_artist(conn_probe_top_d)
    fig.add_artist(conn_probe_bot_d)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=2.5,
            label="$\mathcal{D}_t$",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="*",
            markerfacecolor="gold",
            linestyle="None",
            markersize=5,
            label=r"$\mathbf{x}_{\mathrm{best}}$",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="D",
            markerfacecolor="crimson",
            linestyle="None",
            markersize=3,
            label=r"$\mathbf{x}_{t+1}$",
        ),
        mpatches.Ellipse(
            (0, 0),
            width=1.0,
            height=0.55,
            facecolor="none",
            edgecolor="red",
            linewidth=1.2,
            label="FIM ellipse",
        ),
        # mlines.Line2D([], [], color=_GEOM_COLOR, linestyle="--", linewidth=1.1, label="Zoom box"),
        mpatches.FancyArrowPatch(
            (0.0, 0.5),
            (1.0, 0.5),
            mutation_scale=6.0,
            arrowstyle="->",
            lw=0.8,
            color="darkorange",
            label=r"$\nabla_{\mathbf{x}} \alpha(\mathbf{x})$",
        ),
        mpatches.FancyArrowPatch(
            (0.0, 0.5),
            (1.0, 0.5),
            mutation_scale=6.0,
            arrowstyle="->",
            lw=0.8,
            color="purple",
            label=r"$g_t(\mathbf{x})^{-1}\nabla_{\mathbf{x}} \alpha(\mathbf{x})$",
        ),
    ]
    legend_x = (
        0.5 * (x_cols[0] + (x_cols[-1] + panel_w)) - 0.025
    )  # center of panel block
    legend_y = -0.05  # tune only vertical legend position
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        framealpha=0.92,
        bbox_to_anchor=(legend_x, legend_y),
        columnspacing=0.2,
        handletextpad=0.25,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.5,
        handler_map={
            mpatches.Ellipse: HandlerPatch(patch_func=_legend_ellipse),
            mpatches.FancyArrowPatch: HandlerPatch(patch_func=_legend_arrow),
        },
    )
    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01
        )
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        fig.savefig(out.with_suffix(".pgf"), bbox_inches="tight", pad_inches=0.01)
    return fig
