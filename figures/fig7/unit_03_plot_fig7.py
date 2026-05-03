"""Figure 7 plotting: mean / variance / LogEI with FIM ellipses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

_ELLIPSE_COLOR = "crimson"

# Colorbar controls (edit these to tune visuals; no CLI plumbing).
COLORBAR_TICKS_MAX = 3  # lower, middle, upper
# Colorbar thickness (tune these).
# - `fraction`: relative width of the colorbar w.r.t. the axes.
# - `aspect`: length/width ratio; smaller => thicker.
COLORBAR_FRACTION = 0.16
COLORBAR_ASPECT = 10
COLORBAR_SHRINK = 0.75
COLORBAR_PAD = 0.05
COLORBAR_ANCHOR = (0.0, 0.0)  # keep colorbar bottom aligned with subplot bottom
COLORBAR_PANCHOR = (1.0, 0.0)
COLORBAR_EXP_X = -0.05
COLORBAR_EXP_Y = 1.2
COLORBAR_EXP_HA = "left"
COLORBAR_EXP_VA = "bottom"
COLORBAR_EXP_FONTSIZE = 5

# Colorbar tick formatting: use scientific notation with integer mantissa.
# Tick labels become e.g. "-4" and the shared exponent becomes the colorbar title.
COLORBAR_SCI_TICKS = True

# Axis-label placement controls.
X1_LABELPAD = -1.2
X2_LABELPAD = -0.25
X1_LABEL_Y = -0.16
X2_LABEL_X = -0.16

# Heatmap rendering controls.
HEATMAP_LEVELS = 25

# Observation marker + legend controls.
OBS_DOT_SIZE = 5
OBS_DOT_EDGEWIDTH = 0.35
LEGEND_Y = -0.15
LEGEND_FRAMEON = True
LEGEND_FACE_COLOR = "white"
LEGEND_EDGE_COLOR = "black"
LEGEND_FRAME_ALPHA = 0.8
LEGEND_FRAME_LINEWIDTH = 0.2


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
            "text.usetex": True,
            "mathtext.fontset": "cm",
            "pgf.rcfonts": False,
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )


def _cbar_ticks(
    cbar, vmin: float, vmax: float, max_ticks: int = COLORBAR_TICKS_MAX
) -> None:
    # Exactly three ticks: lower / middle / upper.
    if max_ticks <= 1 or vmax <= vmin:
        ticks = np.array([vmin], dtype=float)
    elif max_ticks == 2:
        ticks = np.array([vmin, vmax], dtype=float)
    else:
        mid = 0.5 * (vmin + vmax)
        ticks = np.array([vmin, mid, vmax], dtype=float)

    cbar.set_ticks(ticks.tolist())

    if not COLORBAR_SCI_TICKS:
        # Default decimals formatter (kept for optional fallback).
        span = float(vmax - vmin)
        step = span / max(1, (len(ticks) - 1))
        decimals = (
            max(1, int(np.ceil(-np.log10(max(1e-30, abs(step)))))) if span > 0 else 1
        )
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
        return

    # Scientific ticks: integer mantissa; shared exponent is the title.
    maxabs = float(np.max(np.abs(ticks)))
    if maxabs <= 0.0 or not np.isfinite(maxabs):
        exp = 0
        scale = 1.0
    else:
        exp = int(np.floor(np.log10(maxabs)))
        scale = float(10.0**exp) if exp != 0 else 1.0

    mant_int = [int(np.round(float(t) / scale)) for t in ticks]
    cbar.ax.set_yticklabels([str(m) for m in mant_int])

    # Always show exponent for uniformity (including x10^0).
    cbar.ax.set_title("")
    cbar.ax.text(
        COLORBAR_EXP_X,
        COLORBAR_EXP_Y,
        rf"$\times 10^{{{exp}}}$",
        transform=cbar.ax.transAxes,
        ha=COLORBAR_EXP_HA,
        va=COLORBAR_EXP_VA,
        fontsize=COLORBAR_EXP_FONTSIZE,
        clip_on=False,
    )


def _gmat(g: dict) -> np.ndarray:
    return np.array([[g["g11"], g["g12"]], [g["g12"], g["g22"]]], dtype=np.float64)


def _ellipse_with_eig_floor(
    g_matrix: np.ndarray,
    center: np.ndarray,
    rho: float,
    eig_floor_scale: float,
    n_points: int = 100,
) -> np.ndarray:
    """Ellipse points for delta^T G delta = rho^2 with dynamic eig floor.

    Floor is computed as eps = eig_floor_scale * mean(eigenvalues).
    """
    evals_raw, evecs = np.linalg.eigh(g_matrix)
    eps = float(max(1e-14, eig_floor_scale * float(np.mean(evals_raw))))
    evals = np.clip(evals_raw, eps, None)
    axes = rho / np.sqrt(evals)
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    ellipse = evecs @ np.diag(axes) @ circle
    return center[:, None] + ellipse


def _contourf_clean(
    ax: plt.Axes,
    X1: np.ndarray,
    X2: np.ndarray,
    Z: np.ndarray,
    levels: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """Contourf without vector seam artifacts in PDF outputs."""
    z_clean = np.asarray(Z, dtype=np.float64)
    z_clean = np.where(np.isfinite(z_clean), z_clean, vmin)
    z_clean = np.clip(z_clean, vmin, vmax)
    level_min = float(levels[0])
    level_max = float(levels[-1])
    eps = max(1e-12, 1e-9 * max(1.0, abs(level_max - level_min)))
    levels_safe = np.linspace(level_min - eps, level_max + eps, len(levels))
    cs = ax.contourf(
        X1,
        X2,
        z_clean,
        levels=levels_safe,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        antialiased=False,
        corner_mask=False,
    )
    return cs


def plot_fig7(
    X1: np.ndarray,
    X2: np.ndarray,
    mean_gp: np.ndarray,
    mean_ibnn: np.ndarray,
    mean_bnn: np.ndarray,
    var_gp: np.ndarray,
    var_ibnn: np.ndarray,
    var_bnn: np.ndarray,
    logei_gp: np.ndarray,
    logei_ibnn: np.ndarray,
    logei_bnn: np.ndarray,
    x_train: np.ndarray,
    anchors: list[np.ndarray],
    gx_gp: list[dict],
    gx_ibnn: list[dict],
    gx_bnn: list[dict],
    rho: float = 0.08,
    ellipse_scale_gp: float = 1.0,
    ellipse_scale_ibnn: float = 1.0,
    ellipse_scale_bnn: float = 1.0,
    heatmap_floor_pct_gp: float = 80.0,
    heatmap_floor_pct_ibnn: float = 80.0,
    heatmap_floor_pct_bnn: float = 80.0,
    eig_floor_scale: float = 1.0,
    figsize: tuple[float, float] = (2.0, 1.5),
    savepath: str | None = None,
) -> plt.Figure:
    """3x3 panels: predicted mean, predictive variance, and LogEI."""
    _apply_style()
    fig, axes = plt.subplots(
        3,
        3,
        figsize=figsize,
        constrained_layout=False,
        sharex="col",
        sharey="row",
        gridspec_kw={"wspace": 0.4, "hspace": 0.4},
    )
    # Tighten outer margins so labels/tick text don't waste figure area.
    fig.subplots_adjust(left=0.04, right=0.995, top=0.91, bottom=0.075)

    titles = [r"GP-SE", r"GP-IBNN", r"BNN"]
    mean_grids = [mean_gp, mean_ibnn, mean_bnn]
    var_grids = [var_gp, var_ibnn, var_bnn]
    logei_grids = [logei_gp, logei_ibnn, logei_bnn]
    gx_panels = [gx_gp, gx_ibnn, gx_bnn]
    scales = [ellipse_scale_gp, ellipse_scale_ibnn, ellipse_scale_bnn]
    floor_pcts = [heatmap_floor_pct_gp, heatmap_floor_pct_ibnn, heatmap_floor_pct_bnn]

    top_axes = axes[0, :]
    middle_axes = axes[1, :]
    bottom_axes = axes[2, :]

    x_lo = float(X1.min())
    x_hi = float(X1.max())
    y_lo = float(X2.min())
    y_hi = float(X2.max())

    for col in range(3):
        ax_top = top_axes[col]
        ax_mid = middle_axes[col]
        ax_bot = bottom_axes[col]

        mean_grid = mean_grids[col]
        var_grid = var_grids[col]
        logei_grid = logei_grids[col]

        # Top row: predicted mean (no percentile clipping)
        mvmin = float(np.min(mean_grid))
        mvmax = float(np.max(mean_grid))
        mean_levels = np.linspace(mvmin, mvmax, HEATMAP_LEVELS)
        mean_plot = np.nan_to_num(mean_grid, nan=mvmin, posinf=mvmax, neginf=mvmin)
        mmesh = _contourf_clean(
            ax_top, X1, X2, mean_plot, mean_levels, "viridis", mvmin, mvmax
        )
        mcbar = fig.colorbar(
            mmesh,
            ax=ax_top,
            pad=COLORBAR_PAD,
            fraction=COLORBAR_FRACTION,
            aspect=COLORBAR_ASPECT,
            shrink=COLORBAR_SHRINK,
            anchor=COLORBAR_ANCHOR,
            panchor=COLORBAR_PANCHOR,
        )
        _cbar_ticks(mcbar, mvmin, mvmax, max_ticks=COLORBAR_TICKS_MAX)
        ax_top.set_title(titles[col], pad=2)
        ax_top.set_xlim(x_lo, x_hi)
        ax_top.set_ylim(y_lo, y_hi)
        if col == 0:
            ax_top.set_ylabel(r"$x_2$", labelpad=X2_LABELPAD)
            ax_top.yaxis.set_label_coords(X2_LABEL_X, 0.5)
            ax_top.set_yticks([0.0, 1.0])
            ax_top.set_yticklabels(["0", "1"])
            ax_top.tick_params(labelleft=True)
        else:
            ax_top.tick_params(labelleft=False)
        ax_top.tick_params(labelbottom=False)
        ax_top.set_aspect("auto")
        ax_top.scatter(
            x_train[:, 0],
            x_train[:, 1],
            facecolors="white",
            edgecolors="black",
            linewidths=OBS_DOT_EDGEWIDTH,
            s=OBS_DOT_SIZE,
            zorder=8,
            clip_on=True,
        )

        # Middle row: predictive variance (no percentile clipping)
        vvmin = float(np.min(var_grid))
        vvmax = float(np.max(var_grid))
        var_levels = np.linspace(vvmin, vvmax, HEATMAP_LEVELS)
        var_plot = np.nan_to_num(var_grid, nan=vvmin, posinf=vvmax, neginf=vvmin)
        vmesh = _contourf_clean(
            ax_mid, X1, X2, var_plot, var_levels, "plasma", vvmin, vvmax
        )
        vcbar = fig.colorbar(
            vmesh,
            ax=ax_mid,
            pad=COLORBAR_PAD,
            fraction=COLORBAR_FRACTION,
            aspect=COLORBAR_ASPECT,
            shrink=COLORBAR_SHRINK,
            anchor=COLORBAR_ANCHOR,
            panchor=COLORBAR_PANCHOR,
        )
        _cbar_ticks(vcbar, vvmin, vvmax, max_ticks=COLORBAR_TICKS_MAX)
        if col == 0:
            ax_mid.set_ylabel(r"$x_2$", labelpad=X2_LABELPAD)
            ax_mid.yaxis.set_label_coords(X2_LABEL_X, 0.5)
            ax_mid.set_yticks([0.0, 1.0])
            ax_mid.set_yticklabels(["0", "1"])
            ax_mid.tick_params(labelleft=True)
        else:
            ax_mid.tick_params(labelleft=False)
        ax_mid.tick_params(labelbottom=False)
        ax_mid.set_xlim(x_lo, x_hi)
        ax_mid.set_ylim(y_lo, y_hi)
        ax_mid.set_aspect("auto")
        ax_mid.scatter(
            x_train[:, 0],
            x_train[:, 1],
            facecolors="white",
            edgecolors="black",
            linewidths=OBS_DOT_EDGEWIDTH,
            s=OBS_DOT_SIZE,
            zorder=8,
            clip_on=True,
        )

        # Bottom row: LogEI + overlays
        lvmin = np.percentile(logei_grid, float(floor_pcts[col]))
        lvmax = np.percentile(logei_grid, 98)
        # For AF, apply the lower-floor explicitly before contourf.
        logei_plot = np.clip(logei_grid, lvmin, lvmax)
        logei_plot = np.nan_to_num(logei_plot, nan=lvmin, posinf=lvmax, neginf=lvmin)
        af_levels = np.linspace(lvmin, lvmax, HEATMAP_LEVELS)
        lmesh = _contourf_clean(
            ax_bot, X1, X2, logei_plot, af_levels, "magma", lvmin, lvmax
        )
        lcbar = fig.colorbar(
            lmesh,
            ax=ax_bot,
            pad=COLORBAR_PAD,
            fraction=COLORBAR_FRACTION,
            aspect=COLORBAR_ASPECT,
            shrink=COLORBAR_SHRINK,
            anchor=COLORBAR_ANCHOR,
            panchor=COLORBAR_PANCHOR,
        )
        _cbar_ticks(lcbar, lvmin, lvmax, max_ticks=COLORBAR_TICKS_MAX)
        ax_bot.set_xlim(x_lo, x_hi)
        ax_bot.set_ylim(y_lo, y_hi)
        ax_bot.set_xlabel(r"$x_1$", labelpad=X1_LABELPAD)
        ax_bot.xaxis.set_label_coords(0.5, X1_LABEL_Y)
        if col == 0:
            ax_bot.set_ylabel(r"$x_2$", labelpad=X2_LABELPAD)
            ax_bot.yaxis.set_label_coords(X2_LABEL_X, 0.5)
            ax_bot.set_yticks([0.0, 1.0])
            ax_bot.set_yticklabels(["0", "1"])
            ax_bot.tick_params(labelleft=True)
        else:
            ax_bot.tick_params(labelleft=False)
        ax_bot.set_aspect("auto")
        ax_bot.scatter(
            x_train[:, 0],
            x_train[:, 1],
            facecolors="white",
            edgecolors="black",
            linewidths=OBS_DOT_EDGEWIDTH,
            s=OBS_DOT_SIZE,
            zorder=8,
            clip_on=True,
        )

        rho_panel = float(rho) * float(scales[col])
        for anchor, gdict in zip(anchors, gx_panels[col]):
            ell = _ellipse_with_eig_floor(
                _gmat(gdict),
                center=anchor,
                rho=rho_panel,
                eig_floor_scale=float(eig_floor_scale),
            )
            for ax_draw in (ax_top, ax_mid, ax_bot):
                ax_draw.plot(
                    ell[0],
                    ell[1],
                    color=_ELLIPSE_COLOR,
                    linewidth=1.2,
                    linestyle="-",
                    zorder=10,
                )

    # Row labels on the far left (rotated), no gray group boxes.
    left_x = min(
        a.get_position().x0 for a in [top_axes[0], middle_axes[0], bottom_axes[0]]
    )
    label_x = left_x - 0.11
    row_centers = [
        0.5 * (top_axes[0].get_position().y0 + top_axes[0].get_position().y1),
        0.5 * (middle_axes[0].get_position().y0 + middle_axes[0].get_position().y1),
        0.5 * (bottom_axes[0].get_position().y0 + bottom_axes[0].get_position().y1),
    ]
    for y, text in zip(row_centers, ["Mean", "Variance", "LogEI"]):
        fig.text(
            label_x,
            y,
            text,
            rotation=90,
            ha="center",
            va="center",
            color="black",
        )

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=OBS_DOT_EDGEWIDTH,
            markersize=4.2,
            label=r"$\mathcal{D}_t$",
        ),
        Line2D(
            [],
            [],
            color=_ELLIPSE_COLOR,
            linewidth=1.2,
            linestyle="-",
            label="FIM ellipse",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=2,
        frameon=LEGEND_FRAMEON,
        facecolor=LEGEND_FACE_COLOR,
        edgecolor=LEGEND_EDGE_COLOR,
        framealpha=LEGEND_FRAME_ALPHA,
        handlelength=1.2,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    if fig.legends:
        fig.legends[-1].get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)

    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01
        )
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        fig.savefig(out.with_suffix(".pgf"), bbox_inches="tight", pad_inches=0.01)

    return fig
