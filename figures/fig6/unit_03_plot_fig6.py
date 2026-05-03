"""Figure 6: LogEI + TuRBO (GP-SE) / FIM ellipse / FITR (diag of same G as ellipse)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.ticker import FormatStrFormatter

from neurips_viz.fig3.unit_03_plot_fig3 import compute_metric_ellipse

_TURBO_COLOR = "cornflowerblue"
_FIM_COLOR = "crimson"
_FITR_COLOR = "darkorange"


def _apply_style() -> None:
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


def _cbar_ticks(cbar, vmin: float, vmax: float, n: int = 4) -> None:
    ticks = np.linspace(vmin, vmax, max(2, n))
    span = vmax - vmin
    decimals = max(1, int(np.ceil(-np.log10(span / n)))) if span > 0 else 1
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def _draw_anchor_overlays(
    ax: plt.Axes,
    center: np.ndarray,
    g_matrix: np.ndarray,
    rho: float,
    turbo_box: dict | None,
    fitr_box: dict,
    linestyle: str,
    linewidth: float = 1.3,
) -> None:
    if turbo_box is not None:
        lo, hi = turbo_box["lo"], turbo_box["hi"]
        ax.add_patch(
            mpatches.Rectangle(
                (lo[0], lo[1]),
                hi[0] - lo[0],
                hi[1] - lo[1],
                linewidth=linewidth,
                edgecolor=_TURBO_COLOR,
                facecolor="none",
                linestyle=linestyle,
                zorder=10,
            )
        )

    ell_pts = compute_metric_ellipse(g_matrix, center=center, rho=rho)
    ax.plot(
        ell_pts[0],
        ell_pts[1],
        color=_FIM_COLOR,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=10,
    )

    lo, hi = fitr_box["lo"], fitr_box["hi"]
    ax.add_patch(
        mpatches.Rectangle(
            (lo[0], lo[1]),
            hi[0] - lo[0],
            hi[1] - lo[1],
            linewidth=linewidth,
            edgecolor=_FITR_COLOR,
            facecolor="none",
            linestyle=linestyle,
            zorder=10,
        )
    )


def _legend_ellipse(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return mpatches.Ellipse(
        (xdescent + 0.5 * width, ydescent + 0.5 * height),
        width=0.9 * width,
        height=0.55 * height,
        fill=False,
        edgecolor=orig_handle.get_edgecolor(),
        linewidth=orig_handle.get_linewidth(),
    )


def plot_fig6(
    X1: np.ndarray,
    X2: np.ndarray,
    logei_gp: np.ndarray,
    logei_ibnn: np.ndarray,
    logei_bnn: np.ndarray,
    # Per-panel overlay specs: two anchors (best, random) × dicts
    turbo_best: dict | None,
    turbo_random: dict | None,
    fitr_gp_best: dict,
    fitr_gp_random: dict,
    fitr_ibnn_best: dict,
    fitr_ibnn_random: dict,
    fitr_bnn_best: dict,
    fitr_bnn_random: dict,
    g_gp_best: dict,
    g_gp_random: dict,
    g_ibnn_best: dict,
    g_ibnn_random: dict,
    g_bnn_best: dict,
    g_bnn_random: dict,
    x_best: np.ndarray,
    x_random: np.ndarray,
    x_train: np.ndarray,
    rho: float = 0.14,
    fim_ellipse_scale: float = 1.0,
    figsize: tuple[float, float] = (4.5, 1.25),
    savepath: str | None = None,
) -> plt.Figure:
    """Three-panel figure: GP-SE (+TuRBO), GP-IBNN, deep ensemble BNN.

    Ellipses trace δᵀ G δ = ρ² with **effective** ρ_plot = rho * fim_ellipse_scale
    (geometry unchanged; scale is display-only for visibility).
    """
    _apply_style()
    rho_draw = float(rho) * float(fim_ellipse_scale)
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    titles = [
        r"(a) GP-SE",
        r"(b) GP-IBNN",
        r"(c) Deep BNN",
    ]
    grids = [logei_gp, logei_ibnn, logei_bnn]
    cmaps = ["magma", "magma", "magma"]

    def gmat(d: dict) -> np.ndarray:
        return np.array(
            [[d["g11"], d["g12"]], [d["g12"], d["g22"]]],
            dtype=np.float64,
        )

    panels = [
        (
            turbo_best,
            turbo_random,
            fitr_gp_best,
            fitr_gp_random,
            g_gp_best,
            g_gp_random,
        ),
        (None, None, fitr_ibnn_best, fitr_ibnn_random, g_ibnn_best, g_ibnn_random),
        (None, None, fitr_bnn_best, fitr_bnn_random, g_bnn_best, g_bnn_random),
    ]

    for col, ax in enumerate(axes):
        grid = grids[col]
        tb, tr, fb, fr, gb, gr = panels[col]

        vmin = np.percentile(grid, 2)
        vmax = np.percentile(grid, 98)
        mesh = ax.pcolormesh(
            X1, X2, grid, cmap=cmaps[col], vmin=vmin, vmax=vmax, shading="auto"
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        _cbar_ticks(cbar, vmin, vmax)

        ax.set_title(titles[col], pad=2)
        ax.set_xlim(float(X1.min()), float(X1.max()))
        ax.set_ylim(float(X2.min()), float(X2.max()))
        ax.set_xlabel(r"$x_1$", labelpad=1)
        if col == 0:
            ax.set_ylabel(r"$x_2$", labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_aspect("auto")

        _draw_anchor_overlays(
            ax,
            x_best,
            gmat(gb),
            rho_draw,
            tb,
            fb,
            linestyle="-",
        )
        _draw_anchor_overlays(
            ax,
            x_random,
            gmat(gr),
            rho_draw,
            tr,
            fr,
            linestyle="--",
        )

        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c="black",
            s=12,
            zorder=8,
            clip_on=True,
        )
        ax.scatter(
            [x_best[0]],
            [x_best[1]],
            marker="*",
            s=40,
            c="gold",
            edgecolors="black",
            linewidths=0.7,
            zorder=11,
        )
        ax.scatter(
            [x_random[0]],
            [x_random[1]],
            marker="D",
            s=10,
            c="white",
            edgecolors="black",
            linewidths=0.7,
            zorder=11,
        )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=3,
            label=r"$\mathcal{D}_t$",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="*",
            markerfacecolor="gold",
            linestyle="None",
            markersize=6,
            label=r"$\mathbf{x}_\mathrm{best}$",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="D",
            markerfacecolor="white",
            linestyle="None",
            markersize=2,
            label=r"$\mathbf{x}_\mathrm{random}$",
        ),
        mpatches.Patch(
            facecolor="none",
            edgecolor=_TURBO_COLOR,
            linewidth=1.3,
            label="TuRBO",
        ),
        mpatches.Ellipse(
            (0, 0),
            1.0,
            0.55,
            fill=False,
            edgecolor=_FIM_COLOR,
            linewidth=1.3,
            label=r"FIM ellipse",
        ),
        mpatches.Patch(
            facecolor="none",
            edgecolor=_FITR_COLOR,
            linewidth=1.3,
            label=r"FITR",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=8,
        framealpha=0.92,
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=0.45,
        handletextpad=0.25,
        borderpad=0.25,
        labelspacing=0.15,
        handlelength=1.5,
        handler_map={
            mpatches.Ellipse: HandlerPatch(patch_func=_legend_ellipse),
        },
    )

    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03
        )
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)

    return fig
