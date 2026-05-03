"""Figure 1: conventional BO, manifold trace, C_alpha heatmap, and bound panel."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import ConnectionPatch, Rectangle

from .unit_01_gp_model import eval_objective
from .unit_02_metric_field import compute_fisher_rao_distance_grid

_CMAP = "plasma"
# Green for endpoint markers (same in both panes so reader links them).
_X_START_COLOR = "pink"
_X_END_COLOR = "darkgreen"

# Orange for the highlighted geodesic θ_best → θ_next.
_GEO_COLOR = "darkorange"

font_size = 7.5
legend_font_size = 7

diamond_size = 35
star_size = 100
x_marker_size = 80

circle_size = 12

# Centralized annotation placement controls (all in data coordinates).
# Edit these values to move labels without touching plotting logic.
ANNOTATION_POS = {
    "panel_a": {
        "x0": {"dx": 0.01, "dy": 0.18},
        "x1": {"dx": -0.2, "dy": -0.35},
    },
    "panel_b": {
        "x0": {"dx": -0.8, "dy": -0.05},
        "x1": {"dx": 0.1, "dy": 0.1},
        "d_fr": {"dx": -4, "dy": 0.415},
        # Keep labels inside axes while still allowing explicit control.
        "clip_margin": {"x": 0.05, "y": 0.05},
        "d_fr_clip": {"x_left": 0.08, "x_right": 0.15, "y_low": -1, "y_high": 0.05},
    },
}

# Slightly tall for bottom fig.legend (avoids clipping long math entries, e.g. $f_{\mathrm{true}}$).
FIG1_FIGSIZE_INCHES = (6.0, 2.0)

LEGEND_BBOX_TO_ANCHOR = (0.5, -0.01)

# Title–axes gap in **points** for panels A and D (``set_title(..., pad=...)`` / ``axes.titlepad``).
FIG1_TITLE_PAD = 4.5

# Tighter horizontal gap between subplots (esp. panel C vs bound panel D).
FIG1_CONSTRAINED_LAYOUT_W_PAD = 0.006

# Panel B LogEI colorbar: shorter bar; label above bar, left-aligned with bar edge.
FIG1_LOGEI_CBAR_FRACTION = 0.038
FIG1_LOGEI_CBAR_PAD = 0.02
FIG1_LOGEI_CBAR_SHRINK = 0.68

# Optional manual box-corner overrides for zero-trace edge regions.
# Each box is specified by two opposite corners in axis data coordinates:
# ((x0, y0), (x1, y1)).
# Set any entry to None to use automatic placement from trace==0 regions.
ZERO_TRACE_BOX_CORNERS = {
    "panel_a": {
        "left": ((0.00, -2.25), (0.25, 2.0)),
        "right": ((0.75, -2.25), (1.00, 2.0)),
    },
    "panel_d": {
        "left": ((0.00, 0.1e-9), (0.25, 1e-5)),
        "right": ((0.75, 0.1e-9), (1.00, 1e-5)),
    },
}

# Optional manual corners for the Figure-level gray box around manifold-view panels
# (in figure-fraction coordinates): (x0, y0, x1, y1). Set to None for auto placement.
MANIFOLD_VIEW_GROUP_BOX_CORNERS: tuple[float, float, float, float] | None = (
    0.43,
    0.1,
    1.0,
    0.83,
)
MANIFOLD_VIEW_GROUP_BOX_PAD = 0.012

# ---------------------------------------------------------------------------
# Colormap legend proxy
# ---------------------------------------------------------------------------


class ColormapLine:
    """Legend proxy drawn as a horizontal plasma gradient."""

    def __init__(self, cmap_name: str = _CMAP, lw: float = 2.5):
        self.cmap_name = cmap_name
        self.lw = lw


class HandlerColormapLine(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        n = 18
        xs = np.linspace(xdescent, xdescent + width, n + 1)
        y = ydescent + 0.5 * height
        segs = np.stack(
            [
                np.column_stack([xs[:-1], np.full(n, y)]),
                np.column_stack([xs[1:], np.full(n, y)]),
            ],
            axis=1,
        )
        lc = LineCollection(
            segs,
            cmap=plt.get_cmap(orig_handle.cmap_name),
            norm=Normalize(0.0, 1.0),
            linewidths=orig_handle.lw,
            transform=trans,
        )
        lc.set_array(np.linspace(0.0, 1.0, n))
        return [lc]


# ---------------------------------------------------------------------------
# Manifold geometry helpers
# ---------------------------------------------------------------------------


def _integer_exponent_yticks(
    ymin: float, ymax: float, max_ticks: int = 4
) -> list[float]:
    """Up to ``max_ticks`` powers of ten with integer exponents spanning [ymin, ymax]."""
    if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymin <= 0 or ymax <= 0:
        return [1e-8, 1e-4, 1.0, 1e4]
    lo = float(np.log10(ymin))
    hi = float(np.log10(ymax))
    if hi < lo:
        lo, hi = hi, lo
    L = int(np.floor(lo))
    U = int(np.ceil(hi))
    if U < L:
        L, U = U, L
    ks = list(range(L, U + 1))
    if not ks:
        k = int(round(0.5 * (lo + hi)))
        return [10.0**k]
    if len(ks) <= max_ticks:
        return [10.0**k for k in ks]
    idx = np.linspace(0, len(ks) - 1, max_ticks)
    picked: list[int] = []
    for i in idx:
        k = ks[int(round(float(i)))]
        if k not in picked:
            picked.append(k)
    if len(picked) < 2:
        picked = [ks[0], ks[-1]]
    return [10.0**k for k in picked]


def _fr_distance(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """Fisher-Rao distance between two univariate Gaussians."""
    s1, s2 = max(float(sigma1), 1e-12), max(float(sigma2), 1e-12)
    inner = 1.0 + ((mu1 - mu2) ** 2 + 2.0 * (s1 - s2) ** 2) / (4.0 * s1 * s2)
    return float(np.sqrt(2.0) * np.arccosh(max(inner, 1.0 + 1e-15)))


def _zero_trace_edge_regions(
    x: np.ndarray, tr_g: np.ndarray, tol: float = 1e-14
) -> list[tuple[float, float]]:
    """Return left/right edge x-intervals where tr(G) is numerically zero."""
    x = np.asarray(x, dtype=np.float64)
    tr_g = np.asarray(tr_g, dtype=np.float64)
    if x.size == 0 or tr_g.size != x.size:
        return []
    mask = np.isfinite(tr_g) & (tr_g <= tol)
    if not np.any(mask):
        return []

    regions: list[tuple[float, float]] = []

    left_end = 0
    while left_end < mask.size and mask[left_end]:
        left_end += 1
    if left_end > 0:
        regions.append((float(x[0]), float(x[left_end - 1])))

    right_start = mask.size - 1
    while right_start >= 0 and mask[right_start]:
        right_start -= 1
    if right_start < mask.size - 1:
        regions.append((float(x[right_start + 1]), float(x[-1])))
    return regions


def _add_input_space_boxes(ax: plt.Axes, x_regions: list[tuple[float, float]]) -> None:
    """Draw subtle dashed boxes near axis bottom for marked x-regions."""
    if not x_regions:
        return
    y_lo, y_hi = ax.get_ylim()
    y_span = y_hi - y_lo
    box_y = y_lo + 0.03 * y_span
    box_h = 0.18 * y_span
    for x0, x1 in x_regions:
        if x1 <= x0:
            continue
        ax.add_patch(
            Rectangle(
                (x0, box_y),
                x1 - x0,
                box_h,
                facecolor="none",
                edgecolor="#8a8a8a",
                linewidth=0.9,
                linestyle="--",
                alpha=0.9,
                zorder=8,
            )
        )


def _add_input_space_boxes_with_corners(
    ax: plt.Axes, x_regions: list[tuple[float, float]], panel_key: str
) -> None:
    """
    Draw dashed boxes for zero-trace regions with optional manual corners.

    Manual corners are read from ZERO_TRACE_BOX_CORNERS[panel_key]["left"|"right"].
    If unset (None), automatic bottom-of-axis placement is used.
    """
    cfg = ZERO_TRACE_BOX_CORNERS.get(panel_key, {})
    if not x_regions:
        return

    # left region then right region
    ordered = sorted(x_regions, key=lambda r: r[0])
    labels = ["left", "right"]
    manual_drawn = False
    for idx, region in enumerate(ordered[:2]):
        label = labels[idx]
        manual = cfg.get(label)
        if manual is None:
            continue
        (x0, y0), (x1, y1) = manual
        if x1 <= x0 or y1 <= y0:
            continue
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none",
                edgecolor="black",
                linewidth=0.9,
                linestyle="--",
                alpha=0.6,
                zorder=8,
            )
        )
        manual_drawn = True

    # Draw automatic boxes only for regions without manual corners.
    if not manual_drawn or cfg.get("left") is None or cfg.get("right") is None:
        auto_regions: list[tuple[float, float]] = []
        for idx, region in enumerate(ordered[:2]):
            label = labels[idx]
            if cfg.get(label) is None:
                auto_regions.append(region)
        _add_input_space_boxes(ax, auto_regions)


def _box_center(panel_key: str, side: str) -> tuple[float, float] | None:
    """Return center point (in data coords) of configured manual box."""
    corners = ZERO_TRACE_BOX_CORNERS.get(panel_key, {}).get(side)
    if corners is None:
        return None
    (x0, y0), (x1, y1) = corners
    if x1 <= x0 or y1 <= y0:
        return None
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _add_cross_panel_box_arrows(
    fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes
) -> None:
    """
    Draw light curved arrows connecting panel-A and AF-bound gray boxes.

    Left box maps to left box, and right box maps to right box.
    """
    side_to_rad = {"left": 0.24, "right": -0.24}
    for side in ("left", "right"):
        a_corners = ZERO_TRACE_BOX_CORNERS.get("panel_a", {}).get(side)
        d_corners = ZERO_TRACE_BOX_CORNERS.get("panel_d", {}).get(side)
        if a_corners is not None and d_corners is not None:
            (ax0, ay0), (ax1, ay1) = a_corners
            (dx0, dy0), (dx1, dy1) = d_corners
            if side == "left":
                # left: bottom-right  -> bottom-left, positive bend
                p_a = (ax1 - 0.02, ay0)
                p_d = (dx0 + 0.1, dy0 + 0.5e-8)
            else:
                # right: top-right -> top-left, negative bend
                p_a = (ax1 - 0.05, ay1)
                p_d = (dx0 + 0.05, 1e-6)
        else:
            # Fallback for non-manual box placement.
            p_a = _box_center("panel_a", side)
            p_d = _box_center("panel_d", side)
            if p_a is None or p_d is None:
                continue
        connector = ConnectionPatch(
            xyA=p_a,
            coordsA=ax_left.transData,
            xyB=p_d,
            coordsB=ax_right.transData,
            arrowstyle="-|>",
            shrinkA=2.0,
            shrinkB=2.0,
            mutation_scale=10.0,
            lw=1.0,
            color="black",
            alpha=0.6,
            connectionstyle=f"arc3,rad={side_to_rad[side]}",
            zorder=7,
        )
        fig.add_artist(connector)


def _add_manifold_view_group_box(
    fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes
) -> None:
    """Add light gray figure-level group box and title for manifold panels."""
    if MANIFOLD_VIEW_GROUP_BOX_CORNERS is None:
        bb_l = ax_left.get_position()
        bb_r = ax_right.get_position()
        x0 = min(bb_l.x0, bb_r.x0) - MANIFOLD_VIEW_GROUP_BOX_PAD
        y0 = min(bb_l.y0, bb_r.y0) - MANIFOLD_VIEW_GROUP_BOX_PAD
        x1 = (
            max(bb_l.x0 + bb_l.width, bb_r.x0 + bb_r.width)
            + MANIFOLD_VIEW_GROUP_BOX_PAD
        )
        y1 = (
            max(bb_l.y0 + bb_l.height, bb_r.y0 + bb_r.height)
            + MANIFOLD_VIEW_GROUP_BOX_PAD
        )
    else:
        x0, y0, x1, y1 = MANIFOLD_VIEW_GROUP_BOX_CORNERS

    fig.add_artist(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
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
        x0 + 0.5 * (x1 - x0),
        y1 + 0.004,
        "Manifold view",
        ha="center",
        va="bottom",
        fontsize=font_size,
        color="black",
        transform=fig.transFigure,
    )


def _geodesic_between(
    mu1: float, sigma1: float, mu2: float, sigma2: float, n_pts: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytic geodesic on M² between (μ1,σ1) and (μ2,σ2).

    In (u, v) = (μ, σ/√2) coords the Gaussian manifold is the Poincaré
    upper half-plane and geodesics are Euclidean semicircles centred on v=0.
    """
    u1, v1 = float(mu1), float(sigma1) / np.sqrt(2.0)
    u2, v2 = float(mu2), float(sigma2) / np.sqrt(2.0)
    if abs(u1 - u2) < 1e-8:
        # Vertical geodesic: exponential interpolation in σ.
        ts = np.linspace(0.0, 1.0, n_pts)
        sig = (
            max(float(sigma1), 1e-12)
            * (max(float(sigma2), 1e-12) / max(float(sigma1), 1e-12)) ** ts
        )
        return np.full(n_pts, mu1), sig
    c = 0.5 * ((u1**2 + v1**2) - (u2**2 + v2**2)) / (u1 - u2)
    r = np.hypot(u1 - c, v1)
    theta1 = np.arctan2(v1, u1 - c)  # in (0, π)
    theta2 = np.arctan2(v2, u2 - c)  # in (0, π)
    thetas = np.linspace(theta1, theta2, n_pts)
    mu_path = c + r * np.cos(thetas)
    sigma_path = np.maximum(r * np.sin(thetas), 1e-12) * np.sqrt(2.0)
    return mu_path, sigma_path


# ---------------------------------------------------------------------------
# Panel A — Conventional BO view
# ---------------------------------------------------------------------------


def plot_panel_a(
    ax: plt.Axes, gp_data: dict, geom_data: dict, shared_norm: Normalize
) -> None:
    """
    GP posterior + true function + observations + θ_best + θ_next.
    LogEI shown as a thick plasma-coloured horizontal strip at the panel bottom.
    Endpoints: x=0 → darkgreen plus (+), x=1 → darkgreen cross (×), annotated.
    """
    bounds = gp_data["bounds"]
    x_t = geom_data["x_test"]
    mu = geom_data["mu"]
    sigma = geom_data["sigma"]
    obs_x = geom_data["obs_x"]
    obs_y = geom_data["obs_y_std"]
    logei = geom_data["logei_clip"]
    x_peak = geom_data["x_peak"]
    peak_idx = geom_data["peak_idx"]

    x_eval = torch.from_numpy(x_t[:, None]).to(dtype=torch.float64)
    y_true = (
        eval_objective(x_eval, bounds).squeeze(-1).cpu().numpy() - gp_data["y_mean"]
    ) / gp_data["y_std"]

    # GP posterior
    ax.fill_between(
        x_t, mu - 2 * sigma, mu + 2 * sigma, color="#6baed6", alpha=0.22, zorder=1
    )
    ax.plot(x_t, mu, color="#1f77b4", lw=2.0, zorder=2)
    ax.plot(x_t, y_true, color="saddlebrown", ls="-", lw=1.2, alpha=0.95, zorder=2)

    # Observations
    ax.scatter(obs_x, obs_y, s=circle_size, c="black", zorder=7)

    # θ_best
    best_idx = int(np.argmax(obs_y))
    ax.scatter(
        [obs_x[best_idx]],
        [obs_y[best_idx]],
        marker="D",
        s=diamond_size,
        c="red",
        edgecolors="black",
        lw=0.8,
        zorder=9,
    )

    # θ_next
    ax.axvline(x=x_peak, color="#888888", ls=":", lw=0.9, alpha=0.8, zorder=3)
    ax.scatter(
        [x_peak],
        [mu[peak_idx]],
        marker="*",
        s=star_size,
        c="gold",
        edgecolors="black",
        lw=0.8,
        zorder=10,
    )

    # Endpoint markers: x=0 → plus (+), x=1 → cross (×), both darkgreen
    ax.scatter(
        [x_t[0]],
        [mu[0]],
        marker="+",
        s=x_marker_size,
        c=_X_START_COLOR,
        linewidths=2.2,
        zorder=11,
    )
    ax.scatter(
        [x_t[-1]],
        [mu[-1]],
        marker="x",
        s=x_marker_size,
        c=_X_END_COLOR,
        linewidths=2.2,
        zorder=11,
    )
    # ax.annotate(
    #     "$x=0$",
    #     xy=(x_t[0], mu[0]),
    #     xytext=(
    #         x_t[0] + ANNOTATION_POS["panel_a"]["x0"]["dx"],
    #         mu[0] + ANNOTATION_POS["panel_a"]["x0"]["dy"],
    #     ),
    #     color=_EP_COLOR,
    #     fontsize=font_size,
    #     zorder=12,
    # )
    # ax.annotate(
    #     "$x=1$",
    #     xy=(x_t[-1], mu[-1]),
    #     xytext=(
    #         x_t[-1] + ANNOTATION_POS["panel_a"]["x1"]["dx"],
    #         mu[-1] + ANNOTATION_POS["panel_a"]["x1"]["dy"],
    #     ),
    #     color=_EP_COLOR,
    #     fontsize=font_size,
    #     zorder=12,
    # )

    # Thick plasma-coloured strip as LineCollection (links to colourbar)
    y_strip = -2.12
    pts = np.array([x_t, np.full_like(x_t, y_strip)]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc_strip = LineCollection(
        segs, cmap=_CMAP, norm=shared_norm, linewidths=8, zorder=6, alpha=1.0
    )
    lc_strip.set_array((logei[:-1] + logei[1:]) / 2.0)
    ax.add_collection(lc_strip)

    y_cand = np.concatenate(
        (y_true, mu - 2.0 * sigma, mu + 2.0 * sigma, obs_y.astype(np.float64))
    )
    pad_y = 0.05 * max(float(np.nanmax(y_cand) - np.nanmin(y_cand)), 1e-6)
    y_lo = min(float(np.nanmin(y_cand)) - pad_y, y_strip - 0.28)
    y_hi = float(np.nanmax(y_cand)) + pad_y

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Input $x$", labelpad=1.0)
    ax.set_ylabel("Output $y$", labelpad=1.0)
    ax.set_title("Surrogate and AF", pad=FIG1_TITLE_PAD)
    ax.grid(alpha=0.18)
    x_regions = _zero_trace_edge_regions(x_t, geom_data["tr_gbar"])
    _add_input_space_boxes_with_corners(ax, x_regions, panel_key="panel_a")


# ---------------------------------------------------------------------------
# Panel B — Statistical manifold view
# ---------------------------------------------------------------------------


def plot_panel_b(ax: plt.Axes, geom_data: dict, shared_norm: Normalize):
    """
    (μ, log₁₀ σ) statistical manifold.

    Background:
      • Dashed steelblue contours — Fisher-Rao rings d_FR = const from θ_best.
        These are the analogue of Euclidean circles, but on the curved manifold.
      • Faint cornflower arcs — fan of geodesics showing manifold curvature.
    Foreground:
      • Highlighted orange geodesic from θ_best to θ_next — the manifold
        "straight-line" path that the BO step traces, together with the FR ring
        that passes through θ_next, labelled with d_FR.  The same step that
        looks like an arbitrary jump in Panel A is a single geodesic hop here.
      • φ(X) scatter coloured by LogEI (same plasma cmap as Panel A).
      • Green P-crosses at x=0 and x=1 matching Panel A.
    """
    mu = geom_data["mu"]
    sigma = geom_data["sigma"]
    logei = geom_data["logei_clip"]
    mu_best = geom_data["mu_best"]
    sigma_best = geom_data["sigma_best"]
    obs_mu = geom_data["obs_mu"]
    log10_obs_sig = geom_data["log10_obs_sigma"]
    mu_min = geom_data["mu_min"]
    mu_max = geom_data["mu_max"]
    sig_floor = geom_data["sigma_floor_plot"]
    sig_max_grid = geom_data["sig_max_grid"]
    # geodesics = geom_data["geodesics"]
    peak_idx = geom_data["peak_idx"]

    def td(s):
        return np.log10(np.maximum(s, 1e-12))

    sig_disp = td(sigma)
    sig_disp_best = td(sigma_best)

    # ── Fisher-Rao distance rings ──────────────────────────────────────────
    n_grid = 400
    mu_g = np.linspace(mu_min, mu_max, n_grid)
    sig_g = np.linspace(sig_floor, sig_max_grid, n_grid)
    MU, SIG = np.meshgrid(mu_g, sig_g)
    D_FR = compute_fisher_rao_distance_grid(MU, SIG, mu_best, sigma_best)

    # cs = ax.contour(
    #     MU,
    #     td(SIG),
    #     D_FR,
    #     levels=[0.5, 1.0, 1.5, 2.0, 2.5],
    #     colors="steelblue",
    #     linewidths=0.8,
    #     linestyles="--",
    #     alpha=0.40,
    #     zorder=2,
    # )
    # ax.clabel(cs, fmt=r"$d_{FR}=%.1f$", fontsize=7, inline=True)

    # # ── Geodesic fan (background, faint) ──────────────────────────────────
    # for mu_geo, log10_sig_geo in geodesics:
    #     ax.plot(
    #         mu_geo,
    #         log10_sig_geo,
    #         color="cornflowerblue",
    #         lw=0.7,
    #         alpha=0.25,
    #         zorder=3,
    #         solid_capstyle="round",
    #     )

    # ── φ(X) curve coloured by LogEI ──────────────────────────────────────
    ax.plot(mu, sig_disp, "-", color="gray", lw=0.7, alpha=0.25, zorder=4)
    sc = ax.scatter(
        mu,
        sig_disp,
        c=logei,
        cmap=_CMAP,
        norm=shared_norm,
        s=circle_size,
        linewidths=0,
        zorder=5,
    )

    # Observed data images
    ax.scatter(obs_mu, log10_obs_sig, c="black", s=circle_size, zorder=9)

    # θ_best
    ax.scatter(
        [mu_best],
        [sig_disp_best],
        marker="D",
        s=diamond_size,
        c="red",
        edgecolors="black",
        lw=0.8,
        zorder=12,
    )

    # θ_next
    ax.scatter(
        [mu[peak_idx]],
        [sig_disp[peak_idx]],
        marker="*",
        s=star_size,
        c="gold",
        edgecolors="black",
        lw=0.8,
        zorder=13,
    )

    # ── Highlighted geodesic θ_best → θ_next + matching FR ring ──────────
    mu_nx = float(mu[peak_idx])
    sig_nx = float(sigma[peak_idx])
    d_fr = _fr_distance(mu_best, sigma_best, mu_nx, sig_nx)

    mu_geo_hl, sig_geo_hl = _geodesic_between(mu_best, sigma_best, mu_nx, sig_nx)
    valid = (sig_geo_hl > sig_floor * 0.5) & (mu_geo_hl > mu_min - 0.1)
    if valid.sum() > 5:
        ax.plot(
            mu_geo_hl[valid],
            td(sig_geo_hl[valid]),
            color=_GEO_COLOR,
            lw=2.2,
            alpha=0.90,
            zorder=7,
            solid_capstyle="round",
        )
        # Label at midpoint of the arc
        mid = valid.sum() // 2
        mx = mu_geo_hl[valid][mid]
        my = td(sig_geo_hl[valid])[mid]
        _dfr_tx = np.clip(
            mx + ANNOTATION_POS["panel_b"]["d_fr"]["dx"],
            mu_min + ANNOTATION_POS["panel_b"]["d_fr_clip"]["x_left"],
            mu_max - ANNOTATION_POS["panel_b"]["d_fr_clip"]["x_right"],
        )
        _dfr_ty = np.clip(
            my + ANNOTATION_POS["panel_b"]["d_fr"]["dy"],
            ANNOTATION_POS["panel_b"]["d_fr_clip"]["y_low"],
            ANNOTATION_POS["panel_b"]["d_fr_clip"]["y_high"],
        )
        # ax.annotate(
        #     f"$d_{{FR}}={d_fr:.2f}$",
        #     xy=(mx, my),
        #     xytext=(_dfr_tx, _dfr_ty),
        #     fontsize=font_size,
        #     color=_GEO_COLOR,
        #     bbox=dict(
        #         boxstyle="round,pad=0.2",
        #         facecolor="white",
        #         edgecolor="black",
        #         linewidth=0.8,
        #     ),
        #     arrowprops=dict(
        #         arrowstyle="->",
        #         color=_GEO_COLOR,
        #         lw=0.8,
        #         connectionstyle="arc3,rad=0.25",
        #     ),
        #     zorder=14,
        # )

    # FR ring at exactly d_fr_next — the "trust boundary" for this BO step
    ax.contour(
        MU,
        td(SIG),
        D_FR,
        levels=[d_fr],
        colors=[_GEO_COLOR],
        linewidths=1.5,
        linestyles="--",
        alpha=0.65,
        zorder=6,
    )

    # ── Endpoint markers matching Panel A: x=0 → plus, x=1 → cross ──────
    ax.scatter(
        [mu[0]],
        [sig_disp[0]],
        marker="+",
        s=x_marker_size,
        c=_X_START_COLOR,
        linewidths=2.2,
        zorder=11,
    )
    ax.scatter(
        [mu[-1]],
        [sig_disp[-1]],
        marker="x",
        s=x_marker_size,
        c=_X_END_COLOR,
        linewidths=2.2,
        zorder=11,
    )
    # # Clamp annotation positions so they stay inside the axes.
    # y_lo, y_hi = -1.2, 0.1
    # x_lo_ax, x_hi_ax = mu_min, mu_max

    # def _clamp_annot(xv, yv, dx, dy):
    #     x_margin = ANNOTATION_POS["panel_b"]["clip_margin"]["x"]
    #     y_margin = ANNOTATION_POS["panel_b"]["clip_margin"]["y"]
    #     return (
    #         np.clip(xv + dx, x_lo_ax + x_margin, x_hi_ax - x_margin),
    #         np.clip(yv + dy, y_lo + y_margin, y_hi - y_margin),
    #     )

    # ax0_tx, ax0_ty = _clamp_annot(
    #     mu[0],
    #     sig_disp[0],
    #     ANNOTATION_POS["panel_b"]["x0"]["dx"],
    #     ANNOTATION_POS["panel_b"]["x0"]["dy"],
    # )
    # ax1_tx, ax1_ty = _clamp_annot(
    #     mu[-1],
    #     sig_disp[-1],
    #     ANNOTATION_POS["panel_b"]["x1"]["dx"],
    #     ANNOTATION_POS["panel_b"]["x1"]["dy"],
    # )
    # ax.annotate(
    #     "$x=0$",
    #     xy=(mu[0], sig_disp[0]),
    #     xytext=(ax0_tx, ax0_ty),
    #     color=_EP_COLOR,
    #     fontsize=font_size,
    #     zorder=12,
    # )
    # ax.annotate(
    #     "$x=1$",
    #     xy=(mu[-1], sig_disp[-1]),
    #     xytext=(ax1_tx, ax1_ty),
    #     color=_EP_COLOR,
    #     fontsize=font_size,
    #     zorder=12,
    # )

    ax.set_xlim(mu_min, mu_max)
    ax.set_ylim(-1.2, 0.1)
    ax.set_xlabel(r"Mean $\mu$", labelpad=1.0)
    ax.set_ylabel(r"Std. dev. $\log_{10}\Sigma$", labelpad=1.0)
    ax.set_title(r"Posterior map $\varphi_t(x)$", pad=FIG1_TITLE_PAD)
    ax.grid(alpha=0.12)
    return sc


def plot_panel_c_manifold(ax: plt.Axes, geom_data: dict):
    """
    C_alpha as a function of (μ, σ) on the statistical manifold (LogEI sensitivity),
    with the same φ(X) trace and markers as panel B for alignment.
    """
    mu = geom_data["mu"]
    sigma = geom_data["sigma"]
    mu_best = geom_data["mu_best"]
    sigma_best = geom_data["sigma_best"]
    obs_mu = geom_data["obs_mu"]
    log10_obs_sig = geom_data["log10_obs_sigma"]
    mu_min = geom_data["mu_min"]
    mu_max = geom_data["mu_max"]
    peak_idx = geom_data["peak_idx"]

    sig_hm = geom_data["c_alpha_hm_sigma"]
    c_field = np.asarray(geom_data["c_alpha_hm_values"], dtype=np.float64)
    c_pos = np.where(c_field > 0.0, c_field, np.nan)
    c_trace_raw = np.asarray(geom_data["c_alpha_mc"], dtype=np.float64)
    pos_flat = c_pos[np.isfinite(c_pos) & (c_pos > 0)]
    tr_flat = c_trace_raw[np.isfinite(c_trace_raw) & (c_trace_raw > 0)]
    vmin = float(np.nanpercentile(pos_flat, 5) if pos_flat.size else 1e-300)
    if tr_flat.size:
        vmin = float(min(vmin, np.nanpercentile(tr_flat, 5)))
    vmin = max(vmin, 1e-300)
    vmax = float(
        max(
            np.nanpercentile(pos_flat, 99.5) if pos_flat.size else vmin * 10,
            np.nanpercentile(tr_flat, 99.9) if tr_flat.size else vmin * 10,
            vmin * 10,
        )
    )

    def td(s):
        return np.log10(np.maximum(s, 1e-12))

    sig_disp = td(sigma)
    sig_disp_best = td(sigma_best)
    y0 = float(np.log10(max(sig_hm[0], 1e-12)))
    y1 = float(np.log10(max(sig_hm[-1], 1e-12)))

    im = ax.imshow(
        c_pos,
        origin="lower",
        extent=(mu_min, mu_max, y0, y1),
        aspect="auto",
        interpolation="bilinear",
        cmap="cividis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        zorder=1,
    )

    # Faint guide, then φ(x) in constant color to avoid heatmap/colorbar ambiguity.
    ax.plot(mu, sig_disp, "-", color="white", lw=0.45, alpha=0.22, zorder=4)
    ax.plot(
        mu,
        sig_disp,
        color="#f3e55d",
        lw=1.2,
        alpha=0.95,
        zorder=8,
        solid_capstyle="round",
    )
    ax.scatter(obs_mu, log10_obs_sig, c="black", s=circle_size, zorder=9)
    ax.scatter(
        [mu_best],
        [sig_disp_best],
        marker="D",
        s=diamond_size,
        c="red",
        edgecolors="black",
        lw=0.8,
        zorder=12,
    )
    ax.scatter(
        [mu[peak_idx]],
        [sig_disp[peak_idx]],
        marker="*",
        s=star_size,
        c="gold",
        edgecolors="black",
        lw=0.8,
        zorder=13,
    )

    # No geodesic / d_FR overlays in panel C; keep this panel focused on C_alpha heatmap.

    ax.scatter(
        [mu[0]],
        [sig_disp[0]],
        marker="+",
        s=x_marker_size,
        c=_X_START_COLOR,
        linewidths=2.0,
        zorder=11,
    )
    ax.scatter(
        [mu[-1]],
        [sig_disp[-1]],
        marker="x",
        s=x_marker_size,
        c=_X_END_COLOR,
        linewidths=2.0,
        zorder=11,
    )

    ax.set_xlim(mu_min, mu_max)
    ax.set_ylim(-1.2, 0.1)
    ax.set_xlabel(r"Mean $\mu$", labelpad=1.0)
    ax.set_title(r"AF sensitivity $C_\alpha(\varphi_t(x))$", pad=FIG1_TITLE_PAD)
    # ax.set_ylabel(r"$\log_{10}\Sigma$", labelpad=1.0)
    ax.grid(alpha=0.10, color="white", linewidth=0.3)
    return im


def plot_panel_c(ax: plt.Axes, geom_data: dict) -> dict:
    """Bound panel: ||grad_x alpha|| <= sqrt(C_alpha * tr(G))."""
    x_t = geom_data["x_test"]
    _f = 1e-8
    grad_alpha_norm = np.maximum(geom_data["grad_alpha_norm"], _f)
    upper_bound_sqrt = np.maximum(geom_data["upper_bound_sqrt"], _f)
    c_alpha = np.maximum(geom_data["c_alpha_mc"], _f)
    tr_g = np.maximum(geom_data["tr_gbar"], _f)
    grad_line = ax.plot(
        x_t,
        grad_alpha_norm,
        color="magenta",
        lw=1.2,
        label=r"$\|\nabla\alpha(\mathbf{x})\|_2$",
        zorder=5,
    )[0]
    bound_line = ax.plot(
        x_t,
        upper_bound_sqrt,
        color="blue",
        lw=1.2,
        ls="--",
        label=r"$\sqrt{C_\alpha \mathrm{tr}\,G}$",
        zorder=5,
    )[0]
    c_alpha_line = ax.plot(
        x_t,
        c_alpha,
        color="cyan",
        lw=1.0,
        alpha=0.9,
        label=r"$C_\alpha$",
        zorder=3,
    )[0]
    tr_g_line = ax.plot(
        x_t,
        tr_g,
        color="red",
        lw=1.0,
        alpha=0.9,
        label=r"$\mathrm{tr}\,G$",
        zorder=3,
    )[0]

    ax.set_yscale("log")
    y_all = np.hstack((grad_alpha_norm, upper_bound_sqrt, c_alpha, tr_g))
    y_all = y_all[np.isfinite(y_all) & (y_all > 0)]
    if y_all.size == 0:
        ymin_d, ymax_d = _f, 1.0
    else:
        ymin_d = float(np.min(y_all))
        ymax_d = float(np.max(y_all))
    ymin_vis = 10 ** (np.log10(ymin_d) - 0.08)
    ymax_vis = 10 ** (np.log10(ymax_d) + 0.08)
    ax.set_ylim(ymin_vis, ymax_vis)

    y_ticks = _integer_exponent_yticks(ymin_vis, ymax_vis, max_ticks=4)
    ax.set_yticks(y_ticks)

    def _decade_int_tick(y: float, _pos: int | None = None) -> str:
        if not np.isfinite(y) or y <= 0:
            return ""
        k = int(round(np.log10(y)))
        if not np.isclose(y, 10.0**k, rtol=0.0, atol=1e-15 * max(1.0, 10.0**k)):
            return ""
        return f"${k}$"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_decade_int_tick))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="y", which="major", pad=1.0)

    # Ordinate is log10; tick labels are exponents k (value = 10^k). Compact label on the
    # right edge so we do not reserve top-left canvas or collide with the panel title.
    ax.yaxis.set_label_position("left")
    ax.set_ylabel(r"$\log_{10}$ Scale", labelpad=0.5, fontsize=font_size)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Input $x$", labelpad=1.0)
    ax.set_title("AF gradient bound", pad=FIG1_TITLE_PAD)
    ax.grid(alpha=0.18, which="major")
    x_regions = _zero_trace_edge_regions(x_t, geom_data["tr_gbar"])
    _add_input_space_boxes_with_corners(ax, x_regions, panel_key="panel_d")
    return {
        "grad": grad_line,
        "bound": bound_line,
        "c_alpha": c_alpha_line,
        "tr_g": tr_g_line,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_fig1(gp_data: dict, geom_data: dict) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
            "text.usetex": True,
            "mathtext.fontset": "cm",
            "pgf.rcfonts": False,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
            "axes.titlepad": FIG1_TITLE_PAD,
        }
    )

    shared_norm = Normalize(vmin=-100.0, vmax=0.0)

    fig, (ax_a, ax_d, ax_b, ax_c) = plt.subplots(
        1, 4, figsize=FIG1_FIGSIZE_INCHES, constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=FIG1_CONSTRAINED_LAYOUT_W_PAD)

    plot_panel_a(ax_a, gp_data, geom_data, shared_norm)
    panel_c_handles = plot_panel_c(ax_d, geom_data)
    sc = plot_panel_b(ax_b, geom_data, shared_norm)
    im_calpha = plot_panel_c_manifold(ax_c, geom_data)
    _add_cross_panel_box_arrows(fig, ax_a, ax_d)
    _add_manifold_view_group_box(fig, ax_b, ax_c)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_box_aspect(1)

    cbar = fig.colorbar(
        sc,
        ax=ax_b,
        fraction=FIG1_LOGEI_CBAR_FRACTION,
        pad=FIG1_LOGEI_CBAR_PAD,
        shrink=FIG1_LOGEI_CBAR_SHRINK,
    )
    cbar.ax.text(
        0.0,
        1.03,
        r"LogEI",
        transform=cbar.ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=font_size,
        clip_on=False,
    )

    fig.colorbar(im_calpha, ax=ax_c, fraction=0.048, pad=0.04)

    # Legend anchored to Panel 2 right side.
    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2.0),
        Line2D([0], [0], color="saddlebrown", ls="-", lw=1.2, alpha=0.95),
        ColormapLine(_CMAP, lw=2.7),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor="black",
            markersize=25,
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="D",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=25,
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="*",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=35,
        ),
        # Line2D(
        #     [0],
        #     [0],
        #     linestyle="None",
        #     marker="+",
        #     markeredgecolor=_EP_COLOR,
        #     markersize=9,
        #     markeredgewidth=2.2,
        # ),
        # Line2D(
        #     [0],
        #     [0],
        #     linestyle="None",
        #     marker="x",
        #     markeredgecolor=_EP_COLOR,
        #     markersize=7,
        #     markeredgewidth=2.2,
        # ),
        Line2D([0], [0], color=_GEO_COLOR, lw=2.0),
        Line2D([0], [0], color=panel_c_handles["grad"].get_color(), lw=1.9),
        Line2D(
            [0],
            [0],
            color=panel_c_handles["bound"].get_color(),
            lw=1.9,
            ls="--",
        ),
        Line2D([0], [0], color=panel_c_handles["c_alpha"].get_color(), lw=1.2),
        Line2D([0], [0], color=panel_c_handles["tr_g"].get_color(), lw=1.2),
    ]
    legend_labels = [
        "GP",
        r"$f_{\mathrm{true}}$",
        "LogEI",
        r"$\mathcal{D}_t$",
        r"$x_{\mathrm{next}}$",
        r"$x_{\mathrm{best}}$",
        "Geodesic",
        r"$\|\nabla_{x}\alpha(\mathbf{x})\|_2$",
        r"$\sqrt{C_\alpha(\varphi_t(x)) \mathrm{tr}(g_t(x))}$",
        r"$C_\alpha(\varphi_t(x))$",
        r"$\mathrm{tr}(g_t(x))$",
    ]
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        ncol=11,
        markerscale=0.2,
        framealpha=0.7,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.45,
        borderaxespad=0.1,
        labelspacing=0.1,
        handler_map={ColormapLine: HandlerColormapLine()},
    )
    return fig
