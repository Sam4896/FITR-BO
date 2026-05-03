"""Figure 1 geometric quantities: trace metric and manifold primitives."""

from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement


def _safe_sigma(sigma: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    return np.maximum(sigma, floor)


def _compute_trace_gbar(
    dmu_dx: np.ndarray, dsigma_dx: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    sigma_safe = _safe_sigma(sigma)
    return (dmu_dx**2) / (sigma_safe**2) + 2.0 * (dsigma_dx**2) / (sigma_safe**2)


def _compute_logei_bound_terms(
    model,
    x_test: torch.Tensor,
    best_f: float,
) -> dict:
    """Compute ||grad_x LogEI|| and C_alpha from closed-form LogEI(mu, sigma)."""
    normal_dist = torch.distributions.Normal(
        torch.zeros(1, dtype=x_test.dtype, device=x_test.device),
        torch.ones(1, dtype=x_test.dtype, device=x_test.device),
    )
    grad_alpha_norm = []
    c_alpha_mc = []

    for i in range(x_test.shape[0]):
        x_i = x_test[i : i + 1].clone().detach().requires_grad_(True)
        posterior = model.posterior(x_i)
        mu = posterior.mean.reshape(())
        sigma = posterior.variance.clamp_min(1e-12).sqrt().reshape(()).clamp(min=1e-12)

        # Closed-form LogEI(μ,σ) = log(σ) + log(zΦ(z)+φ(z)), z=(μ-best_f)/σ.
        z = (mu - best_f) / sigma
        phi_z = torch.exp(normal_dist.log_prob(z))
        Phi_z = normal_dist.cdf(z)
        inner = z * Phi_z + phi_z
        alpha = torch.log(sigma) + torch.log(inner.clamp(min=1e-300))

        d_alpha_dx = torch.autograd.grad(alpha, x_i, retain_graph=True)[0].reshape(())
        grad_alpha_norm.append(float(torch.abs(d_alpha_dx).detach().cpu().item()))

        d_alpha_dmu, d_alpha_dsigma = torch.autograd.grad(alpha, (mu, sigma))
        # g_F^{-1}=diag(sigma^2, sigma^2/2) in (mu, sigma) coordinates.
        ginv_quad = sigma**2 * d_alpha_dmu**2 + 0.5 * sigma**2 * d_alpha_dsigma**2
        c_alpha_mc.append(float(ginv_quad.detach().cpu().item()))

    return {
        "grad_alpha_norm": np.asarray(grad_alpha_norm, dtype=np.float64),
        "c_alpha_mc": np.asarray(c_alpha_mc, dtype=np.float64),
    }


def _compute_tangent_vectors(
    mu: np.ndarray,
    sigma: np.ndarray,
    dmu_dx: np.ndarray,
    dsigma_dx: np.ndarray,
    tr_gbar: np.ndarray,
    n_arrows: int = 4,
) -> dict:
    # Pick representative points by quantiles of tr(G_bar).
    q = np.linspace(0.15, 0.85, n_arrows)
    targets = np.quantile(tr_gbar, q)
    idx = np.array([int(np.argmin(np.abs(tr_gbar - t))) for t in targets], dtype=int)

    # Fisher norm of tangent equals sqrt(tr(G_bar)).
    fr_norm = np.sqrt(np.maximum(tr_gbar[idx], 1e-12))
    direction = np.stack([dmu_dx[idx], dsigma_dx[idx]], axis=1)
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction_norm = np.maximum(direction_norm, 1e-12)
    direction_unit = direction / direction_norm

    # Length in display space scales with FR norm.
    length = 0.03 + 0.10 * (fr_norm / np.maximum(fr_norm.max(), 1e-12))
    arrows = direction_unit * length[:, None]

    return {
        "idx": idx,
        "x": mu[idx],
        "y": sigma[idx],
        "u": arrows[:, 0],
        "v": arrows[:, 1],
        "fr_norm": fr_norm,
    }


def integrate_geodesic(
    mu0: float,
    sigma0: float,
    v_mu0: float,
    v_sigma0: float,
    n_steps: int = 600,
    t_end: float = 1.0,
    sigma_floor: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """RK4 integration of the Fisher-Rao geodesic ODE on M² (Gaussian family).

    Christoffel symbols:
        d²μ/dt²  =  (2/σ)(μ')(σ')
        d²σ/dt²  = -(1/2σ)(μ')² + (1/σ)(σ')²
    """
    dt = t_end / n_steps
    s = np.array([mu0, sigma0, v_mu0, v_sigma0], dtype=np.float64)
    mu_path = [mu0]
    sigma_path = [sigma0]

    def ode(state: np.ndarray) -> np.ndarray:
        _, sig, vu, vs = state
        sig_s = max(sig, sigma_floor)
        return np.array(
            [
                vu,
                vs,
                (2.0 / sig_s) * vu * vs,
                -(1.0 / (2.0 * sig_s)) * vu**2 + (1.0 / sig_s) * vs**2,
            ]
        )

    for _ in range(n_steps):
        k1 = ode(s)
        k2 = ode(s + 0.5 * dt * k1)
        k3 = ode(s + 0.5 * dt * k2)
        k4 = ode(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        s[1] = max(s[1], sigma_floor)
        mu_path.append(s[0])
        sigma_path.append(s[1])

    return np.array(mu_path), np.array(sigma_path)


def compute_fisher_rao_distance_grid(
    mu_grid: np.ndarray,
    sigma_grid: np.ndarray,
    mu_ref: float,
    sigma_ref: float,
    min_sigma: float = 1e-9,
) -> np.ndarray:
    """Fisher–Rao distance from θ_ref to each (μ, σ) on a grid (same formula as manifold_view reference)."""
    s = np.maximum(sigma_grid, min_sigma)
    s_ref = max(sigma_ref, min_sigma)
    inner = 1.0 + ((mu_grid - mu_ref) ** 2 + 2.0 * (s - s_ref) ** 2) / (4.0 * s * s_ref)
    inner = np.maximum(inner, 1.0 + 1e-15)
    return np.sqrt(2.0) * np.arccosh(inner)


def _c_alpha_manifold_field(
    mu_grid: np.ndarray,
    sigma_grid: np.ndarray,
    best_f: float,
) -> np.ndarray:
    """
    Acquisition sensitivity C_alpha(μ, σ) for closed-form LogEI on the Gaussian
    predictive manifold (same Fisher inverse as in _compute_logei_bound_terms).

    With α = log σ + log(zΦ(z)+φ(z)), z=(μ−f*)/σ, one gets
    C_alpha = (Φ/w)^2 + (1/2)(φ/w)^2  where w = zΦ+φ (no σ prefactor after cancellation).
    """
    mu_t = torch.tensor(mu_grid, dtype=torch.float64)
    sig_t = torch.tensor(sigma_grid, dtype=torch.float64).clamp(min=1e-12)
    z = (mu_t - best_f) / sig_t
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=torch.float64), torch.ones((), dtype=torch.float64)
    )
    Phi = normal.cdf(z)
    phi = torch.exp(normal.log_prob(z))
    w = (z * Phi + phi).clamp(min=1e-300)
    c_alpha = (Phi / w) ** 2 + 0.5 * (phi / w) ** 2
    return c_alpha.detach().cpu().numpy().astype(np.float64)


def _compute_hyperbolic_grid(
    mu_min: float,
    mu_max: float,
    sigma_max: float,
    n_mu_lines: int = 11,
    sigma0: float = 0.005,
    delta: float = 0.18,
) -> dict:
    """Horizontal lines at sigma_k = sigma0 * exp(k * delta) for uniform geodesic spacing."""
    mu_lines = np.linspace(mu_min, mu_max, n_mu_lines)
    sigma_lines = []
    s = sigma0
    while s <= sigma_max * 1.05:
        sigma_lines.append(s)
        s *= np.exp(delta)
    return {"mu_lines": np.array(mu_lines), "sigma_lines": np.array(sigma_lines)}


def compute_fig1_geometry(
    model,
    X_train: torch.Tensor,
    Y_std_train: torch.Tensor,
    bounds: torch.Tensor,
    n_test: int = 200,
) -> dict:
    """Compute all numeric quantities needed by both figure panels."""
    _ = bounds
    dtype = X_train.dtype
    x_test = torch.linspace(0, 1, n_test, dtype=dtype).unsqueeze(-1)

    with torch.no_grad():
        posterior = model.posterior(x_test)
        mu = posterior.mean.squeeze(-1).cpu().numpy()
        sigma = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()

    x_np = x_test.squeeze(-1).cpu().numpy()
    dmu_dx = np.gradient(mu, x_np)
    dsigma_dx = np.gradient(sigma, x_np)
    tr_gbar = _compute_trace_gbar(dmu_dx, dsigma_dx, sigma)

    best_f = float(Y_std_train.max().item())
    mc_terms = _compute_logei_bound_terms(
        model=model,
        x_test=x_test,
        best_f=best_f,
    )
    grad_alpha_norm = mc_terms["grad_alpha_norm"]
    _plot_floor = 1e-8
    c_alpha_mc = np.maximum(np.maximum(mc_terms["c_alpha_mc"], 0.0), _plot_floor)
    tr_g_floored = np.maximum(tr_gbar, _plot_floor)
    upper_bound_sqrt = np.sqrt(c_alpha_mc * tr_g_floored)

    acqf = LogExpectedImprovement(model, best_f=best_f)
    with torch.no_grad():
        logei = acqf(x_test.unsqueeze(-2)).squeeze(-1).cpu().numpy()

    peak_idx = int(np.argmax(logei))
    x_peak = float(x_np[peak_idx])

    with torch.no_grad():
        obs_post = model.posterior(X_train)
        obs_mu = obs_post.mean.squeeze(-1).cpu().numpy()
        obs_sigma = obs_post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()

    best_idx = int(Y_std_train.squeeze().argmax().item())
    with torch.no_grad():
        pb = model.posterior(X_train[best_idx : best_idx + 1])
        mu_best = float(pb.mean.squeeze().item())
        sigma_best = float(pb.variance.clamp_min(1e-12).sqrt().squeeze().item())

    log10_sigma = np.log10(np.maximum(sigma, 1e-12))
    log10_obs_sigma = np.log10(np.maximum(obs_sigma, 1e-12))

    # Geodesic fan from θ_best: shoot rays in evenly spaced directions on M².
    sigma_floor_plot = 0.03
    sig_max_grid = float(np.max(sigma) * 1.5)
    mu_lo = float(np.min(mu) - 0.5)
    mu_hi = float(np.max(mu) + 0.5)
    geodesics = []
    for angle in np.linspace(np.pi * 0.05, np.pi * 0.95, 10):
        speed = 1.5
        v_mu = speed * np.cos(angle)
        v_sig = speed * np.sin(angle) * np.sqrt(2.0)
        if sigma_best + 0.1 * v_sig < 1e-6:
            continue
        try:
            mu_geo, sig_geo = integrate_geodesic(
                mu_best, sigma_best, v_mu, v_sig, n_steps=800
            )
            valid = (
                (sig_geo > sigma_floor_plot)
                & (sig_geo < sig_max_grid * 1.1)
                & (mu_geo > mu_lo - 0.2)
                & (mu_geo < mu_hi + 0.2)
            )
            if valid.sum() > 5:
                geodesics.append(
                    (mu_geo[valid], np.log10(np.maximum(sig_geo[valid], 1e-12)))
                )
        except Exception:
            pass

    tangents = _compute_tangent_vectors(
        mu, sigma, dmu_dx, dsigma_dx, tr_gbar, n_arrows=4
    )
    hi_idx = int(np.argmax(tr_gbar))
    lo_idx = int(np.argmin(tr_gbar))

    tr_gbar_vmax95 = float(np.percentile(tr_gbar, 95))
    tr_gbar_clipped = np.clip(tr_gbar, 0.0, tr_gbar_vmax95)

    sigma_min_ax = 0.0
    sigma_max = float(np.max(sigma) * 1.2)
    mu_min = mu_lo
    mu_max = mu_hi
    grid = _compute_hyperbolic_grid(mu_min, mu_max, sigma_max)

    # C_alpha(μ, σ) heatmap on the same (μ, σ) window as panel B (manifold-native).
    n_calpha = 160
    mu_calpha = np.linspace(mu_min, mu_max, n_calpha, dtype=np.float64)
    sig_calpha = np.linspace(sigma_floor_plot, sig_max_grid, n_calpha, dtype=np.float64)
    MU_ca, SIG_ca = np.meshgrid(mu_calpha, sig_calpha, indexing="xy")
    c_alpha_field = _c_alpha_manifold_field(MU_ca, SIG_ca, best_f)

    return {
        "x_test": x_np,
        "mu": mu,
        "sigma": sigma,
        "log10_sigma": log10_sigma,
        "dmu_dx": dmu_dx,
        "dsigma_dx": dsigma_dx,
        "tr_gbar": tr_gbar,
        "grad_alpha_norm": grad_alpha_norm,
        "c_alpha_mc": c_alpha_mc,
        "upper_bound_sqrt": upper_bound_sqrt,
        "tr_gbar_clipped": tr_gbar_clipped,
        "tr_gbar_vmax95": tr_gbar_vmax95,
        "logei": logei,
        "logei_clip": np.clip(logei, -100.0, 0.0),
        "x_peak": x_peak,
        "peak_idx": peak_idx,
        "obs_x": X_train.squeeze(-1).cpu().numpy(),
        "obs_y_std": Y_std_train.squeeze(-1).cpu().numpy(),
        "obs_mu": obs_mu,
        "obs_sigma": obs_sigma,
        "log10_obs_sigma": log10_obs_sigma,
        "mu_best": mu_best,
        "sigma_best": sigma_best,
        "tangents": tangents,
        "high_idx": hi_idx,
        "low_idx": lo_idx,
        "mu_min": mu_min,
        "mu_max": mu_max,
        "sigma_min_ax": sigma_min_ax,
        "sigma_max": sigma_max,
        "sigma_floor_plot": sigma_floor_plot,
        "sig_max_grid": sig_max_grid,
        "geodesics": geodesics,
        "grid": grid,
        "c_alpha_hm_mu": mu_calpha,
        "c_alpha_hm_sigma": sig_calpha,
        "c_alpha_hm_values": c_alpha_field,
    }
