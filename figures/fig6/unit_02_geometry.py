"""LogEI, FIM (G_x), TuRBO (ARD), and FITR aligned with the same pullback G for Fig. 6."""

from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor

from poc.deep_ensemble import DeepEnsembleModel
from src.riemannTuRBO.eps_config import EpsConfig, EpsMode, compute_eps_from_eigs
from src.riemannTuRBO.identity import ARDLengthscaleTransform


def _model_dtype_and_bounds(
    model, bounds: torch.Tensor
) -> tuple[torch.dtype, torch.Tensor]:
    if isinstance(model, DeepEnsembleModel):
        dtype = next(model.parameters()).dtype
        return dtype, bounds.to(dtype=dtype, device=bounds.device)
    return torch.float64, bounds


def compute_logei_grid(
    model,
    X_flat: np.ndarray,
    bounds: torch.Tensor,
    best_f: float | None = None,
) -> np.ndarray:
    """LogEI on a flattened raw-domain grid (normalizes to the model input space)."""
    if best_f is None:
        if not hasattr(model, "train_targets") or model.train_targets is None:
            raise ValueError("best_f is required when model has no train_targets")
        best_f = float(model.train_targets.max().item())
    acqf = LogExpectedImprovement(model=model, best_f=best_f)
    dtype, b = _model_dtype_and_bounds(model, bounds)
    candidates_raw = torch.from_numpy(X_flat).to(dtype=dtype, device=b.device)
    candidates_norm = normalize(candidates_raw, bounds=b)
    with torch.no_grad():
        vals = acqf(candidates_norm.unsqueeze(1)).squeeze(-1)
    return vals.detach().cpu().numpy()


def compute_gx_at_point(
    model,
    x_raw: np.ndarray,
    bounds: torch.Tensor,
) -> dict:
    """Pullback Fisher metric G_x for Gaussian predictive N(μ,σ²); aligns with Fig. 3."""
    dtype, b = _model_dtype_and_bounds(model, bounds)
    model.eval()
    x_t = torch.tensor(x_raw, dtype=dtype, device=b.device).view(1, 2).requires_grad_(True)
    x_model = normalize(x_t, bounds=b)
    post = model.posterior(x_model)
    mu_s = post.mean.squeeze()
    var_s = post.variance.squeeze().clamp_min(
        torch.tensor(1e-18, dtype=dtype, device=b.device)
    )

    grad_mu = torch.autograd.grad(mu_s, x_t, retain_graph=True, create_graph=False)[0].squeeze(0)
    grad_var = torch.autograd.grad(var_s, x_t, retain_graph=False, create_graph=False)[0].squeeze(0)

    gm = grad_mu.detach().cpu().numpy()
    gv = grad_var.detach().cpu().numpy()
    var_val = float(var_s.detach())

    g11 = float(gm[0] * gm[0] / var_val + gv[0] * gv[0] / (2.0 * var_val**2))
    g12 = float(gm[0] * gm[1] / var_val + gv[0] * gv[1] / (2.0 * var_val**2))
    g22 = float(gm[1] * gm[1] / var_val + gv[1] * gv[1] / (2.0 * var_val**2))

    return {
        "g11": g11,
        "g12": g12,
        "g22": g22,
        "grad_mu": gm,
        "grad_var": gv,
        "mu": float(post.mean.squeeze().detach()),
        "sigma": float(var_val**0.5),
    }


def _tensor_bounds_to_raw_dict(bb: Tensor, bounds: torch.Tensor) -> dict:
    """bb: [2, D] in normalized space → lo/hi arrays in raw space."""
    b64 = bounds.to(dtype=torch.float64)
    lo_norm, hi_norm = bb[0].double(), bb[1].double()
    lo_raw = unnormalize(lo_norm.unsqueeze(0), bounds=b64).squeeze(0)
    hi_raw = unnormalize(hi_norm.unsqueeze(0), bounds=b64).squeeze(0)
    lo_np = lo_raw.detach().cpu().numpy()
    hi_np = hi_raw.detach().cpu().numpy()
    return {"lo": lo_np, "hi": hi_np, "hw": (hi_np - lo_np) / 2.0}


def compute_fitr_tr_from_same_G(
    g_dict: dict,
    x_raw: np.ndarray,
    bounds: torch.Tensor,
    tr_length: float,
    eps_cfg: EpsConfig | None = None,
) -> dict:
    """Axis-aligned FITR box from diag(G) of the **same** G as the FIM ellipse.

    Uses the DiagGradRMSTransform recipe (``1/√(G_ii+ε)``, arithmetic mean normalize,
    then geometric mean volume normalize, then ``± length/2``) so overlays match the
    implemented diagonal FRTR, but with ``G_11,G_22`` taken from ``compute_gx_at_point``
    instead of MC score-RMS.
    """
    if eps_cfg is None:
        eps_cfg = EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12)

    b = bounds.to(dtype=torch.float64)
    x_norm = normalize(
        torch.tensor(x_raw, dtype=torch.float64).view(1, 2), bounds=b
    ).squeeze(0)

    diag_G = torch.tensor(
        [g_dict["g11"], g_dict["g22"]], dtype=torch.float64, device=b.device
    )
    eps = compute_eps_from_eigs(diag_G, eps_cfg)
    weights = 1.0 / torch.sqrt(diag_G + eps)
    weights = weights / weights.mean()
    log_w = torch.log(weights + 1e-16)
    geom_mean = torch.exp(log_w.mean())
    weights = weights / geom_mean

    hw = weights * tr_length / 2.0
    lower = torch.clamp(x_norm - hw, 0.0, 1.0)
    upper = torch.clamp(x_norm + hw, 0.0, 1.0)
    bb = torch.stack([lower, upper])
    return _tensor_bounds_to_raw_dict(bb, bounds)


def compute_turbo_tr_ard(
    model_se,
    x_raw: np.ndarray,
    bounds: torch.Tensor,
    tr_length: float,
) -> dict:
    """TuRBO-style TR via ARDLengthscaleTransform (GP lengthscales)."""
    dtype = torch.float64
    b = bounds.to(dtype=dtype)
    x_norm = normalize(torch.tensor(x_raw, dtype=dtype).view(1, 2), bounds=b).squeeze(0)
    eps_cfg = EpsConfig(mode=EpsMode.FIXED, eps=1.0)
    ard = ARDLengthscaleTransform(model_se, sampler=None, eps_cfg=eps_cfg)
    bb = ard(x_norm, tr_length)
    return _tensor_bounds_to_raw_dict(bb, bounds)
