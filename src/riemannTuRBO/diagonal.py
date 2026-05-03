"""
Diagonal Transform Operators for Riemannian Trust Regions
=========================================================

This module provides axis-aligned (diagonal) transform operators:
1. DiagGradMean: Uses gradient of posterior mean
2. DiagGradRMS: Uses RMS of Fisher gradients (recommended)
3. FiniteDiff: Model-agnostic via finite differences

These operators scale each dimension independently based on local sensitivity.
They are fast to compute and provide exact z-bounds since the transform
is axis-aligned.

Comparison
----------
- DiagGradMean: One backward pass, fast but uses only point estimate
- DiagGradRMS: Multiple backward passes (S samples), more robust
- FiniteDiff: 2*D forward passes, works with any model

All preserve axis alignment, so:
- The trust region is a hyper-rectangle (not a parallelepiped)
- Z-bounds can be computed exactly
- No rotation/correlation between dimensions is captured

Use LowRankSVD if you need to capture cross-dimensional correlations.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor
from botorch.sampling import MCSampler

from .base import AxisAlignedTR
from .eps_config import EpsConfig, compute_eps_from_eigs
from .utils import get_fisher_grads_from_posterior, ensure_x_shape_for_posterior


logger = logging.getLogger("DiagonalTransform")


# =============================================================================
# DiagGradMean Transform
# =============================================================================


class DiagGradMeanTransform(AxisAlignedTR):
    """
    Diagonal preconditioner from gradient of posterior mean.

    Computes G_ii = (∂μ/∂x_i)² where μ is the posterior mean.
    The transform scales each dimension by 1/√(G_ii + ε).
    """

    name = "DiagGradMean"

    def __init__(
        self,
        model,
        sampler: Optional[MCSampler] = None,
        *,
        eps_cfg: Optional[EpsConfig] = None,
        volume_normalize: bool = True,
        use_lp_bounds: bool = False,
    ):
        super().__init__(
            model,
            sampler,
            eps_cfg=eps_cfg,
            volume_normalize=volume_normalize,
            use_lp_bounds=use_lp_bounds,
        )

    def _compute_weights(self, x_center: Tensor) -> Tensor:
        """Compute weights from posterior mean gradient."""
        # Ensure shape (1, D) for posterior call
        x_for_grad = (
            ensure_x_shape_for_posterior(x_center).detach().clone().requires_grad_(True)
        )

        # Get posterior mean gradient
        post = self.model.posterior(x_for_grad)
        mu = post.mean.sum()
        grad = (
            torch.autograd.grad(mu, x_for_grad, retain_graph=False, create_graph=False)[
                0
            ]
            .squeeze()
            .detach()
        )  # Extract (D,) from (1, D)

        # Diagonal metric: G_ii = grad_i²
        diag_G = grad.pow(2)

        # Compute adaptive epsilon
        eps = compute_eps_from_eigs(diag_G, self.config.eps_cfg)

        weights = 1.0 / torch.sqrt(diag_G + eps)

        logger.info(
            f"DiagGradMean: grad_norm={grad.norm().item():.4e}, "
            f"diag_G_range=[{diag_G.min():.4e}, {diag_G.max():.4e}]"
        )
        self.eps_used = eps

        # Update diagnostics
        self.diagnostics.update(
            {
                "diag_G": diag_G.tolist(),
                "grad_norm": grad.norm().item(),
                "weights": weights.tolist(),
            }
        )

        return weights


# =============================================================================
# DiagGradRMS Transform
# =============================================================================


class DiagGradRMSTransform(AxisAlignedTR):
    """
    Diagonal preconditioner from RMS of Fisher gradients (True Fisher Diagonal).

    Computes diag(E[g g^T]) where g = ∇log p(y|x) are Fisher gradients.
    This estimates the diagonal of the True Fisher Information Matrix.

    diag_G_i = (1/S) Σ_s (g_{s,i})²
    """

    name = "DiagGradRMS"

    def __init__(
        self,
        model,
        sampler: MCSampler,
        *,
        eps_cfg: Optional[EpsConfig] = None,
        volume_normalize: bool = True,
        use_lp_bounds: bool = False,
        observation_noise: bool = True,
    ):
        super().__init__(
            model,
            sampler,
            eps_cfg=eps_cfg,
            volume_normalize=volume_normalize,
            use_lp_bounds=use_lp_bounds,
        )
        self.observation_noise = observation_noise

    def _compute_weights(self, x_center: Tensor) -> Tensor:
        """Compute weights from RMS of Fisher gradients."""
        # Ensure shape (1, D) for posterior call
        x_for_grad = ensure_x_shape_for_posterior(x_center)
        # Get Fisher gradients: shape (S, N, D)
        fisher_grads = get_fisher_grads_from_posterior(
            self.model,
            x_for_grad,
            self.sampler,
            observation_noise=self.observation_noise,
        )

        S, N, D = fisher_grads.shape

        # Diagonal of metric: G_ii = E[g_i²]
        # Square and average over samples
        diag_G = fisher_grads.pow(2).mean(dim=0)  # (N, D)

        # If N > 1, average over batch (typically N=1 for single center)
        if N > 1:
            diag_G = diag_G.mean(dim=0)
        else:
            diag_G = diag_G.squeeze(0)

        # Compute adaptive epsilon (scale-invariant damping; avoids extreme anisotropy)
        eps = compute_eps_from_eigs(diag_G, self.config.eps_cfg)

        weights = 1.0 / torch.sqrt(diag_G + eps)
        weights = weights / weights.mean()

        # Diagnostics
        grad_norms = fisher_grads.norm(dim=-1)  # (S, N)
        logger.info(
            f"DiagGradRMS: S={S}, grad_norm_mean={grad_norms.mean():.4e}, "
            f"diag_G_range=[{diag_G.min():.4e}, {diag_G.max():.4e}]"
        )
        self.eps_used = eps

        # Update diagnostics
        self.diagnostics.update(
            {
                "diag_G": diag_G.tolist(),
                "num_samples": S,
                "grad_norm_mean": grad_norms.mean().item(),
                "grad_norm_std": grad_norms.std().item(),
                "weights": weights.tolist(),
            }
        )

        return weights


# =============================================================================
# FiniteDiff Transform (Model-Agnostic Diagonal)
# =============================================================================


class FiniteDiffTransform(AxisAlignedTR):
    """
    Model-agnostic diagonal preconditioner using finite differences.

    Estimates gradients via central differences:
        ∂μ/∂x_i ≈ [μ(x + h*e_i) - μ(x - h*e_i)] / (2h)
    """

    name = "FiniteDiff"

    def __init__(
        self,
        model,
        sampler: Optional[MCSampler] = None,
        *,
        eps_cfg: Optional[EpsConfig] = None,
        volume_normalize: bool = True,
        use_lp_bounds: bool = False,
        fd_h: float = 1e-4,
    ):
        self.fd_h = fd_h
        super().__init__(
            model,
            sampler,
            eps_cfg=eps_cfg,
            volume_normalize=volume_normalize,
            use_lp_bounds=use_lp_bounds,
        )

    def _compute_weights(self, x_center: Tensor) -> Tensor:
        """Compute weights using finite differences."""
        from .utils import get_posterior_mean_scalar

        h = self.fd_h
        x0 = x_center
        dim = x_center.shape[-1]
        device = x_center.device
        dtype = x_center.dtype

        # Estimate gradient via central differences
        grad = torch.zeros(dim, device=device, dtype=dtype)

        for d in range(dim):
            e = torch.zeros(dim, device=device, dtype=dtype)
            e[d] = 1.0

            # Clamp to bounds [0, 1] and ensure shape (1, D) for posterior
            xp = ensure_x_shape_for_posterior(torch.clamp(x0 + h * e, 0.0, 1.0))
            xm = ensure_x_shape_for_posterior(torch.clamp(x0 - h * e, 0.0, 1.0))

            # Get posterior means
            mp = get_posterior_mean_scalar(self.model, xp)
            mm = get_posterior_mean_scalar(self.model, xm)

            grad[d] = (mp - mm) / (2.0 * h)

        # Diagonal metric
        diag_G = grad.pow(2)

        # Adaptive epsilon
        eps = compute_eps_from_eigs(diag_G, self.config.eps_cfg)

        weights = 1.0 / torch.sqrt(diag_G + eps)

        logger.info(
            f"FiniteDiff: h={h:.1e}, grad_norm={grad.norm():.4e}, "
            f"diag_G_range=[{diag_G.min():.4e}, {diag_G.max():.4e}]"
        )
        self.eps_used = eps

        # Update diagnostics
        self.diagnostics.update(
            {
                "diag_G": diag_G.tolist(),
                "grad_fd": grad.tolist(),
                "fd_h": h,
                "weights": weights.tolist(),
            }
        )

        return weights
