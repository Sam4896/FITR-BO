"""
Identity and ARD Lengthscale Transform Operators
================================================

This module provides:
1. IdentityTransform: Isotropic (no preconditioning) - baseline
2. ARDLengthscaleTransform: Uses GP's ARD lengthscales (TuRBO-style)

These are the simplest transforms, useful as baselines and for comparison.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import Tensor
from botorch.sampling import MCSampler

from .base import AxisAlignedTR
from .eps_config import EpsConfig, EpsMode, compute_eps_from_eigs


logger = logging.getLogger("IdentityTransform")


# =============================================================================
# Identity Transform (Baseline)
# =============================================================================


class IdentityTransform(AxisAlignedTR):
    """
    Identity (isotropic) transform - no preconditioning.

    The trust region is a hypercube in x-space, scaled uniformly.
    """

    name = "Identity"

    def __init__(
        self,
        model,
        sampler: Optional[MCSampler] = None,
        *,
        eps_cfg: Optional[EpsConfig] = None,
        volume_normalize: bool = True,
        use_lp_bounds: bool = False,
    ):
        # Default to FIXED eps=1.0 for identity
        if eps_cfg is None:
            eps_cfg = EpsConfig(mode=EpsMode.FIXED, eps=1.0)
        super().__init__(
            model,
            sampler,
            eps_cfg=eps_cfg,
            volume_normalize=volume_normalize,
            use_lp_bounds=use_lp_bounds,
        )

    def _compute_weights(self, x_center: Tensor) -> Tensor:
        """Compute uniform weights for identity transform."""
        dim = x_center.shape[-1]
        device = x_center.device
        dtype = x_center.dtype
        eps = compute_eps_from_eigs(
            torch.zeros(dim, device=device, dtype=dtype),
            self.config.eps_cfg,
        )
        scale = 1.0 / math.sqrt(eps)
        weights = torch.full((dim,), scale, device=device, dtype=dtype)

        logger.info(f"Identity: scale={scale:.4f}")
        self.eps_used = eps

        # Update diagnostics
        self.diagnostics.update(
            {
                "scale": scale,
            }
        )
        return weights


# =============================================================================
# ARD Lengthscale Transform (TuRBO-style)
# =============================================================================


class ARDLengthscaleTransform(AxisAlignedTR):
    """
    Preconditioner from GP's ARD lengthscales.
    """

    name = "ARDLengthscale"

    def __init__(
        self,
        model,
        sampler: Optional[MCSampler] = None,
        *,
        eps_cfg: Optional[EpsConfig] = None,
        use_lp_bounds: bool = False,
        **kwargs,
    ):
        super().__init__(
            model,
            sampler,
            eps_cfg=eps_cfg,
            volume_normalize=False,
            use_lp_bounds=use_lp_bounds,
        )

    def _compute_weights(self, x_center: Tensor) -> Tensor:
        """
        Compute weights from ARD lengthscales (matches original TuRBO exactly).

        Original TuRBO calculation:
        1. weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        2. weights = weights / weights.mean()  # Arithmetic mean normalization
        3. weights = weights / torch.prod(weights.pow(1.0 / len(weights)))  # Geometric mean normalization
        4. Use weights directly (no epsilon, no sqrt, no inversion)

        Note: The base class will apply volume normalization again if enabled,
        but original TuRBO applies geometric mean normalization only once.
        To match exactly, we apply both normalizations here, and the base class
        volume normalization will be a no-op (since geometric mean is already 1).
        """
        try:
            # Try base_kernel.lengthscale first (original TuRBO uses this)
            if hasattr(self.model, "covar_module") and hasattr(
                self.model.covar_module, "base_kernel"
            ):
                ls = self.model.covar_module.base_kernel.lengthscale.squeeze().detach()
            elif hasattr(self.model, "covar_module"):
                # Fallback to covar_module.lengthscale if base_kernel not available
                ls = self.model.covar_module.lengthscale.squeeze().detach()
            elif hasattr(self.model, "model") and hasattr(
                self.model.model, "covar_module"
            ):
                if hasattr(self.model.model.covar_module, "base_kernel"):
                    ls = self.model.model.covar_module.base_kernel.lengthscale.squeeze().detach()
                else:
                    ls = self.model.model.covar_module.lengthscale.squeeze().detach()
            else:
                raise AttributeError("Cannot find lengthscale")

            ls = ls.to(device=x_center.device, dtype=x_center.dtype)
        except Exception as e:
            raise ValueError(
                f"ARDLengthscaleTransform requires a model with accessible lengthscales. "
                f"Error: {e}"
            )

        # Step 1: Arithmetic mean normalization (original TuRBO)
        weights = ls / ls.mean()

        # Step 2: Geometric mean normalization (original TuRBO)
        # This is equivalent to: weights / torch.prod(weights.pow(1.0 / len(weights)))
        # But computed more numerically stable using logs
        log_weights = torch.log(weights + 1e-16)
        geom_mean = torch.exp(log_weights.mean())
        weights = weights / geom_mean

        # Original TuRBO uses weights directly (no epsilon, no sqrt, no inversion)
        # The weights represent the trust region scaling factors directly
        # Larger lengthscale = larger weight = larger trust region in that dimension
        #
        # Note: After geometric mean normalization, the geometric mean is 1.0,
        # so if the base class applies volume normalization again, it will be a no-op.

        logger.info(
            f"ARDLengthscale: ls_range=[{ls.min():.4f}, {ls.max():.4f}], "
            f"weight_range=[{weights.min():.4f}, {weights.max():.4f}], "
            f"geom_mean={geom_mean:.6f}"
        )

        # No epsilon used in original TuRBO
        self.eps_used = 0.0

        # Update diagnostics
        self.diagnostics.update(
            {
                "lengthscales_raw": ls.tolist(),
                "weights": weights.tolist(),
            }
        )

        return weights
