# TODO: The multi-output case is not implemented yet. It will be implemented in the future.
"""
Base Class for Trust Regions
============================

This module defines the abstract base classes for Trust Regions in Bayesian Optimization.

Hierarchy:
1. TrustRegion (ABC): Root abstract class.
2. AxisAlignedTR (TrustRegion): For diagonal transforms (fast path, X-space bounds).
3. RotatedTR (TrustRegion): For rotated transforms (slow path, Z-space operator).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from botorch.sampling import MCSampler

from .eps_config import EpsConfig
from .utils import (
    probe_linear_operator_matrix,
    geometric_mean_singular_value,
    ensure_x_center_1d,
)

logger = logging.getLogger("TrustRegionBase")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TransformConfig:
    """
    Base configuration for transform operators.

    Attributes
    ----------
    eps_cfg : EpsConfig
        Epsilon (damping) configuration.
    volume_normalize : bool
        If True, normalize operator so geometric mean of singular values = 1.
        This decouples shape from size, letting `length` control TR size.
    use_lp_bounds : bool
        If True, use Linear Programming to find the minimal circumscribing z-box
        (tighter but slower). If False, use conservative corner mapping (faster).
    include_polytope_constraints : bool
        If True, compute and include polytope inequality constraints for rotated
        transforms. These can be passed to optimize_acqf for direct polytope sampling.
        Default False.
    """

    eps_cfg: EpsConfig = field(default_factory=EpsConfig)
    volume_normalize: bool = True


# =============================================================================
# Transform Operator (Result Container for RotatedTR)
# =============================================================================


@dataclass
class TransformOperator:
    """
    Riemannian transform operator with computed properties.
    Used mainly by RotatedTR.
    """

    operator: Callable[[Tensor], Tensor]
    inv_sqrt_diag: Tensor
    principal_directions: Optional[Tensor]
    eps_used: float
    z_bounds_lower: Tensor
    z_bounds_upper: Tensor
    is_axis_aligned: bool
    diagnostics: dict = field(default_factory=dict)
    inequality_constraints: Optional[list[tuple[Tensor, Tensor, float]]] = None

    def __call__(self, z: Tensor) -> Tensor:
        return self.operator(z)

    @property
    def z_bounds(self) -> Tuple[Tensor, Tensor]:
        return self.z_bounds_lower, self.z_bounds_upper


# =============================================================================
# Root Abstract Base Class
# =============================================================================


class TrustRegion(ABC):
    """
    Abstract base class for all Trust Regions.
    """

    name: str = "base"

    def __init__(
        self,
        model,
        sampler: Optional[MCSampler] = None,
        *,
        eps_cfg: Optional[EpsConfig] = None,
        volume_normalize: bool = True,
        **kwargs,  # absorb legacy params (use_lp_bounds, include_polytope_constraints)
    ):
        self.model = model
        self.sampler = sampler

        self.config = TransformConfig(
            eps_cfg=eps_cfg if eps_cfg is not None else EpsConfig(),
            volume_normalize=volume_normalize,
        )
        # Store diagnostics here (populated after __call__)
        self.diagnostics: dict = {}
        self.eps_used: float = 0.0


# =============================================================================
# AxisAlignedTR (Fast Path)
# =============================================================================


class AxisAlignedTR(TrustRegion):
    """
    Base class for axis-aligned (diagonal) transforms.

    The Trust Region is exactly a hyper-rectangle in X-space.
    This class provides a fast path: `__call__()` directly returns
    X-space bounds without going through z-space transformation.
    """

    @property
    def is_axis_aligned(self) -> bool:
        return True

    @abstractmethod
    def _compute_weights(self, x_center: Tensor) -> Tensor:
        """
        Compute per-dimension scaling weights (before volume normalization).

        Parameters
        ----------
        x_center : Tensor
            Center point (D,) tensor.

        Returns
        -------
        Tensor
            (D,) tensor where weights[d] is the scaling factor for dimension d.
            Typically 1/sqrt(G_dd + eps).
        """
        pass

    def __call__(self, x_center: Tensor, length: float) -> Tensor:
        """
        Compute trust region bounds for the given center and length.

        Parameters
        ----------
        x_center : Tensor
            Center point (D,) tensor.
        length : float
            Trust region length.

        Returns
        -------
        Tensor
            [2, D] bounds tensor (lower, upper) in X-space.

        Must handle:
        1. Scaling (based on metric diagonal)
        2. Volume Normalization (geometric mean = 1)
        3. Clipping to unit cube [0, 1]
        """
        weights, x_center = self._get_volume_normalized_weights(x_center)

        tr_length_adjusted_weights = self._get_tr_length_adjusted_weights(
            weights, length
        )

        # 3. Construct bounds (use 1d center so bounds are [2, D] not [2, 1, D])
        lower = torch.clamp(x_center - tr_length_adjusted_weights, 0.0, 1.0)
        upper = torch.clamp(x_center + tr_length_adjusted_weights, 0.0, 1.0)

        bounds = torch.stack([lower, upper])

        # 5. Compute diagnostics after computation
        self._compute_diagnostics(weights)

        return bounds

    def _get_volume_normalized_weights(self, x_center: Tensor) -> Tensor:
        """
        Get volume normalized weights for the given center.

        Returns
        -------
        weights : Tensor
            (D,) tensor where weights[d] is the scaling factor for dimension d.
            Typically 1/sqrt(G_dd + eps).
        x_center : Tensor
            (D,) tensor of the normalized x_center. It is important to return the normalized x_center because it is used to compute the bounds in the __call__ method.
        """
        # Normalize x_center to (D,)
        x_center = ensure_x_center_1d(x_center)  # to make calculating the bounds easier

        # 1. Get raw weights
        weights = self._compute_weights(x_center)

        # 2. Volume Normalize
        if self.config.volume_normalize:
            log_w = torch.log(weights + 1e-16)
            geom_mean = torch.exp(log_w.mean())
            weights = weights / geom_mean

        return weights, x_center

    def _get_tr_length_adjusted_weights(self, weights: Tensor, length: float) -> Tensor:
        """
        Get TR length adjusted weights for the given weights and length.
        """
        return weights * length / 2.0

    def _compute_diagnostics(self, weights: Tensor):
        """Compute and store diagnostics after weights are computed."""
        if weights.min() > 0:
            anisotropy = (weights.max() / weights.min()).item()
        else:
            anisotropy = float("inf")

        self.diagnostics["true_anisotropy"] = anisotropy

    def compute_true_anisotropy(self) -> float:
        """Get stored anisotropy from diagnostics."""
        return self.diagnostics.get("true_anisotropy", float("inf"))


# =============================================================================
# RotatedTR (Slow Path)
# =============================================================================


class RotatedTR(TrustRegion):
    """
    Base class for rotated (non-axis-aligned) transforms.

    The Trust Region is a rotated ellipsoid/polytope.
    Must operate in latent Z-space with operator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operator: Optional[TransformOperator] = None

    @property
    def is_axis_aligned(self) -> bool:
        return False

    @property
    def operator(self) -> TransformOperator:
        if self._operator is None:
            raise RuntimeError("Transform not computed. Call __call__() first.")
        return self._operator

    @property
    def z_bounds(self) -> Tuple[Tensor, Tensor]:
        return self.operator.z_bounds

    @property
    def z_bounds_lower(self) -> Tensor:
        return self.operator.z_bounds_lower

    @property
    def z_bounds_upper(self) -> Tensor:
        return self.operator.z_bounds_upper

    @abstractmethod
    def _compute_raw_operator(
        self, x_center: Tensor
    ) -> Tuple[Callable[[Tensor], Tensor], Tensor, Optional[Tensor], float, dict]:
        """
        Compute the raw operator and related quantities.

        Parameters
        ----------
        x_center : Tensor
            Center point (D,) tensor.

        Returns
        -------
        Tuple containing:
            - operator: Callable mapping z -> x_delta
            - inv_sqrt_diag: Tensor of inverse sqrt diagonal elements
            - principal_directions: Optional principal directions
            - eps_used: float epsilon value used
            - diagnostics: dict of diagnostic information
        """
        pass

    def __call__(self, x_center: Tensor, length: float) -> TransformOperator:
        """
        Compute trust region operator for the given center and length.

        Parameters
        ----------
        x_center : Tensor
            Center point (D,) tensor.
        length : float
            Trust region length.

        Returns
        -------
        TransformOperator
            The computed transform operator with z-bounds and diagnostics.
        """
        # Store length for z-bounds computation
        self._current_length = length
        self._current_x_center = x_center

        logger.debug(f"Computing {self.name} transform at center {x_center.tolist()}")

        op_raw, inv_sqrt_diag, principal_dirs, eps_used, diag = (
            self._compute_raw_operator(x_center)
        )

        self.eps_used = eps_used
        self.diagnostics.update(diag)

        if self.config.volume_normalize:
            op, inv_sqrt_diag = self._apply_volume_normalization(
                op_raw, inv_sqrt_diag, x_center
            )
        else:
            op = op_raw

        # Compute z-bounds (always rotated logic for RotatedTR)
        inequality_constraints = None
        z_lower, z_upper, inequality_constraints = (
            self._compute_z_bounds_rotated_and_constraints(op, x_center, length)
        )

        # Ensure true_anisotropy is in diagnostics
        if "true_anisotropy" not in diag:
            true_anisotropy = self.compute_true_anisotropy(op, x_center)
            diag["true_anisotropy"] = true_anisotropy
            self.diagnostics["true_anisotropy"] = true_anisotropy

        self._operator = TransformOperator(
            operator=op,
            inv_sqrt_diag=inv_sqrt_diag,
            principal_directions=principal_dirs,
            eps_used=eps_used,
            z_bounds_lower=z_lower,
            z_bounds_upper=z_upper,
            is_axis_aligned=False,
            diagnostics=diag,
            inequality_constraints=inequality_constraints,
        )

        return self._operator

    def _apply_volume_normalization(self, op_raw, inv_sqrt_diag, x_center: Tensor):
        dim = x_center.shape[-1]
        device = x_center.device
        dtype = x_center.dtype
        W = probe_linear_operator_matrix(op_raw, dim, device, dtype)
        gm = geometric_mean_singular_value(W)
        beta = 1.0 / gm

        def op_normalized(z: Tensor) -> Tensor:
            return op_raw(z) * beta

        return op_normalized, inv_sqrt_diag * beta

    def _compute_z_bounds_rotated_and_constraints(
        self, op, x_center: Tensor, length: float
    ):
        dim = x_center.shape[-1]
        device = x_center.device
        dtype = x_center.dtype
        z_lower = -torch.ones(dim, device=device, dtype=dtype)
        z_upper = torch.ones(dim, device=device, dtype=dtype)
        return z_lower, z_upper, None  # no polytope constraints

    def compute_true_anisotropy(
        self,
        op: Optional[Callable[[Tensor], Tensor]] = None,
        x_center: Optional[Tensor] = None,
    ) -> float:
        # Use analytic eigs if available (set by LowRankSVDTransform)
        if hasattr(self, "_vn_eigs") and self._vn_eigs is not None:
            eigs = self._vn_eigs.abs()
            eig_min = eigs.min()
            if eig_min > 0:
                return (eigs.max() / eig_min).item()
            return float("inf")
        return float("inf")
