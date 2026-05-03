"""
Riemannian Acquisition Function Wrapper
=======================================

This module provides RiemannianWrappedAcquisitionFunction, a wrapper
that transforms any BoTorch acquisition function to operate in the
Riemannian latent space Z for rotated transforms (RotatedTR).

The wrapper handles:
1. Sampling from constrained z-bounds (computed by the transform)
2. Mapping z to x via the transform operator
3. Clamping out-of-bounds x (for rotated transforms)
4. Collecting diagnostics (clamp rate, etc.)

Design
------
This wrapper is specifically for RotatedTR transforms (e.g., LowRankSVD).
AxisAlignedTR transforms use direct X-space optimization and do not need this wrapper.

The transform provides:
- The operator z -> x_delta
- The z-bounds (circumscribed around the feasible polytope)
- Polytope inequality constraints (optional)

Z-Bounds Handling
-----------------
For rotated transforms (LowRankSVD):
- z_bounds are circumscribed around the feasible polytope
- Some samples may produce x outside [0,1]
- These are clamped per-dimension
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from botorch.acquisition import AcquisitionFunction

from .base import RotatedTR, TransformOperator


logger = logging.getLogger("TrustRegionAcquisition")


# =============================================================================
# Diagnostics
# =============================================================================


@dataclass
class TransformDiagnostics:
    """
    Diagnostics from the z -> x transformation.

    Attributes
    ----------
    num_clamped : int
        Number of points that had at least one dimension clamped.
    total_points : int
        Total number of points transformed.
    points_clamped_ratio : float
        Fraction of points with at least one clamped dimension (num_clamped / total_points).
    mean_dims_clamped_ratio : float
        Mean fraction of dimensions clamped per point, averaged over clamped points only.
    max_violation : float
        Maximum distance outside bounds before clamping.
    z_bounds_used : tuple
        The (z_lower, z_upper) bounds actually used.
    """

    num_clamped: int
    total_points: int
    points_clamped_ratio: float
    """Fraction of points that had at least one dimension clamped."""
    mean_dims_clamped_ratio: float
    """Mean fraction of dimensions clamped, averaged over clamped points only."""
    max_violation: float
    z_bounds_used: tuple


# =============================================================================
# Trust Region Wrapped Acquisition Function
# =============================================================================


class TrustRegionWrappedAcquisitionFunction(AcquisitionFunction):
    """
    Wraps a BoTorch AcquisitionFunction to operate in Trust Region latent space Z.

    This wrapper is specifically for RotatedTR transforms (e.g., LowRankSVD).
    AxisAlignedTR transforms use direct X-space optimization and do not need this wrapper.

    This wrapper handles the complete transformation pipeline:
    1. Sample z from constrained bounds [z_lower, z_upper] (from transform)
    2. Scale z by trust region length
    3. Apply the Riemannian operator: x_delta = transform_op(z * length)
    4. Center: x = x_center + x_delta
    5. Clamp to [0, 1]^D (for rotated transforms, some clamping may occur)

    Optimization happens in Z-space where the bounds are axis-aligned
    (even though the transform has rotations).

    Parameters
    ----------
    acq_function : AcquisitionFunction
        Base acquisition function defined on normalized input space X.
    transform : Union[RotatedTR, TransformOperator]
        Either a RotatedTR instance (must be called first) or its computed operator.
    x_center : Tensor
        Center of trust region in [0, 1]^D. Required.
    length : float
        Trust region length. Required.
    warn_on_clamp : bool
        If True, emit warnings when points are clamped.
    collect_diagnostics : bool
        If True, collect transformation diagnostics in self.last_diagnostics.

    Attributes
    ----------
    z_bounds : Tensor
        Bounds for z-space optimization, shape (2, D).
    last_diagnostics : Optional[TransformDiagnostics]
        Diagnostics from the last call to map_z_to_x.

    Examples
    --------
    >>> from botorch.acquisition import qLogExpectedImprovement
    >>> from botorch.optim import optimize_acqf
    >>> from riemannTuRBO import LowRankSVDTransform
    >>>
    >>> # Create transform and compute operator
    >>> transform = LowRankSVDTransform(model, sampler)
    >>> _ = transform(x_center, 0.5)  # Compute operator
    >>>
    >>> # Wrap acquisition function
    >>> acq = qLogExpectedImprovement(model, best_f=Y.max())
    >>> wrapped = TrustRegionWrappedAcquisitionFunction(acq, transform, x_center, 0.5)
    >>>
    >>> # Optimize in z-space (with polytope constraints if available)
    >>> z_opt, _ = optimize_acqf(
    ...     wrapped,
    ...     bounds=wrapped.z_bounds,
    ...     q=1,
    ...     num_restarts=10,
    ...     raw_samples=512,
    ...     inequality_constraints=wrapped.inequality_constraints,
    ... )
    >>> x_opt = wrapped.map_z_to_x(z_opt)
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        transform: Union[RotatedTR, TransformOperator],
        x_center: Optional[Tensor] = None,
        length: Optional[float] = None,
        warn_on_clamp: bool = True,
        collect_diagnostics: bool = False,
    ):
        super().__init__(model=acq_function.model)

        self.acq_function = acq_function
        self.warn_on_clamp = warn_on_clamp
        self.collect_diagnostics = collect_diagnostics
        self.last_diagnostics: Optional[TransformDiagnostics] = None

        # Handle transform input - only RotatedTR is supported
        if isinstance(transform, RotatedTR):
            # Transform must be called first to get operator
            if not hasattr(transform, "_operator") or transform._operator is None:
                raise RuntimeError(
                    "Transform must be called with __call__(x_center, length) before "
                    "creating TrustRegionWrappedAcquisitionFunction"
                )
            operator = transform.operator
        else:
            # TransformOperator was passed directly
            operator = transform

        # x_center and length are required
        if x_center is None:
            raise ValueError("x_center must be provided")
        if length is None:
            raise ValueError("length must be provided")

        self.x_center = x_center
        self.length = length

        # Ensure x_center is (1, D) for internal use
        from .utils import ensure_x_shape_for_posterior

        self.x_center = ensure_x_shape_for_posterior(self.x_center)

        self.transform_operator = operator
        self.transform_op = operator.operator
        self.z_lower = operator.z_bounds_lower
        self.z_upper = operator.z_bounds_upper

        self.dim = self.x_center.shape[-1]
        self.device = self.x_center.device
        self.dtype = self.x_center.dtype

        # Normalized bounds [0, 1]^D for clamping
        self._x_bounds = (
            torch.zeros(self.dim, device=self.device, dtype=self.dtype),
            torch.ones(self.dim, device=self.device, dtype=self.dtype),
        )

        logger.info(f"TrustRegionWrapper: length={self.length:.4f}")

    @property
    def z_bounds(self) -> Tensor:
        """
        Bounds for z-space optimization, shape (2, D).

        Use with optimize_acqf:
            optimize_acqf(wrapped, bounds=wrapped.z_bounds, ...)
        """
        return torch.stack([self.z_lower, self.z_upper])

    @property
    def inequality_constraints(
        self,
    ) -> Optional[list[tuple[Tensor, Tensor, float]]]:
        """
        Polytope inequality constraints for z-space optimization.

        Returns None if constraints were not computed by the transform.
        Otherwise returns list of (indices, coefficients, rhs) tuples for optimize_acqf.

        Use with optimize_acqf:
            constraints = wrapped.inequality_constraints
            if constraints is not None:
                optimize_acqf(wrapped, bounds=wrapped.z_bounds,
                             inequality_constraints=constraints, ...)
            else:
                optimize_acqf(wrapped, bounds=wrapped.z_bounds, ...)
        """
        return self.transform_operator.inequality_constraints

    def map_z_to_x(self, z: Tensor) -> Tensor:
        """
        Map latent z coordinates to input x coordinates.

        Parameters
        ----------
        z : Tensor
            Latent coordinates, should be within [z_lower, z_upper].
            Shape: (..., D) where ... are batch dimensions.

        Returns
        -------
        Tensor
            Input coordinates in [0, 1]^D, same shape as z.

        Notes
        -----
        The transformation is:
            x = clamp(x_center + transform_op(length * z), [0, 1])

        For rotated transforms, some clamping may occur since z_bounds are
        circumscribed around the feasible polytope.
        """
        # Store original shape
        shape_z = z.shape
        z_flat = z.view(-1, shape_z[-1])

        # 1. Scale z by trust region length
        z_scaled = z_flat * self.length

        # 2. Apply Riemannian transformation
        x_delta = self.transform_op(z_scaled)

        # Reshape back
        x_delta = x_delta.view(shape_z)

        # 3. Add center
        x = self.x_center + x_delta

        # 4. Check bounds and clamp
        lower, upper = self._x_bounds

        # NOTE: Use a small tolerance here. In float64 it's common to see
        # ~1e-16 numerical noise that would otherwise mark points as "clamped"
        # (and yield clamp_rate=100%) even though they are effectively in-bounds.
        tol = 1e-12 if x.dtype == torch.float64 else 1e-6

        below = x < (lower - tol)
        above = x > (upper + tol)
        out_of_bounds = below.any(dim=-1) | above.any(dim=-1)

        # Max violation for diagnostics (relative to tolerance-aware bounds)
        max_below = ((lower - tol) - x).clamp(min=0).max().item()
        max_above = (x - (upper + tol)).clamp(min=0).max().item()
        max_violation = max(max_below, max_above)

        # Clamp
        x = torch.clamp(x, min=lower, max=upper)

        # Diagnostics
        dims_clamped = (below | above)  # (N, D) bool
        num_clamped = out_of_bounds.sum().item()
        total_points = out_of_bounds.numel()
        D = x.shape[-1]

        # Mean fraction of dimensions clamped, over clamped points only
        if num_clamped > 0:
            mean_dims_clamped_ratio = float(
                dims_clamped[out_of_bounds].float().mean().item()
            )
        else:
            mean_dims_clamped_ratio = 0.0

        if self.collect_diagnostics:
            self.last_diagnostics = TransformDiagnostics(
                num_clamped=num_clamped,
                total_points=total_points,
                points_clamped_ratio=num_clamped / max(total_points, 1),
                mean_dims_clamped_ratio=mean_dims_clamped_ratio,
                max_violation=max_violation,
                z_bounds_used=(self.z_lower.tolist(), self.z_upper.tolist()),
            )

        if self.warn_on_clamp and num_clamped > 0:
            warnings.warn(
                f"TrustRegionWrapper: {num_clamped}/{total_points} point(s) clamped "
                f"(mean {mean_dims_clamped_ratio:.1%} of dims per clamped point). "
                f"Max violation: {max_violation:.4f}. "
                f"This is expected for rotated transforms (LowRankSVD) since z_bounds "
                f"are circumscribed around the feasible polytope.",
                UserWarning,
                stacklevel=2,
            )

        return x

    def forward(self, z: Tensor) -> Tensor:
        """
        Evaluate acquisition function on latent input z.

        Parameters
        ----------
        z : Tensor
            Latent coordinates, shape (batch, q, D) or (batch, D).

        Returns
        -------
        Tensor
            Acquisition values.

        Notes
        -----
        Uses the Straight-Through Estimator (STE) for boundary clamping:
        - Forward pass: x_clamped (correct GP evaluation at valid points)
        - Backward pass: gradient of x_raw (no zero-gradient at boundary)

        Without STE, hard clamp kills ∂acqf/∂z when x hits [0,1] boundary,
        causing the optimizer to stall (center_distance=0 every iteration).
        STE preserves gradient signal through the boundary, matching
        quantization-aware training practice (Bengio et al., 2013).
        """
        # Compute raw (unclamped) x — needed for gradient computation
        shape_z = z.shape
        z_flat = z.view(-1, shape_z[-1])
        z_scaled = z_flat * self.length
        x_delta = self.transform_op(z_scaled)
        x_delta = x_delta.view(shape_z)
        x_raw = self.x_center + x_delta

        # Clamp for correct forward evaluation (GP must see valid x in [0,1]^D)
        lower, upper = self._x_bounds
        x_clamped = torch.clamp(x_raw, min=lower, max=upper)

        # STE: forward uses x_clamped, backward uses gradient of x_raw
        # x == x_clamped in value, but ∂x/∂z == ∂x_raw/∂z (no dead gradient)
        x = x_raw + (x_clamped - x_raw).detach()

        return self.acq_function(x)


# =============================================================================
# Factory Function
# =============================================================================


def make_trust_region_acqf(
    acq_function: AcquisitionFunction,
    transform: Union[RotatedTR, TransformOperator],
    x_center: Optional[Tensor] = None,
    length: Optional[float] = None,
    warn_on_clamp: bool = True,
    collect_diagnostics: bool = True,
) -> TrustRegionWrappedAcquisitionFunction:
    """
    Factory function to create a Trust Region-wrapped acquisition function.

    This is the recommended way to create wrapped acquisition functions.

    Parameters
    ----------
    acq_function : AcquisitionFunction
        Base acquisition function.
    transform : Union[RotatedTR, TransformOperator]
        Transform instance (auto-computed) or its operator.
    x_center, length, warn_on_clamp, collect_diagnostics
        See TrustRegionWrappedAcquisitionFunction.

    Returns
    -------
    TrustRegionWrappedAcquisitionFunction
        The wrapped acquisition function.

    Examples
    --------
    >>> acq = qLogExpectedImprovement(model, best_f=Y.max())
    >>> transform = LowRankSVDTransform(model, sampler)
    >>> _ = transform(x_best, 0.5)  # Compute operator
    >>> wrapped = make_trust_region_acqf(acq, transform, x_best, 0.5)
    >>> z_opt, _ = optimize_acqf(wrapped, bounds=wrapped.z_bounds, q=1, ...)
    """
    return TrustRegionWrappedAcquisitionFunction(
        acq_function=acq_function,
        transform=transform,
        x_center=x_center,
        length=length,
        warn_on_clamp=warn_on_clamp,
        collect_diagnostics=collect_diagnostics,
    )
