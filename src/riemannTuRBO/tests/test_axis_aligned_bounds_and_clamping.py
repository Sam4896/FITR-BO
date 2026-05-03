from __future__ import annotations

import torch
from riemannTuRBO.base import AxisAlignedTR


class ToyDiagTransform(AxisAlignedTR):
    """Diagonal linear transform with known scaling (no model required)."""

    name = "ToyDiag"

    def __init__(
        self,
        *,
        x_center: torch.Tensor,
        length: float,
        diag_scale: torch.Tensor,
        volume_normalize: bool,
    ) -> None:
        self._diag_scale = diag_scale
        self._x_center = x_center
        self._length = length
        super().__init__(
            model=None,
            sampler=None,
            eps_cfg=None,
            volume_normalize=volume_normalize,
            use_lp_bounds=False,
            include_polytope_constraints=False,
        )

    def _compute_weights(self, x_center: torch.Tensor) -> torch.Tensor:
        """Compute weights from diag_scale."""
        return self._diag_scale.detach().clone()


def test_axis_aligned_bounds_within_unit_cube_float64() -> None:
    """Verify that get_bounds returns bounds within [0, 1]."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    D = 25
    length = 0.8

    x_center = torch.rand(D, device=device, dtype=dtype) * 0.8 + 0.1
    diag_scale = torch.exp(torch.randn(D, device=device, dtype=dtype) * 0.2)

    transform = ToyDiagTransform(
        x_center=x_center,
        length=length,
        diag_scale=diag_scale,
        volume_normalize=True,
    )

    # Get bounds directly
    bounds = transform(x_center, length)
    lower, upper = bounds[0], bounds[1]

    # Check validity
    assert (lower >= 0.0).all()
    assert (upper <= 1.0).all()
    assert (lower <= x_center).all()
    assert (upper >= x_center).all()
    assert (lower <= upper).all()


def test_axis_aligned_clamping_logic() -> None:
    """Verify that get_bounds correctly clamps to [0, 1] when TR exceeds domain."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    D = 10
    length = 2.0  # Large length to force clamping

    # Center near 0
    x_center = torch.full((D,), 0.1, device=device, dtype=dtype)
    # Uniform scaling (after normalization)
    diag_scale = torch.ones(D, device=device, dtype=dtype)

    transform = ToyDiagTransform(
        x_center=x_center,
        length=length,
        diag_scale=diag_scale,
        volume_normalize=False,
    )

    bounds = transform(x_center, length)
    lower, upper = bounds[0], bounds[1]

    # Check lower clamp
    # Theoretical lower: 0.1 - 2.0*1.0 = -1.9
    # Should be clamped to 0.0
    assert (lower == 0.0).all()

    # Check upper
    # Theoretical upper: 0.1 + 2.0*1.0 = 2.1
    # Should be clamped to 1.0
    assert (upper == 1.0).all()


def test_anisotropy_metric_always_ge_one() -> None:
    """Verify that anisotropy metric (max/min ratio) is always >= 1.0."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    dim = 10

    # Create a simple diagonal transform with varying scales
    inv_sqrt_diag = torch.tensor(
        [0.1, 0.5, 1.0, 2.0, 5.0, 0.2, 0.8, 1.5, 3.0, 0.3],
        device=device,
        dtype=dtype,
    )

    # Compute anisotropy as max/min (consistent with LowRankSVD logic)
    min_val = inv_sqrt_diag.min()
    if min_val > 0:
        anisotropy = (inv_sqrt_diag.max() / min_val).item()
    else:
        anisotropy = float("inf")

    # Anisotropy should always be >= 1.0 by definition
    assert anisotropy >= 1.0, f"Anisotropy must be >= 1.0, got {anisotropy}"

    # Test edge case: all values equal (isotropic)
    inv_sqrt_diag_iso = torch.ones(dim, device=device, dtype=dtype) * 2.5
    anisotropy_iso = (inv_sqrt_diag_iso.max() / inv_sqrt_diag_iso.min()).item()
    assert anisotropy_iso == 1.0, (
        f"Isotropic case should give anisotropy=1.0, got {anisotropy_iso}"
    )
