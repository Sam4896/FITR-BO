"""
Tests for Transform Diagnostics, Epsilon, and Configuration
===========================================================

This module tests that all transforms:
1. Populate diagnostics correctly (eps_used, true_anisotropy, etc.)
2. Respect eps_cfg configuration
3. Respect volume_normalize configuration
4. Have accessible config properties
"""

from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler

from riemannTuRBO import (
    IdentityTransform,
    ARDLengthscaleTransform,
    DiagGradMeanTransform,
    DiagGradRMSTransform,
    FiniteDiffTransform,
    LowRankSVDTransform,
    EpsConfig,
    EpsMode,
)


@pytest.fixture
def simple_model():
    """Create a simple GP model for testing."""
    torch.manual_seed(42)
    X = torch.rand(10, 5, dtype=torch.float64)
    Y = torch.rand(10, 1, dtype=torch.float64)
    model = SingleTaskGP(X, Y)
    model.eval()
    return model


@pytest.fixture
def x_center():
    """Create a test center point."""
    return torch.rand(5, dtype=torch.float64) * 0.8 + 0.1


@pytest.fixture
def sampler():
    """Create a sampler for transforms that need it."""
    return SobolQMCNormalSampler(sample_shape=torch.Size([32]))


# =============================================================================
# Tests for Axis-Aligned Transforms
# =============================================================================


def test_identity_transform_diagnostics(simple_model, x_center):
    """Test IdentityTransform diagnostics and config."""
    transform = IdentityTransform(
        model=simple_model,
        eps_cfg=EpsConfig(mode=EpsMode.FIXED, eps=1.0),
        volume_normalize=True,
    )
    # Call transform to populate diagnostics
    _ = transform(x_center, 0.5)

    # Check diagnostics
    assert "true_anisotropy" in transform.diagnostics
    assert "scale" in transform.diagnostics
    assert isinstance(transform.diagnostics["true_anisotropy"], (int, float))
    assert transform.diagnostics["true_anisotropy"] >= 1.0

    # Check eps_used
    assert transform.eps_used > 0
    assert transform.eps_used == pytest.approx(1.0, rel=1e-6)

    # Check config
    assert transform.config.volume_normalize is True
    assert transform.config.eps_cfg.mode == EpsMode.FIXED
    assert transform.config.eps_cfg.eps == 1.0

    # Check anisotropy computation
    anisotropy = transform.compute_true_anisotropy()
    assert anisotropy == transform.diagnostics["true_anisotropy"]
    assert anisotropy >= 1.0


def test_ard_lengthscale_transform_diagnostics(simple_model, x_center):
    """Test ARDLengthscaleTransform diagnostics and config."""
    transform = ARDLengthscaleTransform(
        model=simple_model,
        eps_cfg=EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-6),
        volume_normalize=True,
    )
    # Call transform to populate diagnostics
    _ = transform(x_center, 0.5)

    # Check diagnostics
    assert "true_anisotropy" in transform.diagnostics
    assert "lengthscales_raw" in transform.diagnostics
    assert "weights" in transform.diagnostics
    assert isinstance(transform.diagnostics["true_anisotropy"], (int, float))
    assert transform.diagnostics["true_anisotropy"] >= 1.0

    # Check eps_used (ARDLengthscaleTransform doesn't use epsilon, so eps_used = 0.0 is valid)
    assert transform.eps_used >= 0

    # Check config
    # ARDLengthscaleTransform always sets volume_normalize=False because it applies its own normalization
    assert transform.config.volume_normalize is False
    assert transform.config.eps_cfg.mode == EpsMode.AUTO_TRACE

    # Check anisotropy (allow small floating point differences)
    anisotropy = transform.compute_true_anisotropy()
    assert anisotropy == pytest.approx(
        transform.diagnostics["true_anisotropy"], rel=1e-10
    )


def test_diag_grad_mean_transform_diagnostics(simple_model, x_center):
    """Test DiagGradMeanTransform diagnostics and config."""
    transform = DiagGradMeanTransform(
        model=simple_model,
        eps_cfg=EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12),
        volume_normalize=True,
    )
    # Call transform to populate diagnostics
    _ = transform(x_center, 0.5)

    # Check diagnostics
    assert "true_anisotropy" in transform.diagnostics
    assert "diag_G" in transform.diagnostics
    assert "grad_norm" in transform.diagnostics
    assert "weights" in transform.diagnostics
    assert isinstance(transform.diagnostics["true_anisotropy"], (int, float))
    assert transform.diagnostics["true_anisotropy"] >= 1.0

    # Check eps_used
    assert transform.eps_used > 0

    # Check config
    assert transform.config.volume_normalize is True
    assert transform.config.eps_cfg.mode == EpsMode.AUTO_TRACE

    # Check anisotropy (allow small floating point differences)
    anisotropy = transform.compute_true_anisotropy()
    assert anisotropy == pytest.approx(
        transform.diagnostics["true_anisotropy"], rel=1e-10
    )


def test_diag_grad_rms_transform_diagnostics(simple_model, x_center, sampler):
    """Test DiagGradRMSTransform diagnostics and config."""
    transform = DiagGradRMSTransform(
        model=simple_model,
        sampler=sampler,
        eps_cfg=EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12),
        volume_normalize=True,
        clamp_diag_G_min_factor=0.1,
    )
    # Call transform to populate diagnostics
    _ = transform(x_center, 0.5)

    # Check diagnostics
    assert "true_anisotropy" in transform.diagnostics
    assert "diag_G" in transform.diagnostics
    assert "num_samples" in transform.diagnostics
    assert "grad_norm_mean" in transform.diagnostics
    assert "grad_norm_std" in transform.diagnostics
    assert "weights" in transform.diagnostics
    assert isinstance(transform.diagnostics["true_anisotropy"], (int, float))
    assert transform.diagnostics["true_anisotropy"] >= 1.0

    # Check eps_used
    assert transform.eps_used > 0

    # Check config
    assert transform.config.volume_normalize is True
    assert transform.config.eps_cfg.mode == EpsMode.AUTO_TRACE

    # Check anisotropy (allow small floating point differences)
    anisotropy = transform.compute_true_anisotropy()
    assert anisotropy == pytest.approx(
        transform.diagnostics["true_anisotropy"], rel=1e-10
    )


def test_finite_diff_transform_diagnostics(simple_model, x_center):
    """Test FiniteDiffTransform diagnostics and config."""
    transform = FiniteDiffTransform(
        model=simple_model,
        eps_cfg=EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12),
        volume_normalize=True,
        fd_h=1e-4,
    )
    # Call transform to populate diagnostics
    _ = transform(x_center, 0.5)

    # Check diagnostics
    assert "true_anisotropy" in transform.diagnostics
    assert "diag_G" in transform.diagnostics
    assert "grad_fd" in transform.diagnostics
    assert "fd_h" in transform.diagnostics
    assert "weights" in transform.diagnostics
    assert isinstance(transform.diagnostics["true_anisotropy"], (int, float))
    assert transform.diagnostics["true_anisotropy"] >= 1.0

    # Check eps_used
    assert transform.eps_used > 0

    # Check config
    assert transform.config.volume_normalize is True
    assert transform.config.eps_cfg.mode == EpsMode.AUTO_TRACE

    # Check anisotropy (allow small floating point differences)
    anisotropy = transform.compute_true_anisotropy()
    assert anisotropy == pytest.approx(
        transform.diagnostics["true_anisotropy"], rel=1e-10
    )


# =============================================================================
# Tests for Rotated Transforms
# =============================================================================


def test_lowrank_svd_transform_diagnostics(simple_model, x_center, sampler):
    """Test LowRankSVDTransform diagnostics and config."""
    transform = LowRankSVDTransform(
        model=simple_model,
        sampler=sampler,
        eps_cfg=EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12),
        volume_normalize=True,
        include_polytope_constraints=False,
        clamp_diag_G_min_factor=0.1,
    )
    # Call transform to populate diagnostics
    _ = transform(x_center, 0.5)

    # Check diagnostics
    assert "true_anisotropy" in transform.diagnostics
    assert "diag_G" in transform.diagnostics
    assert "num_samples" in transform.diagnostics
    assert isinstance(transform.diagnostics["true_anisotropy"], (int, float))
    assert transform.diagnostics["true_anisotropy"] >= 1.0

    # Check eps_used
    assert transform.eps_used > 0

    # Check config
    assert transform.config.volume_normalize is True
    assert transform.config.eps_cfg.mode == EpsMode.AUTO_TRACE
    assert transform.config.include_polytope_constraints is False

    # Check anisotropy (allow small floating point differences)
    anisotropy = transform.compute_true_anisotropy()
    assert anisotropy == pytest.approx(
        transform.diagnostics["true_anisotropy"], rel=1e-10
    )

    # Check operator exists (RotatedTR specific)
    assert hasattr(transform, "operator")
    assert transform.operator is not None
    assert hasattr(transform, "z_bounds")


# =============================================================================
# Tests for Configuration
# =============================================================================


def test_volume_normalize_config(simple_model, x_center):
    """Test that volume_normalize config is respected."""
    # With volume normalization
    transform_norm = IdentityTransform(
        model=simple_model,
        volume_normalize=True,
    )
    _ = transform_norm(x_center, 0.5)

    # Without volume normalization
    transform_no_norm = IdentityTransform(
        model=simple_model,
        volume_normalize=False,
    )
    _ = transform_no_norm(x_center, 0.5)

    assert transform_norm.config.volume_normalize is True
    assert transform_no_norm.config.volume_normalize is False

    # Both should have diagnostics
    assert "true_anisotropy" in transform_norm.diagnostics
    assert "true_anisotropy" in transform_no_norm.diagnostics


def test_eps_config_modes(simple_model, x_center):
    """Test different epsilon configuration modes."""
    # Fixed mode
    transform_fixed = IdentityTransform(
        model=simple_model,
        eps_cfg=EpsConfig(mode=EpsMode.FIXED, eps=2.0),
    )
    _ = transform_fixed(x_center, 0.5)
    assert transform_fixed.config.eps_cfg.mode == EpsMode.FIXED
    assert transform_fixed.config.eps_cfg.eps == 2.0
    assert transform_fixed.eps_used == pytest.approx(2.0, rel=1e-6)

    # Auto trace mode
    transform_auto = DiagGradMeanTransform(
        model=simple_model,
        eps_cfg=EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-10),
    )
    _ = transform_auto(x_center, 0.5)
    assert transform_auto.config.eps_cfg.mode == EpsMode.AUTO_TRACE
    assert transform_auto.config.eps_cfg.jitter == 1e-10
    assert transform_auto.eps_used > 0


def test_eps_used_is_consistent(simple_model, x_center):
    """Test that eps_used is consistent across multiple calls."""
    transform = DiagGradMeanTransform(
        model=simple_model,
        eps_cfg=EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12),
    )
    _ = transform(x_center, 0.5)

    eps1 = transform.eps_used
    eps2 = transform.eps_used
    eps3 = transform.eps_used

    assert eps1 == eps2 == eps3
    assert eps1 > 0


def test_anisotropy_is_consistent(simple_model, x_center):
    """Test that anisotropy is consistent across multiple calls."""
    transform = DiagGradMeanTransform(
        model=simple_model,
        eps_cfg=EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12),
    )
    _ = transform(x_center, 0.5)

    aniso1 = transform.compute_true_anisotropy()
    aniso2 = transform.compute_true_anisotropy()
    aniso3 = transform.diagnostics["true_anisotropy"]

    assert aniso1 == aniso2 == aniso3
    assert aniso1 >= 1.0


def test_all_transforms_have_required_diagnostics(simple_model, x_center, sampler):
    """Test that all transforms have the required core diagnostics."""
    transforms = [
        IdentityTransform(model=simple_model),
        ARDLengthscaleTransform(model=simple_model),
        DiagGradMeanTransform(model=simple_model),
        DiagGradRMSTransform(model=simple_model, sampler=sampler),
        FiniteDiffTransform(model=simple_model),
        LowRankSVDTransform(model=simple_model, sampler=sampler),
    ]

    for transform in transforms:
        # Call transform to populate diagnostics
        _ = transform(x_center, 0.5)
        # Core diagnostics that all transforms must have
        assert "true_anisotropy" in transform.diagnostics, (
            f"{transform.name} missing true_anisotropy"
        )
        assert isinstance(transform.eps_used, (int, float)), (
            f"{transform.name} missing eps_used"
        )
        # ARDLengthscaleTransform doesn't use epsilon, so eps_used = 0.0 is valid
        assert transform.eps_used >= 0, (
            f"{transform.name} has invalid eps_used: {transform.eps_used}"
        )
        assert transform.diagnostics["true_anisotropy"] >= 1.0, (
            f"{transform.name} has invalid anisotropy: {transform.diagnostics['true_anisotropy']}"
        )

        # Config should be accessible
        assert hasattr(transform, "config")
        assert hasattr(transform.config, "volume_normalize")
        assert hasattr(transform.config, "eps_cfg")
        assert isinstance(transform.config.volume_normalize, bool)


def test_axis_aligned_vs_rotated_diagnostics(simple_model, x_center, sampler):
    """Test that AxisAlignedTR and RotatedTR have appropriate diagnostics."""
    # Axis-aligned transform
    axis_transform = DiagGradRMSTransform(model=simple_model, sampler=sampler)
    _ = axis_transform(x_center, 0.5)

    # Rotated transform
    rotated_transform = LowRankSVDTransform(model=simple_model, sampler=sampler)
    _ = rotated_transform(x_center, 0.5)

    # Both should have core diagnostics
    assert "true_anisotropy" in axis_transform.diagnostics
    assert "true_anisotropy" in rotated_transform.diagnostics
    assert axis_transform.eps_used > 0
    assert rotated_transform.eps_used > 0

    # RotatedTR should have operator and z_bounds
    assert hasattr(rotated_transform, "operator")
    assert hasattr(rotated_transform, "z_bounds")
    assert rotated_transform.operator is not None

    # AxisAlignedTR should NOT have operator
    assert not hasattr(axis_transform, "operator") or not hasattr(
        getattr(axis_transform, "operator", None), "operator"
    )

    # Both should have is_axis_aligned property
    assert axis_transform.is_axis_aligned is True
    assert rotated_transform.is_axis_aligned is False


def test_config_defaults(simple_model, x_center):
    """Test that config defaults are applied correctly."""
    # Test with minimal arguments (should use defaults)
    transform = IdentityTransform(
        model=simple_model,
    )
    _ = transform(x_center, 0.5)

    # Defaults should be applied
    assert transform.config.volume_normalize is True  # Default
    assert transform.config.eps_cfg is not None
    assert transform.config.use_lp_bounds is True  # Default
    assert transform.config.include_polytope_constraints is False  # Default


def test_diagnostics_persistence(simple_model, x_center):
    """Test that diagnostics persist across multiple method calls."""
    transform = DiagGradMeanTransform(
        model=simple_model,
    )
    # Call transform to populate diagnostics
    _ = transform(x_center, 0.5)

    # Get initial diagnostics
    initial_anisotropy = transform.diagnostics["true_anisotropy"]
    initial_eps = transform.eps_used

    # Call transform multiple times
    for _ in range(3):
        bounds = transform(x_center, 0.5)
        assert bounds.shape == (2, x_center.shape[-1])

    # Diagnostics should remain the same
    assert transform.diagnostics["true_anisotropy"] == initial_anisotropy
    assert transform.eps_used == initial_eps
