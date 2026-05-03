"""
Tests for REI center selector with grad_rms and ard_lengthscale transforms.

Verifies that:
1. DiagGradRMSTransform (grad_rms) works with REI selector when the TR
   starts for the first time and when it restarts after length collapse.
2. ARDLengthscaleTransform (ard_lengthscale) works with REI selector in
   the same two scenarios.
"""
from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

from riemannTuRBO import TurboState, get_center_selector


# Lazy import to avoid pulling experiments before path is set
def _get_generate_riemannian_batch():
    from experiments.riemann_turbo import generate_riemannian_batch
    return generate_riemannian_batch


@pytest.fixture
def small_model_and_data():
    """Small GP and data for fast REI + transform tests."""
    torch.manual_seed(42)
    dim = 4
    n = 12
    X = torch.rand(n, dim, dtype=torch.float64)
    Y = torch.randn(n, 1, dtype=torch.float64)
    Y_std = (Y - Y.mean()) / (Y.std() + 1e-8)
    model = SingleTaskGP(X, Y_std, outcome_transform=Standardize(m=1))
    model.eval()
    return model, X, Y_std, dim


@pytest.fixture
def rei_selector_small():
    """REI selector (TuRBO-RLogEI) with smaller optimizer budget for fast tests."""
    return get_center_selector(
        "rei",
        n_region=32,
        num_restarts=4,
        raw_samples=64,
        optimizer_options={"batch_limit": 2, "maxiter": 50},
    )


# -----------------------------------------------------------------------------
# grad_rms (DiagGradRMSTransform) + REI
# -----------------------------------------------------------------------------


def test_grad_rms_rei_first_time(small_model_and_data, rei_selector_small):
    """grad_rms + REI: batch generation works when TR starts for the first time."""
    model, X, Y, dim = small_model_and_data
    state = TurboState(dim=dim, q=1, length=0.5)
    generate_riemannian_batch = _get_generate_riemannian_batch()

    X_next, diagnostics = generate_riemannian_batch(
        state=state,
        model=model,
        X=X,
        Y=Y,
        q=1,
        n_candidates=100,
        num_restarts=4,
        raw_samples=32,
        acqf="ts",
        qmc_sample_shape=32,
        center_selector=rei_selector_small,
        transform_method="diag_grad_rms",
        return_diagnostics=True,
    )

    assert X_next.shape == (1, dim)
    assert torch.all(X_next >= 0) and torch.all(X_next <= 1)
    assert "center" in diagnostics
    assert diagnostics["center"].shape == (1, dim)


def test_grad_rms_rei_after_restart(small_model_and_data, rei_selector_small):
    """grad_rms + REI: batch generation works when TR restarts after length collapse."""
    model, X, Y, dim = small_model_and_data
    state = TurboState(dim=dim, q=1, length=0.5)
    generate_riemannian_batch = _get_generate_riemannian_batch()

    restart_selector = get_center_selector("restart", num_samples=64)

    X_next, diagnostics = generate_riemannian_batch(
        state=state,
        model=model,
        X=X,
        Y=Y,
        q=1,
        n_candidates=100,
        num_restarts=4,
        raw_samples=32,
        acqf="ts",
        qmc_sample_shape=32,
        center_selector=restart_selector,
        transform_method="diag_grad_rms",
        return_diagnostics=True,
    )

    assert X_next.shape == (1, dim)
    assert torch.all(X_next >= 0) and torch.all(X_next <= 1)
    assert "center" in diagnostics
    assert diagnostics["center"].shape == (1, dim)


# -----------------------------------------------------------------------------
# ard_lengthscale (ARDLengthscaleTransform) + REI
# -----------------------------------------------------------------------------


def test_ard_lengthscale_rei_first_time(small_model_and_data, rei_selector_small):
    """ard_lengthscale + REI: batch generation works when TR starts for the first time."""
    model, X, Y, dim = small_model_and_data
    state = TurboState(dim=dim, q=1, length=0.5)
    generate_riemannian_batch = _get_generate_riemannian_batch()

    X_next, diagnostics = generate_riemannian_batch(
        state=state,
        model=model,
        X=X,
        Y=Y,
        q=1,
        n_candidates=100,
        num_restarts=4,
        raw_samples=32,
        acqf="ts",
        qmc_sample_shape=32,
        center_selector=rei_selector_small,
        transform_method="ard_lengthscale",
        return_diagnostics=True,
    )

    assert X_next.shape == (1, dim)
    assert torch.all(X_next >= 0) and torch.all(X_next <= 1)
    assert "center" in diagnostics
    assert diagnostics["center"].shape == (1, dim)


def test_ard_lengthscale_rei_after_restart(small_model_and_data, rei_selector_small):
    """ard_lengthscale + REI: batch generation works when TR restarts after length collapse."""
    model, X, Y, dim = small_model_and_data
    state = TurboState(dim=dim, q=1, length=0.5)
    generate_riemannian_batch = _get_generate_riemannian_batch()

    restart_selector = get_center_selector("restart", num_samples=64)

    X_next, diagnostics = generate_riemannian_batch(
        state=state,
        model=model,
        X=X,
        Y=Y,
        q=1,
        n_candidates=100,
        num_restarts=4,
        raw_samples=32,
        acqf="ts",
        qmc_sample_shape=32,
        center_selector=restart_selector,
        transform_method="ard_lengthscale",
        return_diagnostics=True,
    )

    assert X_next.shape == (1, dim)
    assert torch.all(X_next >= 0) and torch.all(X_next <= 1)
    assert "center" in diagnostics
    assert diagnostics["center"].shape == (1, dim)
