from __future__ import annotations

import torch

from riemannTuRBO.lowrank_svd import LowRankSVDTransform


def _make_noinit_transform() -> LowRankSVDTransform:
    """Create a LowRankSVDTransform instance without running __init__ (no GP needed)."""
    t = object.__new__(LowRankSVDTransform)
    return t


def test_clamp_eigenvalues_applies_eps_floor() -> None:
    """Test that _clamp_eigenvalues applies minimum eps floor."""
    t = _make_noinit_transform()
    eigs = torch.tensor([1e-8, 1e-4, 1e-2, 1.0], dtype=torch.float64)
    eps = 1e-10
    clamped = t._clamp_eigenvalues(eigs, eps)

    # All eigenvalues should be >= eps
    assert torch.all(clamped >= eps)
    # Values already >= eps should be unchanged
    assert torch.allclose(clamped[1:], eigs[1:])
    # Values < eps should be clamped to eps
    assert torch.isclose(clamped[0], torch.tensor(eps, dtype=eigs.dtype))


def test_clamp_eigenvalues_handles_zeros() -> None:
    """Test that _clamp_eigenvalues handles zero eigenvalues."""
    t = _make_noinit_transform()
    eigs = torch.tensor([0.0, 1.0, 10.0], dtype=torch.float64)
    eps = 1e-6
    clamped = t._clamp_eigenvalues(eigs, eps)

    # Zero should be clamped to eps, others unchanged
    expected = torch.tensor([1e-6, 1.0, 10.0], dtype=torch.float64)
    assert torch.allclose(clamped, expected)
