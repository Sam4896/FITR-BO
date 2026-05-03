"""
Test BOUNCE integration: run_bounce_optimization runs and returns the expected result shape.

Uses a small synthetic objective (maximize -||x||^2 on [0,1]^2) so the test is fast.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Ensure repo root and src are on path (conftest does repo_root, src_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
repo_root = os.path.abspath(os.path.join(src_dir, ".."))
for path in (repo_root, src_dir):
    if path not in sys.path:
        sys.path.insert(0, path)


def _dummy_objective(x: torch.Tensor) -> torch.Tensor:
    """Maximize -||x-0.5||^2 on [0,1]^d (peak at 0.5)."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return -(x - 0.5).pow(2).sum(dim=-1, keepdim=True)


def _bounce_available() -> bool:
    if not (Path(src_dir) / "old_src" / "BOUNCE" / "bounce").exists():
        return False
    try:
        import gin  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _bounce_available(), reason="BOUNCE package or gin-config not available")
def test_bounce_run_returns_correct_shape():
    from src.old_src.BOUNCE import run_bounce_optimization

    results = run_bounce_optimization(
        fun=_dummy_objective,
        dim=2,
        max_evals=20,
        n_init=5,
        batch_size=1,
        seed=0,
        device=torch.device("cpu"),
    )

    assert "Y" in results
    assert "X" in results
    assert "n_evals" in results
    assert "best_value" in results
    assert "method_name" in results
    assert results["method_name"] == "bounce"

    Y = results["Y"]
    X = results["X"]
    n_evals = results["n_evals"]
    best_value = results["best_value"]

    assert n_evals == len(Y)
    assert n_evals == len(X)
    assert n_evals >= 5
    assert len(X[0]) == 2
    # Best value should be near 0 (max of -||x-0.5||^2 on [0,1]^2 is 0 at x=0.5)
    assert best_value <= 0.1
    assert best_value >= -1.0


@pytest.mark.skipif(not _bounce_available(), reason="BOUNCE package or gin-config not available")
def test_bounce_callable_benchmark():
    """CallableBenchmark evaluates the callable and exposes continuous [0,1]^d."""
    from src.old_src.BOUNCE.bounce.callable_benchmark import CallableBenchmark

    def f(x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1, keepdim=True)

    bench = CallableBenchmark(dim=3, fun=f)
    assert bench.dim == 3
    assert bench.representation_dim == 3
    assert bench.is_continuous

    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = bench(x)
    assert out.shape == (2,)
    assert out[0].item() == 0.0
    assert out[1].item() == 3.0
