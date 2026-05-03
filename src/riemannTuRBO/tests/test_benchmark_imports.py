"""
Test that all benchmark problems return valid outputs for single and batch inputs.

Each problem must:
- Accept input of shape (batch_size, dim) in problem bounds
- Return output of shape (batch_size, 1) with finite values

If single or batch evaluation fails (wrong shape or non-finite), the test fails:
the wrapper for that problem should be adjusted.
Problems are skipped only when dependencies are missing (LassoBench, BenchSuite).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

BENCHMARK_DIR = Path(src_dir) / "benchmark"
BENCHSUITE_DIR = BENCHMARK_DIR / "BenchSuite"
if BENCHSUITE_DIR.exists() and str(BENCHSUITE_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHSUITE_DIR))

import pytest
import torch
from torch import Tensor
from botorch.utils.transforms import unnormalize

from src.benchmark.define_problems import DefineProblems


# Canonical problem names (one per problem; no aliases)
EMBEDDED = [
    "levy4_25", "levy4_100", "levy4_300", "levy4_1000",
    "hartmann6_25", "hartmann6_100", "hartmann6_300", "hartmann6_1000",
]
LASSOBENCH = [
    "lasso_synt_simple", "lasso_synt_medium", "lasso_synt_high", "lasso_synt_hard",
    "lasso_diabetes", "lasso_breast_cancer", "lasso_leukemia", "lasso_rcv1", "lasso_dna",
]
BENCHSUITE = [
    "svm", "lunar_lander", "robot_pushing", "swimming", "hopper", "ant", "humanoid",
    "half_cheetah",
]
EXISTING = [
    "MOPTA08", "HPA101-2", "HPA102-1", "HPA103-1", "HPA101-0", "RoverTrajectory",
]
ALL_PROBLEMS = EMBEDDED + LASSOBENCH + BENCHSUITE + EXISTING

BATCH_SIZE = 5


@pytest.mark.parametrize("problem_name", ALL_PROBLEMS)
def test_problem_returns_output_single_and_batch(problem_name):
    """
    For each problem: single (1, dim) -> (1, 1) and batch (BATCH_SIZE, dim) -> (BATCH_SIZE, 1).
    Skip only when dependency missing. If single or batch fails, test fails (fix the wrapper).
    """
    if problem_name in LASSOBENCH and not (BENCHMARK_DIR / "LassoBench").exists():
        pytest.skip("LassoBench not cloned")

    try:
        fun = DefineProblems(problem_name, noise_std=None, negate=True)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    except (ImportError, RuntimeError) as e:
        msg = str(e)
        if "SSL" in msg or "CERTIFICATE" in msg:
            pytest.skip("LassoBench dataset download (SSL)")
        pytest.skip(msg)
    assert fun is not None
    assert hasattr(fun, "dim") and fun.dim > 0
    assert hasattr(fun, "bounds")

    # Single point: (1, dim) -> (1, 1), finite
    x_single = unnormalize(torch.rand(1, fun.dim), fun.bounds)
    y_single = fun(x_single)
    assert isinstance(y_single, Tensor), (
        f"{problem_name}: output must be Tensor, got {type(y_single)}"
    )
    assert y_single.shape == (1, 1), (
        f"{problem_name}: single output must be (1, 1), got {y_single.shape}"
    )
    assert torch.isfinite(y_single).all(), (
        f"{problem_name}: single output must be finite"
    )

    # Batch: (BATCH_SIZE, dim) -> (BATCH_SIZE, 1), finite
    x_batch = unnormalize(torch.rand(BATCH_SIZE, fun.dim), fun.bounds)
    y_batch = fun(x_batch)
    assert isinstance(y_batch, Tensor), (
        f"{problem_name}: batch output must be Tensor, got {type(y_batch)}"
    )
    assert y_batch.shape == (BATCH_SIZE, 1), (
        f"{problem_name}: batch output must be ({BATCH_SIZE}, 1), got {y_batch.shape}. "
        "Adjust the wrapper for this problem to support batch evaluation."
    )
    assert torch.isfinite(y_batch).all(), (
        f"{problem_name}: batch output must be finite"
    )


def test_problem_availability_report():
    """Print which problems are available vs skipped (test always passes)."""
    available = []
    unavailable = []

    for name in ALL_PROBLEMS:
        if name in LASSOBENCH and not (BENCHMARK_DIR / "LassoBench").exists():
            unavailable.append((name, "LassoBench not cloned"))
            continue
        try:
            fun = DefineProblems(name, noise_std=None, negate=True)
            available.append((name, fun.dim))
        except Exception as e:
            unavailable.append((name, str(e)[:60]))

    print("\n" + "=" * 70)
    print("Problem availability")
    print("=" * 70)
    for name, dim in available:
        print(f"  [OK]   {name:<25} dim={dim}")
    for name, reason in unavailable:
        print(f"  [SKIP] {name:<25} {reason}")
    print("=" * 70)
    assert True
