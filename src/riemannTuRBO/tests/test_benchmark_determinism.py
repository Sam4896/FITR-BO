"""
Test deterministic nature of all benchmark problems from define_problems.py.

- Without seed: problem is created with seed=None; same input is evaluated twice
  on the same instance. We only assert that both evaluations succeed and return
  valid outputs (no assertion that outputs are equal).
- With seed: two problem instances are created with the same explicit seed;
  the same input is evaluated on both. We assert that the two outputs are equal
  (determinism when seed is set).
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


# Canonical problem names (must match test_benchmark_imports.ALL_PROBLEMS)
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

# Fixed seeds for reproducible inputs and for determinism test
INPUT_SEED = 0
DETERMINISM_SEED = 12345

# Tolerance for float equality when asserting determinism
RTOL = 1e-5
ATOL = 1e-6

# BenchSuite RL tasks use environment rollouts that are not seeded via DefineProblems;
# same seed + same input can still yield different rewards. We only assert determinism
# for problems that support it.
NON_DETERMINISTIC_EVEN_WITH_SEED = set(BENCHSUITE)


def _skip_if_unavailable(problem_name: str):
    """Raise pytest.skip if problem dependencies are missing or creation fails."""
    if problem_name in LASSOBENCH and not (BENCHMARK_DIR / "LassoBench").exists():
        pytest.skip("LassoBench not cloned")
    try:
        DefineProblems(problem_name, noise_std=None, negate=True, seed=42)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    except (ImportError, RuntimeError) as e:
        msg = str(e)
        if "SSL" in msg or "CERTIFICATE" in msg:
            pytest.skip("LassoBench dataset download (SSL)")
        pytest.skip(msg)


@pytest.mark.parametrize("problem_name", ALL_PROBLEMS)
def test_determinism_without_seed(problem_name):
    """
    Without setting the seed: create problem with seed=None, evaluate the same
    input twice on the same instance. Only assert that both evaluations succeed
    and return valid (finite) outputs; do not require outputs to be equal.
    """
    _skip_if_unavailable(problem_name)

    fun = DefineProblems(problem_name, noise_std=None, negate=True, seed=None)
    assert fun is not None and hasattr(fun, "dim") and fun.dim > 0

    torch.manual_seed(INPUT_SEED)
    x = unnormalize(torch.rand(1, fun.dim), fun.bounds)

    y1 = fun(x)
    y2 = fun(x)

    for name, y in [("first", y1), ("second", y2)]:
        assert isinstance(y, Tensor), (
            f"{problem_name}: {name} output must be Tensor, got {type(y)}"
        )
        assert y.shape == (1, 1), (
            f"{problem_name}: {name} output must be (1, 1), got {y.shape}"
        )
        assert torch.isfinite(y).all(), (
            f"{problem_name}: {name} output must be finite"
        )


@pytest.mark.parametrize("problem_name", ALL_PROBLEMS)
def test_determinism_with_seed(problem_name):
    """
    With setting the seed: create two problem instances with the same explicit
    seed, evaluate the same input on both. For problems that support it (all
    except BenchSuite RL tasks), assert that the two outputs are equal.
    BenchSuite tasks are not required to be deterministic (env rollouts not seeded).
    """
    _skip_if_unavailable(problem_name)

    fun1 = DefineProblems(
        problem_name, noise_std=None, negate=True, seed=DETERMINISM_SEED
    )
    fun2 = DefineProblems(
        problem_name, noise_std=None, negate=True, seed=DETERMINISM_SEED
    )
    assert fun1.dim == fun2.dim

    torch.manual_seed(INPUT_SEED)
    x = unnormalize(torch.rand(1, fun1.dim), fun1.bounds)

    y1 = fun1(x)
    y2 = fun2(x)

    assert isinstance(y1, Tensor) and isinstance(y2, Tensor)
    assert y1.shape == (1, 1) and y2.shape == (1, 1)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all(), (
        f"{problem_name}: outputs must be finite"
    )
    if problem_name not in NON_DETERMINISTIC_EVEN_WITH_SEED:
        assert torch.allclose(y1, y2, rtol=RTOL, atol=ATOL), (
            f"{problem_name}: same seed and same input should give same output; "
            f"got y1={y1.item():.10f} vs y2={y2.item():.10f}"
        )
