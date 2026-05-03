"""
BoTorch Wrappers for LassoBench Problems
=========================================

This module provides BoTorch BaseTestProblem wrappers for all LassoBench
synthetic and real-world benchmarks. These wrappers allow LassoBench problems
to be used seamlessly with BoTorch optimization algorithms.

All wrappers:
- Convert inputs from torch.Tensor to numpy arrays
- Handle LassoBench's [-1, 1] input bounds
- Return torch.Tensor outputs compatible with BoTorch
- Support both single point and batch evaluation

Usage:
------
    from src.benchmark.lassobench_wrappers import LassoSyntSimple
    problem = LassoSyntSimple(noise_std=None, negate=True)
    x = torch.rand(1, problem.dim) * 2 - 1  # [-1, 1] bounds
    y = problem(x)
"""

import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem

# Fix SSL for dataset downloads (applied at module level)
try:
    import sys
    import os

    ssl_fix_path = os.path.join(os.path.dirname(__file__), "fix_libsvm_ssl.py")
    if os.path.exists(ssl_fix_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("fix_libsvm_ssl", ssl_fix_path)
        if spec and spec.loader:
            fix_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fix_module)
except (ImportError, Exception):
    # SSL fix not critical, will fail later if needed
    pass

try:
    import LassoBench
except ImportError:
    LassoBench = None


class BaseLassoBenchWrapper(BaseTestProblem):
    """
    Base class for LassoBench wrappers.

    All LassoBench problems use [-1, 1] bounds and return cross-validation loss.
    """

    def __init__(self, lasso_bench, noise_std=None, negate=True):
        """
        Initialize the wrapper.

        Parameters
        ----------
        lasso_bench : LassoBench.SyntheticBenchmark or LassoBench.RealBenchmark
            The LassoBench problem instance
        noise_std : float, optional
            Standard deviation of observation noise (not used by LassoBench)
        negate : bool, optional
            If True, negate the objective (for maximization)
        """
        self.lasso_bench = lasso_bench
        self.dim = lasso_bench.n_features
        # LassoBench uses [-1, 1] bounds
        self._bounds = [(-1.0, 1.0) for _ in range(self.dim)]

        # Set continuous_inds for newer BoTorch versions
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []

        super().__init__(noise_std=noise_std, negate=negate)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the LassoBench problem.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (batch_size, dim) with values in [-1, 1]

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, 1) with cross-validation loss
        """
        # Handle both single point and batch inputs
        if X.dim() == 1:
            X = X.unsqueeze(0)

        batch_size = X.shape[0]
        results = []

        for i in range(batch_size):
            # Convert to numpy array with double precision (LassoBench requirement)
            x_np = X[i].cpu().double().numpy().reshape(-1)

            # Ensure values are in [-1, 1] range
            x_np = np.clip(x_np, -1.0, 1.0)

            # Evaluate using LassoBench
            try:
                loss = self.lasso_bench.evaluate(x_np)
            except Exception as e:
                # If evaluation fails, return a large penalty value
                loss = 1e10
                print(f"Warning: LassoBench evaluation failed: {e}")

            # Convert to float and store
            if isinstance(loss, (np.ndarray, np.generic)):
                loss = float(loss.item() if hasattr(loss, "item") else float(loss))
            else:
                loss = float(loss)

            results.append(loss)

        # Return as (batch_size, 1) tensor
        result_tensor = torch.tensor(results, dtype=X.dtype, device=X.device)
        return result_tensor.unsqueeze(-1)


# ============================================================================
# Synthetic Benchmarks
# ============================================================================


class LassoSyntSimple(BaseLassoBenchWrapper):
    """LassoBench synthetic simple benchmark (60 dimensions)."""

    def __init__(self, noise_std=None, negate=True, noise=False, seed=42):
        if LassoBench is None:
            raise ImportError("LassoBench is not installed. Run: poetry install")
        bench = LassoBench.SyntheticBenchmark(
            pick_bench="synt_simple", noise=noise, seed=seed
        )
        super().__init__(bench, noise_std=noise_std, negate=negate)


class LassoSyntMedium(BaseLassoBenchWrapper):
    """LassoBench synthetic medium benchmark (100 dimensions)."""

    def __init__(self, noise_std=None, negate=True, noise=False, seed=42):
        if LassoBench is None:
            raise ImportError("LassoBench is not installed. Run: poetry install")
        bench = LassoBench.SyntheticBenchmark(
            pick_bench="synt_medium", noise=noise, seed=seed
        )
        super().__init__(bench, noise_std=noise_std, negate=negate)


class LassoSyntHigh(BaseLassoBenchWrapper):
    """LassoBench synthetic high benchmark (300 dimensions)."""

    def __init__(self, noise_std=None, negate=True, noise=False, seed=42):
        if LassoBench is None:
            raise ImportError("LassoBench is not installed. Run: poetry install")
        bench = LassoBench.SyntheticBenchmark(
            pick_bench="synt_high", noise=noise, seed=seed
        )
        super().__init__(bench, noise_std=noise_std, negate=negate)


class LassoSyntHard(BaseLassoBenchWrapper):
    """LassoBench synthetic hard benchmark (1000 dimensions)."""

    def __init__(self, noise_std=None, negate=True, noise=False, seed=42):
        if LassoBench is None:
            raise ImportError("LassoBench is not installed. Run: poetry install")
        bench = LassoBench.SyntheticBenchmark(
            pick_bench="synt_hard", noise=noise, seed=seed
        )
        super().__init__(bench, noise_std=noise_std, negate=negate)


# ============================================================================
# Real-World Benchmarks
# ============================================================================


class LassoDiabetes(BaseLassoBenchWrapper):
    """LassoBench Diabetes dataset benchmark (8 dimensions)."""

    def __init__(self, noise_std=None, negate=True, seed=42):
        if LassoBench is None:
            raise ImportError("LassoBench is not installed. Run: poetry install")
        bench = LassoBench.RealBenchmark(pick_data="diabetes", seed=seed)
        super().__init__(bench, noise_std=noise_std, negate=negate)


class LassoBreastCancer(BaseLassoBenchWrapper):
    """LassoBench Breast Cancer dataset benchmark (10 dimensions)."""

    def __init__(self, noise_std=None, negate=True, seed=42):
        if LassoBench is None:
            raise ImportError("LassoBench is not installed. Run: poetry install")
        bench = LassoBench.RealBenchmark(pick_data="breast_cancer", seed=seed)
        super().__init__(bench, noise_std=noise_std, negate=negate)


class LassoLeukemia(BaseLassoBenchWrapper):
    """LassoBench Leukemia dataset benchmark (7129 dimensions)."""

    def __init__(self, noise_std=None, negate=True, seed=42):
        if LassoBench is None:
            raise ImportError("LassoBench is not installed. Run: poetry install")
        bench = LassoBench.RealBenchmark(pick_data="leukemia", seed=seed)
        super().__init__(bench, noise_std=noise_std, negate=negate)


class LassoRCV1(BaseLassoBenchWrapper):
    """LassoBench RCV1 dataset benchmark (19959 dimensions)."""

    def __init__(self, noise_std=None, negate=True, seed=42):
        if LassoBench is None:
            raise ImportError("LassoBench is not installed. Run: poetry install")
        bench = LassoBench.RealBenchmark(pick_data="rcv1", seed=seed)
        super().__init__(bench, noise_std=noise_std, negate=negate)


class LassoDNA(BaseLassoBenchWrapper):
    """LassoBench DNA dataset benchmark (180 dimensions)."""

    def __init__(self, noise_std=None, negate=True, seed=42):
        if LassoBench is None:
            raise ImportError("LassoBench is not installed. Run: poetry install")
        bench = LassoBench.RealBenchmark(pick_data="dna", seed=seed)
        super().__init__(bench, noise_std=noise_std, negate=negate)
