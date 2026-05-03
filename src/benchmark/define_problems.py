"""
DefineProblems - Benchmark Problem Factory
===========================================

This module provides a unified interface for creating benchmark problems
for Bayesian optimization experiments. All problems are wrapped as
BoTorch BaseTestProblem instances, providing a consistent API.

Usage:
------
    from src.benchmark.define_problems import DefineProblems
    import torch
    from botorch.utils.transforms import unnormalize

    # Create a problem instance
    problem = DefineProblems("levy4_100", noise_std=None, negate=True)

    # Evaluate using the public __call__ method (expects unnormalized inputs)
    x = unnormalize(torch.rand(1, problem.dim), problem.bounds)
    y = problem(x)  # Returns tensor of shape (1, 1)

    # For batch evaluation
    x_batch = unnormalize(torch.rand(5, problem.dim), problem.bounds)
    y_batch = problem(x_batch)  # Returns tensor of shape (5, 1)

Available Problems:
------------------

Embedded Synthetic Benchmarks (from "Vanilla BO in High Dimensions"):
    - "levy4_25", "levy4_100", "levy4_300", "levy4_1000"
        Levy function embedded in 25D, 100D, 300D, 1000D
        Dimensions: 25, 100, 300, 1000

    - "hartmann6_25", "hartmann6_100", "hartmann6_300", "hartmann6_1000"
        Hartmann6 function embedded in 25D, 100D, 300D, 1000D
        Dimensions: 25, 100, 300, 1000

LassoBench Benchmarks:
    Synthetic:
    - "lasso_synt_simple" or "lassosyntsimple"
        LassoBench synthetic simple benchmark
        Dimension: 60

    - "lasso_synt_medium" or "lassosyntmedium"
        LassoBench synthetic medium benchmark
        Dimension: 100

    - "lasso_synt_high" or "lassosynthigh"
        LassoBench synthetic high benchmark
        Dimension: 300

    - "lasso_synt_hard" or "lassosynthard"
        LassoBench synthetic hard benchmark
        Dimension: 1000

    Real-World:
    - "lasso_diabetes" or "lassodiabetes"
        Diabetes dataset from LassoBench
        Dimension: 8

    - "lasso_breast_cancer" or "lassobreastcancer"
        Breast Cancer dataset from LassoBench
        Dimension: 10

    - "lasso_leukemia" or "lassoleukemia"
        Leukemia dataset from LassoBench
        Dimension: ~7129 (varies with train/test split)

    - "lasso_rcv1" or "lassorcv1"
        RCV1 dataset from LassoBench
        Dimension: ~47236 (varies with train/test split)

    - "lasso_dna" or "lassodna"
        DNA dataset from LassoBench
        Dimension: 180

    Requires: LassoBench library (included)
    Note: Real-world benchmarks require dataset downloads (SSL fix included)

BenchSuite Benchmarks:
    - "svm"
        Support Vector Machine hyperparameter optimization
        Dimension: 388

    - "lunar_lander" or "lunarlander"
        Lunar Lander control optimization
        Dimension: 12

    - "robot_pushing" or "robotpushing"
        Robot pushing task optimization
        Dimension: 14

    - "swimming"
        MuJoCo Swimmer optimization
        Dimension: 16

    - "hopper"
        MuJoCo Hopper optimization
        Dimension: 33

    - "ant"
        MuJoCo Ant optimization
        Dimension: 888

    - "humanoid"
        MuJoCo Humanoid optimization
        Dimension: 6392

    Requires: BenchSuite library (included)

Existing Benchmarks:
    - "MOPTA08"
        MOPTA08 benchmark
        Dimension: 124

    - "HPA101-2", "HPA102-1", "HPA103-1", "HPA101-0"
        HPA (High-Performance Airfoil) benchmarks
        Dimensions: Varies by problem

    - "RoverTrajectory"
        Rover trajectory optimization
        Dimension: 60 (default, can be specified)

BoTorch Built-in Functions:
    Any BoTorch test function name can be used, e.g.:
    - "Ackley", "Branin", "Rosenbrock", etc.
    Use with dim parameter for high-dimensional versions.

Parameters:
----------
    problem : str
        Name of the problem to create

    dim : int, optional (default=0)
        Dimension for BoTorch built-in functions. Ignored for named problems.

    dim_emb : int, optional (default=0)
        Embedding dimension (currently unused)

    noise_std : float, optional (default=None)
        Standard deviation of observation noise. If None, no noise is added.

    negate : bool, optional (default=True)
        If True, negate the objective (for maximization problems).
        BoTorch optimizers minimize, so set negate=True for maximization.

    n_div : int, optional (default=4)
        Number of divisions for HPA problems.

    seed : int, optional (default=None)
        Random seed for benchmarks that use one (e.g. LassoBench). If None, 42 is used.
        Same seed + same input => repeatable output.

Returns:
-------
    BaseTestProblem
        A BoTorch BaseTestProblem instance with:
        - .dim : problem dimension
        - .bounds : problem bounds (list of (lower, upper) tuples)
        - .__call__(x) : evaluation method (public API)
            Input: torch.Tensor of shape (batch_size, dim) with unnormalized values
            Output: torch.Tensor of shape (batch_size, 1)

Example:
--------
    # Create and evaluate a problem
    problem = DefineProblems("levy4_100", noise_std=None, negate=True)
    print(f"Problem dimension: {problem.dim}")
    print(f"Bounds: {problem.bounds[:3]}...")  # Show first 3 bounds

    # Single point evaluation
    import torch
    from botorch.utils.transforms import unnormalize
    x = unnormalize(torch.rand(1, problem.dim), problem.bounds)
    y = problem(x)
    print(f"Evaluation result: {y.item()}")

    # Batch evaluation
    x_batch = unnormalize(torch.rand(10, problem.dim), problem.bounds)
    y_batch = problem(x_batch)
    print(f"Batch evaluation shape: {y_batch.shape}")

Notes:
------
    - Always use the public __call__ method for evaluation, not _evaluate_true
    - The __call__ method expects unnormalized inputs (in problem bounds)
    - Use botorch.utils.transforms.unnormalize to convert from [0,1]^d to bounds
    - All problems return torch.Tensor outputs
    - For problems requiring external dependencies, ensure they are installed via poetry
"""

import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem

from src.benchmark.mopta08.mopta08 import Mopta08
from src.benchmark.ebo_rover.rover_function import RoverTrajectory
from src.benchmark.hpa.hpa.problem import *
from src.benchmark.vanilla_bo_benchmarks import (
    Levy4_25,
    Levy4_100,
    Levy4_300,
    Levy4_1000,
    Hartmann6_25,
    Hartmann6_100,
    Hartmann6_300,
    Hartmann6_1000,
)
from src.benchmark.lassobench_wrappers import (
    LassoSyntSimple,
    LassoSyntMedium,
    LassoSyntHigh,
    LassoSyntHard,
    LassoDiabetes,
    LassoBreastCancer,
    LassoLeukemia,
    LassoRCV1,
    LassoDNA,
)

# BenchSuite wrappers (if BenchSuite is available)
try:
    from src.benchmark.benchsuite_lassobench_wrappers import (
        SVM,
        LunarLander,
        RobotPushing,
        Swimming,
        Hopper,
        Ant,
        Humanoid,
    )
except ImportError:
    # BenchSuite not available, define dummy classes
    SVM = None
    LunarLander = None
    RobotPushing = None
    Swimming = None
    Hopper = None
    Ant = None
    Humanoid = None

try:
    from src.benchmark.benchsuite_subprocess_wrapper import create_benchsuite_subprocess_problem
except ImportError:
    create_benchsuite_subprocess_problem = None


def _benchsuite_problem(problem_name: str, noise_std=None, negate=True):
    """Use in-process wrapper if available, else subprocess (external BenchSuite project)."""
    if SVM is not None:
        return None  # caller will use the specific class
    if create_benchsuite_subprocess_problem is not None:
        return create_benchsuite_subprocess_problem(problem_name, noise_std=noise_std, negate=negate)
    return None


class BotorchHPA(BaseTestProblem):
    def __init__(
        self,
        problem_name,
        n_div=4,
        level=1,
        NORMALIZED=True,
        noise_std=None,
        negate=True,
    ):
        self.hpa = eval(
            problem_name
            + "(n_div="
            + str(n_div)
            + ", level="
            + str(level)
            + ", NORMALIZED="
            + str(NORMALIZED)
            + ")"
        )
        self.nx = self.hpa.nx
        self.nf = self.hpa.nf
        self.ng = self.hpa.ng
        if NORMALIZED:
            self.lb = np.zeros(self.nx)
            self.ub = np.ones(self.nx)
        else:
            self.lb = self.hpa.lbound
            self.ub = self.hpa.ubound
        self.dim = self.nx
        self._bounds = [(l, u) for (l, u) in zip(self.lb, self.ub)]
        # Set continuous_inds before calling super() for newer BoTorch versions
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=negate)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Handle both single point and batch inputs
        if X.dim() == 1:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]

        results = []
        for i in range(batch_size):
            x = X[i].cpu().numpy().reshape(-1)
            if self.ng > 0:
                f, g = self.hpa(x)
            else:
                f = self.hpa(x)
            # Handle numpy array/scalar conversion
            if isinstance(f, (np.ndarray, np.generic)):
                f = float(f.item() if f.size == 1 else f)
            else:
                f = float(f)
            results.append(f)

        # Return as (batch_size, 1) tensor
        result_tensor = torch.tensor(results, dtype=X.dtype, device=X.device)
        return result_tensor.unsqueeze(-1)


def DefineProblems(problem, dim=0, dim_emb=0, noise_std=None, negate=True, n_div=4, seed=None):
    # LassoBench and other seed-based benchmarks: use given seed or default 42 for repeatability
    _seed = seed if seed is not None else 42
    if "HPA" in problem:
        name = problem[:6]
        level = int(problem[-1])
        return BotorchHPA(name, n_div, level, noise_std=noise_std, negate=True)
    elif "MOPTA08" in problem:
        return Mopta08(noise_std=noise_std, negate=True)
    elif "RoverTrajectory" in problem:
        if len(problem) > 15:
            n_points = int(problem[15:])
        else:
            n_points = 30
        return RoverTrajectory(n=n_points, noise_std=noise_std, negate=False)
    elif problem == "levy4_25":
        return Levy4_25(noise_std=noise_std, negate=negate)
    elif problem == "levy4_100":
        return Levy4_100(noise_std=noise_std, negate=negate)
    elif problem == "levy4_300":
        return Levy4_300(noise_std=noise_std, negate=negate)
    elif problem == "levy4_1000":
        return Levy4_1000(noise_std=noise_std, negate=negate)
    elif problem == "hartmann6_25":
        return Hartmann6_25(noise_std=noise_std, negate=negate)
    elif problem == "hartmann6_100":
        return Hartmann6_100(noise_std=noise_std, negate=negate)
    elif problem == "hartmann6_300":
        return Hartmann6_300(noise_std=noise_std, negate=negate)
    elif problem == "hartmann6_1000":
        return Hartmann6_1000(noise_std=noise_std, negate=negate)
    # LassoBench Synthetic Benchmarks (seed for repeatable train/test splits and data)
    elif problem == "lasso_synt_simple" or problem == "lassosyntsimple":
        return LassoSyntSimple(noise_std=noise_std, negate=negate, seed=_seed)
    elif problem == "lasso_synt_medium" or problem == "lassosyntmedium":
        return LassoSyntMedium(noise_std=noise_std, negate=negate, seed=_seed)
    elif problem == "lasso_synt_high" or problem == "lassosynthigh":
        return LassoSyntHigh(noise_std=noise_std, negate=negate, seed=_seed)
    elif problem == "lasso_synt_hard" or problem == "lassosynthard":
        return LassoSyntHard(noise_std=noise_std, negate=negate, seed=_seed)
    # LassoBench Real-World Benchmarks
    elif problem == "lasso_diabetes" or problem == "lassodiabetes":
        return LassoDiabetes(noise_std=noise_std, negate=negate, seed=_seed)
    elif problem == "lasso_breast_cancer" or problem == "lassobreastcancer":
        return LassoBreastCancer(noise_std=noise_std, negate=negate, seed=_seed)
    elif problem == "lasso_leukemia" or problem == "lassoleukemia":
        return LassoLeukemia(noise_std=noise_std, negate=negate, seed=_seed)
    elif problem == "lasso_rcv1" or problem == "lassorcv1":
        return LassoRCV1(noise_std=noise_std, negate=negate, seed=_seed)
    elif problem == "lasso_dna" or problem == "lassodna":
        return LassoDNA(noise_std=noise_std, negate=negate, seed=_seed)
    # BenchSuite Benchmarks (in-process if available, else subprocess to external project)
    elif problem == "svm":
        sub = _benchsuite_problem("svm", noise_std=noise_std, negate=negate)
        return sub if sub is not None else SVM(noise_std=noise_std, negate=negate)
    elif problem == "lunar_lander" or problem == "lunarlander":
        sub = _benchsuite_problem("lunar_lander", noise_std=noise_std, negate=negate)
        return sub if sub is not None else LunarLander(noise_std=noise_std, negate=negate)
    elif problem == "robot_pushing" or problem == "robotpushing":
        sub = _benchsuite_problem("robot_pushing", noise_std=noise_std, negate=negate)
        return sub if sub is not None else RobotPushing(noise_std=noise_std, negate=negate)
    elif problem == "swimming":
        sub = _benchsuite_problem("swimming", noise_std=noise_std, negate=negate)
        return sub if sub is not None else Swimming(noise_std=noise_std, negate=negate)
    elif problem == "hopper":
        sub = _benchsuite_problem("hopper", noise_std=noise_std, negate=negate)
        return sub if sub is not None else Hopper(noise_std=noise_std, negate=negate)
    elif problem == "ant":
        sub = _benchsuite_problem("ant", noise_std=noise_std, negate=negate)
        return sub if sub is not None else Ant(noise_std=noise_std, negate=negate)
    elif problem == "humanoid":
        sub = _benchsuite_problem("humanoid", noise_std=noise_std, negate=negate)
        return sub if sub is not None else Humanoid(noise_std=noise_std, negate=negate)
    elif problem == "half_cheetah" or problem == "halfcheetah":
        sub = _benchsuite_problem("half_cheetah", noise_std=noise_std, negate=negate)
        if sub is not None:
            return sub
        raise FileNotFoundError(
            "BenchSuite not found. half_cheetah is only available via external BenchSuite "
            "(set BENCHSUITE_ROOT or clone BenchSuite in parent dir of natural_bo)."
        )
    else:
        # Botorch functions
        if dim > 0:
            try:
                fun = eval(
                    problem
                    + "(dim="
                    + str(dim)
                    + ", noise_std="
                    + str(noise_std)
                    + ", negate="
                    + str(negate)
                    + ")"
                )
            except (TypeError, NameError):
                fun = eval(
                    problem
                    + "(noise_std="
                    + str(noise_std)
                    + ", negate="
                    + str(negate)
                    + ")"
                )
            return fun
        else:
            return eval(
                problem
                + "(noise_std="
                + str(noise_std)
                + ", negate="
                + str(negate)
                + ")"
            )
