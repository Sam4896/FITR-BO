"""
Run BenchSuite benchmarks via subprocess (main.py) when the in-process
wrapper is not available. BenchSuite can live in a separate project with its
own Python venv (e.g. parent dir of natural_bo); this project never imports
BenchSuite or adds it to sys.path.

Search order for BenchSuite root:
  1. Environment variable BENCHSUITE_ROOT (absolute path)
  2. Parent of natural_bo repo: <natural_bo_parent>/BenchSuite
  3. Inside this repo: src/benchmark/BenchSuite (optional)

Usage: DefineProblems("swimming", ...) uses this when benchsuite_lassobench_wrappers
fails to import but BenchSuite is found at one of the above locations.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem


# natural_bo repo root: this file is in .../natural_bo/src/benchmark/
_THIS_FILE = Path(__file__).resolve()
_NATURAL_BO_ROOT = _THIS_FILE.parent.parent.parent

# (main.py --name, dim, lb, ub). main.py expects x in [0,1]; we convert from [lb,ub].
BENCHSUITE_SUBPROCESS_CONFIG = {
    "swimming": ("swimmer", 16, -1.0, 1.0),
    "hopper": ("hopper", 33, -1.4, 1.4),
    "ant": ("ant", 888, -1.0, 1.0),
    "humanoid": ("humanoid", 6392, -1.0, 1.0),
    "half_cheetah": ("halfcheetah", 102, -1.0, 1.0),
    "halfcheetah": ("halfcheetah", 102, -1.0, 1.0),
    "lunar_lander": ("lunarlander", 12, 0.0, 1.0),
    "lunarlander": ("lunarlander", 12, 0.0, 1.0),
    "robot_pushing": ("robotpushing", 14, 0.0, 1.0),
    "robotpushing": ("robotpushing", 14, 0.0, 1.0),
    "svm": ("svm", 388, 0.0, 1.0),
}

# Per-problem subprocess timeout (seconds). SVM and other slow benchmarks need more than default.
BENCHSUITE_SUBPROCESS_TIMEOUT = {
    "svm": 600,  # 388 dims; single evaluation can exceed 120s
}
DEFAULT_SUBPROCESS_TIMEOUT = 120


def get_benchsuite_dir() -> Path | None:
    """
    Return path to BenchSuite project root (where main.py lives).
    Does NOT add anything to sys.path; used only for subprocess.
    """
    # 1. Explicit env (e.g. export BENCHSUITE_ROOT=/home/user/BenchSuite)
    root = os.environ.get("BENCHSUITE_ROOT")
    if root:
        p = Path(root).resolve()
        if p.is_dir() and (p / "main.py").is_file():
            return p
    # 2. Sibling of natural_bo: <parent>/BenchSuite
    parent_benchsuite = _NATURAL_BO_ROOT.parent / "BenchSuite"
    if parent_benchsuite.is_dir() and (parent_benchsuite / "main.py").is_file():
        return parent_benchsuite
    # 3. Inside this repo (optional)
    in_repo = _THIS_FILE.parent / "BenchSuite"
    if in_repo.is_dir() and (in_repo / "main.py").is_file():
        return in_repo
    return None


def _env_with_mujoco(benchsuite_dir: Path) -> dict:
    """Environment with MuJoCo paths and patchelf (conda) for subprocess."""
    env = os.environ.copy()
    mujoco_path = benchsuite_dir / "data" / "mujoco210"
    if mujoco_path.is_dir():
        bin_path = mujoco_path / "bin"
        nvidia = "/usr/lib/nvidia"
        existing = (env.get("LD_LIBRARY_PATH") or "").split(os.pathsep)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in [str(bin_path), nvidia] + existing if p.strip())
        env["MUJOCO_PY_MUJOCO_PATH"] = str(mujoco_path)
    # mujoco-py (PyPI) needs patchelf on first build; add conda env bin if present
    conda_bin = benchsuite_dir.parent / "miniconda3" / "envs" / "benchsuite" / "bin"
    if conda_bin.is_dir():
        env["PATH"] = os.pathsep.join([str(conda_bin), env.get("PATH", "")])
    return env


def run_benchsuite_main(
    benchsuite_dir: Path,
    main_name: str,
    x_normalized: list[float],
    timeout: int = 120,
) -> float:
    """
    Call BenchSuite main.py with --name and -x (values in [0,1]).
    Uses that project's .venv/bin/python when present. Returns the printed float.
    """
    venv_python = benchsuite_dir / ".venv" / "bin" / "python"
    if venv_python.is_file():
        cmd = [str(venv_python), "main.py", "--name", main_name, "-x"]
    else:
        cmd = ["poetry", "run", "python", "main.py", "--name", main_name, "-x"]
    cmd.extend(str(v) for v in x_normalized)
    env = _env_with_mujoco(benchsuite_dir)
    result = subprocess.run(
        cmd,
        cwd=str(benchsuite_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        msg = result.stderr or result.stdout
        if not (msg and msg.strip()):
            msg = (
                f"no output (returncode={result.returncode}). "
                "Run BenchSuite main.py manually from the BenchSuite dir, e.g.: "
                f".venv/bin/python main.py --name {main_name} -x <{len(x_normalized)} values in [0,1]>"
            )
        else:
            msg = f"returncode={result.returncode}; {msg.strip()}"
        raise RuntimeError(f"BenchSuite main.py failed: {msg}")
    # main.py prints only the result number; some benchmarks (e.g. SVM) print other lines first
    lines = (result.stdout or "").strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            if line.startswith("["):
                import ast
                out = ast.literal_eval(line)
                return float(out[0] if isinstance(out, list) else out)
            return float(line)
        except ValueError:
            continue
    raise ValueError(f"Could not parse BenchSuite output (no numeric line): {result.stdout!r}")


class BenchSuiteSubprocessProblem(BaseTestProblem):
    """
    BoTorch BaseTestProblem that evaluates via BenchSuite's main.py in a
    subprocess (external project, its own venv). No imports from BenchSuite
    in this process.
    """

    def __init__(
        self,
        problem_key: str,
        main_name: str,
        dim: int,
        lb: float | np.ndarray,
        ub: float | np.ndarray,
        benchsuite_dir: Path | None = None,
        noise_std=None,
        negate=True,
    ):
        self._problem_key = problem_key
        self._main_name = main_name
        self.dim = dim
        if isinstance(lb, (int, float)):
            lb = np.full(dim, float(lb))
        if isinstance(ub, (int, float)):
            ub = np.full(dim, float(ub))
        self._lb = np.asarray(lb, dtype=np.float64)
        self._ub = np.asarray(ub, dtype=np.float64)
        self._benchsuite_dir = benchsuite_dir or get_benchsuite_dir()
        if self._benchsuite_dir is None:
            raise FileNotFoundError(
                "BenchSuite not found. Set BENCHSUITE_ROOT to the BenchSuite project root "
                "(e.g. parent of natural_bo: /path/to/BenchSuite), or clone it there."
            )
        self._bounds = np.vstack((self._lb, self._ub)).T
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        self._timeout = BENCHSUITE_SUBPROCESS_TIMEOUT.get(
            problem_key, DEFAULT_SUBPROCESS_TIMEOUT
        )
        super().__init__(noise_std=noise_std, negate=negate)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if X.dim() == 1:
            X = X.unsqueeze(0)
        results = []
        for i in range(X.shape[0]):
            x = X[i].cpu().numpy()
            span = self._ub - self._lb
            span[span == 0] = 1.0
            x_norm = ((x - self._lb) / span).tolist()
            y = run_benchsuite_main(
                self._benchsuite_dir,
                self._main_name,
                x_norm,
                timeout=self._timeout,
            )
            results.append(y)
        y_tensor = torch.tensor(results, dtype=X.dtype, device=X.device)
        return y_tensor.unsqueeze(-1)


def create_benchsuite_subprocess_problem(
    problem_name: str,
    noise_std=None,
    negate=True,
) -> BenchSuiteSubprocessProblem | None:
    """
    If problem_name is a BenchSuite problem and the external project is found,
    return a subprocess-based problem; else return None.
    """
    cfg = BENCHSUITE_SUBPROCESS_CONFIG.get(problem_name)
    if cfg is None:
        return None
    main_name, dim, lb, ub = cfg
    benchsuite_dir = get_benchsuite_dir()
    if benchsuite_dir is None:
        return None
    return BenchSuiteSubprocessProblem(
        problem_key=problem_name,
        main_name=main_name,
        dim=dim,
        lb=lb,
        ub=ub,
        benchsuite_dir=benchsuite_dir,
        noise_std=noise_std,
        negate=negate,
    )
