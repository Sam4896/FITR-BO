"""
BOUNCE integration for natural_bo benchmarks.

Callers (e.g. run_benchmark_riemann_tr_paper) pass an evaluation callable that takes
normalized inputs [0,1]^d and returns objective values (e.g. eval_objective closure with
the problem). Bounce uses that callable to evaluate candidates; no internal eval_objective.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import gin
import torch

# Ensure "bounce" package is importable from this directory
_BOUNCE_ROOT = Path(__file__).resolve().parent
if str(_BOUNCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOUNCE_ROOT))

from bounce.bounce import Bounce
from bounce.callable_benchmark import CallableBenchmark

logger = logging.getLogger(__name__)

__all__ = ["run_bounce_optimization", "CallableBenchmark"]


def run_bounce_optimization(
    fun: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    max_evals: int = 300,
    n_init: int = 30,
    batch_size: int = 1,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    initial_target_dimensionality: Optional[int] = None,
    number_new_bins_on_split: int = 2,
    problem_name: Optional[str] = None,
    problem_string: Optional[str] = None,
    optimal_value: Optional[float] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> dict:
    """
    Run Bounce on a continuous [0,1]^d objective.

    fun: callable that evaluates the objective on normalized inputs [0,1]^d.
         Callers should pass the eval_objective closure with the problem, e.g.
         lambda x: ensure_y_shape_n1(eval_objective(func=problem, x_normalized=x)).
         Returns (n,1) or (n,) with values to maximize.
    """
    log = logger_instance or logger
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if device.type == "cuda" else "cpu"

    # Seed only used for initial points (initial_seed passed to Bounce); rest of run is independent.

    # Bounce minimizes; fun returns values we maximize. Pass -fun for minimization.
    def _min_obj(x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=device, dtype=dtype)
        return -fun(x)

    benchmark = CallableBenchmark(
        dim=dim,
        fun=_min_obj,
        noise_std=None,
        optimal_value=optimal_value,
    )

    if initial_target_dimensionality is None:
        initial_target_dimensionality = min(5, dim)

    results_dir = tempfile.mkdtemp(prefix="bounce_run_")
    dtype_str = "float64" if dtype == torch.float64 else "float32"

    gin.parse_config_files_and_bindings(
        [],
        [
            f"Bounce.number_initial_points={n_init}",
            f"Bounce.initial_target_dimensionality={initial_target_dimensionality}",
            f"Bounce.number_new_bins_on_split={number_new_bins_on_split}",
            f"Bounce.maximum_number_evaluations={max_evals}",
            f"Bounce.batch_size={batch_size}",
            f"Bounce.results_dir={results_dir!r}",
            f"Bounce.device={device_str!r}",
            f"Bounce.dtype={dtype_str!r}",
        ],
    )

    def _iteration_callback(iteration, n_evals, best_min, batch_y, best_before_min):
        best_value = -best_min
        if batch_y is None:
            log.info(f"\n--- Initial (n_evals={n_evals}) ---")
            log.info(f"  Best value: {best_value:.4f}")
            return
        best_before = -best_before_min if best_before_min is not None else best_value
        current_batch = [-y for y in batch_y]
        improvement = best_value - best_before
        batch_size_this = len(current_batch)
        log.info(f"\n--- Iteration {iteration} (n_evals={n_evals}) ---")
        log.info(f"State before update: best={best_before:.4f}")
        log.info(f"Evaluating {batch_size_this} candidate(s)...")
        log.info(f"Evaluation results: current batch={current_batch}")
        log.info("State after update:")
        log.info(f"  Best value: {best_value:.4f} (improvement: {improvement:+.4f})")
        if improvement > 0:
            log.info("  SUCCESS: Found better value")
        else:
            log.info("  NO IMPROVEMENT: Best value unchanged")

    try:
        bounce_instance = Bounce(benchmark=benchmark, initial_seed=seed)
        log.info("Running Bounce (continuous [0,1]^d)...")
        bounce_instance.run(iteration_callback=_iteration_callback)
    finally:
        gin.clear_config()

    X = bounce_instance.x_up_global.detach().cpu()
    Y_stored = bounce_instance.fx_global.detach().cpu()
    Y = -Y_stored
    if Y.dim() == 1:
        Y = Y.unsqueeze(1)

    Y_list = Y.cpu().numpy().tolist()
    X_list = X.cpu().numpy().tolist()
    best_value = float(Y.max().item())
    n_evals = len(Y)

    results = {
        "transform_method": "bounce",
        "method_name": "bounce",
        "acqf": "ei",
        "Y": Y_list,
        "X": X_list,
        "centers": [],
        "state_history": [],
        "diagnostics_history": [],
        "n_evals": n_evals,
        "best_value": best_value,
        "optimal_value": optimal_value,
    }
    if problem_name is not None:
        results["problem_name"] = problem_name
    if problem_string is not None:
        results["problem_string"] = problem_string

    try:
        import shutil
        shutil.rmtree(results_dir, ignore_errors=True)
    except Exception:
        pass

    return results
