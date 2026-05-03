"""
Experiment runner for Riemannian TuRBO: config and single-trial run.

Used by run_benchmark_riemann_tr_paper for multi-trial benchmarks.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Optional
from pathlib import Path

import torch
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from riemann_turbo import (
    TurboState,
    generate_riemannian_batch,
    get_initial_points,
    eval_objective,
    ensure_y_shape_n1,
)
from experiments.utils import setup_device, save_results
from src.riemannTuRBO import (
    EpsConfig,
    EpsMode,
    CenterSelector,
    BestObservedSelector,
    RestartCenterSelector,
    get_center_selector,
)

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ExperimentConfig:
    """Experiment configuration for a single BO run."""

    dim: int
    batch_size: int
    n_init: int
    max_evals: int
    num_restarts: int = 10
    raw_samples: int = 512
    qmc_sample_shape: int = 128
    acqf: str = "logei"
    seed: int = 0
    device: torch.device = None
    dtype: torch.dtype = torch.float64
    observation_noise: bool = True
    volume_normalize: bool = True
    n_candidates: Optional[int] = None
    center_selector_type: Optional[str] = None
    eps_cfg: Optional[EpsConfig] = None
    use_raasp: bool = True
    problem_name: Optional[str] = None
    problem_string: Optional[str] = None
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.eps_cfg is None:
            self.eps_cfg = EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12)
        if self.device is None:
            self.device = setup_device()


def _resolve_center_selector(
    center_selector_type: str,
    restart_triggered: bool,
    is_dsp: bool,
    logger: logging.Logger,
) -> CenterSelector:
    """Resolve selector object for the current iteration."""
    if is_dsp:
        return BestObservedSelector()
    if restart_triggered and center_selector_type in ("best", "last"):
        logger.info(
            "Restart triggered: using RestartCenterSelector (global posterior-mean search)."
        )
        return RestartCenterSelector(num_samples=512)
    if restart_triggered:
        logger.info(
            "Restart triggered: using configured selector '%s'.",
            center_selector_type,
        )
    return get_center_selector(center_selector_type)


def run_single_method(
    config: ExperimentConfig,
    transform_method: str,
    fun: Callable,
    logger: logging.Logger,
) -> dict:
    """
    Run a single (transform_method, center_selector) experiment.
    """
    method_name = transform_method
    is_dsp = (transform_method or "").lower() == "dsp"
    center_selector_type = config.center_selector_type or "best"
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running {method_name}")
    logger.info(f"{'=' * 80}")

    device = config.device
    dtype = config.dtype
    dim = config.dim

    # Initialize state with update strategy
    state = TurboState(dim=dim, q=config.batch_size)

    # Generate initial points
    logger.info(f"Generating {config.n_init} initial points...")
    X = get_initial_points(
        dim=dim,
        n_pts=config.n_init,
        device=device,
        dtype=dtype,
        seed=config.seed,
    )
    try:
        Y = ensure_y_shape_n1(eval_objective(func=fun, x_normalized=X))
    except (PermissionError, FileNotFoundError) as e:
        logger.error(
            f"Failed to evaluate problem (permission/executable issue): {e}\n"
            f"For MOPTA08, ensure the .bin files are in src/benchmark/mopta08/ and have execute permissions.",
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate problem: {e}", exc_info=True)
        raise

    # Standardize Y
    Y_mean = Y.mean()
    Y_std = Y.std()
    Y_standardized = (Y - Y_mean) / (Y_std + 1e-8)

    logger.info("Initial data:")
    logger.info(f"  Best value: {Y.max().item():.4f}")
    logger.info(f"  Mean value: {Y.mean().item():.4f}")
    logger.info(f"  Std value: {Y.std().item():.4f}")
    logger.info(f"  Min value: {Y.min().item():.4f}")
    logger.info(f"  Number of initial points: {len(Y)}")
    if hasattr(fun, "optimal_value") and fun.optimal_value is not None:
        logger.info(f"  Optimal value: {fun.optimal_value:.4f}")
        logger.info(f"  Gap to optimal: {Y.max().item() - fun.optimal_value:.4f}")

    # Storage
    Y_all = [Y.clone()]
    X_all = [X.clone()]
    state_history = []
    diagnostics_history = []
    centers_history = []  # Store centers for each iteration

    # Fit initial model
    logger.info("Fitting initial GP model...")
    model = SingleTaskGP(X, Y_standardized)
    model.to(device=device, dtype=dtype)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Optimization loop
    n_evals = len(X)
    iteration = 0
    # Compute n_candidates if not provided
    if config.n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
    else:
        n_candidates = config.n_candidates

    while n_evals < config.max_evals:
        iteration += 1
        best_before = Y.max().item()
        success_counter_before = state.success_counter
        failure_counter_before = state.failure_counter
        length_before = state.length

        problem_label = (
            getattr(config, "problem_name", None)
            or getattr(config, "problem_string", None)
            or "N/A"
        )
        logger.info(
            f"\n--- [{problem_label}] Iteration {iteration} (n_evals={n_evals}) ---"
        )
        logger.info(
            f"State before update: best={best_before:.4f}, length={length_before:.4f}, "
            f"successes={success_counter_before}, failures={failure_counter_before}"
        )

        # Update state best value
        state.best_value = Y.max().item()

        # Generate batch
        try:
            selector = _resolve_center_selector(
                center_selector_type=center_selector_type,
                restart_triggered=state.restart_triggered,
                is_dsp=is_dsp,
                logger=logger,
            )
            logger.info(
                "Center selector resolved: %s (restart=%s)",
                type(selector).__name__,
                state.restart_triggered,
            )

            logger.info(f"Generating batch of {config.batch_size} candidates...")
            X_next, diagnostics = generate_riemannian_batch(
                state=state,
                model=model,
                X=X,
                Y=Y_standardized,
                q=config.batch_size,
                n_candidates=n_candidates,
                num_restarts=config.num_restarts,
                raw_samples=config.raw_samples,
                acqf=config.acqf,
                qmc_sample_shape=config.qmc_sample_shape,
                center_selector=selector,
                transform_method=transform_method,
                eps_cfg=config.eps_cfg,
                volume_normalize=config.volume_normalize,
                observation_noise=config.observation_noise,
                return_diagnostics=True,
                use_raasp=config.use_raasp,
            )
            state.restart_triggered = False

            # Log diagnostics from batch generation (skip TR-related when DSP)
            if not is_dsp:
                logger.info("Batch generation diagnostics:")
                logger.info(f"  Transform method: {transform_method}")
                logger.info(f"  Epsilon used: {diagnostics.get('eps_used', 'N/A'):.4e}")
                logger.info(f"  Anisotropy: {diagnostics.get('anisotropy', 'N/A'):.4f}")
                if "anisotropy_before_clamp" in diagnostics:
                    logger.info(
                        f"  Anisotropy (before clamp): {diagnostics['anisotropy_before_clamp']:.4f}"
                    )
                    logger.info(
                        f"  Anisotropy (after clamp): {diagnostics['anisotropy_after_clamp']:.4f}"
                    )
                if "condition_number" in diagnostics:
                    logger.info(
                        f"  Condition number: {diagnostics['condition_number']:.4f}"
                    )
                if "points_clamped_ratio" in diagnostics:
                    logger.info(
                        f"  Clamped points: {diagnostics['points_clamped_ratio']:.2%} "
                        f"(mean {diagnostics.get('mean_dims_clamped_ratio', 0.0):.2%} of dims per clamped point)"
                    )
                if "n_clamped" in diagnostics:
                    logger.info(
                        f"  Number of eigenvalues clamped: {diagnostics['n_clamped']}"
                    )
                if "volume_normalize_factor" in diagnostics:
                    logger.info(
                        f"  Volume normalization factor: {diagnostics['volume_normalize_factor']:.4e}"
                    )

            # Evaluate
            logger.info(f"Evaluating {config.batch_size} candidate(s)...")
            try:
                Y_next = ensure_y_shape_n1(
                    eval_objective(func=fun, x_normalized=X_next)
                )
            except (PermissionError, FileNotFoundError) as e:
                logger.error(
                    f"Failed to evaluate problem (permission/executable issue): {e}\n"
                    f"For MOPTA08, ensure the .bin files are in src/benchmark/mopta08/ and have execute permissions.",
                    exc_info=True,
                )
                raise
            except Exception as e:
                logger.error(f"Failed to evaluate problem: {e}", exc_info=True)
                raise

            logger.info(f"Evaluation results: current batch={Y_next}")

            # Append data
            X = torch.cat((X, X_next), dim=0)
            Y = torch.cat((Y, Y_next), dim=0)
            Y_standardized = (Y - Y_mean) / (Y_std + 1e-8)

            # Store
            Y_all.append(Y_next.clone())
            X_all.append(X_next.clone())
            state_history.append(
                {
                    "length": state.length,
                    "best_value": state.best_value,
                    "success_counter": state.success_counter,
                    "failure_counter": state.failure_counter,
                }
            )
            diagnostics_history.append(diagnostics)
            # Store center (convert tensor to list for JSON serialization)
            center = diagnostics.get("center", None)
            if center is not None:
                if isinstance(center, torch.Tensor):
                    center_flat = center.flatten().to(device=device, dtype=dtype)
                    if centers_history:
                        prev_center = torch.tensor(
                            centers_history[-1],
                            device=device,
                            dtype=dtype,
                        ).flatten()
                        dist = (center_flat - prev_center).norm().item()
                        logger.info(
                            "Center distance to previous: %.6f (L2)",
                            dist,
                        )
                    centers_history.append(center.detach().cpu().numpy().tolist())
                else:
                    centers_history.append(center)
            else:
                centers_history.append(None)

            # Update state (may set state.restart_triggered for next iteration)
            state.update(Y_next, X_next=X_next)

            # Save intermediate results (observed data) incrementally
            if config.output_dir is not None:
                try:
                    # Build observed_data structure from current history
                    current_observed_data = {
                        "initial": {
                            "inputs": X_all[0].cpu().numpy().tolist(),
                            "outputs": Y_all[0].cpu().numpy().tolist(),
                        },
                        "iterations": [
                            {
                                "iteration": i + 1,
                                "inputs": X_all[i + 1].cpu().numpy().tolist(),
                                "outputs": Y_all[i + 1].cpu().numpy().tolist(),
                            }
                            for i in range(len(X_all) - 1)
                        ],
                    }
                    save_results(
                        current_observed_data,
                        config.output_dir / f"{config.seed}_observed_data.json",
                    )
                except Exception as e:
                    logger.warning(f"Failed to save intermediate results: {e}")

            # Log state update results
            best_after = Y.max().item()
            success_counter_after = state.success_counter
            failure_counter_after = state.failure_counter
            length_after = state.length
            improvement = best_after - best_before

            logger.info("State after update:")
            logger.info(
                f"  Best value: {best_after:.4f} (improvement: {improvement:+.4f})"
            )
            logger.info(f"  TR length: {length_after:.4f} (was {length_before:.4f})")
            logger.info(
                f"  Success counter: {success_counter_after} (was {success_counter_before}, "
                f"change: {success_counter_after - success_counter_before:+d})"
            )
            logger.info(
                f"  Failure counter: {failure_counter_after} (was {failure_counter_before}, "
                f"change: {failure_counter_after - failure_counter_before:+d})"
            )

            # Determine if this was a success or failure
            if improvement > 0:
                logger.info(
                    f"  SUCCESS: Found better value (improvement: {improvement:.4f})"
                )
            else:
                logger.info("  NO IMPROVEMENT: Best value unchanged")

            # Refit model
            logger.info("Refitting GP model...")
            model = SingleTaskGP(X, Y_standardized)
            model.to(device=device, dtype=dtype)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            n_evals = len(X)

            # Comprehensive summary log message (omit TR KPIs when DSP)
            if is_dsp:
                log_msg = (
                    f"Summary: Best={best_after:.4f} (Δ{improvement:+.4f}), "
                    f"Successes={success_counter_after}, Failures={failure_counter_after}"
                )
            else:
                log_msg = (
                    f"Summary: Best={best_after:.4f} (Δ{improvement:+.4f}), "
                    f"Length={length_after:.4f}, "
                    f"Successes={success_counter_after}, Failures={failure_counter_after}, "
                    f"Epsilon={diagnostics.get('eps_used', 0):.4e}, "
                    f"Anisotropy={diagnostics.get('anisotropy', 0):.4f}"
                )
                if "anisotropy_before_clamp" in diagnostics:
                    log_msg += (
                        f", Aniso(before)={diagnostics['anisotropy_before_clamp']:.2f}, "
                        f"Aniso(after)={diagnostics['anisotropy_after_clamp']:.2f}"
                    )
                if "condition_number" in diagnostics:
                    log_msg += f", Cond={diagnostics['condition_number']:.2f}"
                if "points_clamped_ratio" in diagnostics:
                    log_msg += (
                        f", Clamp={diagnostics['points_clamped_ratio']:.2%}"
                        f"(dims={diagnostics.get('mean_dims_clamped_ratio', 0.0):.2%})"
                    )
                if "n_clamped" in diagnostics:
                    log_msg += f", N_clamped={diagnostics['n_clamped']}"

            logger.info(log_msg)

        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
            break

    # Get optimal value if available (benchmark problems may not have this)
    optimal_value = getattr(fun, "optimal_value", None)

    # Final results - comprehensive JSON with all data
    # Convert diagnostics to JSON-serializable format
    diagnostics_history_json = []
    for diag in diagnostics_history:
        diag_json = {}
        for key, value in diag.items():
            if isinstance(value, torch.Tensor):
                diag_json[key] = value.detach().cpu().numpy().tolist()
            elif hasattr(value, "item"):  # Scalar tensor
                diag_json[key] = (
                    value.item() if hasattr(value, "item") else float(value)
                )
            else:
                diag_json[key] = value
        diagnostics_history_json.append(diag_json)

    # Build observed_data: initial (input-output pairs) + per-iteration batches
    observed_data = {
        "initial": {
            "inputs": X_all[0].cpu().numpy().tolist(),
            "outputs": Y_all[0].cpu().numpy().tolist(),
        },
        "iterations": [
            {
                "iteration": i,
                "inputs": X_all[i].cpu().numpy().tolist(),
                "outputs": Y_all[i].cpu().numpy().tolist(),
            }
            for i in range(1, len(X_all))
        ],
    }

    results = {
        "transform_method": transform_method,
        "method_name": method_name,
        "acqf": config.acqf,
        "Y": torch.cat(Y_all, dim=0).cpu().numpy().tolist(),
        "X": torch.cat(X_all, dim=0).cpu().numpy().tolist(),
        "observed_data": observed_data,
        "centers": centers_history,  # All centers for each iteration
        "state_history": state_history,
        "diagnostics_history": diagnostics_history_json,  # Full diagnostics with all fields
        "n_evals": n_evals,
        "best_value": Y.max().item(),
        "optimal_value": optimal_value,
    }

    # Add problem info if available
    if config.problem_name is not None:
        results["problem_name"] = config.problem_name
    if config.problem_string is not None:
        results["problem_string"] = config.problem_string

    # Final summary with comprehensive statistics
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{method_name} completed:")
    logger.info(f"{'=' * 80}")
    logger.info("Final Results:")
    logger.info(f"  Best value: {results['best_value']:.4f}")
    if optimal_value is not None:
        logger.info(f"  Optimal value: {results['optimal_value']:.4f}")
    logger.info(f"  Total evaluations: {results['n_evals']}")

    # Calculate statistics from state history
    if state_history:
        final_state = state_history[-1]
        total_successes = final_state.get("success_counter", 0)
        total_failures = final_state.get("failure_counter", 0)
        final_length = final_state.get("length", 0)

        logger.info("\nState Statistics:")
        logger.info(f"  Total successes: {total_successes}")
        logger.info(f"  Total failures: {total_failures}")
        logger.info(
            f"  Success rate: {total_successes / (total_successes + total_failures) * 100:.2f}%"
            if (total_successes + total_failures) > 0
            else "  Success rate: N/A"
        )
        if not is_dsp:
            logger.info(f"  Final TR length: {final_length:.4f}")

        # Calculate improvement over iterations
        if len(state_history) > 1:
            initial_best = state_history[0].get("best_value", results["best_value"])
            total_improvement = results["best_value"] - initial_best
            logger.info(
                f"  Total improvement: {total_improvement:.4f} (from {initial_best:.4f} to {results['best_value']:.4f})"
            )

    # Diagnostics summary (skip TR-related when DSP)
    if diagnostics_history and not is_dsp:
        logger.info("\nDiagnostics Summary:")
        eps_values = [
            d.get("eps_used", 0) for d in diagnostics_history if "eps_used" in d
        ]
        anisotropy_values = [
            d.get("anisotropy", 0) for d in diagnostics_history if "anisotropy" in d
        ]
        clamp_rates = [
            d.get("points_clamped_ratio", 0)
            for d in diagnostics_history
            if "points_clamped_ratio" in d
        ]
        dims_clamped_rates = [
            d.get("mean_dims_clamped_ratio", 0)
            for d in diagnostics_history
            if "mean_dims_clamped_ratio" in d
        ]

        if eps_values:
            logger.info(
                f"  Epsilon: min={min(eps_values):.4e}, max={max(eps_values):.4e}, mean={sum(eps_values) / len(eps_values):.4e}"
            )
        if anisotropy_values:
            logger.info(
                f"  Anisotropy: min={min(anisotropy_values):.4f}, max={max(anisotropy_values):.4f}, mean={sum(anisotropy_values) / len(anisotropy_values):.4f}"
            )
        if clamp_rates:
            logger.info(
                f"  Clamped points ratio: min={min(clamp_rates):.2%}, max={max(clamp_rates):.2%}, mean={sum(clamp_rates) / len(clamp_rates):.2%}"
            )
        if dims_clamped_rates:
            logger.info(
                f"  Mean dims clamped ratio (per clamped point): min={min(dims_clamped_rates):.2%}, max={max(dims_clamped_rates):.2%}, mean={sum(dims_clamped_rates) / len(dims_clamped_rates):.2%}"
            )

    logger.info(f"{'=' * 80}\n")

    return results
