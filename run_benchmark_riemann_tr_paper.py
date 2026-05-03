"""
Riemannian TuRBO Multi-Trial Benchmarks (riemann_tr_paper)
==========================================================

Runs benchmark problems with multiple seeds. For each problem, runs each
(transform, center_selector) method over n_trials seeds. Saves per-seed
best_observed CSV, observed_data JSON, config, and log. Writes comparison plots (min-max and mean ± std).

Output: riemann_tr_paper / {problem_name} / {acqf}_{timestamp} / {method_name} /
  - {seed}_best_observed.csv, {seed}_observed_data.json, config_seed_{seed}.json, experiment_seed_{seed}.log
  - results_comparison_multi_trial_*.png

Usage:
  python -m experiments.run_benchmark_riemann_tr_paper --problems HPA101-0 --n_trials 11
  python -m experiments.run_benchmark_riemann_tr_paper --problems HPA101-0 --methods "diag_grad_rms:best,diag_grad_rms:rei,dsp"
  python -m experiments.run_benchmark_riemann_tr_paper --problems MOPTA08 --methods bounce --max_evals 50
  python -m experiments.run_benchmark_riemann_tr_paper --problems levy4_25 --methods "lowrank_svd:best,diag_grad_rms:best"

Bounce requires: poetry add gin-config xgboost (or install -e .[bounce] and xgboost).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.curdir))

import torch

from experiment_runner import ExperimentConfig, run_single_method
from riemann_turbo import eval_objective, ensure_y_shape_n1
from experiments.utils import (
    build_observed_data_from_flat,
    get_or_create_commit_id,
    get_results_base_dir,
    ensure_experiment_dir,
    plot_multi_trial_comparison,
    save_best_observed_csv,
    save_config,
    save_results,
    setup_device,
    setup_logging,
)
from src.benchmark.define_problems import DefineProblems
from src.riemannTuRBO import EpsConfig, EpsMode

# Bounce: external method via wrapper (experiment_runner and riemann_turbo unchanged)
METHOD_BOUNCE = "bounce"
try:
    from src.old_src.BOUNCE import run_bounce_optimization as _run_bounce_optimization
except Exception:
    _run_bounce_optimization = None


# Benchmark definitions: (problem_name, problem_string, dim)
# Same style as run_benchmark_HPA101-0.py etc.
BENCHMARKS = [
    ("HPA101-0", "HPA101-0", 17),
    ("HPA101-2", "HPA101-2", 108),
    ("HPA102-1", "HPA102-1", 32),
    ("HPA103-1", "HPA103-1", 32),
    ("lasso_dna", "lasso_dna", 180),
    ("RoverTrajectory", "RoverTrajectory", 60),
    ("MOPTA08", "MOPTA08", 124),
    ("lunarlander", "lunarlander", 12),
    ("robot_pushing", "robot_pushing", 14),
    ("svm", "svm", 388),
    ("swimming", "swimming", 16),
    ("hopper", "hopper", 33),
    ("ant", "ant", 888),
    ("humanoid", "humanoid", 6392),
    ("half_cheetah", "half_cheetah", 102),
    ("levy4_25", "levy4_25", 25),
    ("levy4_100", "levy4_100", 100),
    ("levy4_300", "levy4_300", 300),
    ("levy4_1000", "levy4_1000", 1000),
    ("hartmann6_25", "hartmann6_25", 25),
    ("hartmann6_100", "hartmann6_100", 100),
    ("hartmann6_300", "hartmann6_300", 300),
    ("hartmann6_1000", "hartmann6_1000", 1000),
    ("lasso_synt_simple", "lasso_synt_simple", 60),
    ("lasso_synt_medium", "lasso_synt_medium", 100),
    ("lasso_synt_high", "lasso_synt_high", 300),
    ("lasso_synt_hard", "lasso_synt_hard", 1000),
    ("lasso_diabetes", "lasso_diabetes", 8),
    ("lasso_breast_cancer", "lasso_breast_cancer", 10),
    ("lasso_leukemia", "lasso_leukemia", 7129),
    ("lasso_rcv1", "lasso_rcv1", 47236),
]

NUM_TRIALS = 5
# Default experiment settings (overridable via CLI)
DEFAULT_MAX_EVALS = 300
DEFAULT_BATCH_SIZE = 1
DEFAULT_N_INIT = 30
DEFAULT_NUM_RESTARTS = 5
DEFAULT_RAW_SAMPLES = 128
DEFAULT_QMC_SAMPLE_SHAPE = 256
DEFAULT_OBSERVATION_NOISE = True
# Default methods: (transform, center_selector). DSP/Bounce have no center selector (None).
MethodTuple = Tuple[str, Optional[str]]
DEFAULT_METHODS: List[MethodTuple] = [
    # ("diag_grad_rms", "rei"),
    ("dsp", None),  # DSP: full [0,1]^d bounds, logEI, no trust region
    # ("diag_grad_rms", "best"),
    # (
    #     "lowrank_svd",
    #     "best",
    # ),  # Rotated TR (cheap: analytic volume norm, no polytope MCMC)
    # ("ard_lengthscale", "rei"),
    # ("bounce", None),
    ("diag_grad_rms", "rei", "no_obs_noise"),
    ("diag_grad_rms", "rei", "obs_noise"),
]


def _str2bool(v: str) -> bool:
    """Parse CLI booleans passed as strings (e.g. true/false)."""
    if isinstance(v, bool):
        return v
    v_l = v.strip().lower()
    if v_l in {"1", "true", "t", "yes", "y"}:
        return True
    if v_l in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {v!r}. Expected true/false."
    )


def _method_display_name(
    transform: str,
    center_selector: str | None,
) -> str:
    """Unique display name for a method (used as folder name)."""
    if (transform or "").lower() in ("dsp", METHOD_BOUNCE) or center_selector is None:
        return transform
    return f"{transform}_{center_selector}"


def _run_single_seed(
    config: ExperimentConfig,
    transform_method: str,
    fun: Callable,
    seed: int,
    seed_logger: logging.Logger,
) -> dict:
    """Run one (transform, center_selector) for one seed. Returns results dict (Y, X, best_value, ...)."""
    is_bounce = (transform_method or "").lower() == METHOD_BOUNCE
    if is_bounce and _run_bounce_optimization is not None:
        optimal_val = getattr(fun, "optimal_value", None)

        # Pass eval_objective closure so Bounce uses the same evaluation as other models (normalized inputs)
        def eval_callable(x):
            return ensure_y_shape_n1(eval_objective(func=fun, x_normalized=x))

        # Same experiment parameters as other methods; seed used for initial points and RNG
        return _run_bounce_optimization(
            fun=eval_callable,
            dim=config.dim,
            max_evals=config.max_evals,
            n_init=config.n_init,
            batch_size=config.batch_size,
            seed=seed,
            device=config.device,
            dtype=config.dtype,
            problem_name=config.problem_name,
            problem_string=config.problem_string,
            optimal_value=optimal_val,
            logger_instance=seed_logger,
        )
    return run_single_method(
        config=config,
        transform_method=transform_method,
        fun=fun,
        logger=seed_logger,
    )


def run_experiment_multi_trial(
    config: ExperimentConfig,
    fun: Callable,
    experiment_base_dir: Path,
    seeds: List[int],
    methods: List[MethodTuple],
    logger: Optional[logging.Logger] = None,
    experiment_name: Optional[str] = None,
) -> dict:
    """
    Run each (transform, center_selector) method over multiple seeds.
    Saves per-seed CSV, config, log; then writes comparison plots (min-max and mean ± std).
    Returns all_results[method_display_name][seed] = results dict.
    """
    log = logger or logging.getLogger(__name__)
    experiment_base_dir = Path(experiment_base_dir)
    experiment_base_dir.mkdir(parents=True, exist_ok=True)

    config_file_base = experiment_base_dir / "config_reference.json"
    additional_fields = {
        "methods": [list(m) for m in methods],
        "n_seeds": len(seeds),
        "seeds": list(seeds),
    }
    if experiment_name:
        additional_fields["experiment_name"] = experiment_name
    commit_id = get_or_create_commit_id(
        src_dir=Path("src"),
        experiment_name=experiment_name or "riemann_tr_paper",
        dry_run=False,
    )
    if commit_id:
        log.info(f"Source code commit ID: {commit_id}")
        additional_fields["git_commit_id"] = commit_id
    # Config saving is metadata; don't fail the whole run if it errors.
    try:
        save_config(config, config_file_base, additional_fields=additional_fields)
    except Exception as e:
        log.warning(f"Non-fatal: failed to save reference config: {e}")

    all_results = {}
    for method_tuple in methods:
        transform_method, center_selector_type = method_tuple[0], method_tuple[1]
        method_name = _method_display_name(transform_method, center_selector_type)
        if (
            transform_method or ""
        ).lower() == METHOD_BOUNCE and _run_bounce_optimization is None:
            log.warning(
                "Method 'bounce' requested but BOUNCE not available. Install: poetry add gin-config xgboost. Skipping."
            )
            continue
        method_dir = experiment_base_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        all_results[method_name] = {}

        for seed in seeds:
            seed_config = dataclasses.replace(
                config,
                seed=seed,
                center_selector_type=center_selector_type or "best",
                output_dir=method_dir,
            )
            seed_logger = setup_logging(
                method_dir,
                log_file_name=f"experiment_seed_{seed}.log",
            )
            try:
                results = _run_single_seed(
                    config=seed_config,
                    transform_method=transform_method,
                    fun=fun,
                    seed=seed,
                    seed_logger=seed_logger,
                )
                all_results[method_name][seed] = results
                save_best_observed_csv(
                    results["Y"], method_dir / f"{seed}_best_observed.csv"
                )
                # Save observed input-output pairs (initial + per-iteration)
                observed_data = results.get("observed_data")
                if observed_data is None:
                    observed_data = build_observed_data_from_flat(
                        results["X"],
                        results["Y"],
                        n_init=seed_config.n_init,
                        batch_size=seed_config.batch_size,
                    )
                # save_results(
                #     observed_data,
                #     method_dir / f"{seed}_observed_data.json",
                # )
                # Config is useful metadata, but must not block writing results.
                try:
                    save_config(seed_config, method_dir / f"config_seed_{seed}.json")
                except Exception as e:
                    log.warning(
                        f"Non-fatal: failed to save config for method={method_name} seed={seed}: {e}"
                    )
            except Exception as e:
                log.error(
                    f"Failed method={method_name} seed={seed}: {e}",
                    exc_info=True,
                )

    if all_results:
        optimal_value = None
        for method_results in all_results.values():
            for res in method_results.values():
                if res.get("optimal_value") is not None:
                    optimal_value = res["optimal_value"]
                    break
            if optimal_value is not None:
                break
        plot_title_base = "Riemannian TuRBO Comparison (multi-trial)"
        if config.problem_name:
            plot_title_base = f"{config.problem_name} - {plot_title_base}"
        if config.acqf:
            plot_title_base = f"{plot_title_base} (acqf={config.acqf})"
        plot_multi_trial_comparison(
            experiment_base_dir,
            title=f"{plot_title_base} [min-max]",
            optimal_value=optimal_value,
            acqf=config.acqf,
            band="minmax",
        )
        plot_multi_trial_comparison(
            experiment_base_dir,
            title=f"{plot_title_base} [mean ± std]",
            optimal_value=optimal_value,
            acqf=config.acqf,
            band="std",
        )

    log.info("Multi-trial experiment completed.")
    log.info(f"Results saved under: {experiment_base_dir}")
    return all_results


def _parse_methods_arg(methods_str: str | None) -> list[MethodTuple]:
    """Parse --methods: 'transform:selector' or 'transform'. E.g. diag_grad_rms:last,identity:last,dsp."""
    if not methods_str or not methods_str.strip():
        return DEFAULT_METHODS.copy()
    methods: List[MethodTuple] = []
    for part in [s.strip() for s in methods_str.split(",") if s.strip()]:
        parts = part.split(":")
        if len(parts) >= 2:
            methods.append((parts[0].strip(), parts[1].strip() or None))
        else:
            methods.append((parts[0].strip(), None))
    return methods


def run_one_problem(
    problem_name: str,
    problem_string: str,
    dim: int,
    seeds: list[int],
    output_dir_name: str = "riemann_tr_paper",
    dir_suffix: str | None = None,
    acqfs: list[str] | None = None,
    methods: list[MethodTuple] | None = None,
    max_evals: int = DEFAULT_MAX_EVALS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_init: int = DEFAULT_N_INIT,
    num_restarts: int = DEFAULT_NUM_RESTARTS,
    raw_samples: int = DEFAULT_RAW_SAMPLES,
    qmc_sample_shape: int = DEFAULT_QMC_SAMPLE_SHAPE,
    n_candidates: int | None = None,
    observation_noise: bool = DEFAULT_OBSERVATION_NOISE,
    volume_normalize: bool = True,
    cuda_device: int | None = None,
    use_raasp: bool = True,
):
    """Run multi-trial experiment for one benchmark problem.

    methods: list of (transform, center_selector).
    E.g. [("diag_grad_rms", "last"), ("identity", "last"), ("dsp", None)].
    """
    if acqfs is None:
        acqfs = ["logei"]
    if methods is None:
        methods = DEFAULT_METHODS.copy()

    device = setup_device(cuda_device=cuda_device)
    dtype = torch.float64
    eps_cfg = EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12)

    results_base = get_results_base_dir()
    experiment_base = results_base / output_dir_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        fun = DefineProblems(
            problem_string,
            dim=0,
            dim_emb=0,
            noise_std=None,
            negate=True,
        ).to(dtype=dtype, device=device)
    except Exception as e:
        print(f"Failed to load problem {problem_string}: {e}", file=sys.stderr)
        return

    problem_dir_name = f"{problem_name}_{dir_suffix}" if dir_suffix else problem_name
    for acqf in acqfs:
        acqf_dir = f"{acqf}_{timestamp}"
        experiment_base_dir = ensure_experiment_dir(
            experiment_base, problem_dir_name, acqf_dir
        )
        root_logger = setup_logging(experiment_base_dir, log_file_name="run.log")

        root_logger.info("=" * 80)
        root_logger.info(f"Riemannian TuRBO multi-trial: {problem_name}")
        root_logger.info("=" * 80)
        root_logger.info(f"Problem: {problem_string} (dim={dim})")
        root_logger.info(f"Seeds: {seeds}")
        root_logger.info(f"Acquisition function: {acqf}")
        root_logger.info(f"Methods (transform, center_selector): {methods}")

        # Single base config: run_experiment_multi_trial uses dataclasses.replace() per (method, seed).
        config = ExperimentConfig(
            dim=dim,
            batch_size=batch_size,
            n_init=n_init,
            max_evals=max_evals,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            qmc_sample_shape=qmc_sample_shape,
            seed=seeds[0],
            device=device,
            dtype=dtype,
            acqf=acqf,
            observation_noise=observation_noise,
            volume_normalize=volume_normalize,
            n_candidates=n_candidates,
            center_selector_type=None,
            eps_cfg=eps_cfg,
            use_raasp=use_raasp,
            problem_name=problem_name,
            problem_string=problem_string,
        )

        experiment_name = f"{output_dir_name}_{problem_name}_{acqf_dir}"
        run_experiment_multi_trial(
            config=config,
            fun=fun,
            experiment_base_dir=experiment_base_dir,
            seeds=seeds,
            methods=methods,
            logger=root_logger,
            experiment_name=experiment_name,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run Riemannian TuRBO benchmarks with multiple seeds (riemann_tr_paper)."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=NUM_TRIALS,
        help=f"Number of seeds/trials per problem (default: {NUM_TRIALS}).",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        required=True,
        help="Problem names to run (required). Example: HPA101-0 RoverTrajectory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="riemann_tr_paper",
        help="Top-level output directory name (default: riemann_tr_paper).",
    )
    parser.add_argument(
        "--dir_suffix",
        type=str,
        default=None,
        metavar="MSG",
        help="If set, the root save folder per problem is named PROBLEM_MSG (e.g. MOPTA08_msg). Default: none.",
    )
    # Experiment settings
    parser.add_argument(
        "--max_evals",
        type=int,
        default=DEFAULT_MAX_EVALS,
        help=f"Maximum number of evaluations (default: {DEFAULT_MAX_EVALS}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size q (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=DEFAULT_N_INIT,
        help=f"Number of initial random points (default: {DEFAULT_N_INIT}).",
    )
    parser.add_argument(
        "--num_restarts",
        type=int,
        default=DEFAULT_NUM_RESTARTS,
        help=f"Number of restarts for acquisition optimization (default: {DEFAULT_NUM_RESTARTS}).",
    )
    parser.add_argument(
        "--raw_samples",
        type=int,
        default=DEFAULT_RAW_SAMPLES,
        help=f"Raw samples for acquisition optimization (default: {DEFAULT_RAW_SAMPLES}).",
    )
    parser.add_argument(
        "--qmc_sample_shape",
        type=int,
        default=DEFAULT_QMC_SAMPLE_SHAPE,
        help=f"QMC sample shape (default: {DEFAULT_QMC_SAMPLE_SHAPE}).",
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=None,
        metavar="N",
        help="Number of candidates (default: min(5000, max(2000, 200*dim))).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated methods: 'transform:selector' or 'transform:selector:obs_noise' for diag_grad_rms. "
        "E.g. diag_grad_rms:rei:obs_noise,diag_grad_rms:rei:no_obs_noise,dsp,bounce.",
    )
    parser.add_argument(
        "--volume_normalize",
        type=_str2bool,
        default=True,
        help="Enable/disable volume normalization (default: True). Pass true/false.",
    )
    parser.add_argument(
        "--acqf",
        type=str,
        default="logei",
        choices=["logei", "ei", "ts"],
        help="Acquisition function (default: logei).",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=None,
        metavar="N",
        help="CUDA device index to use (optional). If not set, device is auto-selected.",
    )
    parser.add_argument(
        "--observation_noise",
        action="store_true",
        default=False,
        help="Include observation noise in Fisher gradient computation (DiagGradRMS). Default: off.",
    )
    parser.add_argument(
        "--use_raasp",
        type=_str2bool,
        default=True,
        help="Use RAASP warm-starting for RotatedTR gradient path (default: True). Pass true/false.",
    )
    args = parser.parse_args()

    # If needed, you can generate trial seeds via RNG; currently we use a fixed
    # seed list for reproducibility.
    seeds = [0, 1, 2, 3, 4]
    problems_to_run = args.problems
    methods = _parse_methods_arg(args.methods)
    benchmark_map = {b[0]: (b[0], b[1], b[2]) for b in BENCHMARKS}
    for name in problems_to_run:
        if name not in benchmark_map:
            print(f"Unknown problem: {name}. Skipping.", file=sys.stderr)
            continue
        problem_name, problem_string, dim = benchmark_map[name]
        run_one_problem(
            problem_name=problem_name,
            problem_string=problem_string,
            dim=dim,
            seeds=seeds,
            output_dir_name=args.output_dir,
            dir_suffix=args.dir_suffix,
            acqfs=[args.acqf],
            methods=methods,
            max_evals=args.max_evals,
            batch_size=args.batch_size,
            n_init=args.n_init,
            num_restarts=args.num_restarts,
            raw_samples=args.raw_samples,
            qmc_sample_shape=args.qmc_sample_shape,
            n_candidates=args.n_candidates,
            observation_noise=args.observation_noise,
            volume_normalize=args.volume_normalize,
            cuda_device=args.cuda_device,
            use_raasp=args.use_raasp,
        )


if __name__ == "__main__":
    main()
