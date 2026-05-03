# Bayesian Optimization with Fisher Information Geometry: Gradient Bounds and Trust-Region Methods

This repository contains the codebase for the NeurIPS submission:
**"Bayesian Optimization with Fisher Information Geometry: Gradient Bounds and Trust-Region Methods"**.

## Project structure

- `run_benchmark_riemann_tr_paper.py`: Main script to run the multi-trial benchmark suite used in the paper.
- `riemann_turbo.py`: Core optimization logic and trust-region behavior used by experiment scripts.
- `experiment_runner.py`: Shared experiment orchestration utilities and method dispatch.
- `src/riemannTuRBO/`: Core package implementation for geometry-aware trust-region BO.
- `src/benchmark/`: Benchmark wrappers and problem definitions.
- `experiments/`: Reusable experiment utility modules and helpers.
- `figures/`: Code used to generate paper figures.

## Dependencies and attribution

This codebase uses external source components from:

- Regional Expected Improvement (REI): [Nobuo-Namura/regional-expected-improvement](https://github.com/Nobuo-Namura/regional-expected-improvement)
- BOUNCE algorithm: [lpapenme/bounce](https://github.com/lpapenme/bounce/tree/main)
- Lasso benchmark suite: [ksehic/LassoBench](https://github.com/ksehic/LassoBench)

Please cite the corresponding works when reusing those components.

## Installation

Requirements: Python `>=3.10,<3.13`.

Install with Poetry (recommended):

```bash
poetry install
```

For tests and development extras:

```bash
poetry install --with pytest
```

## Running paper benchmarks

The primary script for paper experiments is `run_benchmark_riemann_tr_paper.py`.

Example commands from repository root:

```bash
python run_benchmark_riemann_tr_paper.py --problems HPA101-0 --n_trials 11
python run_benchmark_riemann_tr_paper.py --problems HPA101-0 --methods "diag_grad_rms:best,diag_grad_rms:rei,dsp"
python run_benchmark_riemann_tr_paper.py --problems MOPTA08 --methods bounce --max_evals 50
```

Results are written under `riemann_tr_paper/{problem_name}/{acqf}_{timestamp}/{method_name}/` and include per-seed logs, configs, trajectories, and comparison plots.

## Optional benchmark assets

- HPA problems: place the HPA assets under `src/benchmark/hpa/` as expected by the wrappers.
- MOPTA08: place the MOPTA08 binaries under `src/benchmark/mopta08/`.

## License

This project is distributed under the BSD 3-Clause License. See `LICENSE`.
