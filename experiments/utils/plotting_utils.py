"""
Plotting utilities for experiment visualization.

This module provides utilities to create and save comparison plots
for experiment results.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_results_comparison(
    all_results: Dict[str, Dict],
    experiment_dir: Path,
    title: str,
    optimal_value: Optional[float] = None,
    acqf: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 150,
) -> Path:
    """
    Plot and save comparison plots for experiment results.

    Args:
        all_results: Dictionary mapping method names to results dictionaries.
            Each results dict must have a "Y" key with evaluation values.
        experiment_dir: Directory to save the plot.
        title: Plot title.
        optimal_value: Optional optimal value to display as a horizontal line.
        acqf: Optional acquisition function name (used in filename).
        figsize: Figure size tuple (width, height).
        dpi: DPI for saved figure.

    Returns:
        Path: Path to the saved plot file.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each method
    for method_name, results in all_results.items():
        Y = results["Y"]
        # Convert to numpy if needed
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)

        # Compute cumulative maximum (best so far)
        fx = np.maximum.accumulate(Y)
        ax.plot(fx, marker="", lw=2, label=method_name)

    # Add optimal value line if available
    if optimal_value is not None:
        ax.axhline(
            y=optimal_value, color="k", linestyle="--", lw=2, label="Global optimum"
        )

    ax.set_xlabel("Number of evaluations", fontsize=14)
    ax.set_ylabel("Best function value", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(left=0)

    plt.tight_layout()

    # Determine filename
    if acqf:
        plot_file = experiment_dir / f"results_comparison_{acqf}.png"
    else:
        plot_file = experiment_dir / "results_comparison.png"

    plt.savefig(plot_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison plot to {plot_file}")
    return plot_file


def plot_multi_trial_comparison(
    experiment_base_dir: Path,
    title: str,
    optimal_value: Optional[float] = None,
    acqf: Optional[str] = None,
    figsize: tuple = (10, 6),
    dpi: int = 150,
    plot_filename: Optional[str] = None,
    band: str = "minmax",
) -> Path:
    """
    Plot mean and shaded band of best observed Y across seeds for each transform.

    Expects directory layout:
        experiment_base_dir / {transform_name} / {seed}_best_observed.csv

    Each CSV has columns: n_evals, best_observed_y.

    Args:
        experiment_base_dir: Directory containing one subdir per transform.
        title: Plot title.
        optimal_value: Optional optimal value (horizontal line).
        acqf: Optional acquisition function (for filename).
        figsize: Figure size.
        dpi: DPI for saved figure.
        plot_filename: Output filename. If None, uses band to set default.
        band: Shaded region type: "minmax" (min to max) or "std" (mean ± std).

    Returns:
        Path to the saved plot file.
    """
    experiment_base_dir = Path(experiment_base_dir)
    if plot_filename is None:
        plot_filename = f"results_comparison_multi_trial_{band}.png"
    fig, ax = plt.subplots(figsize=figsize)

    transform_dirs = sorted(
        d for d in experiment_base_dir.iterdir() if d.is_dir()
    )

    for transform_dir in transform_dirs:
        method_name = transform_dir.name
        csv_files = sorted(
            transform_dir.glob("*_best_observed.csv"),
            key=lambda p: int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else 0,
        )
        if not csv_files:
            continue

        curves = []
        for csv_file in csv_files:
            data = np.genfromtxt(
                csv_file, delimiter=",", skip_header=1, dtype=float
            )
            if data.ndim == 1:
                data = np.atleast_2d(data)
            n_evals = data[:, 0]
            best_y = data[:, 1]
            curves.append((n_evals, best_y))

        # Align by minimum length so all runs contribute
        min_len = min(len(c[0]) for c in curves)
        stacked = np.array([c[1][:min_len] for c in curves])
        n_evals_common = curves[0][0][:min_len]
        n_seeds = stacked.shape[0]

        mean_y = np.mean(stacked, axis=0)
        try:
            if band == "std":
                # With one seed there is no std; plot single run as mean line only
                if n_seeds <= 1:
                    band_low = mean_y
                    band_high = mean_y
                else:
                    std_y = np.std(stacked, axis=0)
                    band_low = mean_y - std_y
                    band_high = mean_y + std_y
            else:
                band_low = np.min(stacked, axis=0)
                band_high = np.max(stacked, axis=0)
        except (ValueError, TypeError, FloatingPointError) as e:
            logger.warning(
                f"Band computation failed for {method_name} ({e}); plotting mean only."
            )
            band_low = mean_y
            band_high = mean_y

        ax.plot(n_evals_common, mean_y, lw=2, label=method_name)
        ax.fill_between(n_evals_common, band_low, band_high, alpha=0.25)

    if optimal_value is not None:
        ax.axhline(
            y=optimal_value, color="k", linestyle="--", lw=2, label="Global optimum"
        )

    ax.set_xlabel("Number of evaluations", fontsize=14)
    ax.set_ylabel("Best function value", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plot_file = experiment_base_dir / plot_filename
    plt.savefig(plot_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    band_label = "min-max" if band == "minmax" else "mean ± std"
    logger.info(f"Saved multi-trial comparison plot ({band_label}) to {plot_file}")
    return plot_file
