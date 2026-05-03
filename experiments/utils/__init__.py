"""
Utilities for experiments.

This package contains utility modules for experiments:
- device_utils: GPU/device management
- commit_util: Git commit tracking for reproducibility
- path_utils: Consistent path handling across systems
- config_utils: Configuration save/load utilities
- result_utils: Result serialization utilities
- plotting_utils: Plotting and visualization utilities
- logging_utils: Logging setup utilities
"""

from .device_utils import setup_device
from .commit_util import (
    get_or_create_commit_id,
    commit_source_changes,
    get_current_commit_id,
)
from .path_utils import (
    get_project_root,
    get_results_base_dir,
    get_source_dir,
    ensure_experiment_dir,
    get_experiment_dir_from_path,
)
from .config_utils import save_config, load_config
from .result_utils import (
    save_results,
    load_results,
    serialize_results,
    save_best_observed_csv,
    build_observed_data_from_flat,
)
from .plotting_utils import plot_results_comparison, plot_multi_trial_comparison
from .logging_utils import setup_logging

__all__ = [
    # Device utilities
    "setup_device",
    # Commit utilities
    "get_or_create_commit_id",
    "commit_source_changes",
    "get_current_commit_id",
    # Path utilities
    "get_project_root",
    "get_results_base_dir",
    "get_source_dir",
    "ensure_experiment_dir",
    "get_experiment_dir_from_path",
    # Config utilities
    "save_config",
    "load_config",
    # Result utilities
    "save_results",
    "load_results",
    "serialize_results",
    "save_best_observed_csv",
    "build_observed_data_from_flat",
    # Plotting utilities
    "plot_results_comparison",
    "plot_multi_trial_comparison",
    # Logging utilities
    "setup_logging",
]
