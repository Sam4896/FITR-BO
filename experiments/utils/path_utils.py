"""
Path utilities for consistent path handling across different systems.

This module provides utilities to handle paths consistently, especially
for experiment directories that may be created in different locations
depending on the system (laptop vs cluster).
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory.

    Assumes the project root contains both 'src' and 'experiments' directories.

    Returns:
        Path: Project root directory.

    Raises:
        RuntimeError: If project root cannot be determined.
    """
    current = Path.cwd().resolve()
    
    # Check if we're already at the root (has both src and experiments)
    if (current / "src").exists() and (current / "experiments").exists():
        return current
    
    # Walk up the directory tree
    for parent in current.parents:
        if (parent / "src").exists() and (parent / "experiments").exists():
            return parent
    
    raise RuntimeError(
        f"Could not find project root. Current directory: {current}. "
        "Expected to find 'src' and 'experiments' directories."
    )


def get_results_base_dir() -> Path:
    """
    Get the base directory for experiment results.

    This function checks for results directories in common locations:
    1. Project root / "results"
    2. Project root / "experiments" / "results"
    
    If neither exists, it returns the project root / "results" as default.

    Returns:
        Path: Base directory for results.
    """
    project_root = get_project_root()
    
    # Check common locations
    candidates = [
        project_root / "results",
        project_root / "experiments" / "results",
    ]
    
    # Return first existing directory, or create the default
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Default to project root / results
    default = project_root / "results"
    default.mkdir(parents=True, exist_ok=True)
    return default


def get_source_dir(relative_path: str = "src/riemannTuRBO") -> Path:
    """
    Get the absolute path to a source directory.

    Args:
        relative_path: Relative path from project root (default: "src/riemannTuRBO").

    Returns:
        Path: Absolute path to the source directory.

    Raises:
        ValueError: If the source directory doesn't exist.
    """
    project_root = get_project_root()
    source_path = project_root / relative_path
    
    if not source_path.exists():
        raise ValueError(
            f"Source directory does not exist: {source_path}. "
            f"Project root: {project_root}"
        )
    
    return source_path.resolve()


def ensure_experiment_dir(base_dir: Path, *subdirs: str) -> Path:
    """
    Ensure an experiment directory exists and return its path.

    Args:
        base_dir: Base directory for experiments.
        *subdirs: Subdirectory components (e.g., "problem_name", "acqf").

    Returns:
        Path: Path to the experiment directory (created if needed).
    """
    experiment_dir = base_dir
    for subdir in subdirs:
        experiment_dir = experiment_dir / subdir
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir.resolve()


def get_experiment_dir_from_path(path: Path) -> Optional[Path]:
    """
    Get the experiment directory from a given path.

    This function tries to find the experiment directory by looking for
    common markers (config.json, experiment.log) in the path or its parents.

    Args:
        path: Path that might be within an experiment directory.

    Returns:
        Optional[Path]: Experiment directory if found, None otherwise.
    """
    path = path.resolve()
    
    # Check if path itself is an experiment dir
    if (path / "config.json").exists() or (path / "experiment.log").exists():
        return path
    
    # Check parents
    for parent in path.parents:
        if (parent / "config.json").exists() or (parent / "experiment.log").exists():
            return parent
    
    return None
