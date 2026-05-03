"""
Logging utilities for experiments.

This module provides utilities to set up logging for experiments
in a consistent way.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(
    experiment_dir: Path,
    log_level: int = logging.INFO,
    log_file_name: str = "experiment.log",
) -> logging.Logger:
    """
    Set up logging to both file and console.

    Creates a named logger for the experiment to avoid duplicate messages
    when multiple experiments run in the same process.

    Parameters
    ----------
    experiment_dir : Path
        Directory to store log file.
    log_level : int
        Logging level (default: INFO).
    log_file_name : str
        Name of the log file (default: "experiment.log").

    Returns
    -------
    logging.Logger
        Logger configured for the experiment.
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log_file = experiment_dir / log_file_name

    # Use root logger with simple format (old style)
    logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(log_level)

    # Create simple formatters (old format - no logger name)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s",
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the named logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
