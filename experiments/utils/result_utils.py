"""
Result serialization utilities for experiments.

This module provides utilities to serialize experiment results to JSON
and CSV in a consistent format.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def serialize_value(value: Any) -> Any:
    """
    Serialize a value to a JSON-compatible format.

    Handles:
    - torch.Tensor -> list
    - np.ndarray -> list
    - numpy scalars -> Python scalars
    - Other types -> as-is

    Args:
        value: Value to serialize.

    Returns:
        JSON-serializable value.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, (np.bool_)):
        return bool(value)
    else:
        return value


def serialize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize experiment results to JSON-compatible format.

    Args:
        results: Results dictionary containing experiment outputs.

    Returns:
        Dict[str, Any]: Serialized results dictionary.
    """
    serialized = {}

    for key, value in results.items():
        if isinstance(value, dict):
            serialized[key] = serialize_results(value)
        elif isinstance(value, list):
            serialized[key] = [
                serialize_value(item)
                if isinstance(item, (torch.Tensor, np.ndarray))
                else serialize_results(item)
                if isinstance(item, dict)
                else serialize_value(item)
                for item in value
            ]
        else:
            serialized[key] = serialize_value(value)

    return serialized


def save_results(results: Dict[str, Any], results_file: Path) -> None:
    """
    Save experiment results to a JSON file.

    Args:
        results: Results dictionary to save.
        results_file: Path to the results file.

    Raises:
        IOError: If the file cannot be written.
    """
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Serialize results
    serialized_results = serialize_results(results)

    try:
        with open(results_file, "w") as f:
            json.dump(serialized_results, f, indent=2)
        logger.info(f"Saved results to {results_file}")
    except IOError as e:
        logger.error(f"Failed to save results to {results_file}: {e}")
        raise


def load_results(results_file: Path) -> Dict[str, Any]:
    """
    Load experiment results from a JSON file.

    Args:
        results_file: Path to the results file.

    Returns:
        Dict[str, Any]: Results dictionary.

    Raises:
        FileNotFoundError: If the results file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    results_file = Path(results_file)

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        logger.info(f"Loaded results from {results_file}")
        return results
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {results_file}: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to read results from {results_file}: {e}")
        raise


def build_observed_data_from_flat(
    X: Union[list, np.ndarray],
    Y: Union[list, np.ndarray],
    n_init: int,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Build observed_data structure (initial + iterations) from flat X, Y arrays.

    Used when the runner does not provide observed_data (e.g. Bounce).
    Assumes first n_init rows are initial design, then each batch_size rows
    form one iteration.

    Args:
        X: All inputs, list of lists or (n, d) array.
        Y: All outputs, list of lists or (n, 1) array.
        n_init: Number of initial points.
        batch_size: Number of points per BO iteration (q).

    Returns:
        Dict with "initial" (inputs, outputs) and "iterations" (list of
        {iteration, inputs, outputs}).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = len(X)
    if n < n_init:
        initial_inputs = X.tolist()
        initial_outputs = Y.tolist()
        iterations = []
    else:
        initial_inputs = X[:n_init].tolist()
        initial_outputs = Y[:n_init].tolist()
        iterations = []
        i = 0
        start = n_init
        while start + batch_size <= n:
            iterations.append(
                {
                    "iteration": i + 1,
                    "inputs": X[start : start + batch_size].tolist(),
                    "outputs": Y[start : start + batch_size].tolist(),
                }
            )
            start += batch_size
            i += 1
        if start < n:
            iterations.append(
                {
                    "iteration": i + 1,
                    "inputs": X[start:n].tolist(),
                    "outputs": Y[start:n].tolist(),
                }
            )
    return {
        "initial": {"inputs": initial_inputs, "outputs": initial_outputs},
        "iterations": iterations,
    }


def save_best_observed_csv(
    Y: Union[list, np.ndarray],
    csv_path: Path,
) -> None:
    """
    Write best-so-far (cumulative max) of Y to CSV: n_evals, best_observed_y.

    Args:
        Y: Evaluation values (flattened).
        csv_path: Path to the output CSV file.
    """
    Y = np.asarray(Y).flatten()
    best_y = np.maximum.accumulate(Y)
    n_evals = np.arange(1, len(Y) + 1, dtype=int)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_evals", "best_observed_y"])
        for n, y in zip(n_evals, best_y):
            w.writerow([n, float(y)])
