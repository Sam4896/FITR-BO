"""
Configuration utilities for saving and loading experiment configs.

This module provides utilities to save and load experiment configurations
in a consistent format.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def save_config(
    config: Any,
    config_file: Path,
    additional_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save experiment configuration to a JSON file.

    Handles conversion of dataclasses, torch types, and other non-serializable
    types to JSON-compatible formats.

    Args:
        config: Configuration object (typically a dataclass).
        config_file: Path to the config file to save.
        additional_fields: Optional dictionary of additional fields to add to config.

    Raises:
        IOError: If the file cannot be written.
    """
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    if hasattr(config, "__dataclass_fields__"):
        config_dict = asdict(config)
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise ValueError(f"Config must be a dataclass or dict, got {type(config)}")

    # Convert nested dataclasses
    if "problem_config" in config_dict:
        if hasattr(config_dict["problem_config"], "__dataclass_fields__"):
            config_dict["problem_config"] = asdict(config_dict["problem_config"])

    # Convert nested dataclasses (eps_cfg)
    if "eps_cfg" in config_dict and config_dict["eps_cfg"] is not None:
        if hasattr(config_dict["eps_cfg"], "__dataclass_fields__"):
            config_dict["eps_cfg"] = asdict(config_dict["eps_cfg"])
        # Handle enum values in eps_cfg
        if (
            isinstance(config_dict["eps_cfg"], dict)
            and "mode" in config_dict["eps_cfg"]
        ):
            mode = config_dict["eps_cfg"]["mode"]
            if hasattr(mode, "value"):
                config_dict["eps_cfg"]["mode"] = mode.value
            elif hasattr(mode, "name"):
                config_dict["eps_cfg"]["mode"] = mode.name

    # Convert torch types to strings
    if "device" in config_dict:
        config_dict["device"] = str(config_dict["device"])
    if "dtype" in config_dict:
        config_dict["dtype"] = str(config_dict["dtype"])

    # Add additional fields
    if additional_fields:
        config_dict.update(additional_fields)

    # Save to file
    try:
        with open(config_file, "w") as f:
            # Some configs may contain non-JSON-native objects (e.g. Path).
            # Use `default=str` to keep experiments from failing mid-run.
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Saved config to {config_file}")
    except IOError as e:
        logger.error(f"Failed to save config to {config_file}: {e}")
        raise


def load_config(config_file: Path) -> Dict[str, Any]:
    """
    Load experiment configuration from a JSON file.

    Args:
        config_file: Path to the config file to load.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        logger.info(f"Loaded config from {config_file}")
        return config_dict
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {config_file}: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to read config from {config_file}: {e}")
        raise
