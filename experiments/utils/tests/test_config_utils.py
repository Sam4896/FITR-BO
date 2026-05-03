"""
Unit tests for config_utils.
"""

import json
import pytest
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from experiments.utils.config_utils import save_config, load_config


@dataclass
class ProblemConfigForTest:
    """Test problem config dataclass."""
    name: str
    dim: int
    problem_string: str


@dataclass
class ExperimentConfigForTest:
    """Test experiment config dataclass."""
    problem_config: ProblemConfigForTest
    batch_size: int = 1
    max_evals: int = 500
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float64


class TestSaveConfig:
    """Tests for save_config function."""

    def test_saves_dataclass_config(self, tmp_path):
        """Test saving a dataclass config."""
        config_file = tmp_path / "config.json"
        problem = ProblemConfigForTest(name="test", dim=10, problem_string="test")
        config = ExperimentConfigForTest(problem_config=problem)
        
        save_config(config, config_file)
        
        assert config_file.exists()
        with open(config_file) as f:
            data = json.load(f)
        assert data["problem_config"]["name"] == "test"
        assert data["batch_size"] == 1

    def test_saves_dict_config(self, tmp_path):
        """Test saving a dict config."""
        config_file = tmp_path / "config.json"
        config = {"batch_size": 1, "max_evals": 500}
        
        save_config(config, config_file)
        
        assert config_file.exists()
        with open(config_file) as f:
            data = json.load(f)
        assert data["batch_size"] == 1

    def test_saves_additional_fields(self, tmp_path):
        """Test saving config with additional fields."""
        config_file = tmp_path / "config.json"
        config = {"batch_size": 1}
        additional = {"git_commit_id": "abc123", "experiment_name": "test"}
        
        save_config(config, config_file, additional_fields=additional)
        
        with open(config_file) as f:
            data = json.load(f)
        assert data["git_commit_id"] == "abc123"
        assert data["experiment_name"] == "test"
        assert data["batch_size"] == 1

    def test_converts_torch_types(self, tmp_path):
        """Test that torch types are converted to strings."""
        config_file = tmp_path / "config.json"
        config = ExperimentConfigForTest(
            problem_config=ProblemConfigForTest(name="test", dim=10, problem_string="test"),
            device=torch.device("cuda:0"),
            dtype=torch.float64,
        )
        
        save_config(config, config_file)
        
        with open(config_file) as f:
            data = json.load(f)
        assert isinstance(data["device"], str)
        assert isinstance(data["dtype"], str)

    def test_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        config_file = tmp_path / "nested" / "dir" / "config.json"
        config = {"test": "value"}
        
        save_config(config, config_file)
        
        assert config_file.exists()

    def test_serializes_path_objects(self, tmp_path):
        """Test that Path values inside dict configs serialize cleanly."""
        config_file = tmp_path / "config.json"
        out_dir = tmp_path / "some_output_dir"
        config = {"batch_size": 1, "output_dir": out_dir}

        save_config(config, config_file)

        assert config_file.exists()
        with open(config_file) as f:
            data = json.load(f)
        assert data["output_dir"] == str(out_dir)

    def test_raises_error_for_invalid_config(self, tmp_path):
        """Test that error is raised for invalid config type."""
        config_file = tmp_path / "config.json"
        
        with pytest.raises(ValueError, match="Config must be a dataclass or dict"):
            save_config("not a config", config_file)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_valid_config(self, tmp_path):
        """Test loading a valid config file."""
        config_file = tmp_path / "config.json"
        config_data = {"batch_size": 1, "max_evals": 500}
        
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        loaded = load_config(config_file)
        assert loaded["batch_size"] == 1
        assert loaded["max_evals"] == 500

    def test_raises_error_when_file_not_exists(self, tmp_path):
        """Test that error is raised when file doesn't exist."""
        config_file = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load_config(config_file)

    def test_raises_error_for_invalid_json(self, tmp_path):
        """Test that error is raised for invalid JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text("invalid json {")
        
        with pytest.raises(json.JSONDecodeError):
            load_config(config_file)
