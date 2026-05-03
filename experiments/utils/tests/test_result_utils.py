"""
Unit tests for result_utils.
"""

import json
import pytest
from pathlib import Path

import numpy as np
import torch

from experiments.utils.result_utils import (
    serialize_value,
    serialize_results,
    save_results,
    load_results,
)


class TestSerializeValue:
    """Tests for serialize_value function."""

    def test_serializes_torch_tensor(self):
        """Test serializing a torch tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = serialize_value(tensor)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_serializes_numpy_array(self):
        """Test serializing a numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = serialize_value(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_serializes_numpy_scalar(self):
        """Test serializing a numpy scalar."""
        scalar = np.float64(3.14)
        result = serialize_value(scalar)
        assert result == 3.14
        assert isinstance(result, float)

    def test_serializes_numpy_integer(self):
        """Test serializing a numpy integer."""
        integer = np.int64(42)
        result = serialize_value(integer)
        assert result == 42
        assert isinstance(result, int)

    def test_serializes_numpy_bool(self):
        """Test serializing a numpy boolean."""
        bool_val = np.bool_(True)
        result = serialize_value(bool_val)
        assert result is True
        assert isinstance(result, bool)

    def test_passes_through_regular_types(self):
        """Test that regular Python types pass through unchanged."""
        assert serialize_value(42) == 42
        assert serialize_value(3.14) == 3.14
        assert serialize_value("string") == "string"
        assert serialize_value(True) is True


class TestSerializeResults:
    """Tests for serialize_results function."""

    def test_serializes_simple_dict(self):
        """Test serializing a simple dictionary."""
        results = {
            "Y": torch.tensor([1.0, 2.0]),
            "best_value": 2.0,
            "n_evals": 10,
        }
        serialized = serialize_results(results)
        
        assert isinstance(serialized["Y"], list)
        assert serialized["Y"] == [1.0, 2.0]
        assert serialized["best_value"] == 2.0
        assert serialized["n_evals"] == 10

    def test_serializes_nested_dict(self):
        """Test serializing nested dictionaries."""
        results = {
            "diagnostics": {
                "anisotropy": torch.tensor(5.0),
                "eps_used": 1e-6,
            },
            "state_history": [{"length": 0.8, "best_value": 1.0}],
        }
        serialized = serialize_results(results)
        
        assert isinstance(serialized["diagnostics"]["anisotropy"], (int, float))
        assert serialized["diagnostics"]["eps_used"] == 1e-6
        assert isinstance(serialized["state_history"], list)

    def test_serializes_list_of_tensors(self):
        """Test serializing a list containing tensors."""
        results = {
            "Y_all": [torch.tensor([1.0]), torch.tensor([2.0])],
        }
        serialized = serialize_results(results)
        
        assert isinstance(serialized["Y_all"], list)
        assert all(isinstance(item, list) for item in serialized["Y_all"])


class TestSaveResults:
    """Tests for save_results function."""

    def test_saves_results_to_file(self, tmp_path):
        """Test saving results to a file."""
        results_file = tmp_path / "results.json"
        results = {
            "Y": torch.tensor([1.0, 2.0, 3.0]),
            "best_value": 3.0,
            "n_evals": 10,
        }
        
        save_results(results, results_file)
        
        assert results_file.exists()
        with open(results_file) as f:
            data = json.load(f)
        assert data["best_value"] == 3.0
        assert isinstance(data["Y"], list)

    def test_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        results_file = tmp_path / "nested" / "dir" / "results.json"
        results = {"test": "value"}
        
        save_results(results, results_file)
        
        assert results_file.exists()


class TestLoadResults:
    """Tests for load_results function."""

    def test_loads_results_from_file(self, tmp_path):
        """Test loading results from a file."""
        results_file = tmp_path / "results.json"
        results_data = {
            "Y": [1.0, 2.0, 3.0],
            "best_value": 3.0,
            "n_evals": 10,
        }
        
        with open(results_file, "w") as f:
            json.dump(results_data, f)
        
        loaded = load_results(results_file)
        assert loaded["best_value"] == 3.0
        assert loaded["Y"] == [1.0, 2.0, 3.0]

    def test_raises_error_when_file_not_exists(self, tmp_path):
        """Test that error is raised when file doesn't exist."""
        results_file = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load_results(results_file)
