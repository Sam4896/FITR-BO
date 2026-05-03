"""
Unit tests for path_utils.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.utils.path_utils import (
    get_project_root,
    get_results_base_dir,
    get_source_dir,
    ensure_experiment_dir,
    get_experiment_dir_from_path,
)


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_finds_project_root_from_current_dir(self, tmp_path):
        """Test finding project root when already at root."""
        # Create a mock project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "experiments").mkdir()
        
        with patch("experiments.utils.path_utils.Path.cwd", return_value=tmp_path):
            root = get_project_root()
            assert root == tmp_path

    def test_finds_project_root_from_subdirectory(self, tmp_path):
        """Test finding project root from a subdirectory."""
        # Create project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "experiments").mkdir()
        (tmp_path / "experiments" / "utils").mkdir(parents=True)
        
        subdir = tmp_path / "experiments" / "utils"
        
        with patch("experiments.utils.path_utils.Path.cwd", return_value=subdir):
            root = get_project_root()
            assert root == tmp_path

    def test_raises_error_when_no_project_root(self, tmp_path):
        """Test that error is raised when project root cannot be found."""
        # Create a directory without src and experiments
        (tmp_path / "some_dir").mkdir()
        
        with patch("experiments.utils.path_utils.Path.cwd", return_value=tmp_path):
            with pytest.raises(RuntimeError, match="Could not find project root"):
                get_project_root()


class TestGetResultsBaseDir:
    """Tests for get_results_base_dir function."""

    def test_returns_existing_results_dir(self, tmp_path):
        """Test returning existing results directory."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        with patch("experiments.utils.path_utils.get_project_root", return_value=tmp_path):
            base_dir = get_results_base_dir()
            assert base_dir == results_dir

    def test_returns_experiments_results_dir(self, tmp_path):
        """Test returning experiments/results directory."""
        exp_results_dir = tmp_path / "experiments" / "results"
        exp_results_dir.mkdir(parents=True)
        
        with patch("experiments.utils.path_utils.get_project_root", return_value=tmp_path):
            base_dir = get_results_base_dir()
            assert base_dir == exp_results_dir

    def test_creates_default_results_dir(self, tmp_path):
        """Test creating default results directory if none exists."""
        with patch("experiments.utils.path_utils.get_project_root", return_value=tmp_path):
            base_dir = get_results_base_dir()
            assert base_dir == tmp_path / "results"
            assert base_dir.exists()


class TestGetSourceDir:
    """Tests for get_source_dir function."""

    def test_returns_absolute_path(self, tmp_path):
        """Test that get_source_dir returns absolute path."""
        src_dir = tmp_path / "src" / "riemannTuRBO"
        src_dir.mkdir(parents=True)
        
        with patch("experiments.utils.path_utils.get_project_root", return_value=tmp_path):
            result = get_source_dir("src/riemannTuRBO")
            assert result == src_dir.resolve()
            assert result.is_absolute()

    def test_raises_error_when_dir_not_exists(self, tmp_path):
        """Test that error is raised when source directory doesn't exist."""
        with patch("experiments.utils.path_utils.get_project_root", return_value=tmp_path):
            with pytest.raises(ValueError, match="Source directory does not exist"):
                get_source_dir("src/nonexistent")


class TestEnsureExperimentDir:
    """Tests for ensure_experiment_dir function."""

    def test_creates_single_level_dir(self, tmp_path):
        """Test creating a single-level directory."""
        result = ensure_experiment_dir(tmp_path, "test")
        assert result == (tmp_path / "test").resolve()
        assert result.exists()

    def test_creates_nested_dirs(self, tmp_path):
        """Test creating nested directories."""
        result = ensure_experiment_dir(tmp_path, "level1", "level2", "level3")
        assert result == (tmp_path / "level1" / "level2" / "level3").resolve()
        assert result.exists()

    def test_returns_absolute_path(self, tmp_path):
        """Test that function returns absolute path."""
        result = ensure_experiment_dir(tmp_path, "test")
        assert result.is_absolute()


class TestGetExperimentDirFromPath:
    """Tests for get_experiment_dir_from_path function."""

    def test_finds_experiment_dir_with_config(self, tmp_path):
        """Test finding experiment dir by config.json."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()
        (exp_dir / "config.json").touch()
        
        result = get_experiment_dir_from_path(exp_dir / "some_file.txt")
        assert result == exp_dir

    def test_finds_experiment_dir_with_log(self, tmp_path):
        """Test finding experiment dir by experiment.log."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()
        (exp_dir / "experiment.log").touch()
        
        result = get_experiment_dir_from_path(exp_dir / "subdir" / "file.txt")
        assert result == exp_dir

    def test_returns_none_when_not_found(self, tmp_path):
        """Test returning None when experiment dir not found."""
        some_dir = tmp_path / "some_dir"
        some_dir.mkdir()
        
        result = get_experiment_dir_from_path(some_dir)
        assert result is None
