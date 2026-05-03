"""
Unit tests for commit_util.

Note: These tests use mocking to avoid actually running git commands.
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from experiments.utils.commit_util import (
    get_git_repo_root,
    get_current_commit_id,
    has_uncommitted_changes,
    commit_source_changes,
    get_or_create_commit_id,
)


class TestGetGitRepoRoot:
    """Tests for get_git_repo_root function."""

    def test_returns_repo_root_when_in_git_repo(self):
        """Test returning repo root when in a git repository."""
        mock_root = Path("/mock/repo/root")
        
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=f"{mock_root}\n", returncode=0)
            result = get_git_repo_root()
            assert result == mock_root

    def test_returns_none_when_not_in_git_repo(self):
        """Test returning None when not in a git repository."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            result = get_git_repo_root()
            assert result is None

    def test_returns_none_when_git_not_available(self):
        """Test returning None when git is not available."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = get_git_repo_root()
            assert result is None


class TestGetCurrentCommitId:
    """Tests for get_current_commit_id function."""

    def test_returns_commit_id(self):
        """Test returning current commit ID."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="abc123\n", returncode=0)
            result = get_current_commit_id()
            assert result == "abc123"

    def test_returns_none_on_error(self):
        """Test returning None on error."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            result = get_current_commit_id()
            assert result is None


class TestHasUncommittedChanges:
    """Tests for has_uncommitted_changes function."""

    def test_returns_true_when_changes_exist(self):
        """Test returning True when uncommitted changes exist."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=" M src/riemannTuRBO/base.py\n", returncode=0)
            result = has_uncommitted_changes(Path("src/riemannTuRBO"))
            assert result is True

    def test_returns_false_when_no_changes(self):
        """Test returning False when no uncommitted changes."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            result = has_uncommitted_changes(Path("src/riemannTuRBO"))
            assert result is False

    def test_returns_false_on_error(self):
        """Test returning False on error."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            result = has_uncommitted_changes(Path("src/riemannTuRBO"))
            assert result is False


class TestCommitSourceChanges:
    """Tests for commit_source_changes function."""

    def test_returns_current_commit_when_no_changes(self, tmp_path):
        """Test returning current commit when no changes exist."""
        src_dir = tmp_path / "src" / "riemannTuRBO"
        src_dir.mkdir(parents=True)
        
        with patch("experiments.utils.commit_util.get_git_repo_root", return_value=tmp_path):
            with patch("experiments.utils.commit_util.has_uncommitted_changes", return_value=False):
                with patch("experiments.utils.commit_util.get_current_commit_id", return_value="abc123"):
                    success, commit_id = commit_source_changes(src_dir, "Test message")
                    assert success is True
                    assert commit_id == "abc123"

    def test_commits_changes_when_they_exist(self, tmp_path):
        """Test committing changes when they exist."""
        src_dir = tmp_path / "src" / "riemannTuRBO"
        src_dir.mkdir(parents=True)
        
        with patch("experiments.utils.commit_util.get_git_repo_root", return_value=tmp_path):
            with patch("experiments.utils.commit_util.has_uncommitted_changes", return_value=True):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    with patch("experiments.utils.commit_util.get_current_commit_id", return_value="def456"):
                        success, commit_id = commit_source_changes(src_dir, "Test message")
                        assert success is True
                        assert commit_id == "def456"
                        # Verify git add and git commit were called
                        assert mock_run.call_count >= 2

    def test_returns_false_when_not_in_git_repo(self, tmp_path):
        """Test returning False when not in a git repository."""
        src_dir = tmp_path / "src" / "riemannTuRBO"
        
        with patch("experiments.utils.commit_util.get_git_repo_root", return_value=None):
            success, commit_id = commit_source_changes(src_dir, "Test message")
            assert success is False
            assert commit_id is None


class TestGetOrCreateCommitId:
    """Tests for get_or_create_commit_id function."""

    def test_uses_default_source_dir_when_none_provided(self):
        """Test using default source directory when None provided."""
        with patch("experiments.utils.commit_util.get_source_dir") as mock_get:
            mock_dir = Path("/mock/src/riemannTuRBO")
            mock_get.return_value = mock_dir
            with patch("experiments.utils.commit_util.commit_source_changes", return_value=(True, "abc123")):
                result = get_or_create_commit_id(experiment_name="test")
                assert result == "abc123"
                mock_get.assert_called_once_with("src/riemannTuRBO")

    def test_handles_string_source_dir(self):
        """Test handling string source directory."""
        with patch("experiments.utils.commit_util.get_source_dir") as mock_get:
            mock_dir = Path("/mock/src/riemannTuRBO")
            mock_get.return_value = mock_dir
            with patch("experiments.utils.commit_util.commit_source_changes", return_value=(True, "abc123")):
                result = get_or_create_commit_id("src/riemannTuRBO", "test")
                assert result == "abc123"

    def test_returns_none_on_failure(self):
        """Test returning None on failure."""
        with patch("experiments.utils.commit_util.get_source_dir") as mock_get:
            mock_dir = Path("/mock/src/riemannTuRBO")
            mock_get.return_value = mock_dir
            with patch("experiments.utils.commit_util.commit_source_changes", return_value=(False, None)):
                result = get_or_create_commit_id(experiment_name="test")
                assert result is None
