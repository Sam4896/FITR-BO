"""
Git commit utility for tracking code state in experiments.

This module provides utilities to commit source code changes and retrieve
commit IDs for experiment reproducibility.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from .path_utils import get_source_dir

logger = logging.getLogger(__name__)


def get_git_repo_root() -> Optional[Path]:
    """Get the root directory of the git repository.

    Returns:
        Optional[Path]: Path to git repo root, or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_current_commit_id() -> Optional[str]:
    """Get the current git commit ID (HEAD).

    Returns:
        Optional[str]: Current commit ID (short hash), or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def has_uncommitted_changes(src_dir_rel: Path) -> bool:
    """Check if there are uncommitted changes in the source directory.

    Args:
        src_dir_rel: Path to the source directory relative to repo root.

    Returns:
        bool: True if there are uncommitted changes, False otherwise.
    """
    try:
        # Check if src_dir is tracked by git
        result = subprocess.run(
            ["git", "status", "--porcelain", str(src_dir_rel)],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip()) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def commit_source_changes(
    src_dir: Path, commit_message: str, dry_run: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Commit changes in the source directory and return the commit ID.

    This function:
    1. Checks if there are uncommitted changes in src_dir
    2. If yes, stages and commits them with the provided message
    3. Returns the commit ID (short hash)

    Args:
        src_dir: Path to the source directory to commit (e.g., src/riemannTuRBO).
            Can be relative to current working directory or absolute.
        commit_message: Commit message to use.
        dry_run: If True, don't actually commit, just check what would happen.

    Returns:
        Tuple[bool, Optional[str]]: (success, commit_id)
            - success: True if commit was successful or no changes needed
            - commit_id: The commit ID (short hash) if successful, None otherwise
    """
    repo_root = get_git_repo_root()
    if repo_root is None:
        logger.warning("Not in a git repository. Cannot commit changes.")
        return False, None

    # Convert to absolute path if relative
    if not src_dir.is_absolute():
        src_dir = Path.cwd() / src_dir

    # Make src_dir relative to repo root
    try:
        src_dir_rel = src_dir.relative_to(repo_root)
    except ValueError:
        logger.warning(
            f"Source directory {src_dir} is not within git repository {repo_root}"
        )
        return False, None

    # Check if there are uncommitted changes
    if not has_uncommitted_changes(src_dir_rel):
        logger.info(f"No uncommitted changes in {src_dir_rel}. Using current commit.")
        commit_id = get_current_commit_id()
        return True, commit_id

    if dry_run:
        logger.info(
            f"[DRY RUN] Would commit changes in {src_dir_rel} with message: {commit_message}"
        )
        return True, None

    try:
        # Stage all changes in src_dir
        logger.info(f"Staging changes in {src_dir_rel}...")
        subprocess.run(
            ["git", "add", str(src_dir_rel)],
            check=True,
            capture_output=True,
        )

        # Commit with the provided message
        logger.info(f"Committing changes with message: {commit_message}")
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            check=True,
            capture_output=True,
        )

        # Get the new commit ID
        commit_id = get_current_commit_id()
        logger.info(f"Successfully committed changes. Commit ID: {commit_id}")
        return True, commit_id

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to commit changes: {e}")
        return False, None


def get_or_create_commit_id(
    src_dir: Optional[Path] = None, experiment_name: str = "", dry_run: bool = False
) -> Optional[str]:
    """
    Get or create a commit ID for the current source code state.

    This is a convenience function that:
    1. Checks if there are uncommitted changes
    2. If yes, commits them with a message based on experiment_name
    3. Returns the commit ID

    Args:
        src_dir: Path to the source directory. If None, uses default "src/riemannTuRBO"
            relative to project root. Can be relative to project root or absolute.
        experiment_name: Name of the experiment (used in commit message).
        dry_run: If True, don't actually commit.

    Returns:
        Optional[str]: Commit ID (short hash), or None if failed.
    """
    # Use default source directory if not provided
    if src_dir is None:
        try:
            src_dir = get_source_dir("src/riemannTuRBO")
        except ValueError:
            logger.warning(
                "Could not find src/riemannTuRBO directory. Using current directory."
            )
            src_dir = Path("src/riemannTuRBO")
    elif isinstance(src_dir, str):
        # If string, try to resolve relative to project root
        try:
            src_dir = get_source_dir(src_dir)
        except ValueError:
            # Fall back to treating as relative to current directory
            src_dir = Path(src_dir)
            if not src_dir.is_absolute():
                src_dir = Path.cwd() / src_dir

    commit_message = (
        f"Experiment: {experiment_name}"
        if experiment_name
        else "Experiment: auto-commit"
    )
    success, commit_id = commit_source_changes(src_dir, commit_message, dry_run=dry_run)
    return commit_id if success else None
