"""
Unit tests for logging_utils.
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import patch

from experiments.utils.logging_utils import setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_creates_logger(self, tmp_path):
        """Test that a logger is created."""
        logger = setup_logging(tmp_path)
        assert isinstance(logger, logging.Logger)

    def test_creates_log_file(self, tmp_path):
        """Test that a log file is created."""
        log_file = tmp_path / "experiment.log"
        setup_logging(tmp_path)
        assert log_file.exists()

    def test_creates_custom_log_file(self, tmp_path):
        """Test creating a custom log file name."""
        log_file = tmp_path / "custom.log"
        setup_logging(tmp_path, log_file_name="custom.log")
        assert log_file.exists()

    def test_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        exp_dir = tmp_path / "nested" / "dir"
        setup_logging(exp_dir)
        assert exp_dir.exists()

    def test_logger_has_file_handler(self, tmp_path):
        """Test that logger has a file handler."""
        logger = setup_logging(tmp_path)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0

    def test_logger_has_console_handler(self, tmp_path):
        """Test that logger has a console handler."""
        logger = setup_logging(tmp_path)
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) > 0

    def test_logger_does_not_propagate(self, tmp_path):
        """Test that logger does not propagate to root."""
        logger = setup_logging(tmp_path)
        assert logger.propagate is False

    def test_logger_uses_specified_level(self, tmp_path):
        """Test that logger uses specified log level."""
        logger = setup_logging(tmp_path, log_level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_writes_to_log_file(self, tmp_path):
        """Test that messages are written to log file."""
        logger = setup_logging(tmp_path)
        logger.info("Test message")
        
        log_file = tmp_path / "experiment.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
