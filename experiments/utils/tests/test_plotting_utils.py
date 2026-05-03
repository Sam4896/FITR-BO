"""
Unit tests for plotting_utils.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np

from experiments.utils.plotting_utils import plot_results_comparison


class TestPlotResultsComparison:
    """Tests for plot_results_comparison function."""

    def test_creates_plot_file(self, tmp_path):
        """Test that a plot file is created."""
        all_results = {
            "method1": {"Y": np.array([1.0, 2.0, 3.0])},
            "method2": {"Y": np.array([2.0, 3.0, 4.0])},
        }
        
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            plot_file = plot_results_comparison(
                all_results, tmp_path, "Test Plot"
            )
            assert isinstance(plot_file, Path)

    def test_uses_acqf_in_filename(self, tmp_path):
        """Test that acqf is used in filename when provided."""
        all_results = {"method1": {"Y": np.array([1.0, 2.0])}}
        
        with patch("matplotlib.pyplot.savefig") as mock_save, patch("matplotlib.pyplot.close"):
            plot_file = plot_results_comparison(
                all_results, tmp_path, "Test", acqf="logei"
            )
            assert "logei" in str(plot_file)

    def test_handles_optimal_value(self, tmp_path):
        """Test that optimal value is handled."""
        all_results = {"method1": {"Y": np.array([1.0, 2.0])}}
        
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            plot_file = plot_results_comparison(
                all_results, tmp_path, "Test", optimal_value=0.0
            )
            assert plot_file is not None

    def test_handles_list_y_values(self, tmp_path):
        """Test handling Y values as lists."""
        all_results = {
            "method1": {"Y": [1.0, 2.0, 3.0]},
        }
        
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            plot_file = plot_results_comparison(
                all_results, tmp_path, "Test"
            )
            assert plot_file is not None

    def test_handles_empty_results(self, tmp_path):
        """Test handling empty results."""
        all_results = {}
        
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            plot_file = plot_results_comparison(
                all_results, tmp_path, "Test"
            )
            assert plot_file is not None
