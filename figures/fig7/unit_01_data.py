"""Figure 7 data: same surrogate setup as Figure 6."""

from __future__ import annotations

from neurips_viz.fig6.unit_01_data import (  # re-export for fig7
    build_deep_ensemble_bnn,
    build_gp_surrogate,
    build_ibnn_surrogate,
    get_training_data,
)

__all__ = [
    "build_gp_surrogate",
    "build_ibnn_surrogate",
    "build_deep_ensemble_bnn",
    "get_training_data",
]
