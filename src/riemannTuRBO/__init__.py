"""
Riemannian TuRBO: Trust Region Bayesian Optimization with Local Geometry
=========================================================================

This package provides Riemannian Trust Region methods for Bayesian Optimization.
The key idea is to shape the trust region using local sensitivity information
from the surrogate model, rather than using axis-aligned boxes.

Main Components
---------------
1. **Transform Operators**: Map latent z-space to input x-space
   - IdentityTransform (baseline)
   - DiagGradMeanTransform / DiagGradRMSTransform (axis-aligned)
   - LowRankSVDTransform (captures rotations)
   - ARDLengthscaleTransform (TuRBO-style)
   - FiniteDiffTransform (model-agnostic)

2. **Acquisition Wrapper**: Transforms any BoTorch acqf to operate in z-space

3. **State Management**: TuRBO-style TR updates

Quick Start
-----------
>>> from riemannTuRBO import LowRankSVDTransform, TrustRegionWrappedAcquisitionFunction
>>> from botorch.sampling import SobolQMCNormalSampler
>>> from botorch.acquisition import qLogExpectedImprovement
>>>
>>> # 1. Create sampler
>>> sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]))
>>>
>>> # 2. Create transform (auto-computes on init)
>>> transform = LowRankSVDTransform(model, x_best, 0.5, sampler)
>>>
>>> # 3. Wrap acquisition function
>>> acq = qLogExpectedImprovement(model, best_f=Y.max())
>>> wrapped = TrustRegionWrappedAcquisitionFunction(acq, transform)
>>>
>>> # 4. Optimize in z-space
>>> z_opt, _ = optimize_acqf(wrapped, bounds=wrapped.z_bounds, q=1, ...)
>>> x_opt = wrapped.map_z_to_x(z_opt)
>>>
>>> # 5. Use transform properties directly
>>> print(f"Epsilon used: {transform.eps_used}")
>>> print(f"Z-bounds: {transform.z_bounds}")
"""

from __future__ import annotations

# Epsilon configuration
from .eps_config import (
    EpsConfig,
    EpsMode,
    compute_eps_from_eigs,
)

# Base classes
from .base import (
    AxisAlignedTR,
    RotatedTR,
    TransformConfig,
    TransformOperator,
    TrustRegion,
)

# Transform implementations
from .diagonal import (
    DiagGradMeanTransform,
    DiagGradRMSTransform,
    FiniteDiffTransform,
)
from .lowrank_svd import (
    LowRankSVDTransform,
)
from .identity import (
    IdentityTransform,
    ARDLengthscaleTransform,
)

# Acquisition wrapper
from .acquisition import (
    TrustRegionWrappedAcquisitionFunction,
    TransformDiagnostics,
    make_trust_region_acqf,
)

# State management
from .state import TurboState

# Center selection
from .center_selection import (
    CenterSelector,
    BestObservedSelector,
    LastObservedSelector,
    RestartCenterSelector,
    REISelector,
    UCBSelector,
    get_center_selector,
)

# Utilities (for advanced users)
from .utils import (
    symmetrize,
    spd_sqrt,
    spd_inverse_sqrt,
    get_fisher_grads_from_posterior,
    probe_linear_operator_matrix,
    geometric_mean_singular_value,
)


__all__ = [
    # Epsilon
    "EpsConfig",
    "EpsMode",
    "compute_eps_from_eigs",
    # Base
    "TrustRegion",
    "AxisAlignedTR",
    "RotatedTR",
    "TransformConfig",
    "TransformOperator",
    # Transforms
    "IdentityTransform",
    "ARDLengthscaleTransform",
    "DiagGradMeanTransform",
    "DiagGradRMSTransform",
    "LowRankSVDTransform",
    "FiniteDiffTransform",
    # Acquisition
    "TrustRegionWrappedAcquisitionFunction",
    "TransformDiagnostics",
    "make_trust_region_acqf",
    # State
    "TurboState",
    # Center selection
    "CenterSelector",
    "BestObservedSelector",
    "LastObservedSelector",
    "RestartCenterSelector",
    "REISelector",
    "UCBSelector",
    "get_center_selector",
    # Utilities
    "symmetrize",
    "spd_sqrt",
    "spd_inverse_sqrt",
    "get_fisher_grads_from_posterior",
    "probe_linear_operator_matrix",
    "geometric_mean_singular_value",
]

__version__ = "0.2.0"
