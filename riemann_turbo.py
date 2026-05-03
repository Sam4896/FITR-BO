"""
Riemannian TuRBO: Trust Region Bayesian Optimization with Local Geometry
=========================================================================

This module provides the main entry point for Riemannian Trust Region BO.
It uses local sensitivity information (Fisher gradients) to shape the
trust region as an anisotropic parallelepiped.

This module is a thin wrapper around the `riemannTuRBO` package, providing
a simple API for batch generation that matches the original TuRBO interface.

For more control, use the `riemannTuRBO` package directly.

Usage
-----
>>> from poc.riemann_turbo import generate_riemannian_batch
>>> from src.riemannTuRBO import TurboState
>>>
>>> state = TurboState(dim=10, q=1)
>>> X_next = generate_riemannian_batch(state, model, X, Y, q=1, acqf="ts")
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union
import logging

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.acquisition import (
    qExpectedImprovement,
    qLogExpectedImprovement,
)
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from botorch.optim.initializers import initialize_q_batch, sample_points_around_best
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import unnormalize

# Import from refactored riemannTuRBO module
from src.riemannTuRBO import (
    # Config
    EpsConfig,
    EpsMode,
    # Transforms (new API - auto-computes on init)
    TrustRegion,
    AxisAlignedTR,
    RotatedTR,
    IdentityTransform,
    DiagGradMeanTransform,
    DiagGradRMSTransform,
    LowRankSVDTransform,
    ARDLengthscaleTransform,
    FiniteDiffTransform,
    # Acquisition
    TrustRegionWrappedAcquisitionFunction,
    # State
    TurboState,
    # Center selection
    BestObservedSelector,
    CenterSelector,
)

logger = logging.getLogger("RiemannianTurbo")


class _DSPBounds:
    """
    DSP (Dimensionally Scaled Priors): full [0, 1]^d bounds, no trust region.
    Uses logEI (or other acqf) on the normalized input range at each iteration.
    """

    _is_dsp = True

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.diagnostics = {}
        self.eps_used = 0.0

    def __call__(self, x_center: Tensor, length: float) -> Tensor:
        d = x_center.shape[-1]
        bounds = torch.tensor(
            [[0.0] * d, [1.0] * d],
            device=x_center.device,
            dtype=x_center.dtype,
        )
        return bounds


_METHOD_TO_TRANSFORM_CLS = {
    "identity": IdentityTransform,
    "diag_grad_mean": DiagGradMeanTransform,
    "diag_grad_rms": DiagGradRMSTransform,
    "lowrank_svd": LowRankSVDTransform,
    "ard_lengthscale": ARDLengthscaleTransform,
    "finite_diff": FiniteDiffTransform,
    "dsp": None,  # Built explicitly; no transform class
}


# Re-export for backward compatibility
__all__ = [
    "TurboState",
    "generate_riemannian_batch",
    "get_initial_points",
    "eval_objective",
    "ensure_y_shape_n1",
]


# =============================================================================
# Batch Generation
# =============================================================================


def generate_riemannian_batch(
    state: TurboState,
    model,
    X: Tensor,
    Y: Tensor,
    q: int,
    n_candidates: int = 1000,
    num_restarts: int = 10,
    raw_samples: int = 512,
    acqf: Literal["ts", "ei", "logei"] = "ts",
    qmc_sample_shape: int = 128,
    center_selector: Optional[CenterSelector] = None,
    transform_method: str = "diag_grad_rms",
    eps_cfg: Optional[EpsConfig] = None,
    volume_normalize: bool = True,
    observation_noise: bool = True,
    return_diagnostics: bool = False,
    use_raasp: bool = True,
) -> Union[Tensor, Tuple[Tensor, dict]]:
    """
    Generate a batch of candidates using Riemannian Trust Region BO.

    This is the main entry point for Riemannian TuRBO candidate generation.

    Parameters
    ----------
    state : TurboState
        Current TuRBO state (contains trust region length).
    model : BoTorch Model
        Fitted surrogate model with .posterior() method.
    X : Tensor
        Observed inputs in normalized space [0, 1]^D, shape (N, D).
        Will be normalized to (N, D) if passed as (D,).
    Y : Tensor
        Observed targets (should be standardized), shape (N, 1).
        Will be normalized to (N, 1) if passed as (N,).
    q : int
        Number of candidates to generate.
    n_candidates : int
        Number of discrete candidates for Thompson Sampling.
    num_restarts : int
        Number of restarts for gradient-based acquisition optimization.
    raw_samples : int
        Number of raw samples for acquisition optimization initialization.
    acqf : {"ts", "ei", "logei"}
        Acquisition function type:
        - "ts": Thompson Sampling (discrete, fast)
        - "ei": q-Expected Improvement (gradient-based)
        - "logei": q-Log Expected Improvement (more stable)
    qmc_sample_shape : int
        Number of MC samples for posterior sampling/acquisition.
    center_selector : CenterSelector or None
        Center selector object used to choose trust region center. If None,
        BestObservedSelector is used. For DSP the center is ignored by bounds.
    transform_method : str
        Which transform method to use. Options:
        - "identity": No preconditioning (baseline)
        - "diag_grad_mean": Diagonal from posterior mean gradient
        - "diag_grad_rms": Diagonal from RMS of Fisher gradients
        - "lowrank_svd": Full low-rank SVD (captures rotations)
        - "ard_lengthscale": From GP's ARD lengthscales
        - "finite_diff": Model-agnostic via finite differences
    eps_cfg : Optional[EpsConfig]
        Epsilon configuration. Defaults to AUTO_TRACE.
    include_polytope_constraints : bool
        If True, and the transform is rotated, pass z-polytope constraints to
        optimize_acqf so BoTorch can sample from the polytope (via hit-and-run)
        instead of sampling from a circumscribed box. Default False.
    return_diagnostics : bool
        If True, returns tuple (X_next, diagnostics_dict).
    observation_noise : bool
        If True, include observation noise when computing Fisher gradients (DiagGradRMS).
        Default True.

    Returns
    -------
    Tensor or Tuple[Tensor, dict]
        Candidate points in normalized space [0, 1]^D.
        If return_diagnostics=True, also returns a dict with:
        - "center": selected trust region center
        - "eps_used": epsilon value used
        - "points_clamped_ratio": fraction of candidates with at least one clamped dimension
        - "mean_dims_clamped_ratio": mean fraction of dims clamped per clamped point
        - "anisotropy": std/mean of scaling factors
        - "z_bounds": the z-space bounds used

    Examples
    --------
    >>> # Simple usage
    >>> X_next = generate_riemannian_batch(state, model, X, Y, q=1, acqf="ts")
    >>>
    >>> # With custom transform
    >>> X_next = generate_riemannian_batch(
    ...     state, model, X, Y, q=1,
    ...     transform_method="diag_grad_rms",
    ...     eps_cfg=EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12),
    ... )
    """
    assert acqf in ["ts", "ei", "logei"], f"Unknown acqf: {acqf}"

    # Normalize shapes: X must be (N, D), Y must be (N, 1)
    if X.dim() == 1:
        X = X.unsqueeze(0)
    Y = ensure_y_shape_n1(Y)

    dim = X.shape[-1]
    device = X.device
    dtype = X.dtype

    if eps_cfg is None:
        eps_cfg = EpsConfig(mode=EpsMode.AUTO_TRACE, jitter=1e-12)

    method_key = transform_method.lower()
    # 1. Select center
    _selector = (
        center_selector if center_selector is not None else BestObservedSelector()
    )
    if method_key == "dsp":
        _selector = BestObservedSelector()  # DSP ignores center, keep selection cheap
    x_center = _selector.select_center(model, X, Y, state.length)
    if x_center.dim() == 1:
        x_center = x_center.unsqueeze(0)
    logger.debug("Center selected by %s", type(_selector).__name__)

    # 2. Create transform (no computation on init)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([qmc_sample_shape]))

    if method_key not in _METHOD_TO_TRANSFORM_CLS:
        raise ValueError(
            f"Unknown transform_method={transform_method!r}. Supported: {sorted(_METHOD_TO_TRANSFORM_CLS.keys())}"
        )
    transform_cls = _METHOD_TO_TRANSFORM_CLS[method_key]

    # DSP: full [0,1]^d bounds, no TR (logEI on normalized input range)
    if method_key == "dsp":
        transform = _DSPBounds(device=X.device, dtype=X.dtype)
    else:
        # Different transforms have different parameter requirements
        if transform_cls == LowRankSVDTransform:
            transform = transform_cls(
                model=model,
                sampler=sampler,
                eps_cfg=eps_cfg,
                volume_normalize=volume_normalize,
            )
        elif transform_cls == DiagGradRMSTransform:
            transform = transform_cls(
                model=model,
                sampler=sampler,
                eps_cfg=eps_cfg,
                volume_normalize=volume_normalize,
                observation_noise=observation_noise,
            )
        else:
            # Other transforms (Identity, DiagGradMean, ARDLengthscale, FiniteDiff)
            transform = transform_cls(
                model=model,
                sampler=sampler,
                eps_cfg=eps_cfg,
                volume_normalize=volume_normalize,
            )

    # Initialize diagnostics dict (will be populated by the batch generation functions)
    diagnostics = {
        "center": x_center.clone(),
        "clamp_rate": 0.0,
        "is_axis_aligned": isinstance(transform, AxisAlignedTR)
        or getattr(transform, "_is_dsp", False),
    }

    # 3. Generate/optimize candidates (transform will be called once inside these functions)
    if acqf == "ts":
        X_next = _thompson_sampling_batch(
            model=model,
            x_center=x_center,
            transform=transform,
            length=state.length,
            q=q,
            n_candidates=n_candidates,
            dim=dim,
            device=device,
            dtype=dtype,
            diagnostics=diagnostics,
        )
    else:
        X_next = _gradient_acqf_batch(
            model=model,
            Y=Y,
            x_center=x_center,
            transform=transform,
            length=state.length,
            q=q,
            acqf_type=acqf,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            diagnostics=diagnostics,
            qmc_sample_shape=qmc_sample_shape,
            use_raasp=use_raasp,
        )

    if return_diagnostics:
        return X_next, diagnostics
    return X_next


def _extract_diagnostics_from_transform(transform: TrustRegion) -> dict:
    """
    Extract diagnostics from a transform after it has been called.

    This function extracts all diagnostic information from a transform,
    avoiding the need to call the transform multiple times just for diagnostics.

    Parameters
    ----------
    transform : TrustRegion
        The transform that has been called with (x_center, length).

    Returns
    -------
    dict
        Dictionary containing all diagnostic information.
    """
    transform_diag = transform.diagnostics
    anisotropy = transform_diag.get("true_anisotropy", float("inf"))

    diag = {
        "eps_used": transform.eps_used,
        "anisotropy": anisotropy,
    }

    # Add z-bounds for rotated transforms
    if isinstance(transform, RotatedTR):
        diag["z_bounds"] = (
            transform.z_bounds_lower.tolist(),
            transform.z_bounds_upper.tolist(),
        )
    else:
        diag["z_bounds"] = None

    # Add transform-specific diagnostics (e.g., eigenvalue clamping info for LowRankSVD)
    if "anisotropy_before_clamp" in transform_diag:
        diag["anisotropy_before_clamp"] = transform_diag["anisotropy_before_clamp"]
    if "anisotropy_after_clamp" in transform_diag:
        diag["anisotropy_after_clamp"] = transform_diag["anisotropy_after_clamp"]
    if "eigs_norm" in transform_diag:
        diag["eigs_norm"] = transform_diag["eigs_norm"]
    if "eigs_norm_clamped" in transform_diag:
        diag["eigs_norm_clamped"] = transform_diag["eigs_norm_clamped"]
    if "condition_number" in transform_diag:
        diag["condition_number"] = transform_diag["condition_number"]

    return diag


def _thompson_sampling_batch(
    model,
    x_center: Tensor,
    transform: TrustRegion,
    length: float,
    q: int,
    n_candidates: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    diagnostics: dict,
) -> Tensor:
    """Generate batch using discrete Thompson Sampling."""

    # FAST PATH: Axis-aligned transforms or DSP (full [0,1]^d)
    is_dsp = getattr(transform, "_is_dsp", False)
    if isinstance(transform, AxisAlignedTR) or is_dsp:
        x_bounds = transform(x_center, length)
        if not is_dsp:
            transform_diag = _extract_diagnostics_from_transform(transform)
            diagnostics.update(transform_diag)
        else:
            diagnostics["eps_used"] = getattr(transform, "eps_used", 0.0)
        diagnostics["points_clamped_ratio"] = 0.0
        diagnostics["mean_dims_clamped_ratio"] = 0.0

        # Normalize x_center to (D,) if needed
        if x_center.dim() > 1:
            x_center_1d = x_center.squeeze()
        else:
            x_center_1d = x_center

        x_lower, x_upper = x_bounds[0], x_bounds[1]

        # Sample perturbations within bounds using Sobol
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(device=device, dtype=dtype)
        pert = x_lower + pert * (x_upper - x_lower)

        # Create a perturbation mask (TuRBO-style)
        # prob_perturb = min(20.0 / dim, 1.0) ensures ~20 dimensions perturbed on average
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb

        # Ensure at least one dimension is perturbed per candidate
        # This is the key TuRBO invariant: every candidate must differ from center in at least one dim
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = True

        # Create candidate points from the center, only perturbing masked dimensions
        x_cand = x_center_1d.expand(n_candidates, dim).clone()
        x_cand[mask] = pert[mask]

        # Thompson Sampling
        with torch.no_grad():
            mps = MaxPosteriorSampling(model=model, replacement=False)
            X_next = mps(x_cand, num_samples=q)

        return X_next

    # SLOW PATH: For rotated transforms, use z-space with operator
    elif isinstance(transform, RotatedTR):
        op_result = transform(x_center, length)
        diagnostics.update(_extract_diagnostics_from_transform(transform))

        x_center_1d = x_center.squeeze() if x_center.dim() > 1 else x_center

        # Sample z ∈ [-1, 1]^D using Sobol
        sobol = SobolEngine(dim, scramble=True)
        z_01 = sobol.draw(n_candidates).to(device=device, dtype=dtype)
        z = z_01 * 2.0 - 1.0

        # Map to x-space via rotation operator, clip to [0, 1]^D
        with torch.no_grad():
            x_delta = op_result.operator(z)  # (n_candidates, D)
        x_cand_raw = x_center_1d + length * x_delta
        x_cand = x_cand_raw.clamp(0.0, 1.0)
        _oob = (x_cand_raw < 0.0) | (x_cand_raw > 1.0)
        _oob_pts = _oob.any(dim=1)
        diagnostics["points_clamped_ratio"] = float(_oob_pts.float().mean().item())
        diagnostics["mean_dims_clamped_ratio"] = (
            float(_oob[_oob_pts].float().mean().item()) if _oob_pts.any() else 0.0
        )

        # TuRBO perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = True
        x_cand_masked = x_center_1d.expand(n_candidates, dim).clone()
        x_cand_masked[mask] = x_cand[mask]

        with torch.no_grad():
            mps = MaxPosteriorSampling(model=model, replacement=False)
            X_next = mps(x_cand_masked, num_samples=q)
        return X_next
    else:
        raise ValueError(f"Unknown transform type: {type(transform)}")


def _gradient_acqf_batch(
    model,
    Y: Tensor,
    x_center: Tensor,
    transform: TrustRegion,
    length: float,
    q: int,
    acqf_type: str,
    num_restarts: int,
    raw_samples: int,
    diagnostics: dict,
    qmc_sample_shape: int,
    use_raasp: bool = True,
) -> Tensor:
    """Generate batch using gradient-based acquisition optimization."""
    Y = ensure_y_shape_n1(Y)

    # Get best_f from (N, 1) tensor
    best_f = Y.max()
    # Create acquisition function
    if acqf_type == "ei":
        acq_func = qExpectedImprovement(
            model,
            best_f=best_f,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([qmc_sample_shape])),
        )
    else:  # logei
        acq_func = qLogExpectedImprovement(
            model,
            best_f=best_f,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([qmc_sample_shape])),
        )

    # FAST PATH: Axis-aligned transforms or DSP (full [0,1]^d), optimize in X-space
    is_dsp = getattr(transform, "_is_dsp", False)
    if isinstance(transform, AxisAlignedTR) or is_dsp:
        # Call transform once to get bounds (DSP returns full cube; TR returns local box)
        x_bounds = transform(x_center, length)

        if not is_dsp:
            transform_diag = _extract_diagnostics_from_transform(transform)
            diagnostics.update(transform_diag)
        else:
            diagnostics["eps_used"] = getattr(transform, "eps_used", 0.0)
        diagnostics["points_clamped_ratio"] = 0.0
        diagnostics["mean_dims_clamped_ratio"] = 0.0

        X_next, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=x_bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        return X_next

    # SLOW PATH: For rotated transforms, use z-space with operator
    elif isinstance(transform, RotatedTR):
        _ = transform(x_center, length)
        diagnostics.update(_extract_diagnostics_from_transform(transform))

        wrapped = TrustRegionWrappedAcquisitionFunction(
            acq_function=acq_func,
            transform=transform,
            x_center=x_center,
            length=length,
            warn_on_clamp=True,
            collect_diagnostics=True,
        )

        dim = x_center.shape[-1]
        device = x_center.device
        dtype = x_center.dtype
        z_bounds = torch.stack(
            [
                -torch.ones(dim, device=device, dtype=dtype),
                torch.ones(dim, device=device, dtype=dtype),
            ]
        )

        # --- RAASP warm-start block (RotatedTR path only) ---
        batch_z_init = None
        if use_raasp:
            try:
                x_bounds = torch.stack(
                    [
                        torch.zeros(dim, device=device, dtype=dtype),
                        torch.ones(dim, device=device, dtype=dtype),
                    ]
                )
                prob_perturb = min(20.0 / dim, 1.0)
                X_raasp = sample_points_around_best(
                    acq_function=acq_func,
                    n_discrete_points=raw_samples,
                    sigma=1e-3,
                    bounds=x_bounds,
                    subset_sigma=0.1,
                    prob_perturb=prob_perturb,
                )

                sobol = SobolEngine(dim, scramble=True)
                z_sobol = (
                    sobol.draw(raw_samples).to(device=device, dtype=dtype) * 2.0 - 1.0
                )

                if X_raasp is not None:
                    # x→z: linear approximation z ≈ (x - x_center) / length
                    x_center_1d = x_center.view(dim)
                    z_raasp = ((X_raasp - x_center_1d) / (length + 1e-16)).clamp(
                        -1.0, 1.0
                    )
                    z_all = torch.cat([z_raasp, z_sobol], dim=0)
                else:
                    z_all = z_sobol

                z_all_q = z_all.unsqueeze(1)  # (N, 1, D) for q=1
                with torch.no_grad():
                    acq_vals = wrapped(z_all_q)
                batch_z_init, _ = initialize_q_batch(
                    X=z_all_q, acq_vals=acq_vals, n=num_restarts
                )
            except Exception as e:
                logger.warning(
                    "RAASP warm-start failed (%s); falling back to raw Sobol.", e
                )
                batch_z_init = None

        z_opt, _ = optimize_acqf(
            acq_function=wrapped,
            bounds=z_bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples if batch_z_init is None else 0,
            batch_initial_conditions=batch_z_init,
        )

        X_next = wrapped.map_z_to_x(z_opt)
        if wrapped.last_diagnostics is not None:
            diagnostics["points_clamped_ratio"] = (
                wrapped.last_diagnostics.points_clamped_ratio
            )
            diagnostics["mean_dims_clamped_ratio"] = (
                wrapped.last_diagnostics.mean_dims_clamped_ratio
            )
        return X_next
    else:
        raise ValueError(f"Unknown transform type: {type(transform)}")


# =============================================================================
# Utilities
# =============================================================================


def ensure_y_shape_n1(Y: Tensor) -> Tensor:
    """
    Ensure Y has shape (n, 1) for use with GP and acquisition functions.

    - Scalar (dim 0) -> (1, 1)
    - 1D (n,) -> (n, 1)
    - 2D (n, 1) -> returned as-is
    - dim > 2 -> ValueError
    """
    if Y.dim() == 0:
        return Y.unsqueeze(0).unsqueeze(-1)
    if Y.dim() == 1:
        return Y.unsqueeze(-1)
    if Y.dim() > 2:
        raise ValueError(f"Y must be (N, 1) or (N,), got shape {Y.shape}")
    return Y


def get_initial_points(
    dim: int,
    n_pts: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> Tensor:
    """
    Generate initial points in [0, 1]^D (repeatable for same seed).

    Uses Sobol sequence when dim <= 21201 (PyTorch SobolEngine limit).
    For dim > 21201 (e.g. lasso_rcv1), falls back to seeded uniform random.
    """
    _SOBOL_MAX_DIM = 21201
    if dim <= _SOBOL_MAX_DIM:
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        return sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    # Fallback: seeded uniform for very high dimensions
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(n_pts, dim, device=device, dtype=dtype, generator=gen)


def eval_objective(func, x_normalized: Tensor) -> Tensor:
    """
    Evaluate objective function on normalized inputs.

    Unnormalizes from [0, 1]^D to function bounds, then evaluates.
    Handles both single points and batches. For problems that don't support
    batch evaluation, evaluates points one at a time.

    Parameters
    ----------
    func : BaseTestProblem
        The problem function to evaluate.
    x_normalized : Tensor
        Normalized input points in [0, 1]^D, shape (n, d) or (d,).

    Returns
    -------
    Tensor
        Function values, shape (n, 1) or (1,).
    """
    # Handle single point vs batch
    was_1d = x_normalized.dim() == 1
    if was_1d:
        x_normalized = x_normalized.unsqueeze(0)

    x_raw = unnormalize(x_normalized, func.bounds)

    # Try batch evaluation first
    try:
        y = func(x_raw)
        y = ensure_y_shape_n1(y)
        # If we got a single value for a batch, evaluate point-by-point
        if x_normalized.shape[0] > 1 and y.shape[0] == 1:
            # Some problems (like RoverTrajectory) aggregate batch inputs
            results = []
            for i in range(x_normalized.shape[0]):
                x_single = x_normalized[i : i + 1]
                x_raw_single = unnormalize(x_single, func.bounds)
                y_single = ensure_y_shape_n1(func(x_raw_single))
                results.append(y_single)
            y = torch.cat(results, dim=0)
        return y
    except (ValueError, RuntimeError) as e:
        # If batch evaluation fails (shape mismatch, etc.), evaluate point-by-point
        error_msg = str(e).lower()
        if "broadcast" in error_msg or "shape" in error_msg or "size" in error_msg:
            results = []
            for i in range(x_normalized.shape[0]):
                x_single = x_normalized[i : i + 1]
                x_raw_single = unnormalize(x_single, func.bounds)
                y_single = ensure_y_shape_n1(func(x_raw_single))
                results.append(y_single)
            y = torch.cat(results, dim=0)
            return y
        else:
            raise
