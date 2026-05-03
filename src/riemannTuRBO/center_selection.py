"""
Center Selection Strategies for Trust Region Bayesian Optimization
==================================================================

This module provides different strategies for selecting the center of
the trust region for each BO iteration.

Strategies
----------
1. BestObserved: Use the best observed point (standard TuRBO)
2. REI: TuRBO-RLogEI — optimize LogRegionalExpectedImprovement over [0,1]^d
   exactly as turbo_rei does (racqf=='REI'): X_dev, optimize_acqf, same options.
3. UCB: Upper Confidence Bound selection
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch.optim import optimize_acqf

try:
    from ..old_src.rei import qRegionalExpectedImprovement
except ImportError:
    from old_src.rei import qRegionalExpectedImprovement

from botorch.sampling.normal import SobolQMCNormalSampler


logger = logging.getLogger("CenterSelection")


# =============================================================================
# Abstract Base Class
# =============================================================================


class CenterSelector(ABC):
    """
    Abstract base class for trust region center selection.

    The center selection strategy determines where to place the trust region
    for the next BO iteration. Different strategies offer different trade-offs
    between exploitation and exploration.
    """

    @abstractmethod
    def select_center(
        self,
        model,
        X: Tensor,
        Y: Tensor,
        tr_length: float,
    ) -> Tensor:
        """
        Select the center of the trust region.

        Parameters
        ----------
        model : BoTorch Model
            The surrogate model.
        X : Tensor
            Observed inputs in normalized space [0, 1]^D, shape (N, D).
        Y : Tensor
            Observed targets (standardized), shape (N, 1). Will be normalized
            to (N, 1) if passed as (N,).
        tr_length : float
            Current trust region length.

        Returns
        -------
        Tensor
            The selected center in normalized space, shape (D,).
        """
        pass


# =============================================================================
# Concrete Implementations
# =============================================================================


class BestObservedSelector(CenterSelector):
    """
    Standard TuRBO center selection: use the best observed point.

    This is the most common and simplest strategy. It places the trust
    region around the current best observation, encouraging exploitation.

    Pros:
    - Simple and fast
    - Good for unimodal functions
    - Naturally focuses on promising regions

    Cons:
    - May miss good regions if the best point is in a local optimum
    - No exploration component
    """

    def select_center(
        self,
        model,
        X: Tensor,
        Y: Tensor,
        tr_length: float,
    ) -> Tensor:
        # Ensure X is (N, D) and Y is (N, 1)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        elif Y.dim() > 2:
            raise ValueError(f"Y must be (N, 1) or (N,), got shape {Y.shape}")

        # Y is now (N, 1), get best index
        best_idx = Y.argmax()
        best_point = X[best_idx, :].clone().unsqueeze(-2)  # Shape (1, D)
        return best_point


class REISelector(CenterSelector):
    """
    TuRBO-RLogEI: center selection by optimizing LogRegionalExpectedImprovement.

    Matches turbo_rei exactly when racqf=='REI':
    - X_dev: Sobol 128 points in [0,1]^d, one moved to 0.5.
    - racq_function = LogRegionalExpectedImprovement(model, best_f, X_dev, length, bounds).
    - optimize_acqf(racq_function, bounds, q=1, num_restarts=10, raw_samples=512,
      options={"batch_limit": 5, "maxiter": 200}, sequential=True).
    - Center = result of optimize_acqf.

    No EI screening, no candidate list: direct optimization of RLogEI over [0,1]^d.

    Parameters
    ----------
    n_region : int
        Number of design points for region (X_dev). turbo_rei uses 128.
    num_restarts : int
        optimize_acqf num_restarts. turbo_rei uses 10.
    raw_samples : int
        optimize_acqf raw_samples. turbo_rei uses 512.
    optimizer_options : dict
        Passed to optimize_acqf options. turbo_rei uses {"batch_limit": 5, "maxiter": 200}.
    seed : Optional[int]
        Seed for Sobol X_dev. If None, no seed (non-deterministic).
    """

    def __init__(
        self,
        n_region: int = 128,
        num_restarts: int = 10,
        raw_samples: int = 512,
        optimizer_options: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        self.n_region = n_region
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.optimizer_options = (
            optimizer_options
            if optimizer_options is not None
            else {"batch_limit": 5, "maxiter": 200}
        )
        self.seed = seed

    def select_center(
        self,
        model,
        X: Tensor,
        Y: Tensor,
        tr_length: float,
    ) -> Tensor:
        # Ensure X is (N, D) and Y is (N, 1)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        elif Y.dim() > 2:
            raise ValueError(f"Y must be (N, 1) or (N,), got shape {Y.shape}")

        dim = X.shape[-1]
        dtype = X.dtype
        device = X.device

        # Bounds [0, 1]^D, same as turbo_rei normal_bounds
        bounds = torch.stack(
            [
                torch.zeros(dim, device=device, dtype=dtype),
                torch.ones(dim, device=device, dtype=dtype),
            ]
        )

        # X_dev exactly as turbo_rei: Sobol n=128, one point moved to center
        sobol = SobolEngine(dimension=dim, scramble=True, seed=self.seed)
        X_dev = sobol.draw(self.n_region).to(dtype=dtype, device=device)
        X_dev[torch.argmin(torch.sum((X_dev - 0.5) ** 2, dim=1)), :] = 0.5

        # LogRegionalExpectedImprovement (RLogEI) exactly as turbo_rei racqf=='REI'
        best_f = Y.max()
        seed = int(
            torch.randint(
                low=0, high=2**16, size=(1,), device=device, dtype=torch.int64
            )
        )
        racq_function = qRegionalExpectedImprovement(
            X_dev=X_dev,
            model=model,
            best_f=best_f,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=seed),
            length=tr_length,
            bounds=bounds,
        )

        # optimize_acqf exactly as turbo_rei: q=1, num_restarts=10, raw_samples=512, options, sequential=True
        candidates, _ = optimize_acqf(
            acq_function=racq_function,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.optimizer_options,
            sequential=True,
        )
        return candidates  # (1, D)


class UCBSelector(CenterSelector):
    """
    UCB-based center selection.

    Selects the center by maximizing the Upper Confidence Bound:
        UCB(x) = μ(x) + β * σ(x)

    This balances exploitation (high mean) with exploration (high uncertainty).

    Parameters
    ----------
    beta : float
        Exploration-exploitation trade-off parameter.
        Higher beta = more exploration.
    num_candidates : int
        Number of Sobol candidates to evaluate.
    """

    def __init__(self, beta: float = 2.0, num_candidates: int = 1000):
        self.beta = beta
        self.num_candidates = num_candidates

    def select_center(
        self,
        model,
        X: Tensor,
        Y: Tensor,
        tr_length: float,
    ) -> Tensor:
        # Ensure X is (N, D) and Y is (N, 1)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        elif Y.dim() > 2:
            raise ValueError(f"Y must be (N, 1) or (N,), got shape {Y.shape}")

        dim = X.shape[-1]
        dtype = X.dtype
        device = X.device

        # Generate candidates
        sobol = SobolEngine(dim, scramble=True)
        candidates = sobol.draw(self.num_candidates).to(dtype=dtype, device=device)

        # Evaluate UCB
        with torch.no_grad():
            posterior = model.posterior(candidates)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.sqrt().squeeze(-1)
            ucb = mean + self.beta * std

        best_idx = ucb.argmax()
        return candidates[best_idx].clone().unsqueeze(-2)  # Shape (1, D)


class LastObservedSelector(CenterSelector):
    """
    Last observed center selection.
    """

    def select_center(
        self,
        model,
        X: Tensor,
        Y: Tensor,
        tr_length: float,
    ) -> Tensor:
        # Ensure X is (N, D) and Y is (N, 1)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        elif Y.dim() > 2:
            raise ValueError(f"Y must be (N, 1) or (N,), got shape {Y.shape}")

        # Y is now (N, 1), get best index
        last_point = X[-1, :].clone().unsqueeze(-2)  # Shape (1, D)
        return last_point


class RestartCenterSelector(CenterSelector):
    """
    Center selector used when TR restart is triggered for position-based selectors.

    Samples points globally in [0, 1]^D and picks the highest posterior mean to
    re-center after trust region collapse.
    """

    def __init__(self, num_samples: int = 512):
        self.num_samples = num_samples

    def select_center(
        self,
        model,
        X: Tensor,
        Y: Tensor,
        tr_length: float,
    ) -> Tensor:
        return select_restart_center(
            model=model,
            dim=X.shape[-1],
            num_samples=self.num_samples,
            device=X.device,
            dtype=X.dtype,
        )


# =============================================================================
# Restart Center Selection
# =============================================================================


def select_restart_center(
    model,
    dim: int,
    num_samples: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Select a new center when restart is triggered.

    Samples points from the whole input space [0, 1]^D and selects the one
    with the highest posterior mean. This helps escape local optima by
    exploring the global landscape.

    Parameters
    ----------
    model : BoTorch Model
        The surrogate model.
    dim : int
        Input dimensionality.
    num_samples : int
        Number of Sobol samples to generate from the whole input space.
        Default 512.
    device : Optional[torch.device]
        Device for tensors. If None, inferred from model.
    dtype : Optional[torch.dtype]
        Data type for tensors. If None, inferred from model.

    Returns
    -------
    Tensor
        The selected center in normalized space, shape (1, D).
    """
    # Infer device and dtype from model if not provided
    if device is None:
        # Try to get device from model
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

    if dtype is None:
        # Try to get dtype from model
        try:
            dtype = next(model.parameters()).dtype
        except (StopIteration, AttributeError):
            dtype = torch.float64

    # Generate Sobol samples from the whole input space [0, 1]^D
    sobol = SobolEngine(dim, scramble=True)
    candidates = sobol.draw(num_samples).to(dtype=dtype, device=device)

    # Evaluate posterior mean for all candidates
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(candidates)
        mean = posterior.mean.squeeze(-1)  # Shape (num_samples,)

    # Select the candidate with highest posterior mean
    best_idx = mean.argmax()
    center = candidates[best_idx].clone().unsqueeze(-2)  # Shape (1, D)

    logger.info(
        f"Restart center selected: sampled {num_samples} points from [0,1]^D, "
        f"selected point with posterior mean {mean[best_idx].item():.4f}"
    )

    return center


# =============================================================================
# Factory Function
# =============================================================================


def get_center_selector(
    selector_type: str = "best",
    **kwargs,
) -> CenterSelector:
    """
    Factory function for center selectors.

    Parameters
    ----------
    selector_type : str
        One of "best", "rei", "ucb".
    **kwargs
        Additional arguments passed to the selector constructor.

    Returns
    -------
    CenterSelector
        The requested selector instance.

    Examples
    --------
    >>> selector = get_center_selector("rei", n_region=128, num_restarts=10, raw_samples=512)
    >>> center = selector.select_center(model, X, Y, tr_length=0.5)
    """
    if selector_type == "best":
        return BestObservedSelector()
    elif selector_type == "rei":
        return REISelector(**kwargs)
    elif selector_type == "ucb":
        return UCBSelector(**kwargs)
    elif selector_type == "last":
        return LastObservedSelector()
    elif selector_type == "restart":
        return RestartCenterSelector(**kwargs)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")
