"""
Utility Functions for Riemannian Trust Regions
==============================================

This module provides:
1. Matrix utilities (symmetrize, SPD sqrt/inverse sqrt)
2. Fisher gradient computation from posterior samples
3. Linear operator probing utilities
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from torch import Tensor
from botorch.posteriors import Posterior
from botorch.sampling import MCSampler
from botorch.posteriors.transformed import TransformedPosterior


logger = logging.getLogger("RiemannianUtils")


# =============================================================================
# Matrix Utilities
# =============================================================================


def symmetrize(A: Tensor) -> Tensor:
    """Force matrix to be symmetric: (A + A^T) / 2."""
    return 0.5 * (A + A.transpose(-1, -2))


def spd_inverse_sqrt(A: Tensor, eps: float) -> Tensor:
    """
    Compute A^{-1/2} robustly using eigendecomposition.

    Parameters
    ----------
    A : Tensor
        Symmetric positive semi-definite matrix, shape (..., D, D).
    eps : float
        Regularization added to eigenvalues before inversion.

    Returns
    -------
    Tensor
        A^{-1/2} of same shape.

    Notes
    -----
    Uses eigendecomposition for numerical stability:
        A = V @ diag(λ) @ V^T
        A^{-1/2} = V @ diag(1/√(λ + ε)) @ V^T
    """
    D = A.shape[-1]
    A = symmetrize(A)
    A = A + eps * torch.eye(D, device=A.device, dtype=A.dtype)

    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)

    inv_sqrt_evals = 1.0 / torch.sqrt(evals)
    result = (evecs * inv_sqrt_evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)

    return symmetrize(result)


def spd_sqrt(A: Tensor, eps: float) -> Tensor:
    """
    Compute A^{1/2} robustly using eigendecomposition.

    Parameters
    ----------
    A : Tensor
        Symmetric positive semi-definite matrix, shape (..., D, D).
    eps : float
        Regularization added to eigenvalues.

    Returns
    -------
    Tensor
        A^{1/2} of same shape.
    """
    D = A.shape[-1]
    A = symmetrize(A)
    A = A + eps * torch.eye(D, device=A.device, dtype=A.dtype)

    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)

    sqrt_evals = torch.sqrt(evals)
    result = (evecs * sqrt_evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)

    return symmetrize(result)


# =============================================================================
# Fisher Gradient Computation
# =============================================================================


def get_fisher_grads_from_samples(
    posterior: Posterior,
    sampler: MCSampler,
    x_eval: Tensor,
    detach_to_cpu: bool = False,
) -> Tensor:
    """
    Compute TRUE Fisher gradients (Score Function) w.r.t. input.
    """
    # 1. Draw samples (these are your "hallucinated" y)
    # We detach them because for the Score Function, y is fixed data.

    """
    This is the most important line.
    By detaching the samples, you treat y as fixed data. When you subsequently feed this into log_prob(sample) and backpropagate w.r.t x_eval, you are computing:
    \nabla_x \mathbb{E}_{y \sim p(y|x)} \left[ \log p(y|x) \right] = \mathbb{E}_{y \sim p(y|x)} \left[ \nabla_x \log p(y|x) \right]
    """
    posterior_samples = sampler(posterior).detach()  # (S, N, O)

    num_mc_samples = posterior_samples.shape[0]
    fisher_grads_list = []

    if isinstance(posterior, TransformedPosterior):
        raise NotImplementedError(
            "TransformedPosterior not supported yet. The transformed posterior does not have a clearly defined log_prob method. It is suggested to transform the data before training the model and not use any outcome transformation."
        )

    for s in range(num_mc_samples):
        # Only retain graph if not the last iteration
        retain = s < (num_mc_samples - 1)

        # 3. Compute Log-Likelihood of the fixed sample
        # This reconnects the graph to x_eval via the distribution parameters (mu, sigma)
        sample = posterior_samples[s, ...]
        log_prob = posterior.log_prob(sample).sum()

        # 4. Differentiate Log-Likelihood (The Score Function)
        grad = torch.autograd.grad(
            log_prob,
            x_eval,
            retain_graph=retain,
        )[0]

        if detach_to_cpu:
            grad = grad.detach().cpu()
        else:
            grad = grad.detach()

        fisher_grads_list.append(grad)

    fisher_grads = torch.stack(fisher_grads_list, dim=0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fisher_grads


def get_fisher_grads_from_posterior(
    model,
    x_in: Tensor,
    sampler: MCSampler,
    detach_to_cpu: bool = False,
    observation_noise: bool = True,
) -> Tensor:
    """
    Compute Fisher gradients of posterior samples at given input points.

    Parameters
    ----------
    model : BoTorch Model
        Model with .posterior() method.
    x_in : Tensor
        Input points, shape (D,) or (N, D). Will be normalized to (N, D).
    sampler : MCSampler
        Sampler for drawing posterior samples.
    detach_to_cpu : bool
        If True, move results to CPU.

    Returns
    -------
    Tensor
        Fisher gradients tensor of shape (S, N, D).
    """
    # Ensure shape (N, D) for posterior call
    x_eval = ensure_x_shape_for_posterior(x_in).detach().clone().requires_grad_(True)
    posterior = model.posterior(x_eval, observation_noise=observation_noise)

    fisher_grads = get_fisher_grads_from_samples(
        posterior=posterior,
        sampler=sampler,
        x_eval=x_eval,
        detach_to_cpu=detach_to_cpu,
    )

    del posterior, x_eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fisher_grads


def get_posterior_mean_scalar(model, X: Tensor) -> Tensor:
    """
    Get scalar posterior mean for each input point.

    Parameters
    ----------
    model : BoTorch Model
        Model with .posterior() method.
    X : Tensor
        Input points, shape (N, D).

    Returns
    -------
    Tensor
        Posterior means, shape (N,).
    """
    with torch.no_grad():
        post = model.posterior(X, observation_noise=False)
        result = post.mean

        # Collapse batch dimensions
        while result.dim() > 2:
            result = result.mean(dim=0)

        return result.squeeze(-1)


# =============================================================================
# Linear Operator Probing
# =============================================================================


def probe_linear_operator_matrix(
    op: Callable[[Tensor], Tensor],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Construct matrix W such that op(z) ≈ z @ W (by probing standard basis).

    This is the STANDARDIZED function for probing linear operators. All code
    that needs to extract a matrix representation from a linear operator should
    use this function to avoid transpose bugs and ensure consistency.

    The operator is probed by applying it to each standard basis vector e_d.
    For row-vector inputs z (shape (1, D)), op(e_d) returns the d-th row of W,
    so we stack these rows to form the full matrix.

    Parameters
    ----------
    op : Callable
        Linear operator z -> x_delta. Should accept row vectors of shape (1, D)
        and return row vectors of shape (1, D).
    dim : int
        Dimensionality.
    device, dtype
        Tensor device and dtype.

    Returns
    -------
    Tensor
        Matrix W of shape (D, D) representing the linear operator.
        The matrix satisfies: op(z) ≈ z @ W for row vectors z.

    Notes
    -----
    This function was standardized after fixing a transpose bug where the
    matrix was being constructed incorrectly. All operator probing should
    go through this function to maintain correctness.
    """
    eye = torch.eye(dim, device=device, dtype=dtype)
    rows = []
    for d in range(dim):
        # For row-vector inputs z (shape (1, D)), op(e_d) returns the d-th row of W.
        v = op(eye[d : d + 1]).squeeze(0)
        rows.append(v)
    return torch.stack(rows, dim=0)


def geometric_mean_singular_value(W: Tensor) -> float:
    """
    Compute geometric mean of singular values (volume scale of linear map).

    This is used for volume-preserving normalization of operators.

    Parameters
    ----------
    W : Tensor
        Matrix of shape (D, D).

    Returns
    -------
    float
        Geometric mean of singular values.
    """
    s = torch.linalg.svdvals(W)
    gm = torch.exp(torch.log(s + 1e-16).mean()).item()
    return float(max(gm, 1e-16))


# =============================================================================
# Shape Normalization Utilities
# =============================================================================


def ensure_x_shape_for_posterior(x: Tensor) -> Tensor:
    """
    Ensure x is in the correct shape for model.posterior() calls.

    Posterior always expects ndim >= 2, where:
    - Last dimension (-1) is the input dimension D
    - Second-to-last dimension (-2) is the number of points N

    This function normalizes x to have at least 2 dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor, can be (D,) or (N, D) or (..., N, D) with any number of batch dims.

    Returns
    -------
    Tensor
        Tensor with ndim >= 2, where shape[-2] is number of points and shape[-1] is input dim.
    """
    if x.dim() == 1:
        # (D,) -> (1, D)
        return x.unsqueeze(-2)

    # If already ndim >= 2, return as-is (posterior can handle batch dimensions)
    return x


def ensure_x_center_1d(x: Tensor) -> Tensor:
    """
    Ensure x_center is 1D (D,) for internal computations.

    This is the inverse of ensure_x_shape_for_posterior - it extracts
    the center point from potentially batched input.

    Parameters
    ----------
    x : Tensor
        Input tensor, can be (D,) or (1, D) or (N, D).

    Returns
    -------
    Tensor
        Center point of shape (D,).
    """
    if x.dim() == 1:
        return x
    elif x.dim() == 2:
        # Take first point if batched
        return x[0]
    else:
        raise ValueError(f"x_center must be 1D (D,) or 2D (N, D), got shape {x.shape}")
