from __future__ import annotations

import math

import torch

from riemannTuRBO.utils import (
    geometric_mean_singular_value,
    probe_linear_operator_matrix,
    spd_inverse_sqrt,
    symmetrize,
)


def test_probe_linear_operator_matrix_recovers_W() -> None:
    torch.manual_seed(0)
    D = 7
    device = torch.device("cpu")
    dtype = torch.float64

    W = torch.randn(D, D, device=device, dtype=dtype)

    def op(z: torch.Tensor) -> torch.Tensor:
        return z @ W

    W_hat = probe_linear_operator_matrix(op, D, device, dtype)
    assert torch.allclose(W_hat, W, atol=1e-12, rtol=1e-12)


def test_spd_inverse_sqrt_whitens_A_plus_eps_I() -> None:
    torch.manual_seed(0)
    D = 9
    device = torch.device("cpu")
    dtype = torch.float64

    # Construct a symmetric PSD matrix with a controlled spectrum.
    Q, _ = torch.linalg.qr(torch.randn(D, D, device=device, dtype=dtype))
    evals = torch.linspace(0.0, 3.0, D, device=device, dtype=dtype)  # includes zeros
    A = Q @ torch.diag(evals) @ Q.t()
    A = symmetrize(A)

    eps = 1e-6
    W = spd_inverse_sqrt(A, eps=eps)

    # spd_inverse_sqrt internally computes (A + eps I)^{-1/2} (after symmetrize).
    A_eps = symmetrize(A) + eps * torch.eye(D, device=device, dtype=dtype)
    I_hat = W @ A_eps @ W
    I = torch.eye(D, device=device, dtype=dtype)
    assert torch.allclose(I_hat, I, atol=5e-10, rtol=5e-10)


def test_geometric_mean_singular_value_scale_invariance() -> None:
    torch.manual_seed(0)
    D = 6
    device = torch.device("cpu")
    dtype = torch.float64

    W = torch.randn(D, D, device=device, dtype=dtype)
    s = 3.7

    gm = geometric_mean_singular_value(W)
    gm_scaled = geometric_mean_singular_value(W * s)

    # scaling a linear map scales all singular values, so gm scales by |s|
    assert math.isfinite(gm)
    assert math.isfinite(gm_scaled)
    assert abs(gm_scaled - (abs(s) * gm)) / max(gm_scaled, 1e-16) < 1e-10
