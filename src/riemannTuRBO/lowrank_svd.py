"""
Low-Rank SVD Transform Operator for Riemannian Trust Regions
=============================================================

This module provides the LowRankSVD transform operator, which captures
both axis-aligned AND rotated anisotropy via SVD of the Fisher gradient matrix.

This is the most powerful preconditioner:
- Captures cross-dimensional correlations (rotations)
- Uses affine-invariant pre-scaling for robustness
- Handles rank-deficient metrics gracefully

Key Insight
-----------
For a scalar objective with gradient g, the metric G = E[g g^T] is rank-1.
SVD of the gradient matrix directly gives us the principal direction of
sensitivity. The regularization ε fills in the null space.

Affine Invariance
-----------------
Before computing the SVD, we pre-scale each dimension by its marginal
sensitivity (diagonal RMS). This makes the method invariant to input scaling
and ensures ε has consistent meaning across dimensions.

Z-Bounds for Rotated Transforms
-------------------------------
Since this transform is NOT axis-aligned, the feasible z-region is a polytope.
We compute the circumscribing axis-aligned box:
1. Map corners of the x-feasible region back to z-space via W^{-1}
2. Take the bounding box of these z-corners
3. Samples from this box may still produce x outside [0,1], requiring clamping
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from botorch.sampling import MCSampler

from .base import RotatedTR
from .eps_config import EpsConfig, compute_eps_from_eigs
from .utils import (
    get_fisher_grads_from_posterior,
    symmetrize,
    spd_inverse_sqrt,
)


logger = logging.getLogger("LowRankSVDTransform")


# =============================================================================
# LowRankSVD Transform
# =============================================================================


class LowRankSVDTransform(RotatedTR):
    """
    Affine-Invariant Low-Rank SVD Preconditioner using True Fisher Gradients.

    Captures both axis-aligned and rotated anisotropy via SVD:
        G = E[g g^T]  (Fisher Information Matrix)
        W = (G + εI)^{-1/2}  (Whitening transform)

    Algorithm
    ---------
    1. Compute Fisher gradients g_s ~ ∇log p(y|x) via MC sampling
    2. Compute diagonal scaling: σ_d = sqrt(E[g_d²])
    3. Normalize gradients: g_norm = g / σ (affine invariance)
    4. SVD on normalized gradients: find principal directions V
    5. Regularize: λ_reg = λ + ε
    6. Build operator: z → x_delta = (z * base_scale + proj_correction) * scaler

    Pros:
    - Captures rotations (non-axis-aligned anisotropy)
    - Affine-invariant (robust to input scaling)
    - Uses full Fisher information

    Cons:
    - Slower than diagonal methods
    - Z-bounds are approximate (circumscribed box)
    - Samples may need clamping after transformation
    """

    name = "LowRankSVD"

    def __init__(
        self,
        model,
        sampler: MCSampler,
        *,
        eps_cfg: Optional[EpsConfig] = None,
        volume_normalize: bool = True,
        normalize_affine_scaler: bool = True,
        use_lowrank_if_efficient: bool = True,
        **kwargs,  # absorb legacy params
    ):
        # Store LowRankSVD-specific options
        self.normalize_affine_scaler = normalize_affine_scaler
        self.use_lowrank_if_efficient = use_lowrank_if_efficient
        self._vn_eigs = None

        # Base class handles the rest
        super().__init__(
            model,
            sampler,
            eps_cfg=eps_cfg,
            volume_normalize=volume_normalize,
        )

    def _apply_volume_normalization(self, op_raw, inv_sqrt_diag, x_center):
        """Analytic volume normalization — O(r+D) instead of O(D²+D³)."""
        if self._vn_eigs is None:
            return super()._apply_volume_normalization(op_raw, inv_sqrt_diag, x_center)
        all_eigs = self._vn_eigs
        log_gm = torch.log(all_eigs.abs() + 1e-16).mean()
        beta = float(torch.exp(-log_gm).item())

        def op_normalized(z: Tensor) -> Tensor:
            return op_raw(z) * beta

        return op_normalized, inv_sqrt_diag * beta

    def _clamp_eigenvalues(
        self,
        eigs: Tensor,
        eps: float,
    ) -> Tensor:
        """Clamp eigenvalues to minimum regularization value."""
        return torch.clamp(eigs, min=eps)

    def _compute_grads_and_scaler(
        self, x_center: Tensor
    ) -> Tuple[Tensor, Tensor, int, Tensor]:
        """
        Shared logic to compute normalized gradients and affine scaler.
        Returns:
            grads_norm: (S_eff, D)
            scaler: (D,)
            S_eff: int
            diag_G: (D,)
        """
        # 1. Get Fisher gradients: shape (S, N, D)
        fisher_grads = get_fisher_grads_from_posterior(
            self.model, x_center, self.sampler
        )
        S, N, D = fisher_grads.shape

        # Flatten to (S_eff, D)
        fisher_grads_flat = fisher_grads.reshape(-1, D)
        S_eff = fisher_grads_flat.shape[0]

        # 2. Compute diagonal scaling (affine invariance fix)
        diag_G = fisher_grads_flat.pow(2).mean(dim=0)  # (D,)

        # Per-dimension RMS gradient scale
        diag_std = torch.sqrt(diag_G + 1e-12)
        scaler = 1.0 / diag_std  # (D,)

        # Optional: volume-preserving normalization (preserves anisotropy ratios)
        if self.normalize_affine_scaler:
            gm = torch.exp(torch.log(scaler + 1e-16).mean())
            scaler = scaler / gm

        # 3. Normalize gradients (affine whitening)
        grads_norm = fisher_grads_flat * scaler.unsqueeze(0)

        return grads_norm, scaler, S_eff, diag_G

    def get_fisher_information_matrix(self, x_center: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the effective Fisher Information Matrix (Metric) at x_center.
        Includes affine scaling and eigenvalue regularization/clamping.
        Returns M such that dist^2 = x^T M x.
        """
        # Ensure x_center is correct shape
        if x_center.dim() == 1:
            x_center = x_center.unsqueeze(0)

        grads_norm, scaler, S_eff, _ = self._compute_grads_and_scaler(x_center)

        # Compute normalized Fisher matrix
        G_norm = (grads_norm.t() @ grads_norm) / float(S_eff)

        # Eigenvalues for adaptive damping/clamping
        eigs_norm, evecs_norm = torch.linalg.eigh(symmetrize(G_norm))
        eps_norm = compute_eps_from_eigs(eigs_norm, self.config.eps_cfg)

        # # Clamp eigenvalues
        eigs_norm_clamped = self._clamp_eigenvalues(eigs_norm, eps_norm)

        # # Reconstruct G_norm_clamped
        G_norm_clamped = (
            evecs_norm @ torch.diag_embed(eigs_norm_clamped) @ evecs_norm.t()
        )
        G_norm_clamped = symmetrize(G_norm_clamped)

        # Un-normalize to get effective metric M in original space
        # M = diag(1/scaler) @ G_norm_clamped @ diag(1/scaler)
        inv_scaler = 1.0 / scaler
        M = torch.diag(inv_scaler) @ G_norm_clamped @ torch.diag(inv_scaler)
        # M = torch.diag(inv_scaler) @ G_norm @ torch.diag(inv_scaler)

        return M, eigs_norm_clamped

    def _compute_raw_operator(
        self, x_center: Tensor
    ) -> Tuple[Callable[[Tensor], Tensor], Tensor, Optional[Tensor], float, dict]:
        """Compute low-rank SVD operator with affine invariance."""
        grads_norm, scaler, S_eff, diag_G = self._compute_grads_and_scaler(x_center)

        # 4. Decide: use SVD trick or explicit inversion
        if self.use_lowrank_if_efficient and S_eff < x_center.shape[-1]:
            return self._compute_lowrank_path(
                grads_norm, scaler, S_eff, diag_G, x_center
            )
        else:
            return self._compute_explicit_path(
                grads_norm, scaler, S_eff, diag_G, x_center
            )

    def _compute_lowrank_path(
        self,
        grads_norm: Tensor,
        scaler: Tensor,
        S_eff: int,
        diag_G: Tensor,
        x_center: Tensor,
    ) -> Tuple[Callable[[Tensor], Tensor], Tensor, Optional[Tensor], float, dict]:
        """
        Use SVD trick when S_eff < D (more efficient).

        The SVD of the gradient matrix directly gives eigenvalues/vectors
        without explicitly forming the D×D metric.
        """
        try:
            _, Sigma, Vh = torch.linalg.svd(grads_norm, full_matrices=False)
            V = Vh.t()  # (D, r)
        except RuntimeError:
            logger.warning("LowRankSVD: SVD failed, falling back to identity")
            return self._make_identity_fallback(diag_G, x_center)

        # Eigenvalues of G_norm = (1/S_eff) * Sigma²
        eigs_norm = Sigma.pow(2) / float(S_eff)

        # Adaptive damping
        eps_norm = compute_eps_from_eigs(eigs_norm, self.config.eps_cfg)

        # Clamp eigenvalues to limit anisotropy
        eigs_norm_clamped = self._clamp_eigenvalues(eigs_norm, eps_norm)

        lambdas = eigs_norm_clamped + eps_norm

        # Build operator
        base_scale = 1.0 / math.sqrt(eps_norm)
        active_scale = (1.0 / torch.sqrt(lambdas)) - base_scale

        def op(z: Tensor) -> Tensor:
            # A. Apply whitening in normalized space
            x_norm = z * base_scale

            # Low-rank correction
            proj = z @ V
            corr = (proj * active_scale) @ V.t()
            x_norm = x_norm + corr

            # B. Restore affine scales
            x_delta = x_norm * scaler
            return x_delta

        inv_sqrt_diag_proxy = scaler * base_scale

        # Store analytic eigenvalues for volume normalization and anisotropy
        D = x_center.shape[-1]
        r = len(lambdas)
        self._vn_eigs = torch.cat(
            [
                1.0 / torch.sqrt(lambdas),
                torch.full(
                    (D - r,), base_scale, device=lambdas.device, dtype=lambdas.dtype
                ),
            ]
        )

        # Step A: Arithmetic mean normalization — matrix analog of DiagGradRMS's weights/weights.mean()
        arith_mean_M = float(self._vn_eigs.mean().item())
        scale_arith = 1.0 / (arith_mean_M + 1e-16)
        op_pre_arith = op

        def op(z, _s=scale_arith):  # noqa: F811
            return op_pre_arith(z) * _s

        self._vn_eigs = self._vn_eigs * scale_arith
        inv_sqrt_diag_proxy = inv_sqrt_diag_proxy * scale_arith

        # Cheap analytic anisotropy from eigenvalues of operator
        eig_max = float(self._vn_eigs.max().item())
        eig_min = float(self._vn_eigs.min().item())
        true_anisotropy = eig_max / (eig_min + 1e-12)

        # Diagnostics
        anisotropy_before = (
            (eigs_norm.max() / eigs_norm.min()).item()
            if eigs_norm.min() > 0
            else float("inf")
        )
        anisotropy_after = (
            (eigs_norm_clamped.max() / eigs_norm_clamped.min()).item()
            if eigs_norm_clamped.min() > 0
            else float("inf")
        )

        diagnostics = {
            "diag_G": diag_G.tolist(),
            "num_samples": S_eff,
            "singular_values": Sigma.tolist(),
            "eigs_norm": eigs_norm.tolist(),
            "eigs_norm_clamped": eigs_norm_clamped.tolist(),
            "eps_norm": eps_norm,
            "path": "lowrank",
            "rank": len(Sigma),
            "condition_number": (lambdas.max() / lambdas.min()).item()
            if lambdas.min() > 0
            else float("inf"),
            "anisotropy_before_clamp": anisotropy_before,
            "anisotropy_after_clamp": anisotropy_after,
            "true_anisotropy": true_anisotropy,  # From actual operator
        }

        logger.info(
            f"LowRankSVD (lowrank path): S_eff={S_eff}, rank={len(Sigma)}, "
            f"eps_norm={eps_norm:.4e}, condition_number={diagnostics['condition_number']:.2f}, "
            f"anisotropy: {anisotropy_before:.2f} -> {anisotropy_after:.2f}"
        )

        return op, inv_sqrt_diag_proxy, V, eps_norm, diagnostics

    def _compute_explicit_path(
        self,
        grads_norm: Tensor,
        scaler: Tensor,
        S_eff: int,
        diag_G: Tensor,
        x_center: Tensor,
    ) -> Tuple[Callable[[Tensor], Tensor], Tensor, Optional[Tensor], float, dict]:
        """
        Explicit metric inversion when S_eff >= D.

        Computes G_norm = (1/S_eff) * G_norm^T @ G_norm, then inverts.
        """
        # Compute normalized Fisher matrix
        G_norm = (grads_norm.t() @ grads_norm) / float(S_eff)

        # Eigenvalues for adaptive damping
        eigs_norm, evecs_norm = torch.linalg.eigh(symmetrize(G_norm))
        eps_norm = compute_eps_from_eigs(eigs_norm, self.config.eps_cfg)

        # Clamp eigenvalues to limit anisotropy
        eigs_norm_clamped = self._clamp_eigenvalues(eigs_norm, eps_norm)

        # Reconstruct G_norm with clamped eigenvalues
        G_norm_clamped = (
            evecs_norm @ torch.diag_embed(eigs_norm_clamped) @ evecs_norm.t()
        )
        G_norm_clamped = symmetrize(G_norm_clamped)

        # Inverse square root using clamped matrix
        W_norm = spd_inverse_sqrt(G_norm_clamped, eps_norm)

        def op(z: Tensor) -> Tensor:
            return (z @ W_norm) * scaler

        # Proxy for visualization
        w_norm_diag = torch.sqrt(torch.diag(W_norm @ W_norm))
        inv_sqrt_diag_proxy = w_norm_diag * scaler

        # Store analytic eigenvalues for volume normalization and anisotropy
        self._vn_eigs = 1.0 / torch.sqrt(eigs_norm_clamped)

        # Step A: Arithmetic mean normalization — matrix analog of DiagGradRMS's weights/weights.mean()
        arith_mean_M = float(self._vn_eigs.mean().item())
        scale_arith = 1.0 / (arith_mean_M + 1e-16)
        op_pre_arith = op

        def op(z, _s=scale_arith):  # noqa: F811
            return op_pre_arith(z) * _s

        self._vn_eigs = self._vn_eigs * scale_arith
        inv_sqrt_diag_proxy = inv_sqrt_diag_proxy * scale_arith

        # Cheap analytic anisotropy
        eig_max = float(self._vn_eigs.max().item())
        eig_min = float(self._vn_eigs.min().item())
        true_anisotropy = eig_max / (eig_min + 1e-12)

        # Diagnostics
        anisotropy_before = (
            (eigs_norm.max() / eigs_norm.min()).item()
            if eigs_norm.min() > 0
            else float("inf")
        )
        anisotropy_after = (
            (eigs_norm_clamped.max() / eigs_norm_clamped.min()).item()
            if eigs_norm_clamped.min() > 0
            else float("inf")
        )

        diagnostics = {
            "diag_G": diag_G.tolist(),
            "num_samples": S_eff,
            "eigs_norm": eigs_norm.tolist(),
            "eigs_norm_clamped": eigs_norm_clamped.tolist(),
            "eps_norm": eps_norm,
            "path": "explicit",
            "condition_number": (
                eigs_norm_clamped.max() / (eigs_norm_clamped.min() + eps_norm)
            ).item(),
            "anisotropy_before_clamp": anisotropy_before,
            "anisotropy_after_clamp": anisotropy_after,
            "true_anisotropy": true_anisotropy,  # From actual operator
        }

        logger.info(
            f"LowRankSVD (explicit path): S_eff={S_eff}, D={x_center.shape[-1]}, "
            f"eps_norm={eps_norm:.4e}, condition_number={diagnostics['condition_number']:.2f}, "
            f"anisotropy: {anisotropy_before:.2f} -> {anisotropy_after:.2f}"
        )

        return op, inv_sqrt_diag_proxy, evecs_norm, eps_norm, diagnostics

    def _make_identity_fallback(
        self, diag_G: Tensor, x_center: Tensor
    ) -> Tuple[Callable[[Tensor], Tensor], Tensor, Optional[Tensor], float, dict]:
        """Create identity operator when SVD fails."""
        dim = x_center.shape[-1]
        device = x_center.device
        dtype = x_center.dtype
        eps = compute_eps_from_eigs(
            torch.zeros(1, device=device, dtype=dtype),
            self.config.eps_cfg,
        )
        scale = 1.0 / math.sqrt(eps)
        inv_sqrt = torch.full((dim,), scale, device=device, dtype=dtype)

        def op(z: Tensor) -> Tensor:
            return z * scale

        diagnostics = {
            "diag_G": diag_G.tolist(),
            "path": "fallback_identity",
            "reason": "SVD failed",
        }

        return op, inv_sqrt, None, eps, diagnostics
