from __future__ import annotations

import torch


def _turbo_style_perturbation_mask(
    n_candidates: int, dim: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Replicates TuRBO's 'ensure at least one perturbed dim' mask logic."""
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        # randint high is exclusive; use dim to allow selecting dim-1 as well
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = True
    return mask


def test_perturbation_mask_has_no_empty_rows_high_dim() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    n_candidates = 256

    for dim in [5, 20, 200, 1000]:
        mask = _turbo_style_perturbation_mask(
            n_candidates=n_candidates, dim=dim, device=device, dtype=dtype
        )
        assert mask.shape == (n_candidates, dim)
        assert (mask.sum(dim=1) >= 1).all()


def test_candidate_construction_changes_at_least_one_dimension() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    n_candidates = 256
    dim = 200

    x_center = torch.rand(dim, device=device, dtype=dtype)
    pert = torch.rand(n_candidates, dim, device=device, dtype=dtype)
    mask = _turbo_style_perturbation_mask(
        n_candidates=n_candidates, dim=dim, device=device, dtype=dtype
    )

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Each row must differ from x_center in at least one coordinate.
    diff = (X_cand != x_center).any(dim=1)
    assert diff.all()


def test_turbo_initial_conditions_in_z_space() -> None:
    """Test that TuRBO-style initial conditions in z-space ensure at least one perturbed dimension."""
    from poc.experiments.riemann_turbo import _generate_turbo_initial_conditions

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    n_candidates = 256

    for dim in [5, 20, 200]:
        # z-space bounds (typically [-1, 1] or tighter)
        z_bounds = torch.tensor([[-1.0] * dim, [1.0] * dim], device=device, dtype=dtype)

        z_init = _generate_turbo_initial_conditions(
            n_candidates=n_candidates,
            dim=dim,
            z_bounds=z_bounds,
            device=device,
            dtype=dtype,
        )

        assert z_init.shape == (n_candidates, dim)
        # Each candidate should have at least one non-zero dimension (perturbed from center z=0)
        non_zero = (z_init != 0.0).any(dim=1)
        assert non_zero.all(), (
            f"Some candidates have all zero dimensions in {dim}D case"
        )
        # All values should be within bounds
        assert (z_init >= z_bounds[0]).all()
        assert (z_init <= z_bounds[1]).all()
