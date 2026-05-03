"""
Embedded Benchmark Functions from "Vanilla Bayesian Optimization Performs Great in High Dimensions"

This module implements the embedded benchmark functions used in:
Hvarfner, C., Hellsten, E. O., & Nardi, L. (2024).
Vanilla Bayesian Optimization Performs Great in High Dimensions.
Proceedings of the 41st International Conference on Machine Learning.

The benchmarks embed low-dimensional functions (Levy4: 4D, Hartmann6: 6D)
into higher-dimensional spaces (25D, 100D, 300D, 1000D) by only using
the first N dimensions and ignoring the rest.
"""

from torch import Tensor
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions import Levy, Hartmann


class EmbeddedLevy4(BaseTestProblem):
    """
    Levy4 function (4D) embedded in a higher-dimensional space.

    Only the first 4 dimensions are used; the remaining dimensions are ignored.
    This creates a function with effective dimensionality 4 but input dimensionality D.

    Args:
        dim: The embedding dimension (25, 100, 300, or 1000)
        noise_std: Standard deviation of the observation noise
        negate: Whether to negate the function (for maximization)
    """

    def __init__(self, dim=25, noise_std=None, negate=True):
        if dim < 4:
            raise ValueError("Embedding dimension must be at least 4")
        self.original_dim = 4
        self.embedding_dim = dim
        self.dim = dim

        # Set continuous_inds before calling super() for newer BoTorch versions
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []

        # Get bounds from a temporary base function (before super().__init__())
        temp_base = Levy(dim=self.original_dim, negate=False)
        base_bounds = temp_base._bounds
        # Extend the bounds: use base bounds for first 4 dims, repeat last bound for rest
        if len(base_bounds) == self.original_dim:
            self._bounds = base_bounds + [
                base_bounds[-1] for _ in range(self.dim - self.original_dim)
            ]
        else:
            # Fallback: use [0, 1] if bounds structure is unexpected
            self._bounds = [(0.0, 1.0) for _ in range(self.dim)]

        super().__init__(noise_std=noise_std, negate=negate)

        # Create the base Levy function (4D) after super().__init__()
        self.base_function = Levy(dim=self.original_dim, negate=False)

        # Store optimal value from base function
        self.optimal_value = self.base_function.optimal_value
        if negate:
            self.optimal_value = -self.optimal_value

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the embedded Levy4 function.

        Args:
            X: Input tensor of shape (..., dim)

        Returns:
            Function values of shape (..., 1)
        """
        # Handle single point input
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # Extract only the first 4 dimensions
        X_4d = X[..., : self.original_dim]

        # Evaluate the base function
        result = self.base_function._evaluate_true(X_4d)

        # Ensure output has correct shape (..., 1)
        if result.dim() == 0:
            result = result.unsqueeze(0).unsqueeze(-1)
        elif result.dim() == 1:
            result = result.unsqueeze(-1)
        elif result.shape[-1] != 1:
            result = result.unsqueeze(-1)

        return result


class EmbeddedHartmann6(BaseTestProblem):
    """
    Hartmann6 function (6D) embedded in a higher-dimensional space.

    Only the first 6 dimensions are used; the remaining dimensions are ignored.
    This creates a function with effective dimensionality 6 but input dimensionality D.

    Args:
        dim: The embedding dimension (25, 100, 300, or 1000)
        noise_std: Standard deviation of the observation noise
        negate: Whether to negate the function (for maximization)
    """

    def __init__(self, dim=25, noise_std=None, negate=True):
        if dim < 6:
            raise ValueError("Embedding dimension must be at least 6")
        self.original_dim = 6
        self.embedding_dim = dim
        self.dim = dim

        # Set continuous_inds before calling super() for newer BoTorch versions
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []

        # Get bounds from a temporary base function (before super().__init__())
        temp_base = Hartmann(dim=self.original_dim, negate=False)
        base_bounds = temp_base._bounds
        # Extend the bounds: use base bounds for first 6 dims, repeat last bound for rest
        if len(base_bounds) == self.original_dim:
            self._bounds = base_bounds + [
                base_bounds[-1] for _ in range(self.dim - self.original_dim)
            ]
        else:
            # Fallback: use [0, 1] if bounds structure is unexpected
            self._bounds = [(0.0, 1.0) for _ in range(self.dim)]

        super().__init__(noise_std=noise_std, negate=negate)

        # Create the base Hartmann function (6D) after super().__init__()
        self.base_function = Hartmann(dim=self.original_dim, negate=False)

        # Store optimal value from base function
        self.optimal_value = self.base_function.optimal_value
        if negate:
            self.optimal_value = -self.optimal_value

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the embedded Hartmann6 function.

        Args:
            X: Input tensor of shape (..., dim)

        Returns:
            Function values of shape (..., 1)
        """
        # Handle single point input
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # Extract only the first 6 dimensions
        X_6d = X[..., : self.original_dim]

        # Evaluate the base function
        result = self.base_function._evaluate_true(X_6d)

        # Ensure output has correct shape (..., 1)
        if result.dim() == 0:
            result = result.unsqueeze(0).unsqueeze(-1)
        elif result.dim() == 1:
            result = result.unsqueeze(-1)
        elif result.shape[-1] != 1:
            result = result.unsqueeze(-1)

        return result


# Convenience classes for specific dimensions as used in the paper
class Levy4_25(EmbeddedLevy4):
    """Levy4 embedded in 25D"""

    def __init__(self, noise_std=None, negate=True):
        super().__init__(dim=25, noise_std=noise_std, negate=negate)


class Levy4_100(EmbeddedLevy4):
    """Levy4 embedded in 100D"""

    def __init__(self, noise_std=None, negate=True):
        super().__init__(dim=100, noise_std=noise_std, negate=negate)


class Levy4_300(EmbeddedLevy4):
    """Levy4 embedded in 300D"""

    def __init__(self, noise_std=None, negate=True):
        super().__init__(dim=300, noise_std=noise_std, negate=negate)


class Levy4_1000(EmbeddedLevy4):
    """Levy4 embedded in 1000D"""

    def __init__(self, noise_std=None, negate=True):
        super().__init__(dim=1000, noise_std=noise_std, negate=negate)


class Hartmann6_25(EmbeddedHartmann6):
    """Hartmann6 embedded in 25D"""

    def __init__(self, noise_std=None, negate=True):
        super().__init__(dim=25, noise_std=noise_std, negate=negate)


class Hartmann6_100(EmbeddedHartmann6):
    """Hartmann6 embedded in 100D"""

    def __init__(self, noise_std=None, negate=True):
        super().__init__(dim=100, noise_std=noise_std, negate=negate)


class Hartmann6_300(EmbeddedHartmann6):
    """Hartmann6 embedded in 300D"""

    def __init__(self, noise_std=None, negate=True):
        super().__init__(dim=300, noise_std=noise_std, negate=negate)


class Hartmann6_1000(EmbeddedHartmann6):
    """Hartmann6 embedded in 1000D"""

    def __init__(self, noise_std=None, negate=True):
        super().__init__(dim=1000, noise_std=noise_std, negate=negate)
