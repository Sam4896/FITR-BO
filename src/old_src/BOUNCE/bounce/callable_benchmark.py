"""
CallableBenchmark: adapter to run Bounce with an arbitrary callable on [0,1]^d.

Used by the natural_bo benchmark wrapper so Bounce can optimize any
continuous objective fun(x) with x in [0,1]^d.
"""

from typing import Callable, Optional

import torch

from bounce.benchmarks import Benchmark
from bounce.util.benchmark import Parameter, ParameterType


class CallableBenchmark(Benchmark):
    """
    Continuous [0,1]^d benchmark that evaluates a given callable.
    The callable receives tensor of shape (n, dim) and should return (n,) or (n, 1).
    """

    def __init__(
        self,
        dim: int,
        fun: Callable[[torch.Tensor], torch.Tensor],
        noise_std: Optional[float] = None,
        optimal_value: Optional[float] = None,
    ):
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=1.0,
                random_sign=1,
            )
            for i in range(dim)
        ]
        super().__init__(parameters=parameters, noise_std=noise_std, flip=False)
        self._fun = fun
        self._optimal_value = optimal_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        out = self._fun(x)
        if out.ndim == 2:
            out = out.squeeze(-1)
        return out

    @property
    def optimal_value(self) -> Optional[float]:
        return self._optimal_value
