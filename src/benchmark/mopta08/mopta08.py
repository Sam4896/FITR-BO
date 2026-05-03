import os
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine

import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem


# The original class is modified by Fujitsu Limited to fit BaseTestProblem in botorch
class Mopta08(BaseTestProblem):
    def __init__(self, noise_std=None, negate=True):
        self.dim = 124
        self._bounds = np.vstack((np.zeros(self.dim), np.ones(self.dim))).T
        # Set continuous_inds before calling super() for newer BoTorch versions
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=negate)

        self.sysarch = 64 if sys.maxsize > 2**32 else 32
        self.machine = machine().lower()

        if self.machine == "armv7l":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_armhf.bin"
        elif self.machine == "x86_64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_elf64.bin"
        elif self.machine == "i386":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_elf32.bin"
        elif self.machine == "amd64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_amd64.exe"
        else:
            raise RuntimeError("Machine with this architecture is not supported")

        self._mopta_exectutable = os.path.join(
            Path(__file__).parent, self._mopta_exectutable
        )
        if not os.path.isfile(self._mopta_exectutable):
            raise FileNotFoundError(
                f"MOPTA08 executable not found: {self._mopta_exectutable!r}. "
                "On Windows you need mopta08_amd64.exe (not provided by BenchSuite; "
                "only Linux/ARM .bin files are available). "
                "On Linux, ensure the .bin file for your architecture is in this directory."
            )
        self.directory_file_descriptor = tempfile.TemporaryDirectory()
        self.directory_name = self.directory_file_descriptor.name

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Handle both single point and batch inputs
        if X.dim() == 1:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]

        results = []
        for i in range(batch_size):
            x = X[i].cpu().numpy()
            with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
                for _x in x:
                    tmp_file.write(f"{_x}\n")
            popen = subprocess.Popen(
                self._mopta_exectutable,
                stdout=subprocess.PIPE,
                cwd=self.directory_name,
            )
            popen.wait()
            output = (
                open(os.path.join(self.directory_name, "output.txt"), "r")
                .read()
                .split("\n")
            )
            output = [x.strip() for x in output]
            output = torch.tensor([float(x) for x in output if len(x) > 0])
            value = output[0]
            constraints = output[1:]
            result = value + 10 * torch.sum(torch.clip(constraints, 0))
            results.append(result)

        # Stack results and ensure correct shape; cast to input dtype/device
        y = torch.stack(results).to(dtype=X.dtype, device=X.device)
        return y.unsqueeze(-1)
