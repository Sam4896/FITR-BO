from __future__ import annotations

import os
import sys


def pytest_configure() -> None:
    """Ensure `riemannTuRBO` and `experiments` are importable without installing."""
    # conftest.py is in: src/riemannTuRBO/tests/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    repo_root = os.path.abspath(os.path.join(src_dir, ".."))
    for path in (repo_root, src_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
