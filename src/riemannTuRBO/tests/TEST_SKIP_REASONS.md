# Test Skip Reasons

## Benchmark problem tests

All benchmark problems are tested in **`test_benchmark_imports.py`**:
- Each problem is checked for **single** input `(1, dim)` → output `(1, 1)` and **batch** input `(BATCH_SIZE, dim)` → output `(BATCH_SIZE, 1)`.
- If single or batch evaluation fails (wrong shape or non-finite), the test **fails**: the wrapper for that problem should be adjusted.
- Tests are **skipped** only when a dependency is missing (e.g. LassoBench not cloned, BenchSuite not found).

## Why tests are skipped

- **MOPTA08** (Windows): skipped when the required `.bin` executable is not found.
- **LassoBench problems**: skipped when `src/benchmark/LassoBench` is not present or dataset download fails (e.g. SSL).
- **BenchSuite problems** (svm, lunar_lander, robot_pushing, swimming, hopper, ant, humanoid, half_cheetah): skipped when BenchSuite is not available (set `BENCHSUITE_ROOT` or clone BenchSuite in the parent directory of natural_bo).

## Running tests

```bash
poetry run pytest src/riemannTuRBO/tests/test_benchmark_imports.py -v
```

See why tests were skipped:
```bash
poetry run pytest src/riemannTuRBO/tests/test_benchmark_imports.py -v -rs
```

Availability report (which problems are OK vs skipped):
```bash
poetry run pytest src/riemannTuRBO/tests/test_benchmark_imports.py::test_problem_availability_report -v -s
```
