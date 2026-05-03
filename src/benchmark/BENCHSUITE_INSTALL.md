# BenchSuite Install & Run (including MuJoCo problems)

How to install and run the [BenchSuite](https://github.com/lpapenme/BenchSuite) benchmarks used by `define_problems.py`, including the **MuJoCo** ones: `swimming`, `hopper`, `ant`, `humanoid`.

---

## Recommended: BenchSuite as a separate project (does not touch natural_bo)

To **keep natural_bo’s venv and Python version unchanged**, run BenchSuite as an **external project** in the **parent directory** of natural_bo:

```text
/path/to/parent/
  natural_bo/     ← this project (its own Python/venv)
  BenchSuite/     ← clone here, with its own poetry/venv (e.g. Python 3.8)
```

1. **Clone BenchSuite** (outside natural_bo):

   ```bash
   cd /path/to/parent   # parent of natural_bo
   git clone https://github.com/lpapenme/BenchSuite.git BenchSuite
   ```

2. **Finish BenchSuite install** in that directory (needs system deps: `swig`, `python3.8-dev`, etc.):

   ```bash
   cd BenchSuite
   sudo apt install swig libglew-dev patchelf python3.8-dev   # if needed
   poetry install
   ```

3. **Use from natural_bo**: no path or venv changes in natural_bo. The code looks for BenchSuite in:
   - `BENCHSUITE_ROOT` (if set), or
   - `<parent of natural_bo>/BenchSuite`.

   So with the layout above, `DefineProblems("swimming")` etc. will call BenchSuite’s `main.py` via subprocess using BenchSuite’s `.venv/bin/python`. See **BenchSuite/README_NATURAL_BO.md** in the BenchSuite repo for details.

---

## MOPTA08 vs MuJoCo: why MuJoCo is not “binaries only”

**MOPTA08** in this repo runs by **downloading standalone binaries** (e.g. `mopta08_elf64.bin`). The binary reads `input.txt`, writes `output.txt`; no Python runtime or extra deps are needed beyond that binary.

**MuJoCo benchmarks in BenchSuite are different.** BenchSuite does **not** provide equivalent standalone executables for swimmer/hopper/ant/humanoid. Those benchmarks are implemented in **Python**:

- They use the `benchsuite.mujoco` module and `benchsuite.utils.mujoco.func_factories`, which in turn use **mujoco-py** (Python bindings to MuJoCo).
- The repo’s `data/mujoco210` folder contains the **MuJoCo 2.1 SDK** (libraries in `bin/`, headers, model files). Those are **libraries** used by mujoco-py at runtime, not “run this and get a number” binaries.

So you **cannot** run the MuJoCo benchmarks by only downloading binaries and data like MOPTA08. You need the full BenchSuite Python environment (clone repo + `poetry install` + env vars). Optionally, natural_bo can call BenchSuite **via subprocess** (see below) so you don’t have to install BenchSuite’s deps into natural_bo’s env; BenchSuite runs in its own process.

---

## 1. Clone BenchSuite

From the **natural_bo** repo root:

```bash
git clone https://github.com/lpapenme/BenchSuite.git src/benchmark/BenchSuite
cd src/benchmark/BenchSuite
```

---

## 2. MuJoCo environment variables (required for MuJoCo problems)

BenchSuite expects MuJoCo 210 under `data/mujoco210`. Set these **before** running BenchSuite (e.g. in your shell or in code):

```bash
export LD_LIBRARY_PATH=${PWD}/data/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=${PWD}/data/mujoco210
```

Use the path to **BenchSuite’s root** (where `data/mujoco210` lives). From natural_bo root that would be:

```bash
export BENCHSUITE_ROOT=/home/user_name/natural_bo/src/benchmark/BenchSuite
export LD_LIBRARY_PATH=$BENCHSUITE_ROOT/data/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=$BENCHSUITE_ROOT/data/mujoco210
```

*(Adjust `BENCHSUITE_ROOT` if your path differs.)*

---

## 3. Install BenchSuite dependencies

In the **BenchSuite** directory:

```bash
cd src/benchmark/BenchSuite
poetry install
```

BenchSuite’s `pyproject.toml` includes:

- **MuJoCo:** `mujoco-py` (from [LeoIV/mujoco-py](https://github.com/LeoIV/mujoco-py)) and `mujoco = 2.2.2`
- Other deps: LassoBench, ebo, torch, scikit-learn, gym, box2d, etc.

---

## 4. Optional system packages (if MuJoCo build fails)

From the [BenchSuite README](https://github.com/lpapenme/BenchSuite#troubleshooting):

```bash
sudo apt install swig libglew-dev patchelf
```

---

## 5. Run BenchSuite directly (standalone)

From **BenchSuite** root, with the env vars above set:

```bash
cd src/benchmark/BenchSuite
poetry run python3 main.py --name swimmer -x 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```

`main.py` expects:

- `--name`: benchmark name (e.g. `swimmer`, `hopper`, `ant`, `humanoid`, `lunarlander`, `robotpushing`, `svm`, …).
- `-x`: one float per dimension (normalized in [0,1]; BenchSuite scales to `lb + (ub - lb) * x`).

MuJoCo benchmark names in BenchSuite:

| define_problems name | BenchSuite `--name` |
|----------------------|---------------------|
| swimming             | `swimmer`           |
| hopper               | `hopper`            |
| ant                  | `ant`               |
| humanoid             | `humanoid`          |

SVM needs libsvm data; BenchSuite sets `LIBSVMDATA_HOME=/tmp/libsvmdata` when running.

---

## 6. Use from natural_bo (DefineProblems)

To have `DefineProblems("swimming")`, etc., work inside **natural_bo**:

1. BenchSuite must be **cloned** at `src/benchmark/BenchSuite`.
2. The **wrapper** module `src/benchmark/benchsuite_lassobench_wrappers.py` must exist and successfully import the BenchSuite classes (SVM, LunarLander, RobotPushing, Swimming, Hopper, Ant, Humanoid) and expose them as BoTorch-style problems.  
   If that file is missing, it needs to be implemented to wrap BenchSuite’s `benchsuite.benchmarks` (e.g. `MujocoSwimmer`, `MujocoHopper`, …) so they have `.dim`, `.bounds`, and `__call__(x)` returning a `(batch_size, 1)` tensor.
3. When running or testing, **either**:
   - Run from a shell where you’ve set `LD_LIBRARY_PATH` and `MUJOCO_PY_MUJOCO_PATH` as in step 2, or  
   - Set them in the same process before importing BenchSuite (e.g. in a test conftest or a small launcher script).

After that, from natural_bo root you can run your benchmarks or experiments; MuJoCo problems will work if the wrapper is in place and BenchSuite imports succeed (or via the subprocess wrapper when BenchSuite is in the parent directory).

---

## Summary

| Step | Action |
|------|--------|
| 1 | `git clone https://github.com/lpapenme/BenchSuite.git src/benchmark/BenchSuite` |
| 2 | Set `LD_LIBRARY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to `.../BenchSuite/data/mujoco210` (and `.../bin` for LD_LIBRARY_PATH) |
| 3 | `cd src/benchmark/BenchSuite && poetry install` |
| 4 | Optional: `sudo apt install swig libglew-dev patchelf` if needed |
| 5 | Run MuJoCo: `poetry run python3 main.py --name swimmer -x 0.0 ...` (16 zeros for swimmer) |
| 6 | For natural_bo: ensure `benchsuite_lassobench_wrappers.py` exists and env vars are set when running/tests |

Reference: [lpapenme/BenchSuite](https://github.com/lpapenme/BenchSuite) (README, main.py, pyproject.toml, benchsuite/benchmarks.py).
