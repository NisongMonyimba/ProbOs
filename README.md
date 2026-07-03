# ProbOS — A Probabilistic Execution Runtime

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-341%20Python%20%2B%2013%20C%2B%2B-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Month%202%20Week%207-Complete-blue?style=for-the-badge)

**Uncertainty is a first-class data type.**

*A probabilistic execution runtime built one week at a time.*

[What is ProbOS?](#what-is-probos) •
[Quick Start](#quick-start) •
[Architecture](#architecture) •
[Current Capabilities](#current-capabilities) •
[How to Run](#how-to-run) •
[Roadmap](#roadmap)

</div>

---

## What is ProbOS?

Most software treats uncertainty as an afterthought — a special case handled
after the real computation is done. **ProbOS treats uncertainty as the default.**

The analogy with a classical operating system is precise:

| Linux | ProbOS |
|-------|--------|
| Process | Stochastic program (distribution over trajectories) |
| Memory address | Random variable node in the execution graph |
| Scheduler | `MonteCarloEngine` (forward) / `ParticleFilter` (inference) |
| System call | `sample`, `observe`, `condition` |
| File | Probability distribution |
| Kernel | `Distribution`/`Model` ABCs + C++/OpenMP engine (pybind11-bound) |

### Why does this matter?

A deterministic battery simulation uses **one** value for activation energy.
ProbOS simulates **5,000 batteries simultaneously** and finds that the
worst-case (P05) battery decomposes **11.6x faster** than the mean.
That tail risk is completely invisible to deterministic models.

---

## Quick Start

### Prerequisites

| Tool | Minimum Version | Install command |
|------|----------------|----------------|
| Ubuntu / WSL | 22.04 | [WSL Install Guide](https://learn.microsoft.com/en-us/windows/wsl/install) |
| Python | 3.11 | `sudo apt-get install python3.11` |
| g++ | 11 | `sudo apt-get install build-essential` |
| CMake | 3.22 | `sudo apt-get install cmake` |
| Ninja | any | `sudo apt-get install ninja-build` |
| Google Test | any | `sudo apt-get install libgtest-dev libgmock-dev` |

### Clone and run (Ubuntu / WSL)

```bash
git clone https://github.com/NisongMonyimba/ProbOs
cd ProbOs
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install SALib matplotlib scipy pybind11 fastapi uvicorn httpx

# Build the C++ kernel + pybind11 extension
PYBIND11_CMAKE_DIR=$(python -m pybind11 --cmakedir)
cmake -B cpp/build -S cpp -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python) -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" -Wno-dev
cmake --build cpp/build

# Run everything
python -m pytest python/tests/   # 341 Python tests
./cpp/build/test_normal          # 13 C++ tests
bash pre_week_audit.sh           # full quality audit (coverage, security, etc.)
```

### Expected output

```
341/341 Python tests passing
13/13   C++ tests passing
mypy strict: 0 errors (full python/ tree)
ruff: 0 warnings
pre_week_audit.sh: 21/21 checks passed
```

### Run the FastAPI service

```bash
uvicorn python.server.main:app --reload
# open http://localhost:8000/docs for interactive API documentation
```

---

## Architecture

### System Layers

```
+-------------------------------------------------------------+
|              python/server/  (FastAPI, Week 7)              |
|   GET  /health        POST /simulate                        |
|   POST /sensitivity    POST /filter                         |
+----------------------------+--------------------------------+
                             |
+----------------------------+--------------------------------+
|              python/pdsl/  (PDSL compiler, Week 4)           |
|   grammar.lark -> parser -> ast_nodes -> codegen -> compiler|
+----------------------------+--------------------------------+
                             |
+----------------------------+--------------------------------+
|                    python/src/  (Core kernel)                 |
|                                                               |
|   Distribution (ABC)       Model (ABC)                       |
|   +-- Normal, LogNormal,   +-- BatteryModel2Cell              |
|       Uniform, Beta            (8-state Arrhenius ODE)        |
|                                                               |
|   MonteCarloEngine   SobolSensitivity   ProvenanceTracker    |
|   ParticleFilter (validated against exact Kalman filter)     |
+----------------------------+--------------------------------+
                             |
             +---------------+---------------+
             v                               v
+------------------------+       +--------------------------+
|  Python engines        |       |  cpp/ (C++20 + pybind11) |
|  numpy + SALib         |       |  BatteryCell             |
|  341 pytest tests      |       |  MonteCarloEngineOMP      |
|                        |       |  probos_cpp extension     |
|                        |       |  7x faster (measured)     |
|                        |       |  13 Google Tests          |
+------------------------+       +--------------------------+
```

### Repository Structure

```
ProbOs/
|
+-- python/
|   +-- src/                          Core kernel
|   |   +-- distributions.py          Distribution ABC + 4 classes
|   |   +-- state.py                  Model ABC
|   |   +-- battery_model.py          BatteryModel2Cell
|   |   +-- parameter_priors.py       15 priors (Kim 2007)
|   |   +-- monte_carlo.py            MonteCarloEngine
|   |   +-- sensitivity.py            SobolSensitivity
|   |   +-- provenance.py             ProvenanceTracker
|   |   +-- particle_filter.py        ParticleFilter (SIR)
|   +-- pdsl/                         PDSL compiler
|   |   +-- grammar.lark, parser.py, ast_nodes.py,
|   |       codegen.py, compiler.py
|   +-- server/                       FastAPI service layer
|   |   +-- main.py, schemas.py
|   +-- examples/                     Cross-discipline demos
|   |   +-- week2_battery_ode.py, week3_mc_battery.py,
|   |       week3_clt_convergence.py, week3_sobol_battery.py,
|   |       week4_option_pricer.py, week4_ed_queue.py,
|   |       week4_clinical_trial.py
|   +-- tests/                        341 tests total
|
+-- cpp/
|   +-- include/distributions/normal.hpp
|   +-- include/kernel/battery_cell.hpp
|   +-- include/kernel/monte_carlo_omp.hpp
|   +-- src/distributions/normal.cpp
|   +-- src/kernel/monte_carlo_omp.cpp
|   +-- src/kernel/benchmark_omp.cpp
|   +-- src/main.cpp
|   +-- bindings/probos_bindings.cpp   pybind11 -> probos_cpp
|   +-- tests/test_normal.cpp          13 Google Tests
|   +-- CMakeLists.txt
|
+-- docs/
|   +-- monthly_plans/                Day-by-day plan + retrospective
|   |   +-- overall/                  per week, per month, PDF-compiled
|   |   +-- month1/week1-4/
|   |   +-- month2/week5-7/
|   +-- standards/quality_standards.md
|   +-- study/study_guide.md
|   +-- audits/                       Archived pre_week_audit.sh reports
|   +-- architecture.md               This project's architecture, in depth
|   +-- retrospectives/
|
+-- manuscript/                       Research paper (LaTeX, publication
|                                      deferred to Year 2 -- see overall plan)
+-- outputs/figures/                  All generated plots
+-- CMakeLists.txt                    Root wrapper -> add_subdirectory(cpp)
+-- pyproject.toml                    Single source of truth for dependencies
+-- pre_week_audit.sh                 Standing quality audit (run before
|                                      every new week's work)
+-- check_ci.sh                       Watch/verify CI status from the terminal
```

---

## Current Capabilities

### The Core Insight: Why `log_pdf` Must Be Analytical

```python
# WRONG: underflows to 0.0 then log gives -inf
log_density = np.log(distribution.pdf(x))

# CORRECT: analytical form, always finite
log_density = distribution.log_pdf(x)
```

| Expression | Value at `x = mu + 50*sigma` | Status |
|-----------|-------|--------|
| `pdf(x)` | `0.0` | Underflow to zero |
| `np.log(pdf(x))` | `-inf` | **WRONG** |
| `log_pdf(x)` analytical | `-1259.44` | **Correct** |

### Monte Carlo + Sensitivity (Month 1)

Running N=5,000 particles through `BatteryModel2Cell`:

| Metric | Value |
|--------|-------|
| `sigma/sqrt(N)` for T1 at N=5000 | 7.30 K |
| CLT log-log slope | -0.569 (theory -0.500) |
| **Ea_SEI** Sobol $S_1$ / $S_T$ | **0.457 / 0.729** (dominates T1 variance) |
| P05 SEI decomposition rate | **11.6x faster** than the P50 mean |

### Sequential Inference (Month 2 Week 5)

`ParticleFilter`'s posterior mean and std are proven — not assumed — to
converge to an exact closed-form Kalman filter solution on a
linear-Gaussian test case, with Monte Carlo error provably shrinking as
particle count grows from N=100 to N=2000.

### C++/OpenMP Kernel + pybind11 (Month 1 Week 4, Month 2 Week 6)

`BatteryCell::forward_step()` matches the Python
`BatteryModel2Cell.forward_batch()` to `rtol=1e-8`–`1e-10`. The bound
`probos_cpp.MonteCarloEngineOMP` runs **~7x faster** than the pure-Python
engine (measured, not claimed).

### PDSL Compiler (Month 1 Week 4)

Declare a stochastic model in a small DSL instead of hand-writing a
`Model` subclass:

```
model battery {
    state    T1 = 298.0
    param    Ea_SEI ~ Normal(1.35e5, 5e3)
    drift    T1 <- arrhenius(Ea_SEI, T1)
}
```

Pipeline: `grammar.lark -> parser.py -> ast_nodes.py -> codegen.py -> compiler.py`.

### REST API (Month 2 Week 7)

```
GET  /health
POST /simulate      -> MonteCarloEngine
POST /sensitivity     -> SobolSensitivity
POST /filter           -> ParticleFilter
```

Resource-exhaustion bounds enforced via Pydantic before any request
reaches kernel code. All three POST endpoints verified to match a
direct Python kernel call at `rtol=1e-10`.

### Cross-Discipline Validation (Month 1 Week 4)

The same kernel, with zero core changes, correctly models:

| Domain | Model | Validated against |
|--------|-------|-------------------|
| Finance | `OptionPricerModel` | Black-Scholes closed form |
| Hospital ops | `EDQueueModel` | M/M/1 queueing theory |
| Clinical trials | `ClinicalTrialModel` | Adaptive Bayesian design theory |

---

## How to Run

### Full quality audit (run before starting any new week's work)

```bash
bash pre_week_audit.sh
```

Checks: full test suite, mypy strict (full tree), ruff, coverage floor
(85%), Hypothesis property-based tests, doctest, bandit security scan,
pip-audit CVE scan, packaging/build verification, reproducibility, git
hygiene. See `docs/standards/quality_standards.md` for the full checklist.

### Watch CI status from the terminal

```bash
bash check_ci.sh
```

### Run a specific example

```bash
.venv/bin/python python/examples/week1_coin_flip.py        # LLN primer
.venv/bin/python python/examples/week1_normal_demo.py      # log_pdf stability demo
.venv/bin/python python/examples/week3_mc_battery.py       # MC fan plot
.venv/bin/python python/examples/week3_sobol_battery.py    # Sobol sensitivity
.venv/bin/python python/examples/week4_option_pricer.py    # Finance example
.venv/bin/python python/examples/week4_ed_queue.py         # Ops example
.venv/bin/python python/examples/week4_clinical_trial.py   # Medicine example
```

### Use the library in your own Python code

```python
from python.src.distributions import Normal
from python.src.battery_model import BatteryModel2Cell
from python.src.parameter_priors import build_battery_priors
from python.src.monte_carlo import MonteCarloEngine

model  = BatteryModel2Cell()
priors = build_battery_priors()
engine = MonteCarloEngine(model, priors, N=5000, n_steps=300, dt=1.0, seed=42)
result = engine.run()

print(f"P05 T1 at t=300: {result.percentiles[0, -1, 0]:.1f} K")
print(f"P50 T1 at t=300: {result.percentiles[1, -1, 0]:.1f} K")
```

### Use the bound C++ kernel directly

```python
import sys
sys.path.insert(0, "cpp/build")
import probos_cpp

engine = probos_cpp.MonteCarloEngineOMP(N=5000, n_steps=300, dt=1.0)
result = engine.run(seed=42)
print(f"Wall time: {result.wall_time_ms:.1f} ms")
```

---

## Roadmap

The full 24-month roadmap, with an honest month-by-month breakdown of
what is DONE, PLANNED, or DIRECTIONAL, lives in
`docs/monthly_plans/overall/main.tex` (compiled PDF alongside). Every
individual week has its own day-by-day plan and retrospective under
`docs/monthly_plans/<month>/<week>/`.

**Status at a glance:**

```
Month 1  [COMPLETE]  Kernel foundation
  Week 1  Distribution ABC
  Week 2  Model ABC + BatteryModel2Cell (Kim 2007 validated)
  Week 3  MonteCarloEngine + SobolSensitivity + ProvenanceTracker
  Week 4  DP profiling, C++/OpenMP kernel, PDSL v0.1,
          3 cross-discipline examples

Month 2  [IN PROGRESS]  Inference + service layer
  Week 5  ParticleFilter (validated vs exact Kalman filter)   [DONE]
  Week 6  pybind11 bindings -- C++ kernel enters Python       [DONE]
  Week 7  FastAPI service layer                                [DONE]
  Week 8  Cross-discipline filtering examples + Month 2
          retrospective                                        [NEXT]

Month 3+  DIRECTIONAL -- to be planned in detail as each
          month approaches, per the process established after
          Month 1 (see docs/monthly_plans/overall/main.tex)
```

---

## Quality Standards

Every layer is validated against a closed-form solution, literature
reference, or cross-implementation comparison before being trusted --
never just "it runs without crashing." Full standards and rationale in
`docs/standards/quality_standards.md`.

**Current numbers:**

```
341/341 Python tests passing
13/13   C++ tests passing
mypy strict: 0 errors (full python/ tree)
ruff: 0 warnings
Coverage: 90%+ (floor: 85%)
bandit: 0 findings
pip-audit: 0 known CVEs
```

---

## Project Information

| Item | Detail |
|------|--------|
| Author | [AUTHOR NAME] |
| Organisation | Reality Computing Corporation |
| Started | June 2026 |
| License | MIT |
| Languages | Python 3.11 + C++20 |
| Build system | CMake 3.22 + Ninja |
| Platform | Ubuntu 22.04 / WSL2 |
| Repository | https://github.com/NisongMonyimba/ProbOs |
| Publication | Deferred to Year 2 (see `docs/monthly_plans/overall/main.tex`) |

---

## License

MIT License. See [LICENSE](LICENSE) for full terms.

---

<div align="center">

*"Uncertainty is not an obstacle to computation. It is the computation."*

**Reality Computing Corporation — 2026**

</div>
