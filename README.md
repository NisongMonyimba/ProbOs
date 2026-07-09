# ProbOS — A Probabilistic Execution Runtime

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-408%20Python%20%2B%205%20C%2B%2B-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Month%203%20Week%2013-Complete-blue?style=for-the-badge)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20it%20now-F5A623?style=for-the-badge&logo=googlechrome&logoColor=white)](https://nisongmonyimba.github.io/ProbOs/)

**Uncertainty is a first-class data type.**

*A probabilistic execution runtime built one week at a time.*

[**Try the Live Demo**](https://nisongmonyimba.github.io/ProbOs/) •
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

The same principle now applies across two independently validated physical
domains — battery thermal runaway and pharmaceutical shelf-life stability
— with a third (automotive/EV safety) positioned to reuse the battery model
directly, since it shares the same underlying chemistry.

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
python -m pytest python/tests/          # 408 Python tests
ctest --test-dir cpp/build               # 5 C++ test executables
bash pre_week_audit.sh                   # full quality audit (coverage, security, etc.)
```

### Expected output

```
408/408 Python tests passing
5/5     C++ test executables passing (test_normal, test_lognormal,
        test_uniform, test_battery_cell, test_monte_carlo_omp)
mypy strict: 0 errors (full python/ tree)
ruff: 0 warnings
pre_week_audit.sh: 16/16 checks passed
```

### Run the FastAPI service

```bash
uvicorn python.server.main:app --reload
# open http://localhost:8000/docs for interactive API documentation
```

### Watch CI status from the terminal

```bash
bash check_ci.sh
```

Matches the exact commit at `HEAD` (not just "most recent run") via a
short polling retry loop — safe to call immediately after `git push`,
no manual delay needed.

---

## Architecture

### System Layers

```
+-------------------------------------------------------------+
|              python/server/  (FastAPI, Month 2 Week 7)      |
|   GET  /health        POST /simulate                        |
|   POST /sensitivity    POST /filter                         |
|   All 4 models registered (Month 3 Week 9)                  |
+----------------------------+--------------------------------+
                             |
+----------------------------+--------------------------------+
|         python/pdsl/  (PDSL compiler, v0.1 Month 1 Week 4,   |
|                         v0.2 control flow Month 3 Week 13)   |
|   grammar.lark -> parser -> ast_nodes -> codegen -> compiler|
|   v0.2 adds: comparison operators, if/then/else conditionals |
|   compiling to vectorised np.where()                         |
+----------------------------+--------------------------------+
                             |
+----------------------------+--------------------------------+
|                    python/src/  (Core kernel)                 |
|                                                               |
|   Distribution (ABC)       Model (ABC)                       |
|   +-- Normal, LogNormal,   +-- BatteryModel2Cell              |
|       Uniform, Beta,           (8-state Arrhenius ODE,        |
|       Empirical                validated: Kim et al. 2007)   |
|                             +-- PharmaStabilityModel           |
|                                 (Avrami kinetics, validated:   |
|                                 Gonzalez-Gonzalez et al. 2023) |
|                                                               |
|   MonteCarloEngine   SobolSensitivity   ProvenanceTracker    |
|   ParticleFilter (validated against exact Kalman filter)     |
|   GPUMonteCarloEngine (CuPy + kernel fusion, honestly          |
|       benchmarked -- C++/OpenMP wins at every tested N)       |
+----------------------------+--------------------------------+
                             |
             +---------------+---------------+
             v                               v
+------------------------+       +--------------------------+
|  Python engines        |       |  cpp/ (C++20 + pybind11) |
|  numpy + SALib         |       |  BatteryCell             |
|  408 pytest tests      |       |  MonteCarloEngineOMP     |
|                        |       |  (real battery_priors,   |
|                        |       |   not a placeholder      |
|                        |       |   perturbation)          |
|                        |       |  Normal, LogNormal,      |
|                        |       |  Uniform distributions   |
|                        |       |  probos_cpp extension    |
|                        |       |  7x faster (measured)    |
|                        |       |  5 test executables      |
+------------------------+       +--------------------------+
```

### Repository Structure

```
ProbOs/
|
+-- python/
|   +-- src/                          Core kernel
|   |   +-- distributions.py          Distribution ABC + 5 classes
|   |   +-- state.py                  Model ABC
|   |   +-- battery_model.py          BatteryModel2Cell (+ GPU/xp dispatch)
|   |   +-- pharma_stability_model.py PharmaStabilityModel (Avrami kinetics)
|   |   +-- parameter_priors.py       15 real battery priors (Kim 2007)
|   |   +-- monte_carlo.py            MonteCarloEngine
|   |   +-- gpu_monte_carlo.py        GPUMonteCarloEngine (CuPy)
|   |   +-- sensitivity.py            SobolSensitivity
|   |   +-- provenance.py             ProvenanceTracker
|   |   +-- particle_filter.py        ParticleFilter (SIR)
|   +-- pdsl/                         PDSL compiler (v0.1 + v0.2 control flow)
|   |   +-- grammar.lark, parser.py, ast_nodes.py,
|   |       codegen.py, compiler.py
|   +-- server/                       FastAPI service layer
|   |   +-- main.py, schemas.py
|   +-- examples/                     Cross-discipline demos
|   |   +-- week2_battery_ode.py, week3_mc_battery.py,
|   |       week3_clt_convergence.py, week3_sobol_battery.py,
|   |       week4_option_pricer.py, week4_ed_queue.py,
|   |       week4_clinical_trial.py
|   +-- tests/                        408 tests total
|
+-- cpp/
|   +-- include/distributions/        normal.hpp, lognormal.hpp, uniform.hpp
|   +-- include/kernel/               battery_cell.hpp, monte_carlo_omp.hpp,
|   |                                 battery_priors.hpp (real 15-param priors)
|   +-- src/distributions/            normal.cpp, lognormal.cpp, uniform.cpp
|   +-- src/kernel/monte_carlo_omp.cpp, benchmark_omp.cpp
|   +-- src/main.cpp
|   +-- bindings/probos_bindings.cpp   pybind11 -> probos_cpp
|   +-- tests/                        5 test executables (test_normal,
|   |                                 test_lognormal, test_uniform,
|   |                                 test_battery_cell, test_monte_carlo_omp)
|   +-- CMakeLists.txt                 -fPIC globally (Debug+ASan+shared-lib fix)
|
+-- docs/
|   +-- vision/                       Mission, vision, industry roadmap
|   +-- monthly_plans/                Day-by-day plan + retrospective
|   |   +-- overall/                  24-month master roadmap, PDF-compiled
|   |   +-- month1/week1-4/, month2/week5-8/, month3/week9-13/
|   |   +-- month4/ ... month12/      Year 1 pre-planning (directional)
|   +-- standards/quality_standards.md
|   +-- study/study_guide.md          Real study log, updated as work happens
|   +-- audits/                       Archived pre_week_audit.sh reports
|   +-- architecture.md               This project's architecture, in depth
|   +-- automotive_ev_positioning.md, automotive_ev_outreach_email.md
|   +-- sbir_phase1_onepager.md, enterprise_pilot_email.md
|
+-- manuscript/                       Research paper (LaTeX, publication
|                                      deferred to Year 2 -- see overall plan)
+-- outputs/figures/                  All generated plots
+-- CMakeLists.txt                    Root wrapper -> add_subdirectory(cpp)
+-- pyproject.toml                    Single source of truth for dependencies
+-- pre_week_audit.sh                 Standing quality audit (run before
|                                      every new week's work)
+-- check_ci.sh                       Watch/verify CI status (matches exact
|                                      commit SHA, no manual delay needed)
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

### Battery Safety (Month 1) — Validated

`BatteryModel2Cell`, an 8-state Arrhenius thermal-runaway ODE with 15
uncertain parameters, validated against Kim et al. (2007) accelerating
rate calorimetry data. Sobol sensitivity identifies the SEI decomposition
activation energy as the dominant risk driver (S1=0.457).

### Sequential Inference (Month 2 Week 5)

`ParticleFilter`'s posterior mean and std are proven — not assumed — to
converge to an exact closed-form Kalman filter solution on a
linear-Gaussian test case, with Monte Carlo error provably shrinking as
particle count grows from N=100 to N=2000.

### C++/OpenMP Kernel + pybind11 (Month 1 Week 4, Month 2 Week 6, Month 3 sweep)

`BatteryCell::forward_step()` matches the Python
`BatteryModel2Cell.forward_batch()` to `rtol=1e-8`–`1e-10`. The bound
`probos_cpp.MonteCarloEngineOMP` runs **~7x faster** than the pure-Python
engine (measured, not claimed). As of Month 3's comprehensive kernel
sweep, the C++ engine draws from the REAL 15-parameter `battery_priors`
(not a simplified placeholder), closing a previously-documented C++/
Python parameter-spread discrepancy at the root — confirmed within
~1% agreement of the Python engine's own spread.

### PDSL Compiler (v0.1 Month 1 Week 4, v0.2 control flow Month 3 Week 13)

Declare a stochastic model in a small DSL instead of hand-writing a
`Model` subclass:

```
model battery {
    state    T1 = 298.0
    param    Ea_SEI ~ Normal(1.35e5, 5e3)
    drift    T1 <- arrhenius(Ea_SEI, T1)
}
```

PDSL v0.2 adds comparison operators and expression-level conditionals:

```
drift x = if x > 5.0 then -1.0 else 1.0
```

correctly compiling to vectorised `np.where()`, verified with a genuine
per-particle divergence proof (different particles taking different
branches within the same `forward_batch()` call, not just "it compiles").

Pipeline: `grammar.lark -> parser.py -> ast_nodes.py -> codegen.py -> compiler.py`.

### GPU Kernel Path (Month 3 Week 10) — Honestly Benchmarked

`GPUMonteCarloEngine` (CuPy + kernel fusion) was built and honestly
benchmarked against CPU/C++. Kernel fusion genuinely improved GPU
performance (up to 7.6x faster at small N), and GPU now beats plain
Python at N>=2000 — but **C++/OpenMP remains the fastest engine at
every tested N**, reported truthfully rather than oversold.

### Pharmaceutical Stability (Month 3 Week 11) — Validated

`PharmaStabilityModel` uses Avrami kinetics, validated against
Gonzalez-Gonzalez et al. (2023), a real, peer-reviewed, ICH-referenced
chlorhexidine stability study — reproducing the paper's real 365-day
degradation percentages to within 1.18 percentage points. Sobol
sensitivity identifies activation energy as overwhelmingly dominant
(S1=0.993); Monte Carlo propagation reveals a real tail risk invisible
to deterministic simulation — the worst-case 5% of outcomes show
complete potency loss within one year.

### Automotive/EV Safety Positioning (Month 3 Week 12)

`BatteryModel2Cell` (already Validated) is positioned, not
re-engineered, for ISO 26262 automotive functional safety — the same
validated model, reframed for a different regulator. Materials
explicitly state what ProbOS does NOT claim (HARA performance, ASIL
assignment, ISO 26262 certification) before describing what it
genuinely provides: quantitative uncertainty evidence supporting a
HARA process's Severity/Exposure judgments.

### REST API (Month 2 Week 7, extended Month 3 Week 9)

```
GET  /health
POST /simulate      -> MonteCarloEngine
POST /sensitivity     -> SobolSensitivity
POST /filter           -> ParticleFilter
```

Resource-exhaustion bounds enforced via Pydantic before any request
reaches kernel code. All four registered models (battery, option
pricer, ED queue, clinical trial) work with `/simulate` and
`/sensitivity`; `BatteryModel2Cell` (the only fully-deterministic
model) is verified against a direct kernel call at `rtol=1e-10`, while
the three genuinely stochastic models each own a private
`np.random.default_rng(seed)` (fixed at the root, Month 3 Week 9) for
genuine reproducibility. `/filter` correctly rejects `clinical_trial`
with a clear error rather than a meaningless result.

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
Results are archived under `docs/audits/`.

### Check system requirements before setting up

```bash
bash check_system_requirements.sh
```

Run this on any machine (does not need to be inside a clone —
just download and run) to see what's present, what's missing, and
what's optional. Separates REQUIRED items (Python 3.11+, g++,
cmake, ninja, Google Test, pdflatex) from OPTIONAL, GPU-only items
(NVIDIA GPU, CuPy — needed only for `python/src/gpu_monte_carlo.py`,
Month 3 Week 10; everything else works fully without a GPU).

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
`docs/monthly_plans/<month>/<week>/`. Months 4-12 have pre-planning
documents (`docs/monthly_plans/month4/` through `month12/`), kept
intentionally directional rather than over-specified in detail this
far in advance, per this project's own established discipline.

**Status at a glance:**

```
Month 1  [COMPLETE]  Kernel foundation
Month 2  [COMPLETE]  Inference + service layer
Month 3  [COMPLETE]  GPU kernel path, pharma domain (validated),
                     automotive positioning, PDSL v0.2 (control flow),
                     comprehensive C++/kernel sweep

Month 4-6   [PLANNED]  Dashboard architecture, core build,
                       automotive framing, hardening + sweep
Month 7     [PLANNED]  Pharma GPU + C++ parity
Month 8-9   [PLANNED]  Medical device domain (research-gate first)
Month 10    [PLANNED]  Nuclear or aerospace domain (research-gate first)
Month 11    [PLANNED]  PDSL v0.3
Month 12    [PLANNED]  Year 1 comprehensive audit
```

See `docs/vision/main.tex` for the full mission, vision, and honest
Validated/Structurally-ready/Aspirational breakdown per industry.

---

## Quality Standards

Every layer is validated against a closed-form solution, literature
reference, or cross-implementation comparison before being trusted —
never just "it runs without crashing." Full standards and rationale in
`docs/standards/quality_standards.md`.

**Current numbers:**

```
408/408 Python tests passing
5/5     C++ test executables passing
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
| Author | Nisong Monyimba |
| Organisation | Reality Computing Corporation |
| Started | June 2026 |
| License | Apache-2.0 |
| Languages | Python 3.11 + C++20 |
| Build system | CMake 3.22 + Ninja |
| Platform | Ubuntu 22.04 / WSL2 |
| Repository | https://github.com/NisongMonyimba/ProbOs |
| Publication | Deferred to Year 2 (see `docs/monthly_plans/overall/main.tex`) |

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for full terms.

---

<div align="center">

*"Uncertainty is not an obstacle to computation. It is the computation."*

**Reality Computing Corporation — 2026**

</div>
