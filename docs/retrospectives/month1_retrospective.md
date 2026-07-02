# ProbOS - Month 1 Retrospective

**Period:** Week 1 to Week 4
**Status at close:** Kernel foundation complete, PDSL v0.1 shipped, 3 cross-discipline examples validated, CI fully green.

## Final numbers

- Python tests: 264/264 passing
- C++ tests: 13/13 passing
- mypy strict: 0 errors
- ruff: 0 warnings
- CI jobs: 4/4 green (Python, C++, LaTeX manuscript, full integration)

## What shipped

### Week 1 - Distribution ABC
- Distribution ABC with sample(), pdf(), log_pdf(), cdf(), ppf()
- Concrete distributions: Normal, LogNormal, Beta, Uniform
- 44 Python tests, 13 C++ tests

### Week 2 - Model ABC + Physics
- Model ABC defining the forward_batch() contract
- BatteryModel2Cell: 8-state Arrhenius thermal-runaway ODE, validated against Kim et al. 2007
- CLT convergence demo, deterministic ODE demo

### Week 3 - Monte Carlo Kernel
- MonteCarloEngine: vectorised particle propagation, percentile tracking
- SobolSensitivity: Sobol S1/ST indices via SALib
- ProvenanceTracker: causal audit trail for individual particle trajectories
- 91 new tests across MC engine, sensitivity, provenance

### Week 4 - Speed, Language, Cross-Discipline
- Monday: DP profiling. Real wins: 1040x SALib cache, 45x CLT subset trick. One real non-win (inv_RT precompute ~1x) documented honestly.
- Tuesday: C++ BatteryCell + OpenMP MonteCarloEngineOMP. Confirmed 7x speedup over Python serial via genuine benchmark.
- Wednesday: PDSL compiler v0.1. Lark grammar to AST to codegen to compiler, 28 tests.
- Literature pass: added Feng 2018, Saltelli 2019, Chopin and Papaspiliopoulos 2020, Naesseth 2019, FDA 2019.
- Repo hygiene: moved generated PNGs into outputs/figures/, fixed a CI break (missing lark dependency), wrote check_ci.sh.
- Saturday: 3 cross-discipline examples proving kernel domain-agnosticism (option pricer, ED queue, clinical trial).
- docs/study/study_guide.md created as a permanent running study guide.

## What went right

1. Honest profiling over claimed wins. The DP session found inv_RT precomputation gave no real speedup and said so directly.
2. Kernel domain-agnosticism validated, not assumed. Three Week 4 examples needed zero kernel changes, only new Model subclasses.
3. CI treated as a first-class deliverable. The lark dependency break was caught and fixed same-session.
4. Test discipline held under example-code growth. Even demo scripts got real test suites (34 tests total).

## What was harder than expected

1. PDSL parser debugging. Unary minus inside function calls required several grammar iterations.
2. Repo hygiene delayed feature work. PNG reorganisation and the resulting CI break took real Week 4 time.
3. Terminal and heredoc fragility. Shell history expansion and long heredocs breaking mid-stream cost debugging cycles.

## Carry-forward risks for Month 2

- No FastAPI or pybind11 yet. The kernel is pure-Python plus a separate C++ binary.
- Particle filter is unbuilt. ProvenanceTracker gives causal post-hoc analysis but no online inference yet.
- PDSL v0.1 has no control flow. Current grammar supports linear drift/diffusion declarations only.
