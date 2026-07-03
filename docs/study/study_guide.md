# ProbOS Study Guide

A running log of books, papers, and resources mapped to the exact code
written each week. Read the listed material BEFORE or ALONGSIDE the
code for that week -- not after. Each entry says which file the
reading explains.

---

## How to use this document

- Entries are appended week by week, in the order code was written.
- Each entry lists: the file(s) it explains, the book/paper, specific
  chapters/sections, and a one-line reason why it matters for that file.
- "Always open" resources (kept at the bottom) are referenced repeatedly
  across many weeks and don't need re-reading each time.

---

## Month 1

### Week 1 -- Distribution ABC

| File | Resource | Chapters | Why |
|------|----------|----------|-----|
| `python/src/distributions.py` | Blitzstein & Hwang, *Introduction to Probability* | Ch 1-4 | Probability axioms, Bayes, Normal/Beta/Uniform distributions |
| `python/src/distributions.py` (log_pdf) | Billingsley, *Probability and Measure* | Ch 1-3 (skim) | Why log-density must be analytical, not log(pdf(x)) |
| `cpp/src/distributions/normal.cpp` | Stroustrup, *A Tour of C++* (3rd ed.) | Ch 1-4 | `<random>`, `<cmath>`, basic C++ classes |

### Week 2 -- Model ABC + BatteryModel2Cell

| File | Resource | Chapters | Why |
|------|----------|----------|-----|
| `python/src/battery_model.py` | Boyce & DiPrima, *Elementary Differential Equations* | Ch 1-3 | Euler method, ODE basics underlying `forward_batch` |
| `python/src/battery_model.py` | Atkins & de Paula, *Physical Chemistry* | Ch 17-18 | Arrhenius equation: `k = A*exp(-Ea/RT)` |
| `python/src/battery_model.py` | Kim, Pesaran & Spotnitz (2007), *J. Power Sources* 170(2) | Full paper | Original source for the 3-reaction battery model and nominal parameters |
| `python/src/battery_model.py` (broadcasting) | VanderPlas, *Python Data Science Handbook* | Ch 2 | NumPy broadcasting -- why `forward_batch` has no Python loops |

### Week 3 -- MC Engine + Sobol + Provenance

| File | Resource | Chapters | Why |
|------|----------|----------|-----|
| `python/src/monte_carlo.py` | Robert & Casella, *Monte Carlo Statistical Methods* | Ch 1-3 | MC integration, CLT convergence -- the `sigma/sqrt(N)` certificate |
| `python/src/sensitivity.py` | Saltelli et al., *Global Sensitivity Analysis: The Primer* | Ch 1-4 | Sobol indices, Saltelli sampling scheme |
| `python/src/sensitivity.py` | Saltelli et al. (2019), *Environmental Modelling & Software* 114 | Full paper | "Why so many published sensitivity analyses are false" -- the SALib reproducibility caveat documented in `test_sensitivity.py` |
| `python/src/provenance.py` | Cormen et al., *CLRS* | Ch 22-23 | BFS, DAGs -- foundation for `query_ancestors` |

### Week 4 -- DP fixes + C++ OpenMP + PDSL

| File | Resource | Chapters | Why |
|------|----------|----------|-----|
| `python/src/sensitivity.py` (caching) | Cormen et al., *CLRS* | Ch 15-16 | Memoization, tabulation -- the `_build_problem()` cache (1040x speedup) |
| `python/examples/week4_clt_convergence_dp.py` | Cormen et al., *CLRS* | Ch 15-16 | Tabulation (subset trick, 45x speedup) |
| `cpp/include/kernel/battery_cell.hpp` | Stroustrup, *A Tour of C++* | Ch 5-8 | Templates, concepts, C++20 features used in the header |
| `cpp/src/kernel/monte_carlo_omp.cpp` | Pacheco, *An Introduction to Parallel Programming* | Ch 1-4 | OpenMP `#pragma omp parallel for`, thread scheduling |
| `cpp/src/kernel/benchmark_omp.cpp` (profiling result) | Drepper, *What Every Programmer Should Know About Memory* (free PDF) | All | Explains WHY OpenMP gave ~1x speedup -- cache/memory bandwidth, not CPU-bound work |
| `python/pdsl/grammar.lark` | Nystrom, *Crafting Interpreters* (free online) | Ch 1-6 | Scanning, parsing, grammar design |
| `python/pdsl/parser.py` | Nystrom, *Crafting Interpreters* | Ch 5-6 | Recursive descent / Lark transformer pattern |
| `python/pdsl/codegen.py` | Appel, *Modern Compiler Implementation in Java* | Ch 1-7 | AST -> target code generation pattern |
| `python/examples/week4_option_pricer.py` | Hull, *Options, Futures, and Other Derivatives* | Ch 13-15 | Black-Scholes, GBM, European call payoff |
| `python/examples/week4_option_pricer.py` | Glasserman, *Monte Carlo Methods in Financial Engineering* | Ch 1-3 | MC for options, exact GBM simulation (not Euler) |
| `python/examples/week4_option_pricer.py` | Shreve, *Stochastic Calculus for Finance II* | Ch 4 | Why GBM has a closed-form solution (Itô's lemma applied to log(S)) |
| `python/examples/week4_ed_queue.py` | Kleinrock, *Queueing Systems Vol. 1* | Ch 2-3 | M/M/1 queue theory, steady-state formulas (rho, L, W) |
| `python/examples/week4_clinical_trial.py` | Berry (2011), *J. Clinical Oncology* | Full paper | Adaptive Bayesian trial design fundamentals |
| `python/examples/week4_clinical_trial.py` | FDA (2019), *Adaptive Designs for Clinical Trials of Drugs and Biologics* | Section IV-V | Current regulatory standard for simulation-based trial design evidence |
| `python/examples/week4_clinical_trial.py` | Gelman et al., *Bayesian Data Analysis* (3rd ed.) | Ch 2-3 | Beta-Binomial conjugacy, posterior updating |

---

## Always-open references (consult repeatedly, no fixed week)

| Resource | Use for |
|----------|---------|
| Blitzstein & Hwang, *Introduction to Probability* | Any new distribution |
| Robert & Casella, *Monte Carlo Statistical Methods* | Any MC engine change |
| Saltelli et al., *Global Sensitivity Analysis* | Any Sobol/sensitivity question |
| Stroustrup, *A Tour of C++* | Any C++ code |
| Eigen documentation (eigen.tuxfamily.org) | Any matrix operation (Month 3+) |
| FastAPI documentation (fastapi.tiangolo.com) | Any API endpoint (Month 2+) |
| CUDA C++ Programming Guide (docs.nvidia.com/cuda) | Any GPU kernel (Month 3+) |
| Nystrom, *Crafting Interpreters* | Any PDSL compiler work |
| Cormen et al., *CLRS* | Any algorithm/complexity question |

---

## Free resources index

| Resource | URL |
|----------|-----|
| Crafting Interpreters | https://craftinginterpreters.com |
| What Every Programmer Should Know About Memory | https://lwn.net/Articles/250967 |
| Boyd & Vandenberghe, Convex Optimization | https://web.stanford.edu/~boyd/cvxbook/ |
| Naesseth, Lindsten, Schön (2019), Elements of SMC | https://arxiv.org/abs/1903.04797 |
| FastAPI docs | https://fastapi.tiangolo.com |
| pybind11 docs | https://pybind11.readthedocs.io |
| Eigen docs | https://eigen.tuxfamily.org |
| Kim et al. (2007) | DOI: 10.1016/j.jpowsour.2007.04.018 |

---

## Month 2

### Week 5 -- Particle Filter Core

| File | Resource | Chapters | Why |
|------|----------|----------|-----|
| `python/src/particle_filter.py` | Naesseth, Lindsten & Schon (2019), *Elements of Sequential Monte Carlo* | Full paper (free PDF) | Unifying predict/update/resample template used directly as this module's docstring structure |
| `python/src/particle_filter.py` | Chopin & Papaspiliopoulos (2020), *An Introduction to Sequential Monte Carlo* | Ch 8-10 | Systematic resampling scheme, ESS-triggered resampling threshold (0.5) |
| `python/src/particle_filter.py` (log-space weights) | Distribution.log_pdf() rationale (Month 1 Week 1) | -- | Same underflow-avoidance discipline applied to particle weights across many timesteps |
| `python/tests/test_particle_filter.py` | Sarkka (2013), *Bayesian Filtering and Smoothing* | Ch 4 | Closed-form 1D Kalman filter recursion, used as ground truth to validate the particle filter numerically |

### Week 6 -- pybind11: C++ Kernel Enters the Python Package

| File | Resource | Chapters | Why |
|------|----------|----------|-----|
| `cpp/bindings/probos_bindings.cpp` | pybind11 documentation, *Basics* + *NumPy* chapters | https://pybind11.readthedocs.io/en/stable/basics.html, .../advanced/pycpp/numpy.html | `py::array_t<double>`, `py::class_`, `def_property_readonly`, capsule-based ownership for zero-copy NumPy array wrapping |
| `cpp/CMakeLists.txt` (pybind11_add_module) | pybind11 CMake integration guide | https://pybind11.readthedocs.io/en/stable/compiling.html#building-with-cmake | `find_package(pybind11)`, `pybind11_add_module()`, linking OpenMP into a Python extension module |
| `python/tests/test_cpp_bindings.py` (cross-validation discipline) | Month 1 Week 2 -- validating BatteryModel2Cell against Kim 2007 ARC data | -- | Same "validate against a known reference before trusting the port" discipline applied to a Python-vs-C++ port instead of a code-vs-paper comparison |

### Pre-Week 7 hardening -- Quality standards + property-based testing

| File | Resource | Chapters | Why |
|------|----------|----------|-----|
| `python/tests/test_distributions_properties.py` | Hypothesis documentation, *What is Property-Based Testing?* + *Writing Strategies* | https://hypothesis.readthedocs.io/en/latest/quickstart.html | `@given`, `st.floats/integers`, `@settings(max_examples=...)`, shrinking behaviour on failure |
| `docs/standards/quality_standards.md` | OWASP Python Security guidance (bandit's rule basis) | B102 (exec), general SAST principles | Why `exec()` on generated-not-raw-input is an accepted, documented risk rather than a finding to silence blindly |
| `pre_week_audit.sh` (pip-audit section) | pip-audit documentation | https://pypi.org/project/pip-audit/ | Dependency CVE scanning as a standing pre-week check, not a one-off |
