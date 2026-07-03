# ProbOS Quality Standards

Permanent reference for what "done" means before starting any new week's
work. `pre_week_audit.sh` automates as much of this as possible;
sections marked MANUAL require a human pass.

This document is the single source of truth for standards. When
`pre_week_audit.sh` changes, this document changes in the same commit.

---

## Philosophy

Checks earn their place by catching real bugs cheaply and repeatably.
A checklist nobody reads is worse than a short one people trust. Add a
check when a real problem happened once and the check would have caught
it (see docs/monthly_plans/month1/main.tex retrospective and Week 6/7
audit findings for precedent). Defer checks that do not apply yet
(container security before there is a container; load testing before
there is a service) rather than running no-ops.

---

## 1. Correctness

- [ ] Full Python test suite passes (`pytest python/tests/`)
- [ ] mypy strict passes on the FULL `python/` tree, not a hardcoded
      file list (Week 6/7 audit found 4 real errors hiding in
      `python/examples/` because CI's mypy list was stale)
- [ ] C++ test suite passes (`cpp/build/test_normal`)
- [ ] pybind11 extension builds and its binding tests pass, OR skip
      gracefully with `pytest.importorskip` if `cpp/build` absent
- [ ] No skipped test without a one-line reason in the test itself
- [ ] Every example script (`python/examples/*.py`) executes end-to-end
      with exit code 0
- [ ] Any new numerical model validated against a closed-form or
      literature ground truth before being trusted (precedent:
      `BatteryModel2Cell` vs Kim 2007, `ParticleFilter` vs exact Kalman
      filter)
- [ ] Property-based tests (Hypothesis) exist for mathematically
      load-bearing invariants: distribution `log_pdf` correctness,
      particle weight normalisation, percentile ordering (P05<=P50<=P95)
- [ ] Docstring examples that claim to be runnable are verified via
      doctest, not just eyeballed

## 2. Code hygiene

- [ ] ruff clean on the FULL `python/` tree
- [ ] No `TODO`/`FIXME`/`HACK`/`XXX` without a tracking note (a line
      referencing an issue, a monthly plan section, or "deferred to
      Month N" is acceptable; a bare marker is not)
- [ ] No bare `except:` clauses
- [ ] No overly broad `except Exception:` that swallows errors silently
      (catching + re-raising, or catching + logging, is fine; catching +
      `pass` is not)
- [ ] Every importable directory under `python/` has `__init__.py`
      (Week 6/7 audit found `python/examples/` missing this)
- [ ] No circular imports
- [ ] No dead code / unused imports (ruff F401 covers this)
- [ ] Test coverage floor: `pytest-cov` reports >= 85% line coverage on
      `python/src/` and `python/pdsl/` (the kernel; example scripts are
      exempt from the floor since they are demonstrations, not kernel)

## 3. Packaging & installability

- [ ] Fresh `git clone` + fresh venv + `pip install -e ".[dev]"`
      succeeds without manual intervention
- [ ] Core kernel + example imports work from a directory OUTSIDE the
      repo (simulates what a uvicorn/gunicorn worker process sees --
      this is the actual condition Week 7's FastAPI service runs under)
- [ ] Every package imported anywhere in the codebase is declared in
      `pyproject.toml` (precedent: Week 4's `lark` CI break, Week 6's
      `pybind11` addition done correctly the second time)
- [ ] `pip-audit` reports no known CVEs in pinned dependency versions

## 4. Build system

- [ ] C++ CMake configure + build succeeds from a clean `build/`
      directory (not just an incrementally-updated one)
- [ ] `cpp/build/` and all other build-artifact directories are
      correctly gitignored, VERIFIED with `git check-ignore -v` against
      an actual built artifact -- not just visual inspection of
      `.gitignore` (a grep miss on a differently-formatted pattern
      caused a false alarm once; the explicit check is the real proof)
- [ ] pybind11/C++ extension builds and is importable from Python

## 5. Reproducibility

- [ ] Same seed -> bit-identical output, verified explicitly with an
      automated test for every stochastic component (`MonteCarloEngine`,
      `ParticleFilter`, `SobolSensitivity`, `probos_cpp.MonteCarloEngineOMP`)
- [ ] Any determinism claim in a docstring is backed by a test, not
      assumed

## 6. Numerical / scientific validity

- [ ] Any new model validated against ground truth (see Correctness
      section) before being trusted for further work
- [ ] Cross-validation between parallel implementations (Python vs
      C++) run and any known discrepancy explicitly documented AND
      tested-for (precedent: `test_cpp_bindings.py`'s
      `test_spread_difference_is_documented_not_hidden`, which fails
      loudly if the known Python/C++ prior-sampling gap silently closes
      or widens without anyone noticing)
- [ ] No benchmark or speedup claim in a docstring/comment without an
      actual measured number backing it (precedent: Week 4 Monday's
      honest finding that `inv_RT` precompute gave ~1x, not a claimed
      speedup)

## 7. Performance

- [ ] Benchmark results for core hot paths are stored (not just
      printed and discarded) so future runs can diff against history
- [ ] No hot path has regressed more than 2x since the last recorded
      benchmark without an explicit, documented reason

## 8. Security

- [ ] No hardcoded secrets/credentials anywhere in the repo (git
      history included, not just current tree)
- [ ] `bandit` static security scan run on `python/src/` and
      `python/pdsl/`, with no HIGH-severity findings unresolved
- [ ] `python/pdsl/compiler.py`'s `exec()` call operates ONLY on
      PDSL-generated Python source (from `codegen.generate()`), never
      on arbitrary user input passed through unmodified -- verify this
      boundary explicitly whenever `compiler.py` changes
- [ ] `pip-audit` dependency CVE scan (also listed under Packaging,
      repeated here because it is a security check first)
- [x] No endpoint accepts unbounded N/n_steps without a sane upper
      limit (resource exhaustion) -- enforced via Pydantic
      Field(ge=..., le=...) constraints in python/server/schemas.py,
      verified by test_server.py::test_simulate_rejects_oversized_N
- [x] Input validated via Pydantic BEFORE reaching kernel code -- all
      four python/server/schemas.py request models enforce bounds at
      the HTTP boundary
- [x] Kernel-raised ValueError (e.g. SobolSensitivity's non-power-of-2
      N_saltelli check) surfaces as HTTP 422, not an unhandled 500 --
      verified by test_server.py::test_sensitivity_rejects_non_power_of_2_N_saltelli
- [x] Unknown model_name surfaces as HTTP 404, not a 500 -- verified
      by test_server.py::test_simulate_rejects_unknown_model

## 9. CI parity

- [ ] Everything `pre_week_audit.sh` checks locally is also checked in
      `.github/workflows/ci.yml` -- no drift between "passes on my
      machine" and "passes in CI"
- [ ] CI's mypy/ruff file lists cover the FULL tree, re-verified
      whenever a new top-level file or directory is added under `python/`

## 10. Documentation

- [ ] `docs/study/study_guide.md` has an entry for any new file that
      needed external reading to understand or write
- [ ] The current month's plan PDF (`docs/monthly_plans/monthN/main.tex`)
      still accurately describes what is about to be built; if scope
      changed, the PDF is updated and recompiled before work continues
- [ ] `README.md` reflects current repo state (no stale install
      instructions, stale file paths, or stale capability claims)

## 11. Git hygiene

- [ ] No tracked file exceeds 5MB (catches accidentally-committed
      binaries/artifacts early)
- [ ] No secrets appear anywhere in git history (`git log -p` spot
      check, or a dedicated secret-scanning tool once the repo is large
      enough to justify one)
- [ ] Commit messages describe WHY a change was made, not just WHAT
      changed (established practice throughout this project's history;
      keep it)
- [ ] `git status --porcelain` reviewed before every commit -- no
      surprise untracked files silently left out of a commit that
      should have included them

## 12. License / dependency compliance (MANUAL, periodic)

- [ ] All dependencies in `pyproject.toml` use permissive licenses
      (MIT/BSD/Apache-2.0) compatible with eventual SBIR/enterprise
      pilot use (see docs/monthly_plans/overall/main.tex for the
      longer-horizon commercial context) -- re-check whenever a new
      dependency is added, full pass once per month

---

## Deferred (tracked, not yet applicable)

These are real and will be added to `pre_week_audit.sh` the week they
become relevant, not before:

- Container/Docker image security scanning -- once ProbOS is
  containerised
- API load/stress testing, rate limiting -- FastAPI service now
  exists (Week 7) with basic input-bound validation; load/stress
  testing and rate limiting itself remain deferred until there is
  real traffic to test against
- Authentication/authorization checks -- once any endpoint requires auth
- Cyclomatic complexity gating -- tracked informationally if useful,
  not blocking; the checks above already catch most real bugs more
  directly than a complexity score would

---

## How this document is used

`pre_week_audit.sh` automates every item marked with an automatable
check above. Run it before starting any new week:

```bash
bash pre_week_audit.sh
```

Exit code 0 means clear to proceed. Non-zero means fix what is reported
before starting new feature work. MANUAL items (license compliance) are
reviewed by a human on the cadence noted next to each.

When a real bug or gap is found (in review, in production, or by
accident) that none of the above would have caught, add a new check
here AND to `pre_week_audit.sh` in the same commit that fixes the bug --
this is how the checklist grows to reflect actual failure modes rather
than hypothetical ones.
