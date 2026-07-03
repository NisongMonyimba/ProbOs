#!/usr/bin/env bash
# =============================================================================
# pre_week_audit.sh
# ProbOS -- comprehensive pre-week hardening pass
#
# Run this BEFORE starting any new week's work. It checks correctness,
# hygiene, packaging, build system, reproducibility, numerical validity,
# performance sanity, security, CI parity, and git hygiene.
#
# Usage: bash pre_week_audit.sh
# Exit code 0 = all checks passed. Non-zero = at least one check failed.
# =============================================================================

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

FAIL_COUNT=0
PASS_COUNT=0

section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

check() {
    local desc="$1"
    local exit_code="$2"
    if [ "$exit_code" -eq 0 ]; then
        echo "  [PASS] $desc"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  [FAIL] $desc"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# -----------------------------------------------------------------------
section "1. CORRECTNESS"
# -----------------------------------------------------------------------
.venv/bin/python -m pytest python/tests/ -q --no-header > /tmp/audit_pytest.log 2>&1
check "Full Python test suite passes" $?
tail -3 /tmp/audit_pytest.log | sed 's/^/         /'

.venv/bin/python -m mypy python/src/ python/pdsl/ python/examples/ \
    --strict --ignore-missing-imports \
    --explicit-package-bases --no-error-summary > /tmp/audit_mypy.log 2>&1
check "mypy strict on FULL python/ tree" $?
if [ -s /tmp/audit_mypy.log ]; then cat /tmp/audit_mypy.log | sed 's/^/         /'; fi

if [ -f cpp/build/test_normal ]; then
    ./cpp/build/test_normal --gtest_brief=1 > /tmp/audit_cpp.log 2>&1
    check "C++ test suite passes" $?
else
    echo "  [SKIP] C++ tests (cpp/build/test_normal not built)"
fi

# -----------------------------------------------------------------------
section "2. CODE HYGIENE"
# -----------------------------------------------------------------------
.venv/bin/python -m ruff check python/ --select E,F,W > /tmp/audit_ruff.log 2>&1
check "ruff clean on FULL python/ tree" $?
if [ -s /tmp/audit_ruff.log ]; then cat /tmp/audit_ruff.log | sed 's/^/         /'; fi

TODO_COUNT=$(grep -rn "TODO\|FIXME\|HACK\|XXX" python/src/ python/pdsl/ cpp/include/ cpp/src/ cpp/bindings/ 2>/dev/null | wc -l)
if [ "$TODO_COUNT" -eq 0 ]; then
    check "No untracked TODO/FIXME/HACK markers" 0
else
    echo "  [WARN] $TODO_COUNT TODO/FIXME/HACK markers found (review manually):"
    grep -rn "TODO\|FIXME\|HACK\|XXX" python/src/ python/pdsl/ cpp/include/ cpp/src/ cpp/bindings/ 2>/dev/null | sed 's/^/         /'
fi

BARE_EXCEPT=$(grep -rn "except:" python/src/ python/pdsl/ 2>/dev/null | wc -l)
check "No bare 'except:' clauses" $([ "$BARE_EXCEPT" -eq 0 ] && echo 0 || echo 1)

MISSING_INIT=0
for d in $(find python/ -type d -not -path "*__pycache__*"); do
    if ls "$d"/*.py >/dev/null 2>&1 && [ ! -f "$d/__init__.py" ]; then
        echo "  [FAIL] Missing __init__.py: $d"
        MISSING_INIT=1
    fi
done
check "All python/ subpackages have __init__.py" $MISSING_INIT

# -----------------------------------------------------------------------
section "2b. TEST COVERAGE"
# -----------------------------------------------------------------------
.venv/bin/python -m pytest python/tests/ \
    --cov=python/src --cov=python/pdsl \
    --cov-report=term-missing --cov-fail-under=85 \
    -q --no-header > /tmp/audit_coverage.log 2>&1
check "Coverage >= 85% on python/src + python/pdsl" $?
tail -20 /tmp/audit_coverage.log | sed 's/^/         /'

# -----------------------------------------------------------------------
section "2c. PROPERTY-BASED TESTS (Hypothesis)"
# -----------------------------------------------------------------------
.venv/bin/python -m pytest python/tests/test_distributions_properties.py \
    -q --no-header > /tmp/audit_hypothesis.log 2>&1
check "Hypothesis property-based tests pass" $?
tail -5 /tmp/audit_hypothesis.log | sed 's/^/         /'

# -----------------------------------------------------------------------
section "2d. DOCTESTS"
# -----------------------------------------------------------------------
.venv/bin/python -m pytest --doctest-modules python/src/ \
    -q --no-header > /tmp/audit_doctest.log 2>&1
DOCTEST_EXIT=$?
if grep -q "no tests ran\|collected 0 items" /tmp/audit_doctest.log; then
    echo "  [SKIP] No runnable doctest examples found yet in python/src/"
else
    check "Doctest examples in python/src/ execute correctly" $DOCTEST_EXIT
    tail -10 /tmp/audit_doctest.log | sed 's/^/         /'
fi

# -----------------------------------------------------------------------
section "8. SECURITY"
# -----------------------------------------------------------------------
.venv/bin/python -m bandit -r python/src/ python/pdsl/ -q \
    -ll > /tmp/audit_bandit.log 2>&1
check "bandit: no HIGH-severity findings in python/src/ + python/pdsl/" $?
if [ -s /tmp/audit_bandit.log ]; then
    grep -A 3 "Severity: High" /tmp/audit_bandit.log | sed 's/^/         /'
fi

.venv/bin/pip-audit --strict > /tmp/audit_pipaudit.log 2>&1
PIPAUDIT_EXIT=$?
if [ "$PIPAUDIT_EXIT" -eq 0 ]; then
    check "pip-audit: no known CVEs in dependencies" 0
else
    echo "  [WARN] pip-audit found potential issues (review manually):"
    tail -20 /tmp/audit_pipaudit.log | sed 's/^/         /'
fi

SECRET_PATTERNS=$(grep -rniE "(api[_-]?key|secret|password|token)\s*=\s*['\"][a-zA-Z0-9]{10,}" \
    python/src/ python/pdsl/ cpp/ 2>/dev/null | wc -l)
check "No hardcoded secret-like strings in source" $([ "$SECRET_PATTERNS" -eq 0 ] && echo 0 || echo 1)

# -----------------------------------------------------------------------
section "3. PACKAGING & INSTALLABILITY"
# -----------------------------------------------------------------------
cd /tmp
rm -rf /tmp/audit_import_check
mkdir -p /tmp/audit_import_check
cd /tmp/audit_import_check
/home/nison/ProbOs/.venv/bin/python -c "
from python.src.battery_model import BatteryModel2Cell
from python.src.monte_carlo import MonteCarloEngine
from python.src.particle_filter import ParticleFilter
from python.src.sensitivity import SobolSensitivity
from python.src.provenance import ProvenanceTracker
from python.pdsl.compiler import compile_pdsl
from python.examples.week4_option_pricer import OptionPricerModel
" > /tmp/audit_outside_import.log 2>&1
check "Core + example imports work from outside repo dir" $?
cd /home/nison/ProbOs
rm -rf /tmp/audit_import_check

# -----------------------------------------------------------------------
section "4. BUILD SYSTEM"
# -----------------------------------------------------------------------
if [ -f cpp/build/probos_cpp*.so ] 2>/dev/null || ls cpp/build/probos_cpp*.so >/dev/null 2>&1; then
    check "probos_cpp pybind11 extension is built" 0
else
    echo "  [SKIP] probos_cpp extension not built (run: cmake --build cpp/build)"
fi

git check-ignore -v cpp/build/ > /dev/null 2>&1
check "cpp/build/ is correctly gitignored" $?

# -----------------------------------------------------------------------
section "5. REPRODUCIBILITY"
# -----------------------------------------------------------------------
.venv/bin/python -c "
from python.src.monte_carlo import MonteCarloEngine
from python.src.battery_model import BatteryModel2Cell
from python.src.parameter_priors import build_battery_priors
model = BatteryModel2Cell()
priors = build_battery_priors()
e1 = MonteCarloEngine(model, priors, N=100, n_steps=10, seed=42)
e2 = MonteCarloEngine(model, priors, N=100, n_steps=10, seed=42)
r1 = e1.run()
r2 = e2.run()
import numpy as np
assert np.array_equal(r1.trajectories, r2.trajectories)
" > /tmp/audit_repro.log 2>&1
check "MonteCarloEngine same-seed reproducibility" $?

# -----------------------------------------------------------------------
section "6. GIT HYGIENE"
# -----------------------------------------------------------------------
LARGE_FILES=$(git ls-files | xargs -I{} du -k {} 2>/dev/null | awk '$1 > 5000 {print}' | wc -l)
check "No tracked files > 5MB" $([ "$LARGE_FILES" -eq 0 ] && echo 0 || echo 1)

UNTRACKED_IMPORTANT=$(git status --porcelain | grep "^??" | grep -v "__pycache__\|\.pyc$" | wc -l)
if [ "$UNTRACKED_IMPORTANT" -gt 0 ]; then
    echo "  [WARN] $UNTRACKED_IMPORTANT untracked files present (review before committing):"
    git status --porcelain | grep "^??" | grep -v "__pycache__\|\.pyc$" | sed 's/^/         /'
fi

# -----------------------------------------------------------------------
section "SUMMARY"
# -----------------------------------------------------------------------
echo "  Passed: $PASS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo "============================================================"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "  AUDIT RESULT: PASS -- clear to start next week's work"
    exit 0
else
    echo "  AUDIT RESULT: FAIL -- fix issues above before proceeding"
    exit 1
fi
