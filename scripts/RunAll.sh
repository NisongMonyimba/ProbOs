#!/usr/bin/env bash
# =============================================================================
# RunAll.sh -- Full ProbOS build and test pipeline
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
VENV="$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT"

GREEN="\033[0;32m"
RED="\033[0;31m"
RESET="\033[0m"
BOLD="\033[1m"

step=0
pass_count=0

run_step() {
    step=$((step + 1))
    local label="$1"
    local cmd="$2"
    printf "Step %d: %-40s" "$step" "$label"
    if (cd "$PROJECT_ROOT" && eval "$cmd") > /tmp/probos_step_${step}.log 2>&1; then
        echo -e "${GREEN}PASS${RESET}"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}FAIL${RESET}"
        echo "--- Output ---"
        cat /tmp/probos_step_${step}.log
        echo "--- End ---"
        exit 1
    fi
}

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  ProbOS Full Pipeline${RESET}"
echo -e "${BOLD}  Project: $PROJECT_ROOT${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""

# Step 1: Check requirements
run_step "Check system requirements" \
    "python3 --version && g++ --version && cmake --version && ninja --version"

# Step 2: Python virtual environment
run_step "Python virtual environment" \
    "[ -d '$PROJECT_ROOT/.venv' ] || python3 -m venv '$PROJECT_ROOT/.venv'"

# Step 3: Install dependencies
run_step "Install Python dependencies" \
    "source '$VENV' && pip install -q -r '$PROJECT_ROOT/requirements.txt'"

# Step 4: Python tests
run_step "Python tests (111 tests)" \
    "source '$VENV' && python -m pytest '$PROJECT_ROOT/python/tests/' -q --tb=short"

# Step 5: Build C++
run_step "Build C++ code (cmake + ninja)" \
    "mkdir -p '$BUILD_DIR' && cmake -S '$PROJECT_ROOT' -B '$BUILD_DIR' -G Ninja -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null && cmake --build '$BUILD_DIR'"

# Step 6: C++ tests
run_step "C++ tests (13 tests)" \
    "cd '$BUILD_DIR' && ctest --output-on-failure -q"

# Step 7: Week 1 examples
run_step "Week 1 examples" \
    "source '$VENV' && python '$PROJECT_ROOT/python/examples/week1_coin_flip.py' && python '$PROJECT_ROOT/python/examples/week1_normal_demo.py'"

# Step 8: Week 2 examples
run_step "Week 2 examples" \
    "source '$VENV' && python '$PROJECT_ROOT/python/examples/week2_battery_ode.py' && python '$PROJECT_ROOT/python/examples/week2_clt_demo.py'"

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "  ${GREEN}${BOLD}ALL $pass_count STEPS PASSED${RESET}"
echo ""
echo "  Week 1: 44 Python + 13 C++ tests"
echo "  Week 2: 67 Python tests (27 state + 40 battery)"
echo "  Total : 111 Python + 13 C++ = 124 tests"
echo -e "${BOLD}============================================================${RESET}"
echo ""
