#!/usr/bin/env bash
# =============================================================================
# check_system_requirements.sh
#
# System requirements checker for ProbOS. Run from the repo root (or
# anywhere -- it does not depend on being inside the repo) to see what
# is present, what is missing, and what is optional (GPU work only).
#
# Usage:
#   bash check_system_requirements.sh
# =============================================================================

set -uo pipefail

PASS=0
FAIL=0
WARN=0

pass() { echo "  [OK]   $1"; PASS=$((PASS+1)); }
fail() { echo "  [MISSING] $1"; FAIL=$((FAIL+1)); }
warn() { echo "  [OPTIONAL, not found] $1"; WARN=$((WARN+1)); }

section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

section "REQUIRED -- Core CPU work (kernel, tests, docs)"

if command -v python3 >/dev/null 2>&1; then
    PYVER=$(python3 --version 2>&1)
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        pass "Python 3.11+ ($PYVER)"
    else
        fail "Python 3.11+ (found: $PYVER -- too old)"
    fi
else
    fail "Python 3.11+ (not found on PATH)"
fi

if command -v git >/dev/null 2>&1; then
    pass "git ($(git --version))"
else
    fail "git (not found)"
fi

if command -v g++ >/dev/null 2>&1; then
    pass "g++ ($(g++ --version | head -1))"
else
    fail "g++ (not found -- needed for the C++/pybind11 kernel)"
fi

if command -v cmake >/dev/null 2>&1; then
    CMAKE_VER=$(cmake --version | head -1)
    pass "cmake ($CMAKE_VER)"
else
    fail "cmake 3.22+ (not found)"
fi

if command -v ninja >/dev/null 2>&1; then
    pass "ninja ($(ninja --version))"
else
    fail "ninja (not found)"
fi

if [ -f /usr/include/gtest/gtest.h ] || ldconfig -p 2>/dev/null | grep -q libgtest; then
    pass "Google Test (libgtest)"
else
    fail "Google Test (not found -- install libgtest-dev libgmock-dev)"
fi

section "REQUIRED -- LaTeX (for compiling docs/monthly_plans, manuscript)"

if command -v pdflatex >/dev/null 2>&1; then
    pass "pdflatex ($(pdflatex --version | head -1))"
else
    fail "pdflatex (not found -- install texlive-full or similar)"
fi

section "REQUIRED -- Disk space and memory (rough guidance)"

if command -v df >/dev/null 2>&1; then
    AVAIL_GB=$(df -BG . 2>/dev/null | tail -1 | awk '"'"'{print $4}'"'"' | tr -d 'G')
    if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -ge 10 ] 2>/dev/null; then
        pass "Disk space (${AVAIL_GB}G available, 10G+ recommended)"
    else
        warn "Disk space (${AVAIL_GB:-unknown}G available -- 10G+ recommended, more if installing GPU deps)"
    fi
fi

if command -v free >/dev/null 2>&1; then
    MEM_GB=$(free -g | awk '"'"'/^Mem:/{print $2}'"'"')
    if [ "$MEM_GB" -ge 4 ] 2>/dev/null; then
        pass "RAM (${MEM_GB}G total, 4G+ recommended)"
    else
        warn "RAM (${MEM_GB:-unknown}G total -- 4G+ recommended)"
    fi
fi

section "OPTIONAL -- GPU work only (Month 3 Week 10 CuPy path)"
echo "  Everything below is ONLY needed if you plan to run"
echo "  GPUMonteCarloEngine (python/src/gpu_monte_carlo.py)."
echo "  ALL other ProbOS functionality works fully without a GPU."
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" | head -1)
    if [ -n "$GPU_NAME" ]; then
        pass "NVIDIA GPU detected: $GPU_NAME ($GPU_MEM)"
        if [ -n "$CUDA_VER" ]; then
            pass "CUDA driver version: $CUDA_VER"
        fi
    else
        warn "nvidia-smi present but no GPU detected"
    fi
else
    warn "NVIDIA GPU / nvidia-smi (not found -- GPU features will be unavailable, everything else still works)"
fi

if python3 -c "import cupy" 2>/dev/null; then
    pass "CuPy installed"
else
    warn "CuPy (not installed -- run: pip install \"cupy-cuda12x[ctk]\" if you have an NVIDIA GPU)"
fi

section "SUMMARY"
echo "  Required checks passed: $PASS"
echo "  Required checks FAILED: $FAIL"
echo "  Optional (GPU) items not present: $WARN"
echo "============================================================"

if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: This machine can run ALL of ProbOS's core functionality."
    if [ "$WARN" -gt 0 ]; then
        echo "  GPU-specific features (Week 10) are unavailable on this machine --"
        echo "  this is fine, they are optional and skip gracefully everywhere else."
    fi
    exit 0
else
    echo "  RESULT: $FAIL required item(s) missing -- see [MISSING] lines above."
    exit 1
fi
