# =============================================================================
# RunProbOS.ps1 -- Windows PowerShell launcher for ProbOS
#
# HOW TO RUN:
#   1. Open File Explorer
#   2. Navigate to: \\wsl.localhost\Ubuntu-22.04\home\nison\ProbOsWeek1
#   3. Right-click RunProbOS.ps1
#   4. Click: Run with PowerShell
# =============================================================================

$WSL_DISTRO  = "Ubuntu-22.04"
$PROJECT_DIR = "/home/nison/ProbOsWeek1"
$RUNALL      = "$PROJECT_DIR/scripts/RunAll.sh"
$RUNTESTS    = "$PROJECT_DIR/scripts/RunTests.sh"

function Invoke-WSL {
    param([string]$ScriptPath, [string]$Label)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Running: $Label" -ForegroundColor Cyan
    Write-Host "  Script : $ScriptPath" -ForegroundColor Gray
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    wsl -d $WSL_DISTRO -- bash -c "cd '$PROJECT_DIR' && chmod +x '$ScriptPath' && bash '$ScriptPath'"
    $code = $LASTEXITCODE
    Write-Host ""
    if ($code -eq 0) {
        Write-Host "  PASSED: $Label" -ForegroundColor Green
    } else {
        Write-Host "  FAILED: $Label (exit code $code)" -ForegroundColor Red
    }
    return $code
}

function Test-WSL {
    $wslCheck = Get-Command wsl -ErrorAction SilentlyContinue
    if (-not $wslCheck) {
        Write-Host "ERROR: WSL not installed." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    $distroCheck = wsl -d $WSL_DISTRO -- echo "ok" 2>$null
    if ($distroCheck -ne "ok") {
        Write-Host "ERROR: WSL distro '$WSL_DISTRO' not found." -ForegroundColor Red
        wsl --list --quiet
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Clear-Host
Write-Host ""
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "  ProbOS -- A Probabilistic Execution Runtime" -ForegroundColor Blue
Write-Host "  Reality Computing Corporation" -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ""
Write-Host "  Project : $PROJECT_DIR" -ForegroundColor Gray
Write-Host "  Distro  : $WSL_DISTRO" -ForegroundColor Gray
Write-Host "  Week 1: Distribution ABC -- 44 Python + 13 C++ tests" -ForegroundColor Gray
Write-Host "  Week 2: Model ABC + BatteryModel2Cell -- 67 Python tests" -ForegroundColor Gray
Write-Host "  Total : 111 Python + 13 C++ = 124 tests" -ForegroundColor Gray
Write-Host ""

Test-WSL

do {
    Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  What would you like to run?" -ForegroundColor White
    Write-Host ""
    Write-Host "  [1]  RunAll.sh    -- Full pipeline (8 steps)" -ForegroundColor Yellow
    Write-Host "         Build C++, run 124 tests, run all examples" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [2]  RunTests.sh  -- Tests only (~30 seconds)" -ForegroundColor Yellow
    Write-Host "         pytest (111) + mypy + ruff + ctest (13)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [3]  Both         -- RunAll.sh then RunTests.sh" -ForegroundColor Yellow
    Write-Host "         Full verification of everything" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [4]  Exit" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray

    $choice = Read-Host "  Enter choice [1/2/3/4]"

    switch ($choice) {

        "1" {
            $code = Invoke-WSL -ScriptPath $RUNALL -Label "RunAll.sh (Full Pipeline)"
            if ($code -eq 0) {
                Write-Host "  All 8 steps passed." -ForegroundColor Green
            }
        }

        "2" {
            $code = Invoke-WSL -ScriptPath $RUNTESTS -Label "RunTests.sh (Test Suite)"
            if ($code -eq 0) {
                Write-Host "  All tests passed. 124 total." -ForegroundColor Green
            }
        }

        "3" {
            Write-Host ""
            Write-Host "  Running RunAll.sh first, then RunTests.sh..." -ForegroundColor Cyan
            Write-Host ""

            $code1 = Invoke-WSL -ScriptPath $RUNALL -Label "RunAll.sh (Full Pipeline)"

            if ($code1 -eq 0) {
                $code2 = Invoke-WSL -ScriptPath $RUNTESTS -Label "RunTests.sh (Test Suite)"
                Write-Host ""
                if ($code2 -eq 0) {
                    Write-Host "============================================================" -ForegroundColor Green
                    Write-Host "  BOTH SCRIPTS PASSED -- Full verification complete" -ForegroundColor Green
                    Write-Host "  RunAll.sh  : 8/8 steps PASS" -ForegroundColor Green
                    Write-Host "  RunTests.sh: 5/5 steps PASS" -ForegroundColor Green
                    Write-Host "  Total tests: 111 Python + 13 C++ = 124" -ForegroundColor Green
                    Write-Host "============================================================" -ForegroundColor Green
                } else {
                    Write-Host "  RunTests.sh FAILED (exit code $code2)" -ForegroundColor Red
                }
            } else {
                Write-Host "  RunAll.sh FAILED (exit code $code1) -- skipping RunTests.sh" -ForegroundColor Red
            }
        }

        "4" {
            Write-Host ""
            Write-Host "  Exiting ProbOS launcher." -ForegroundColor Gray
            Write-Host ""
        }

        default {
            Write-Host ""
            Write-Host "  Invalid choice. Please enter 1, 2, 3, or 4." -ForegroundColor Red
            Write-Host ""
        }
    }

    if ($choice -ne "4") {
        Write-Host ""
        Read-Host "  Press Enter to return to menu"
        Write-Host ""
    }

} while ($choice -ne "4")

Write-Host "  Done. You can close this window." -ForegroundColor Gray
Write-Host ""
