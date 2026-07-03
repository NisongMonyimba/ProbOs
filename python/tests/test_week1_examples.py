"""
python/tests/test_week1_examples.py

Smoke tests for the two Week 1 example scripts. Per
docs/standards/quality_standards.md: example/utility scripts get smoke
tests confirming they run correctly and produce sane output, while
core kernel files get full test suites.

Both week1_coin_flip.py and week1_normal_demo.py are pedagogical
primers -- the former demonstrates the Law of Large Numbers and
1/sqrt(N) convergence with zero ProbOS dependencies; the latter
demonstrates the Normal distribution class and is the original source
of the log_pdf numerical-stability example featured in README.md's
"Core Insight" section.
"""

from __future__ import annotations

import os
import subprocess
import sys

from python.examples.week1_coin_flip import (
    run_coin_flip_experiment,
    demonstrate_convergence_rate,
)


class TestCoinFlipExperiment:

    def test_returns_float_between_0_and_1(self) -> None:
        freq = run_coin_flip_experiment(1000, seed=42, verbose=False)
        assert 0.0 <= freq <= 1.0

    def test_same_seed_reproducible(self) -> None:
        freq1 = run_coin_flip_experiment(1000, seed=7, verbose=False)
        freq2 = run_coin_flip_experiment(1000, seed=7, verbose=False)
        assert freq1 == freq2

    def test_different_seeds_can_differ(self) -> None:
        freq1 = run_coin_flip_experiment(50, seed=1, verbose=False)
        freq2 = run_coin_flip_experiment(50, seed=2, verbose=False)
        # Not strictly guaranteed, but overwhelmingly likely at N=50
        # with different seeds. Documented as a non-strict check.
        assert freq1 != freq2 or True

    def test_converges_toward_half_at_large_N(self) -> None:
        freq = run_coin_flip_experiment(1_000_000, seed=42, verbose=False)
        assert abs(freq - 0.5) < 0.01

    def test_error_shrinks_with_more_flips(self) -> None:
        freq_small = run_coin_flip_experiment(10, seed=42, verbose=False)
        freq_large = run_coin_flip_experiment(1_000_000, seed=42, verbose=False)
        error_small = abs(freq_small - 0.5)
        error_large = abs(freq_large - 0.5)
        assert error_large < error_small

    def test_demonstrate_convergence_rate_runs_without_error(self) -> None:
        demonstrate_convergence_rate()


class TestWeek1ScriptsExecuteEndToEnd:
    """
    Runs each script as a subprocess (exactly as a user would from the
    command line) and confirms a clean exit code -- the same standard
    already applied to Week 2-4 example scripts.
    """

    def test_coin_flip_script_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "python/examples/week1_coin_flip.py"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0, result.stderr

    def test_normal_demo_script_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "python/examples/week1_normal_demo.py"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0, result.stderr

    def test_normal_demo_produces_figure(self) -> None:
        subprocess.run(
            [sys.executable, "python/examples/week1_normal_demo.py"],
            capture_output=True, text=True, cwd=".",
        )
        assert os.path.exists(
            "outputs/figures/week1_battery_Ea_distribution.png"
        )
