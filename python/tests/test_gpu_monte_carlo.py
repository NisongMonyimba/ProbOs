"""
python/tests/test_gpu_monte_carlo.py

Tests for GPUMonteCarloEngine (Month 3 Week 10).

CRITICAL: these tests require an NVIDIA GPU with CuPy installed.
GitHub Actions CI runners have NO GPU access -- these tests skip
gracefully there via pytest.mark.skipif, with the skip reason VISIBLE
in test output (not hidden), per Week 10's own plan document. Only
genuine local-machine runs (confirmed: NVIDIA GeForce RTX 3050 Ti
Laptop GPU) validate this module.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from python.src.battery_model import BatteryModel2Cell
from python.src.monte_carlo import MonteCarloEngine
from python.src.parameter_priors import build_battery_priors

pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason=(
        "CuPy not installed / no GPU available -- GPU tests skip "
        "gracefully in CI (GitHub Actions has no GPU access). Only "
        "genuine local-machine runs validate GPUMonteCarloEngine."
    ),
)


class TestGPUMonteCarloEngineConstruction:

    def test_construction_succeeds_with_valid_args(self) -> None:
        from python.src.gpu_monte_carlo import GPUMonteCarloEngine
        model = BatteryModel2Cell()
        priors = build_battery_priors()
        engine = GPUMonteCarloEngine(model, priors, N=100, n_steps=5, seed=1)
        assert engine.N == 100
        assert engine.n_steps == 5

    def test_mismatched_priors_length_raises(self) -> None:
        from python.src.gpu_monte_carlo import GPUMonteCarloEngine
        model = BatteryModel2Cell()
        with pytest.raises(ValueError):
            GPUMonteCarloEngine(model, priors=[], N=100, n_steps=5)

    def test_invalid_N_raises(self) -> None:
        from python.src.gpu_monte_carlo import GPUMonteCarloEngine
        model = BatteryModel2Cell()
        priors = build_battery_priors()
        with pytest.raises(ValueError):
            GPUMonteCarloEngine(model, priors, N=0, n_steps=5)


class TestGPUMonteCarloEngineRun:

    def test_run_returns_correct_shapes(self) -> None:
        from python.src.gpu_monte_carlo import GPUMonteCarloEngine
        model = BatteryModel2Cell()
        priors = build_battery_priors()
        engine = GPUMonteCarloEngine(model, priors, N=200, n_steps=10, seed=1)
        result = engine.run()

        assert result.trajectories.shape == (200, 11, 8)
        assert result.percentiles.shape == (3, 11, 8)
        assert result.convergence.shape == (8,)
        assert result.n_particles == 200
        assert result.n_steps == 10

    def test_run_returns_plain_numpy_arrays(self) -> None:
        """
        Callers should never need to import CuPy themselves --
        GPUMCResult's array fields must all be plain NumPy arrays,
        copied back from the GPU inside run().
        """
        from python.src.gpu_monte_carlo import GPUMonteCarloEngine
        model = BatteryModel2Cell()
        priors = build_battery_priors()
        engine = GPUMonteCarloEngine(model, priors, N=100, n_steps=5, seed=1)
        result = engine.run()

        assert isinstance(result.trajectories, np.ndarray)
        assert isinstance(result.percentiles, np.ndarray)
        assert isinstance(result.convergence, np.ndarray)
        assert not isinstance(result.trajectories, cp.ndarray)

    def test_wall_time_ms_is_genuinely_measured(self) -> None:
        from python.src.gpu_monte_carlo import GPUMonteCarloEngine
        model = BatteryModel2Cell()
        priors = build_battery_priors()
        engine = GPUMonteCarloEngine(model, priors, N=100, n_steps=5, seed=1)
        result = engine.run()
        assert result.wall_time_ms > 0.0

    def test_matches_cpu_engine_exactly(self) -> None:
        """
        THE key correctness validation: GPUMonteCarloEngine's
        trajectories must match MonteCarloEngine's (CPU) trajectories
        exactly for BatteryModel2Cell, since both draw parameters
        identically (same seed, same Distribution.sample() calls on
        CPU) and forward_batch()'s CuPy dispatch was verified in
        Week 10 Day 2 to produce bit-exact results vs the NumPy path.
        """
        model_cpu = BatteryModel2Cell()
        priors_cpu = build_battery_priors()
        cpu_engine = MonteCarloEngine(
            model_cpu, priors_cpu, N=200, n_steps=15, dt=1.0, seed=42,
        )
        cpu_result = cpu_engine.run()

        from python.src.gpu_monte_carlo import GPUMonteCarloEngine
        model_gpu = BatteryModel2Cell()
        priors_gpu = build_battery_priors()
        gpu_engine = GPUMonteCarloEngine(
            model_gpu, priors_gpu, N=200, n_steps=15, dt=1.0, seed=42,
        )
        gpu_result = gpu_engine.run()

        np.testing.assert_allclose(
            cpu_result.trajectories, gpu_result.trajectories, rtol=1e-8,
        )
        np.testing.assert_allclose(
            cpu_result.percentiles, gpu_result.percentiles, rtol=1e-8,
        )
