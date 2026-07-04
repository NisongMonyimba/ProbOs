"""
Tests for MonteCarloEngine and MCResult.

111 existing tests pass. These ~25 new tests cover:
  - Construction validation
  - Output shapes
  - Mathematical invariants (P05 <= P50 <= P95, sigma/sqrt(N))
  - Physics (temperature rises, concentrations fall)
  - Convergence rate (1/sqrt(N))
  - Reproducibility
"""

from __future__ import annotations

import numpy as np
import pytest

from python.src.battery_model import BatteryModel2Cell
from python.src.distributions import Distribution
from python.src.monte_carlo import MCResult, MonteCarloEngine
from python.src.parameter_priors import build_battery_priors

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> BatteryModel2Cell:
    return BatteryModel2Cell()


@pytest.fixture(scope="module")
def priors() -> list[Distribution]:
    return build_battery_priors()


@pytest.fixture(scope="module")
def small_engine(
    model: BatteryModel2Cell, priors: list[Distribution]
) -> MonteCarloEngine:
    """N=200, n_steps=20 -- fast for most tests."""
    return MonteCarloEngine(model, priors, N=200, n_steps=20, dt=1.0, seed=42)


@pytest.fixture(scope="module")
def small_result(small_engine: MonteCarloEngine) -> MCResult:
    return small_engine.run()


@pytest.fixture(scope="module")
def medium_engine(
    model: BatteryModel2Cell, priors: list[Distribution]
) -> MonteCarloEngine:
    """N=1000, n_steps=60 -- for convergence tests."""
    return MonteCarloEngine(model, priors, N=1000, n_steps=60, dt=1.0, seed=42)


@pytest.fixture(scope="module")
def medium_result(medium_engine: MonteCarloEngine) -> MCResult:
    return medium_engine.run()


# ---------------------------------------------------------------------------
# TestMCEngineConstruction
# ---------------------------------------------------------------------------

class TestMCEngineConstruction:

    def test_valid_construction(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        engine = MonteCarloEngine(model, priors, N=10, n_steps=5, seed=0)
        assert engine is not None

    def test_default_N_is_5000(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        engine = MonteCarloEngine(model, priors)
        assert engine.N == 5000

    def test_default_n_steps_is_300(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        engine = MonteCarloEngine(model, priors)
        assert engine.n_steps == 300

    def test_default_dt_is_1(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        engine = MonteCarloEngine(model, priors)
        assert engine.dt == 1.0

    def test_seed_stored(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        engine = MonteCarloEngine(model, priors, seed=99)
        assert engine.seed == 99

    def test_wrong_prior_count_raises(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        with pytest.raises(ValueError, match="param_dim"):
            MonteCarloEngine(model, priors[:-1])  # one short

    def test_N_zero_raises(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        with pytest.raises(ValueError, match="N must be"):
            MonteCarloEngine(model, priors, N=0)

    def test_N_negative_raises(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        with pytest.raises(ValueError, match="N must be"):
            MonteCarloEngine(model, priors, N=-1)

    def test_dt_zero_raises(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        with pytest.raises(ValueError, match="dt must be"):
            MonteCarloEngine(model, priors, dt=0.0)

    def test_dt_negative_raises(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        with pytest.raises(ValueError, match="dt must be"):
            MonteCarloEngine(model, priors, dt=-1.0)

    def test_n_steps_zero_raises(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        with pytest.raises(ValueError, match="n_steps must be"):
            MonteCarloEngine(model, priors, n_steps=0)


# ---------------------------------------------------------------------------
# TestMCResultShapes
# ---------------------------------------------------------------------------

class TestMCResultShapes:

    def test_trajectories_shape(
        self, small_result: MCResult, model: BatteryModel2Cell
    ) -> None:
        r = small_result
        assert r.trajectories.shape == (200, 21, model.state_dim)

    def test_params_used_shape(
        self, small_result: MCResult, model: BatteryModel2Cell
    ) -> None:
        r = small_result
        assert r.params_used.shape == (200, model.param_dim)

    def test_percentiles_shape(
        self, small_result: MCResult, model: BatteryModel2Cell
    ) -> None:
        r = small_result
        assert r.percentiles.shape == (3, 21, model.state_dim)

    def test_convergence_shape(
        self, small_result: MCResult, model: BatteryModel2Cell
    ) -> None:
        assert small_result.convergence.shape == (model.state_dim,)

    def test_trajectories_dtype(self, small_result: MCResult) -> None:
        assert small_result.trajectories.dtype == np.float64

    def test_params_used_dtype(self, small_result: MCResult) -> None:
        assert small_result.params_used.dtype == np.float64

    def test_n_particles_stored(self, small_result: MCResult) -> None:
        assert small_result.n_particles == 200

    def test_n_steps_stored(self, small_result: MCResult) -> None:
        assert small_result.n_steps == 20

    def test_dt_stored(self, small_result: MCResult) -> None:
        assert small_result.dt == 1.0


# ---------------------------------------------------------------------------
# TestMCResultInvariants
# ---------------------------------------------------------------------------

class TestMCResultInvariants:

    def test_p05_le_p50_everywhere(self, small_result: MCResult) -> None:
        """P05 <= P50 at every timestep and every state variable."""
        assert np.all(small_result.percentiles[0] <= small_result.percentiles[1])

    def test_p50_le_p95_everywhere(self, small_result: MCResult) -> None:
        """P50 <= P95 at every timestep and every state variable."""
        assert np.all(small_result.percentiles[1] <= small_result.percentiles[2])

    def test_convergence_positive(self, small_result: MCResult) -> None:
        assert np.all(small_result.convergence > 0)

    def test_no_nan_in_trajectories(self, small_result: MCResult) -> None:
        assert not np.any(np.isnan(small_result.trajectories))

    def test_no_inf_in_trajectories(self, small_result: MCResult) -> None:
        assert not np.any(np.isinf(small_result.trajectories))

    def test_no_nan_in_percentiles(self, small_result: MCResult) -> None:
        assert not np.any(np.isnan(small_result.percentiles))

    def test_no_nan_in_convergence(self, small_result: MCResult) -> None:
        assert not np.any(np.isnan(small_result.convergence))

    def test_convergence_equals_sigma_over_sqrt_N(
        self, small_result: MCResult
    ) -> None:
        """Verify convergence = std(final_state, ddof=1) / sqrt(N)."""
        final = small_result.trajectories[:, -1, :]   # (N, state_dim)
        expected = np.std(final, axis=0, ddof=1) / np.sqrt(200)
        np.testing.assert_allclose(small_result.convergence, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestMCPhysics
# ---------------------------------------------------------------------------

class TestMCPhysics:

    def test_temperature_P50_rises(self, small_result: MCResult) -> None:
        """Median temperature T1 should rise from onset conditions."""
        T1_p50_start = small_result.percentiles[1, 0, 0]
        T1_p50_end   = small_result.percentiles[1, -1, 0]
        assert T1_p50_end > T1_p50_start, (
            f"P50 T1 did not rise: start={T1_p50_start:.2f} end={T1_p50_end:.2f}"
        )

    def test_temperature_P95_rises_faster_than_P50(
        self, small_result: MCResult
    ) -> None:
        """
        P95 (hottest tail) heats faster than median.
        P05 = coolest 5% of cells -- they rise less, not more.
        The dangerous tail for thermal runaway is P95 (highest temperature).
        """
        p95_rise = (
            small_result.percentiles[2, -1, 0]
            - small_result.percentiles[2, 0, 0]
        )
        p50_rise = (
            small_result.percentiles[1, -1, 0]
            - small_result.percentiles[1, 0, 0]
        )
        assert p95_rise > p50_rise, (
            f"P95 rise={p95_rise:.4f} not > P50 rise={p50_rise:.4f}"
        )

    def test_temperature_P05_rises_less_than_P50(self, small_result: MCResult) -> None:
        """
        P05 (coolest tail) heats slower than median.
        Confirms the fan spreads correctly: P05 < P50 < P95 temperature rise.
        """
        p05_rise = (
            small_result.percentiles[0, -1, 0]
            - small_result.percentiles[0, 0, 0]
        )
        p50_rise = (
            small_result.percentiles[1, -1, 0]
            - small_result.percentiles[1, 0, 0]
        )
        assert p05_rise < p50_rise, (
            f"P05 rise={p05_rise:.4f} not < P50 rise={p50_rise:.4f}"
        )

    def test_concentrations_P50_decrease(self, small_result: MCResult) -> None:
        """SEI concentration c_SEI_1 (index 2) should decrease over time."""
        c_sei_start = small_result.percentiles[1, 0, 2]
        c_sei_end   = small_result.percentiles[1, -1, 2]
        assert c_sei_end < c_sei_start, (
            f"c_SEI_1 did not decrease: start={c_sei_start:.4f} end={c_sei_end:.4f}"
        )

    def test_temperatures_above_absolute_zero(self, small_result: MCResult) -> None:
        """All temperatures must remain physically valid."""
        temps = small_result.trajectories[:, :, 0]   # T1 for all particles, steps
        assert np.all(temps > 0.0)


# ---------------------------------------------------------------------------
# TestMCConvergence
# ---------------------------------------------------------------------------

class TestMCConvergence:

    def test_larger_N_gives_smaller_convergence(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        """Doubling N should roughly halve sigma/sqrt(N)."""
        r_small = MonteCarloEngine(
            model, priors, N=200, n_steps=5, seed=0
        ).run()
        r_large = MonteCarloEngine(
            model, priors, N=800, n_steps=5, seed=0
        ).run()
        # convergence[0] = sigma/sqrt(N) for T1
        assert r_large.convergence[0] < r_small.convergence[0]

    def test_1_over_sqrt_N_rate(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        """
        Ratio of convergence values should be close to sqrt(N2/N1).
        N=200 vs N=800: expected ratio ~ sqrt(800/200) = 2.0
        Allow generous tolerance because we have only 1 sample.
        """
        r200 = MonteCarloEngine(
            model, priors, N=200, n_steps=5, seed=7
        ).run()
        r800 = MonteCarloEngine(
            model, priors, N=800, n_steps=5, seed=7
        ).run()
        ratio = r200.convergence[0] / r800.convergence[0]
        assert 1.2 < ratio < 3.5, (
            f"1/sqrt(N) rate check failed: ratio={ratio:.3f}, expected ~2.0"
        )


# ---------------------------------------------------------------------------
# TestMCReproducibility
# ---------------------------------------------------------------------------

class TestMCReproducibility:

    def test_same_seed_same_trajectories(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        r1 = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=123).run()
        r2 = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=123).run()
        np.testing.assert_array_equal(r1.trajectories, r2.trajectories)

    def test_same_seed_same_params(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        r1 = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=123).run()
        r2 = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=123).run()
        np.testing.assert_array_equal(r1.params_used, r2.params_used)

    def test_different_seed_different_params(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        r1 = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=1).run()
        r2 = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=2).run()
        assert not np.array_equal(r1.params_used, r2.params_used)


# ---------------------------------------------------------------------------
# TestMCCertificate
# ---------------------------------------------------------------------------

class TestMCCertificate:

    def test_certificate_is_string(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        engine = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=0)
        cert = engine.convergence_certificate()
        assert isinstance(cert, str)

    def test_certificate_contains_header(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        engine = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=0)
        cert = engine.convergence_certificate()
        assert "Convergence Certificate" in cert

    def test_certificate_contains_N(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        engine = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=0)
        cert = engine.convergence_certificate()
        assert "N=50" in cert

    def test_certificate_contains_state_rows(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        engine = MonteCarloEngine(model, priors, N=50, n_steps=5, seed=0)
        cert = engine.convergence_certificate()
        for k in range(model.state_dim):
            assert f"state_{k}" in cert
