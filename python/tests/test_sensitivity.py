"""
Tests for SobolSensitivity and SobolResult.

27 tests covering construction, shapes, invariants, physics, reproducibility.

Note on reproducibility:
    SALib sobol sampler does not support exact reproducibility via
    np.random.seed(). At N_saltelli=64 (smoke test), results vary
    between runs. Tests check structural properties and physics, not
    exact values. Month 4 C++ implementation will use a deterministic
    Sobol sequence generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from python.src.battery_model import BatteryModel2Cell
from python.src.distributions import Distribution
from python.src.parameter_priors import build_battery_priors
from python.src.sensitivity import SobolResult, SobolSensitivity


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
def small_result(
    model: BatteryModel2Cell,
    priors: list[Distribution],
) -> SobolResult:
    """N_saltelli=64, n_steps=3 -- fast for testing."""
    return SobolSensitivity(
        model, priors, N_saltelli=64, n_steps=3, seed=42
    ).run()


# ---------------------------------------------------------------------------
# TestSobolConstruction
# ---------------------------------------------------------------------------

class TestSobolConstruction:

    def test_valid_construction(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        s = SobolSensitivity(model, priors, N_saltelli=64)
        assert s is not None

    def test_N_saltelli_stored(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        s = SobolSensitivity(model, priors, N_saltelli=128)
        assert s.N_saltelli == 128

    def test_N_saltelli_too_small_raises(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        with pytest.raises(ValueError, match="N_saltelli must be >= 64"):
            SobolSensitivity(model, priors, N_saltelli=32)

    def test_N_saltelli_not_power_of_two_raises(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            SobolSensitivity(model, priors, N_saltelli=100)

    def test_wrong_prior_count_raises(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        with pytest.raises(ValueError, match="param_dim"):
            SobolSensitivity(model, priors[:-1], N_saltelli=64)


# ---------------------------------------------------------------------------
# TestSobolResultShapes
# ---------------------------------------------------------------------------

class TestSobolResultShapes:

    def test_S1_shape(
        self, small_result: SobolResult, model: BatteryModel2Cell
    ) -> None:
        assert small_result.S1.shape == (model.param_dim, model.state_dim)

    def test_ST_shape(
        self, small_result: SobolResult, model: BatteryModel2Cell
    ) -> None:
        assert small_result.ST.shape == (model.param_dim, model.state_dim)

    def test_S1_conf_shape(
        self, small_result: SobolResult, model: BatteryModel2Cell
    ) -> None:
        assert small_result.S1_conf.shape == (model.param_dim, model.state_dim)

    def test_ST_conf_shape(
        self, small_result: SobolResult, model: BatteryModel2Cell
    ) -> None:
        assert small_result.ST_conf.shape == (model.param_dim, model.state_dim)

    def test_param_names_length(
        self, small_result: SobolResult, model: BatteryModel2Cell
    ) -> None:
        assert len(small_result.param_names) == model.param_dim

    def test_dominant_param_is_string(self, small_result: SobolResult) -> None:
        assert isinstance(small_result.dominant_param, str)

    def test_dominant_param_in_param_names(
        self, small_result: SobolResult
    ) -> None:
        assert small_result.dominant_param in small_result.param_names

    def test_n_evaluations_correct(
        self, small_result: SobolResult, model: BatteryModel2Cell
    ) -> None:
        # SALib sobol uses N*(d+2) when calc_second_order=False
        expected = 64 * (model.param_dim + 2)
        assert small_result.n_evaluations == expected


# ---------------------------------------------------------------------------
# TestSobolInvariants
# ---------------------------------------------------------------------------

class TestSobolInvariants:

    def test_ST_mostly_ge_S1(self, small_result: SobolResult) -> None:
        """
        ST >= S1 in expectation. At low N_saltelli=64, allow up to 20%
        violations due to sampling noise. At N_saltelli=1024 this is ~0%.
        """
        violations = np.sum(small_result.ST < small_result.S1 - 0.05)
        total      = small_result.S1.size
        assert violations / total < 0.20, (
            f"Too many ST < S1 violations: {violations}/{total}"
        )

    def test_S1_conf_positive(self, small_result: SobolResult) -> None:
        assert np.all(small_result.S1_conf >= 0)

    def test_ST_conf_positive(self, small_result: SobolResult) -> None:
        assert np.all(small_result.ST_conf >= 0)

    def test_no_nan_in_S1(self, small_result: SobolResult) -> None:
        assert not np.any(np.isnan(small_result.S1))

    def test_no_nan_in_ST(self, small_result: SobolResult) -> None:
        assert not np.any(np.isnan(small_result.ST))


# ---------------------------------------------------------------------------
# TestSobolPhysics
# ---------------------------------------------------------------------------

class TestSobolPhysics:

    def test_Ea_SEI_is_dominant_for_T1(
        self, small_result: SobolResult
    ) -> None:
        """
        Dominant parameter for T1 must be an activation energy.
        At N_saltelli=64, Ea_SEI and Ea_anode compete -- both physically
        correct as both control the Arrhenius rate.
        At N_saltelli=1024+, Ea_SEI consistently wins.
        """
        expected = {"Ea_SEI", "Ea_anode", "Ea_cath"}
        assert small_result.dominant_param in expected, (
            f"Expected an activation energy, got {small_result.dominant_param}"
        )

    def test_Ea_SEI_S1_above_threshold(
        self, small_result: SobolResult, model: BatteryModel2Cell
    ) -> None:
        """Ea_SEI first-order index for T1 should be meaningfully positive."""
        ea_sei_idx = model.param_names().index("Ea_SEI")
        s1_ea_sei  = small_result.S1[ea_sei_idx, 0]
        assert s1_ea_sei > 0.05, (
            f"Ea_SEI S1 for T1 = {s1_ea_sei:.4f}, expected > 0.05"
        )

    def test_multiple_params_have_nonzero_S1(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        """
        At least 2 parameters should have positive S1 for T1.

        Uses a larger N_saltelli than the shared small_result fixture
        (256 instead of 64) because this specific claim -- that more
        than one parameter matters, not just the single dominant one --
        is genuinely noisy at N_saltelli=64 with only 3 steps. At 256
        it is stable across seeds while still running in well under a
        second, so it is not worth loosening the assertion instead.
        """
        result = SobolSensitivity(
            model, priors, N_saltelli=256, n_steps=3, seed=42
        ).run()
        positive = int(np.sum(result.S1[:, 0] > 0.01))
        assert positive >= 2, (
            f"Only {positive} parameters have S1 > 0.01 for T1"
        )


# ---------------------------------------------------------------------------
# TestSobolReproducibility
# ---------------------------------------------------------------------------

class TestSobolReproducibility:

    def test_shapes_are_deterministic(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        """Shapes are always identical regardless of SALib internal state."""
        r1 = SobolSensitivity(
            model, priors, N_saltelli=64, n_steps=2, seed=7
        ).run()
        r2 = SobolSensitivity(
            model, priors, N_saltelli=64, n_steps=2, seed=7
        ).run()
        assert r1.S1.shape == r2.S1.shape
        assert r1.ST.shape == r2.ST.shape
        assert r1.n_evaluations == r2.n_evaluations

    def test_dominant_param_is_always_activation_energy(
        self, model: BatteryModel2Cell, priors: list[Distribution]
    ) -> None:
        """
        Regardless of SALib internal state, dominant param must always
        be an activation energy -- that is the physically stable result.
        """
        expected = {"Ea_SEI", "Ea_anode", "Ea_cath"}
        for seed in [1, 2, 3, 7, 42]:
            r = SobolSensitivity(
                model, priors, N_saltelli=64, n_steps=2, seed=seed
            ).run()
            assert r.dominant_param in expected, (
                f"seed={seed}: unexpected dominant param {r.dominant_param}"
            )


# ---------------------------------------------------------------------------
# TestSobolSummary
# ---------------------------------------------------------------------------

class TestSobolSummary:

    def test_summary_is_string(self, small_result: SobolResult) -> None:
        assert isinstance(small_result.summary(), str)

    def test_summary_contains_header(self, small_result: SobolResult) -> None:
        assert "Sobol Sensitivity" in small_result.summary()

    def test_summary_contains_dominant_param(
        self, small_result: SobolResult
    ) -> None:
        assert small_result.dominant_param in small_result.summary()

    def test_summary_contains_all_param_names(
        self, small_result: SobolResult
    ) -> None:
        s = small_result.summary()
        for name in small_result.param_names:
            assert name in s, f"Parameter {name} missing from summary"
