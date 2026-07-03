"""
Tests for python/src/battery_model.py -- BatteryModel2Cell.

Test strategy:
  - Interface tests: does it satisfy the Model ABC?
  - Dimension tests: are shapes and names consistent?
  - Physics tests:   does it behave like a real battery?
  - Validation test: does it match Kim 2007 ARC experiment?
  - Scale test:      does it run N=5000 without error?
"""


import numpy as np
import pytest
from numpy.testing import assert_allclose

from python.src.battery_model import BatteryModel2Cell
from python.src.state import Model


# =============================================================================
# FIXTURE
# A pytest fixture is a function that creates a reusable object for tests.
# Every test that takes "model" as an argument gets a fresh BatteryModel2Cell.
# =============================================================================

@pytest.fixture
def model() -> BatteryModel2Cell:
    """Return a fresh BatteryModel2Cell for each test."""
    return BatteryModel2Cell()


@pytest.fixture
def nominal_state(model: BatteryModel2Cell) -> np.ndarray:
    """Return N=1 state tiled from initial_state."""
    return np.tile(model.initial_state(), (1, 1))   # shape (1, 8)


@pytest.fixture
def nominal_params_1(model: BatteryModel2Cell) -> np.ndarray:
    """Return N=1 nominal parameter array, shape (1, 15)."""
    return np.tile(model.nominal_params(), (1, 1))   # shape (1, 15)


# =============================================================================
# TEST CLASS 1: ABC CONTRACT
# =============================================================================

class TestABCContract:
    """BatteryModel2Cell must satisfy the full Model ABC interface."""

    def test_is_instance_of_Model(self, model: BatteryModel2Cell) -> None:
        """BatteryModel2Cell must be a subclass of Model."""
        assert isinstance(model, Model)

    def test_can_be_instantiated(self) -> None:
        """Must construct without error."""
        m = BatteryModel2Cell()
        assert m is not None

    def test_repr_contains_class_name(self, model: BatteryModel2Cell) -> None:
        r = repr(model)
        assert "BatteryModel2Cell" in r

    def test_repr_contains_state_dim(self, model: BatteryModel2Cell) -> None:
        assert "state_dim=8" in repr(model)

    def test_repr_contains_param_dim(self, model: BatteryModel2Cell) -> None:
        assert "param_dim=15" in repr(model)


# =============================================================================
# TEST CLASS 2: DIMENSIONS
# =============================================================================

class TestDimensions:
    """state_dim, param_dim, param_names, initial_state must be consistent."""

    def test_state_dim_is_8(self, model: BatteryModel2Cell) -> None:
        assert model.state_dim == 8

    def test_param_dim_is_15(self, model: BatteryModel2Cell) -> None:
        assert model.param_dim == 15

    def test_param_names_length(self, model: BatteryModel2Cell) -> None:
        """param_names() must have exactly param_dim entries."""
        assert len(model.param_names()) == model.param_dim

    def test_param_names_are_nonempty_strings(
        self, model: BatteryModel2Cell
    ) -> None:
        for name in model.param_names():
            assert isinstance(name, str) and len(name) > 0

    def test_param_names_are_unique(self, model: BatteryModel2Cell) -> None:
        names = model.param_names()
        assert len(names) == len(set(names)), "Duplicate parameter names found"

    def test_initial_state_shape(self, model: BatteryModel2Cell) -> None:
        s0 = model.initial_state()
        assert s0.shape == (8,)

    def test_initial_state_dtype(self, model: BatteryModel2Cell) -> None:
        assert model.initial_state().dtype == np.float64

    def test_nominal_params_shape(self, model: BatteryModel2Cell) -> None:
        p = model.nominal_params()
        assert p.shape == (15,)

    def test_nominal_params_dtype(self, model: BatteryModel2Cell) -> None:
        assert model.nominal_params().dtype == np.float64


# =============================================================================
# TEST CLASS 3: INITIAL STATE
# =============================================================================

class TestInitialState:
    """initial_state() must reflect ARC onset conditions."""

    def test_temperatures_start_at_onset(
        self, model: BatteryModel2Cell
    ) -> None:
        """
        Both cell temperatures must start at 403.15 K (130 C).
        This matches the ARC (Accelerating Rate Calorimetry) onset
        temperature used in Kim 2007.
        """
        s0 = model.initial_state()
        assert_allclose(s0[BatteryModel2Cell.T1], 403.15, rtol=1e-6)
        assert_allclose(s0[BatteryModel2Cell.T2], 403.15, rtol=1e-6)

    def test_all_concentrations_start_at_one(
        self, model: BatteryModel2Cell
    ) -> None:
        """
        All reactant concentrations must start at 1.0 (fully unreacted).
        Indices 2 through 7 are the six concentration state variables.
        """
        s0 = model.initial_state()
        for i in range(2, 8):
            assert_allclose(s0[i], 1.0, rtol=1e-10,
                            err_msg=f"state[{i}] should be 1.0")

    def test_temperatures_are_above_absolute_zero(
        self, model: BatteryModel2Cell
    ) -> None:
        s0 = model.initial_state()
        assert s0[0] > 0.0
        assert s0[1] > 0.0


# =============================================================================
# TEST CLASS 4: NOMINAL PARAMETERS
# =============================================================================

class TestNominalParams:
    """nominal_params() must return physically plausible Kim 2007 values."""

    def test_activation_energies_are_positive(
        self, model: BatteryModel2Cell
    ) -> None:
        """Activation energies must be positive (energy barrier to overcome)."""
        p = model.nominal_params()
        assert p[BatteryModel2Cell.P_EA_SEI] > 0
        assert p[BatteryModel2Cell.P_EA_AN]  > 0
        assert p[BatteryModel2Cell.P_EA_CA]  > 0

    def test_pre_exponentials_are_positive(
        self, model: BatteryModel2Cell
    ) -> None:
        p = model.nominal_params()
        assert p[BatteryModel2Cell.P_A_SEI] > 0
        assert p[BatteryModel2Cell.P_A_AN]  > 0
        assert p[BatteryModel2Cell.P_A_CA]  > 0

    def test_heats_of_reaction_are_positive(
        self, model: BatteryModel2Cell
    ) -> None:
        """Exothermic reactions release heat, so H > 0."""
        p = model.nominal_params()
        assert p[BatteryModel2Cell.P_H_SEI] > 0
        assert p[BatteryModel2Cell.P_H_AN]  > 0
        assert p[BatteryModel2Cell.P_H_CA]  > 0

    def test_cell_mass_is_plausible(self, model: BatteryModel2Cell) -> None:
        """A standard 18650 cell is 40-50 g."""
        p = model.nominal_params()
        m = p[BatteryModel2Cell.P_M_CELL]
        assert 0.030 < m < 0.060, f"Cell mass {m} kg outside 30-60 g range"

    def test_ambient_temperature_is_room_temp(
        self, model: BatteryModel2Cell
    ) -> None:
        """Nominal ambient is 25 C = 298.15 K."""
        p = model.nominal_params()
        T_amb = p[BatteryModel2Cell.P_T_AMB]
        assert_allclose(T_amb, 298.15, rtol=1e-4)

    def test_onset_temperature_is_130C(
        self, model: BatteryModel2Cell
    ) -> None:
        """ARC onset is 130 C = 403.15 K (Kim 2007)."""
        p = model.nominal_params()
        T_onset = p[BatteryModel2Cell.P_T_ONSET]
        assert_allclose(T_onset, 403.15, rtol=1e-4)


# =============================================================================
# TEST CLASS 5: FORWARD_BATCH SHAPE AND DTYPE
# =============================================================================

class TestForwardBatchShape:
    """forward_batch must return the right shape and dtype."""

    def test_output_shape_N1(
        self,
        model: BatteryModel2Cell,
        nominal_state: np.ndarray,
        nominal_params_1: np.ndarray,
    ) -> None:
        """N=1 single particle: output shape must be (1, 8)."""
        new_state = model.forward_batch(nominal_state, nominal_params_1, dt=1.0)
        assert new_state.shape == (1, 8)

    def test_output_shape_N100(self, model: BatteryModel2Cell) -> None:
        """N=100 particles: output shape must be (100, 8)."""
        N = 100
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_params(), (N, 1))
        new_state = model.forward_batch(state, params, dt=1.0)
        assert new_state.shape == (N, 8)

    def test_output_dtype_is_float64(
        self,
        model: BatteryModel2Cell,
        nominal_state: np.ndarray,
        nominal_params_1: np.ndarray,
    ) -> None:
        new_state = model.forward_batch(nominal_state, nominal_params_1, dt=1.0)
        assert new_state.dtype == np.float64

    def test_no_nan_in_output(
        self,
        model: BatteryModel2Cell,
        nominal_state: np.ndarray,
        nominal_params_1: np.ndarray,
    ) -> None:
        new_state = model.forward_batch(nominal_state, nominal_params_1, dt=1.0)
        assert not np.any(np.isnan(new_state)), "NaN found in forward_batch output"

    def test_no_inf_in_output(
        self,
        model: BatteryModel2Cell,
        nominal_state: np.ndarray,
        nominal_params_1: np.ndarray,
    ) -> None:
        new_state = model.forward_batch(nominal_state, nominal_params_1, dt=1.0)
        assert not np.any(np.isinf(new_state)), "Inf found in forward_batch output"


# =============================================================================
# TEST CLASS 6: PHYSICS
# =============================================================================

class TestPhysics:
    """
    Physics sanity checks.
    The battery must behave like a real battery:
      - Temperature rises when reactions are active
      - Concentrations only decrease (reactions are irreversible)
      - Physical bounds are enforced
    """

    def test_temperature_rises_after_one_step(
        self,
        model: BatteryModel2Cell,
        nominal_state: np.ndarray,
        nominal_params_1: np.ndarray,
    ) -> None:
        """
        At ARC onset conditions (T=130C, c=1.0), the SEI reaction is already
        active. Temperature must rise after one 1-second step.
        """
        T_before = nominal_state[0, BatteryModel2Cell.T1]
        new_state = model.forward_batch(nominal_state, nominal_params_1, dt=1.0)
        T_after = new_state[0, BatteryModel2Cell.T1]
        assert T_after > T_before, (
            f"Temperature did not rise: before={T_before:.4f} K, "
            f"after={T_after:.4f} K"
        )

    def test_concentrations_decrease_after_one_step(
        self,
        model: BatteryModel2Cell,
        nominal_state: np.ndarray,
        nominal_params_1: np.ndarray,
    ) -> None:
        """
        All six reactant concentrations must be strictly less than 1.0
        after one time step (reactions consume reactants).
        """
        new_state = model.forward_batch(nominal_state, nominal_params_1, dt=1.0)
        for i in range(2, 8):
            assert new_state[0, i] < 1.0, (
                f"Concentration at index {i} did not decrease: "
                f"{new_state[0, i]:.6f}"
            )

    def test_concentrations_never_go_negative(
        self, model: BatteryModel2Cell
    ) -> None:
        """
        After many steps, concentrations must stay >= 0.
        We run for 10,000 steps with dt=1.0 s (about 2.8 hours).
        """
        N = 10
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_params(), (N, 1))
        for _ in range(10_000):
            state = model.forward_batch(state, params, dt=1.0)
        # Concentrations are in columns 2-7
        concs = state[:, 2:]
        assert np.all(concs >= 0.0), (
            f"Concentration went negative: min = {concs.min():.6e}"
        )

    def test_temperatures_never_go_below_absolute_zero(
        self, model: BatteryModel2Cell
    ) -> None:
        """Temperatures must stay above 0 K (absolute zero) at all times."""
        N = 10
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_params(), (N, 1))
        for _ in range(1_000):
            state = model.forward_batch(state, params, dt=1.0)
        temps = state[:, :2]
        assert np.all(temps > 0.0), (
            f"Temperature went below 0 K: min = {temps.min():.4f}"
        )

    def test_cold_battery_stays_cool(self, model: BatteryModel2Cell) -> None:
        """
        A battery at 25 C (298 K) with T_amb=298 K and very slow reactions
        should barely heat up over 10 steps.

        We test this by setting T_onset very high (1000 K) so reactions
        are negligible, and checking temperature stays near ambient.
        """
        N = 1
        # Start at ambient temperature, not onset
        state = np.tile(model.initial_state(), (N, 1))
        state[:, BatteryModel2Cell.T1] = 298.15
        state[:, BatteryModel2Cell.T2] = 298.15

        params = np.tile(model.nominal_params(), (N, 1))
        # Push onset temperature to 1000 K so reactions are negligible
        params[:, BatteryModel2Cell.P_T_ONSET] = 1000.0
        # Use a very high activation energy to make rates near zero
        params[:, BatteryModel2Cell.P_EA_SEI] = 2.0e6
        params[:, BatteryModel2Cell.P_EA_AN]  = 2.0e6
        params[:, BatteryModel2Cell.P_EA_CA]  = 2.0e6

        for _ in range(10):
            state = model.forward_batch(state, params, dt=1.0)

        T_final = state[0, BatteryModel2Cell.T1]
        # With no reactions and T == T_amb, temperature should not move much
        assert abs(T_final - 298.15) < 1.0, (
            f"Cold battery heated unexpectedly to {T_final:.2f} K"
        )


# =============================================================================
# TEST CLASS 7: KIM 2007 ARC VALIDATION
# =============================================================================

class TestKim2007Validation:
    """
    Validate against the Kim 2007 ARC (Accelerating Rate Calorimetry) test.

    ARC test protocol:
      - Battery starts at onset temperature T_onset = 130 C = 403.15 K
      - All reactants fully charged (c = 1.0)
      - Adiabatic conditions (no heat loss, h_conv = 0)
      - Self-heating rate should be detectable within minutes

    Kim 2007 reports that under these conditions the battery shows
    measurable self-heating within the first few minutes.
    We verify that the nominal model produces a positive self-heating
    rate and that temperature rises at least 1 K over 60 seconds.
    """

    def test_temperature_rises_at_least_1K_in_60s(
        self, model: BatteryModel2Cell
    ) -> None:
        """
        Under adiabatic ARC conditions, temperature must rise >= 1 K in 60 s.
        This is the basic self-heating criterion from Kim 2007.
        """
        N = 1
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_params(), (N, 1))
        # Adiabatic: no convective heat loss
        params[:, BatteryModel2Cell.P_H_CONV] = 0.0

        T_start = state[0, BatteryModel2Cell.T1]
        for _ in range(60):   # 60 steps of 1 second each = 60 seconds
            state = model.forward_batch(state, params, dt=1.0)
        T_end = state[0, BatteryModel2Cell.T1]

        assert T_end > T_start + 1.0, (
            f"Temperature rose only {T_end - T_start:.4f} K in 60 s "
            f"(expected >= 1 K for Kim 2007 ARC validation)"
        )

    def test_onset_temperature_within_5C_of_kim2007(
        self, model: BatteryModel2Cell
    ) -> None:
        """
        The ARC onset temperature is 403.15 K (130 C) in Kim 2007.
        Our nominal_params must agree to within 5 C (5 K).
        """
        p = model.nominal_params()
        T_onset = p[BatteryModel2Cell.P_T_ONSET]
        assert abs(T_onset - 403.15) <= 5.0, (
            f"T_onset = {T_onset:.2f} K is more than 5 K from 403.15 K"
        )

    def test_self_heating_rate_is_positive(
        self, model: BatteryModel2Cell
    ) -> None:
        """
        dT/dt must be positive at ARC onset conditions with h_conv=0.
        If the self-heating rate is zero or negative, the model is wrong.
        """
        N = 1
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_params(), (N, 1))
        params[:, BatteryModel2Cell.P_H_CONV] = 0.0

        T_before = state[0, BatteryModel2Cell.T1]
        new_state = model.forward_batch(state, params, dt=1.0)
        T_after = new_state[0, BatteryModel2Cell.T1]

        dT_dt = (T_after - T_before) / 1.0   # K/s
        assert dT_dt > 0.0, (
            f"Self-heating rate = {dT_dt:.6f} K/s is not positive"
        )


# =============================================================================
# TEST CLASS 8: LARGE BATCH
# =============================================================================

class TestLargeBatch:
    """N=5000 is the production batch size. Must run cleanly."""

    def test_N5000_shape(self, model: BatteryModel2Cell) -> None:
        N = 5000
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_params(), (N, 1))
        new_state = model.forward_batch(state, params, dt=1.0)
        assert new_state.shape == (N, 8)

    def test_N5000_no_nan(self, model: BatteryModel2Cell) -> None:
        N = 5000
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_params(), (N, 1))
        new_state = model.forward_batch(state, params, dt=1.0)
        assert not np.any(np.isnan(new_state))

    def test_N5000_no_inf(self, model: BatteryModel2Cell) -> None:
        N = 5000
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_params(), (N, 1))
        new_state = model.forward_batch(state, params, dt=1.0)
        assert not np.any(np.isinf(new_state))

    def test_N5000_heterogeneous_params(
        self, model: BatteryModel2Cell
    ) -> None:
        """
        Each particle gets a slightly different Ea_SEI.
        Verifies vectorisation works when params differ across particles.
        """
        N = 5000
        rng = np.random.default_rng(seed=42)
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_params(), (N, 1))
        # Add 5% noise to Ea_SEI for each particle
        params[:, BatteryModel2Cell.P_EA_SEI] *= (
            1.0 + 0.05 * rng.standard_normal(N)
        )
        new_state = model.forward_batch(state, params, dt=1.0)
        assert new_state.shape == (N, 8)
        assert not np.any(np.isnan(new_state))
        # Particles should now have different temperatures (not all identical)
        temps = new_state[:, BatteryModel2Cell.T1]
        assert np.std(temps) > 0.0, (
            "All particles identical -- vectorisation may be broken"
        )
