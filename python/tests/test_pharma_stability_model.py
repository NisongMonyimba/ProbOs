"""
python/tests/test_pharma_stability_model.py

Tests for PharmaStabilityModel (Month 3 Week 11).

See python/src/pharma_stability_model.py's module docstring for the
full validation source citation and the honest methodological
limitation.
"""

from __future__ import annotations

import numpy as np

from python.src.pharma_stability_model import PharmaStabilityModel


class TestPharmaStabilityModelShapes:

    def test_state_dim(self) -> None:
        model = PharmaStabilityModel()
        assert model.state_dim == 2

    def test_param_dim(self) -> None:
        model = PharmaStabilityModel()
        assert model.param_dim == 4

    def test_param_names(self) -> None:
        model = PharmaStabilityModel()
        assert model.param_names() == ["Ea", "A", "n", "T"]

    def test_initial_state(self) -> None:
        model = PharmaStabilityModel()
        state = model.initial_state()
        assert state.shape == (2,)
        assert state[0] == 1.0
        assert state[1] == 0.0


class TestPharmaStabilityModelInvariants:

    def test_potency_never_exceeds_one(self) -> None:
        model = PharmaStabilityModel()
        N = 50
        state = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_parameters(), (N, 1))
        for _ in range(100):
            state = model.forward_batch(state, params, dt=1.0)
            assert np.all(state[:, model.POTENCY] <= 1.0)

    def test_potency_never_negative(self) -> None:
        model = PharmaStabilityModel()
        N = 50
        state = np.tile(model.initial_state(), (N, 1))
        params = np.tile(model.nominal_parameters(), (N, 1))
        for _ in range(2000):
            state = model.forward_batch(state, params, dt=1.0)
            assert np.all(state[:, model.POTENCY] >= 0.0)

    def test_potency_monotonically_decreasing(self) -> None:
        model = PharmaStabilityModel()
        state = model.initial_state().reshape(1, -1)
        params = model.nominal_parameters().reshape(1, -1)
        prev_potency = state[0, model.POTENCY]
        for _ in range(365):
            state = model.forward_batch(state, params, dt=1.0)
            current_potency = state[0, model.POTENCY]
            assert current_potency <= prev_potency + 1e-12
            prev_potency = current_potency

    def test_elapsed_time_accumulates_correctly(self) -> None:
        model = PharmaStabilityModel()
        state = model.initial_state().reshape(1, -1)
        params = model.nominal_parameters().reshape(1, -1)
        for day in range(1, 11):
            state = model.forward_batch(state, params, dt=1.0)
            assert abs(state[0, model.ELAPSED_TIME] - day) < 1e-9


class TestPharmaStabilityModelValidation:
    """
    Day 3 validation against REAL reported data from
    Gonzalez-Gonzalez et al. (Molecules 2023, 28(23), 7925):
    DCHX degradation at 365 days: 3.1% (5C), 17.4% (25C), 25.9% (30C).
    """

    def test_matches_real_reported_data_5C(self) -> None:
        model = PharmaStabilityModel()
        state = model.initial_state().reshape(1, -1)
        params = np.array([[18.52, 1.89e10, 1.0, 278.15]])
        for _ in range(365):
            state = model.forward_batch(state, params, dt=1.0)
        expected_potency = 0.969
        assert abs(state[0, model.POTENCY] - expected_potency) < 0.02

    def test_matches_real_reported_data_25C(self) -> None:
        model = PharmaStabilityModel()
        state = model.initial_state().reshape(1, -1)
        params = np.array([[18.52, 1.89e10, 1.0, 298.15]])
        for _ in range(365):
            state = model.forward_batch(state, params, dt=1.0)
        expected_potency = 0.826
        assert abs(state[0, model.POTENCY] - expected_potency) < 0.02

    def test_matches_real_reported_data_30C(self) -> None:
        model = PharmaStabilityModel()
        state = model.initial_state().reshape(1, -1)
        params = np.array([[18.52, 1.89e10, 1.0, 303.15]])
        for _ in range(365):
            state = model.forward_batch(state, params, dt=1.0)
        expected_potency = 0.741
        assert abs(state[0, model.POTENCY] - expected_potency) < 0.02

    def test_higher_temperature_degrades_faster(self) -> None:
        model = PharmaStabilityModel()
        temps = [278.15, 298.15, 303.15]
        final_potencies = []
        for T in temps:
            state = model.initial_state().reshape(1, -1)
            params = np.array([[18.52, 1.89e10, 1.0, T]])
            for _ in range(365):
                state = model.forward_batch(state, params, dt=1.0)
            final_potencies.append(state[0, model.POTENCY])
        assert final_potencies[0] > final_potencies[1] > final_potencies[2]
