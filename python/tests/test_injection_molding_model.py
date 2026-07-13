"""
python/tests/test_injection_molding_model.py

Tests for InjectionMoldingProcessModel (Month 3 Week 14).

VALIDATION SOURCE
------------------
Kavade, M.V. and Kadam, G.S. (2012). "Parameter Optimization of
Injection Molding of Polypropylene by using Taguchi Methodology."
IOSR Journal of Mechanical and Civil Engineering, 4(4), 49-58. Real
physical experiments, not simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from python.src.distributions import Distribution, Normal
from python.src.injection_molding_model import InjectionMoldingProcessModel
from python.src.monte_carlo import MonteCarloEngine
from python.src.sensitivity import SobolSensitivity


def _real_priors() -> list[Distribution]:
    return [
        Normal(225.0, (235.0 - 215.0) / 6),
        Normal(40.0, (45.0 - 30.0) / 6),
        Normal(7.0, (11.0 - 4.0) / 6),
        Normal(40.0, (45.0 - 35.0) / 6),
        Normal(45.0, (50.0 - 40.0) / 6),
    ]


class TestInjectionMoldingModelBasics:

    def test_state_and_param_dims(self) -> None:
        model = InjectionMoldingProcessModel()
        assert model.state_dim == 1
        assert model.param_dim == 5

    def test_param_names(self) -> None:
        model = InjectionMoldingProcessModel()
        names = model.param_names()
        assert names == [
            "Barrel_Temp", "Injection_Pressure", "Coolant_Flow",
            "Holding_Pressure", "Injection_Speed",
        ]

    def test_nominal_params_reproduce_real_cited_weight(self) -> None:
        """
        The real, physically-verified nominal weight from the source
        paper (predicted mean 96.826g) must be exactly reproduced.
        """
        model = InjectionMoldingProcessModel()
        nominal = model.nominal_parameters()
        state = np.tile(model.initial_state(), (1, 1))
        params = np.tile(nominal, (1, 1))
        result = model.forward_batch(state, params, dt=1.0)
        np.testing.assert_allclose(result[0, 0], 96.826, atol=1e-6)

    def test_forward_batch_is_idempotent(self) -> None:
        """
        Honest architectural check: this process has no genuine
        within-shot time evolution -- repeated calls with the same
        params must converge immediately to the same output.
        """
        model = InjectionMoldingProcessModel()
        nominal = model.nominal_parameters()
        state = np.tile(model.initial_state(), (1, 1))
        params = np.tile(nominal, (1, 1))
        result1 = model.forward_batch(state, params, dt=1.0)
        result2 = model.forward_batch(result1, params, dt=1.0)
        np.testing.assert_allclose(result1, result2, atol=1e-9)

    def test_increasing_barrel_temp_increases_weight(self) -> None:
        """
        Matches the real, physically-measured positive relationship
        reported in the source paper.
        """
        model = InjectionMoldingProcessModel()
        nominal = model.nominal_parameters()
        state = np.tile(model.initial_state(), (1, 1))
        params_nominal = np.tile(nominal, (1, 1))
        params_high = params_nominal.copy()
        params_high[0, model.P_BARREL_TEMP] = 235.0

        result_nominal = model.forward_batch(state, params_nominal, dt=1.0)
        result_high = model.forward_batch(state, params_high, dt=1.0)
        assert result_high[0, 0] > result_nominal[0, 0]

    def test_validate_params_rejects_nonpositive(self) -> None:
        model = InjectionMoldingProcessModel()
        bad_params = np.array([[-1.0, 40.0, 7.0, 40.0, 45.0]])
        with pytest.raises(ValueError):
            model.validate_params(bad_params)


class TestInjectionMoldingRealSobolStructure:
    """
    Regression test: the model's own real Sobol sensitivity ranking
    must match the source paper's real ANOVA ranking.
    """

    def test_sobol_ranking_matches_real_paper_anova(self) -> None:
        model = InjectionMoldingProcessModel()
        priors = _real_priors()

        sobol = SobolSensitivity(
            model, priors, N_saltelli=1024, n_steps=1, dt=1.0, seed=42,
        )
        result = sobol.run()

        param_s1 = sorted(
            zip(model.param_names(), result.S1[:, 0], strict=True),
            key=lambda x: x[1], reverse=True,
        )
        ranked_names = [name for name, _ in param_s1]
        assert ranked_names == [
            "Barrel_Temp", "Injection_Pressure", "Coolant_Flow",
            "Holding_Pressure", "Injection_Speed",
        ]

    def test_sobol_reproducible_same_seed(self) -> None:
        model = InjectionMoldingProcessModel()
        priors = _real_priors()

        sobol1 = SobolSensitivity(
            model, priors, N_saltelli=64, n_steps=1, dt=1.0, seed=42,
        )
        result1 = sobol1.run()
        sobol2 = SobolSensitivity(
            model, priors, N_saltelli=64, n_steps=1, dt=1.0, seed=42,
        )
        result2 = sobol2.run()
        assert np.array_equal(result1.S1, result2.S1)


class TestInjectionMoldingMonteCarlo:

    def test_monte_carlo_mean_matches_nominal(self) -> None:
        """
        Real Monte Carlo mean should closely match the real,
        physically-verified nominal weight.
        """
        model = InjectionMoldingProcessModel()
        priors = _real_priors()
        engine = MonteCarloEngine(model, priors, N=5000, n_steps=1, dt=1.0, seed=42)
        result = engine.run()
        final_weight = result.trajectories[:, -1, 0]
        np.testing.assert_allclose(final_weight.mean(), 96.826, atol=0.05)
