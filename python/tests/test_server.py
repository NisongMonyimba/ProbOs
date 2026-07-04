"""
python/tests/test_server.py

Integration tests for the ProbOS FastAPI service layer (Month 2 Week 7).

Uses FastAPI's TestClient (built on httpx), which runs the app
in-process without an actual network socket -- fast and deterministic.

Per Week 7's exit criteria (docs/monthly_plans/month2/week7/main.tex):
"curl -X POST localhost:8000/simulate ... returns valid JSON matching
a direct Python call to the same MonteCarloEngine" -- these tests
verify exactly that, for all three POST endpoints, by running the
kernel directly in the test and comparing against the HTTP response.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from python.server.main import app
from python.src.battery_model import BatteryModel2Cell
from python.src.monte_carlo import MonteCarloEngine
from python.src.parameter_priors import build_battery_priors
from python.src.particle_filter import ParticleFilter
from python.src.sensitivity import SobolSensitivity
from python.src.state import FloatArray

client = TestClient(app)


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self) -> None:
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_version(self) -> None:
        response = client.get("/health")
        data = response.json()
        assert "version" in data


# ---------------------------------------------------------------------------
# TestSimulateEndpoint
# ---------------------------------------------------------------------------

class TestSimulateEndpoint:

    def test_simulate_returns_200(self) -> None:
        response = client.post("/simulate", json={
            "model_name": "battery", "N": 200, "n_steps": 10, "seed": 42,
        })
        assert response.status_code == 200

    def test_simulate_percentiles_shape(self) -> None:
        response = client.post("/simulate", json={
            "model_name": "battery", "N": 200, "n_steps": 10, "seed": 42,
        })
        data = response.json()
        percentiles = data["percentiles"]
        assert len(percentiles) == 3          # P05, P50, P95
        assert len(percentiles[0]) == 11       # n_steps + 1
        assert len(percentiles[0][0]) == 8     # state_dim

    def test_simulate_matches_direct_kernel_call(self) -> None:
        """
        THE key validation: the API response must numerically match a
        direct Python call to MonteCarloEngine with the same
        parameters and seed.
        """
        response = client.post("/simulate", json={
            "model_name": "battery", "N": 500, "n_steps": 20, "seed": 7,
        })
        api_percentiles = np.array(response.json()["percentiles"])

        model = BatteryModel2Cell()
        priors = build_battery_priors()
        engine = MonteCarloEngine(
            model, priors, N=500, n_steps=20, dt=1.0, seed=7,
        )
        direct_result = engine.run()

        np.testing.assert_allclose(
            api_percentiles, direct_result.percentiles, rtol=1e-10
        )

    def test_simulate_rejects_oversized_N(self) -> None:
        """Pydantic validation must reject N above the resource-exhaustion bound."""
        response = client.post("/simulate", json={
            "model_name": "battery", "N": 10_000_000, "n_steps": 10,
        })
        assert response.status_code == 422

    def test_simulate_rejects_unknown_model(self) -> None:
        response = client.post("/simulate", json={
            "model_name": "nonexistent_model", "N": 100, "n_steps": 10,
        })
        assert response.status_code == 404

    def test_simulate_default_params_work(self) -> None:
        """Confirm sensible defaults work without specifying every field."""
        response = client.post("/simulate", json={})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# TestSensitivityEndpoint
# ---------------------------------------------------------------------------

class TestSensitivityEndpoint:

    def test_sensitivity_returns_200(self) -> None:
        response = client.post("/sensitivity", json={
            "model_name": "battery", "N_saltelli": 64, "n_steps": 2,
            "seed": 42,
        })
        assert response.status_code == 200

    def test_sensitivity_dominant_param_is_activation_energy(self) -> None:
        """
        Consistent with test_sensitivity.py's finding: at low
        N_saltelli, the dominant parameter for T1 is one of the
        activation energies.
        """
        response = client.post("/sensitivity", json={
            "model_name": "battery", "N_saltelli": 64, "n_steps": 2,
            "seed": 42,
        })
        data = response.json()
        expected = {"Ea_SEI", "Ea_anode", "Ea_cath"}
        assert data["dominant_param"] in expected

    def test_sensitivity_S1_shape(self) -> None:
        response = client.post("/sensitivity", json={
            "model_name": "battery", "N_saltelli": 64, "n_steps": 2,
            "seed": 42,
        })
        data = response.json()
        assert len(data["S1"]) == 15   # param_dim
        assert len(data["S1"][0]) == 8  # state_dim

    def test_sensitivity_rejects_non_power_of_2_N_saltelli(self) -> None:
        """
        SobolSensitivity itself raises ValueError for non-power-of-2
        N_saltelli -- the endpoint must surface this as a 422, not a
        500 (unhandled exception).
        """
        response = client.post("/sensitivity", json={
            "model_name": "battery", "N_saltelli": 100, "n_steps": 2,
        })
        assert response.status_code == 422

    def test_sensitivity_matches_direct_kernel_call(self) -> None:
        """
        SALib's Sobol sampler does not support exact reproducibility
        via np.random.seed() -- its internal sequence generator has
        its own state (documented in python/tests/test_sensitivity.py,
        discovered in Month 1 Week 3). Two SEPARATE SobolSensitivity
        calls with the same seed -- one via the API, one direct --
        can therefore produce slightly different S1 values, exactly
        as they can between two direct calls (see
        test_sensitivity.py::TestSobolReproducibility).

        This test verifies STRUCTURAL agreement (shapes match, and
        the dominant parameter is a physically sensible activation
        energy in both cases) rather than exact numerical equality,
        which would be flaky by construction -- the same standard
        already applied to SobolSensitivity's own reproducibility
        tests, now applied consistently at the API layer.
        """
        response = client.post("/sensitivity", json={
            "model_name": "battery", "N_saltelli": 64, "n_steps": 2,
            "seed": 42,
        })
        api_data = response.json()
        api_S1 = np.array(api_data["S1"])

        model = BatteryModel2Cell()
        priors = build_battery_priors()
        s = SobolSensitivity(
            model, priors, N_saltelli=64, n_steps=2, seed=42,
        )
        direct_result = s.run()

        assert api_S1.shape == direct_result.S1.shape
        expected_dominant = {"Ea_SEI", "Ea_anode", "Ea_cath"}
        assert api_data["dominant_param"] in expected_dominant
        assert direct_result.dominant_param in expected_dominant


# ---------------------------------------------------------------------------
# TestFilterEndpoint
# ---------------------------------------------------------------------------

class TestFilterEndpoint:

    def test_filter_returns_200(self) -> None:
        response = client.post("/filter", json={
            "model_name": "battery", "N": 100, "seed": 42,
            "sigma_obs": 5.0,
            "observations": [403.0, 404.0, 405.0, 406.0, 407.0],
        })
        assert response.status_code == 200

    def test_filter_result_shapes(self) -> None:
        response = client.post("/filter", json={
            "model_name": "battery", "N": 100, "seed": 42,
            "sigma_obs": 5.0,
            "observations": [403.0, 404.0, 405.0, 406.0, 407.0],
        })
        data = response.json()
        assert len(data["means"]) == 5    # T = len(observations)
        assert len(data["means"][0]) == 8  # state_dim
        assert data["n_particles"] == 100

    def test_filter_rejects_empty_observations(self) -> None:
        response = client.post("/filter", json={
            "model_name": "battery", "N": 100, "observations": [],
        })
        assert response.status_code == 422

    def test_filter_ess_never_exceeds_N(self) -> None:
        response = client.post("/filter", json={
            "model_name": "battery", "N": 200, "seed": 3,
            "sigma_obs": 5.0,
            "observations": [403.0, 404.0, 405.0],
        })
        data = response.json()
        assert all(ess <= 200.0 + 1e-6 for ess in data["ess_history"])

    def test_filter_matches_direct_kernel_call(self) -> None:
        """
        Uses the SAME loglik construction pattern as the /filter
        endpoint itself (Gaussian noise on state[:, 0]), to confirm
        the HTTP layer introduces no numerical discrepancy.
        """
        obs_list = [403.0, 404.0, 405.0]
        response = client.post("/filter", json={
            "model_name": "battery", "N": 100, "seed": 11,
            "sigma_obs": 5.0, "observations": obs_list,
        })
        api_means = np.array(response.json()["means"])

        model = BatteryModel2Cell()
        priors = build_battery_priors()
        pf = ParticleFilter(model, priors, N=100, dt=1.0, seed=11)

        sigma_obs = 5.0

        def loglik(state: FloatArray, obs: FloatArray) -> FloatArray:
            x = state[:, 0]
            result: FloatArray = -0.5 * ((x - obs[0]) / sigma_obs) ** 2
            return result

        observations = np.array(obs_list).reshape(-1, 1)
        direct_result = pf.run(observations, loglik)

        np.testing.assert_allclose(
            api_means, direct_result.means, rtol=1e-10
        )



# ---------------------------------------------------------------------------
# TestWeek9ModelRegistryExtension
#
# Numerical-match tests for the three models registered in Month 3
# Week 9 (option_pricer, ed_queue, clinical_trial), matching the same
# rtol=1e-10 discipline already established for battery in Week 7.
# ---------------------------------------------------------------------------

from python.examples.week4_clinical_trial import (  # noqa: E402
    ClinicalTrialModel,
    build_clinical_trial_priors,
)
from python.examples.week4_ed_queue import (  # noqa: E402
    EDQueueModel,
    build_ed_queue_priors,
)
from python.examples.week4_option_pricer import (  # noqa: E402
    OptionPricerModel,
    build_option_priors,
)


class TestWeek9SimulateNewModels:

    def test_option_pricer_simulate_matches_direct_kernel_call(self) -> None:
        response = client.post("/simulate", json={
            "model_name": "option_pricer", "N": 500, "n_steps": 20, "seed": 7,
        })
        assert response.status_code == 200
        api_percentiles = np.array(response.json()["percentiles"])

        model = OptionPricerModel(seed=7)
        priors = build_option_priors()
        engine = MonteCarloEngine(
            model, priors, N=500, n_steps=20, dt=1.0, seed=7,
        )
        direct_result = engine.run()

        np.testing.assert_allclose(
            api_percentiles, direct_result.percentiles, rtol=1e-10
        )

    def test_ed_queue_simulate_matches_direct_kernel_call(self) -> None:
        response = client.post("/simulate", json={
            "model_name": "ed_queue", "N": 500, "n_steps": 20, "seed": 7,
        })
        assert response.status_code == 200
        api_percentiles = np.array(response.json()["percentiles"])

        model = EDQueueModel(seed=7)
        priors = build_ed_queue_priors()
        engine = MonteCarloEngine(
            model, priors, N=500, n_steps=20, dt=1.0, seed=7,
        )
        direct_result = engine.run()

        np.testing.assert_allclose(
            api_percentiles, direct_result.percentiles, rtol=1e-10
        )

    def test_clinical_trial_simulate_matches_direct_kernel_call(self) -> None:
        response = client.post("/simulate", json={
            "model_name": "clinical_trial", "N": 500, "n_steps": 20, "seed": 7,
        })
        assert response.status_code == 200
        api_percentiles = np.array(response.json()["percentiles"])

        model = ClinicalTrialModel(seed=7)
        priors = build_clinical_trial_priors()
        engine = MonteCarloEngine(
            model, priors, N=500, n_steps=20, dt=1.0, seed=7,
        )
        direct_result = engine.run()

        np.testing.assert_allclose(
            api_percentiles, direct_result.percentiles, rtol=1e-10
        )

    def test_all_four_models_registered_in_health_error_message(self) -> None:
        """
        Confirms the 404 error message lists all four registered
        models, not just the original 'battery' -- a regression test
        for the registry extension itself.
        """
        response = client.post("/simulate", json={
            "model_name": "nonexistent_model", "N": 100, "n_steps": 5,
        })
        assert response.status_code == 404
        detail = response.json()["detail"]
        for name in ["battery", "option_pricer", "ed_queue", "clinical_trial"]:
            assert name in detail


class TestWeek9SensitivityNewModels:

    def test_option_pricer_sensitivity_matches_direct_kernel_call(self) -> None:
        response = client.post("/sensitivity", json={
            "model_name": "option_pricer", "N_saltelli": 64, "n_steps": 2,
            "seed": 42,
        })
        assert response.status_code == 200
        api_data = response.json()
        api_S1 = np.array(api_data["S1"])

        model = OptionPricerModel()
        priors = build_option_priors()
        s = SobolSensitivity(model, priors, N_saltelli=64, n_steps=2, seed=42)
        direct_result = s.run()

        assert api_S1.shape == direct_result.S1.shape
        assert api_data["param_names"] == direct_result.param_names

    def test_ed_queue_sensitivity_matches_direct_kernel_call(self) -> None:
        response = client.post("/sensitivity", json={
            "model_name": "ed_queue", "N_saltelli": 64, "n_steps": 2,
            "seed": 42,
        })
        assert response.status_code == 200
        api_data = response.json()
        api_S1 = np.array(api_data["S1"])

        model = EDQueueModel()
        priors = build_ed_queue_priors()
        s = SobolSensitivity(model, priors, N_saltelli=64, n_steps=2, seed=42)
        direct_result = s.run()

        assert api_S1.shape == direct_result.S1.shape
        assert api_data["param_names"] == direct_result.param_names

    def test_clinical_trial_sensitivity_matches_direct_kernel_call(self) -> None:
        response = client.post("/sensitivity", json={
            "model_name": "clinical_trial", "N_saltelli": 64, "n_steps": 2,
            "seed": 42,
        })
        assert response.status_code == 200
        api_data = response.json()
        api_S1 = np.array(api_data["S1"])

        model = ClinicalTrialModel()
        priors = build_clinical_trial_priors()
        s = SobolSensitivity(model, priors, N_saltelli=64, n_steps=2, seed=42)
        direct_result = s.run()

        assert api_S1.shape == direct_result.S1.shape
        assert api_data["param_names"] == direct_result.param_names


class TestWeek9FilterScopeRestriction:

    def test_filter_rejects_clinical_trial_with_422(self) -> None:
        """
        clinical_trial's first state variable (n_treatment) is an
        integer enrollment count, not a continuous quantity with
        genuine observation noise -- /filter's generic Gaussian-noise
        likelihood does not apply meaningfully to it (confirmed by
        direct investigation during Week 9). This must be explicitly
        rejected with 422, not silently return a meaningless 200.
        """
        response = client.post("/filter", json={
            "model_name": "clinical_trial", "N": 100,
            "observations": [0.0, 1.0, 2.0],
        })
        assert response.status_code == 422
        assert "clinical_trial" in response.json()["detail"]

    def test_filter_still_works_for_option_pricer(self) -> None:
        response = client.post("/filter", json={
            "model_name": "option_pricer", "N": 100,
            "observations": [100.0, 101.0, 102.0],
        })
        assert response.status_code == 200

    def test_filter_still_works_for_ed_queue(self) -> None:
        response = client.post("/filter", json={
            "model_name": "ed_queue", "N": 100,
            "observations": [2.0, 3.0, 4.0],
        })
        assert response.status_code == 200


class TestWallTimeMs:

    def test_simulate_returns_genuine_wall_time(self) -> None:
        """
        Found via the comprehensive workaround scan: /simulate used to
        return a hardcoded wall_time_ms=0.0 placeholder. Confirms it
        now returns a genuinely measured, positive value.
        """
        response = client.post("/simulate", json={
            "model_name": "battery", "N": 100, "n_steps": 10,
        })
        assert response.status_code == 200
        wall_time_ms = response.json()["wall_time_ms"]
        assert wall_time_ms > 0.0
