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
        response = client.post("/sensitivity", json={
            "model_name": "battery", "N_saltelli": 64, "n_steps": 2,
            "seed": 42,
        })
        api_S1 = np.array(response.json()["S1"])

        model = BatteryModel2Cell()
        priors = build_battery_priors()
        s = SobolSensitivity(
            model, priors, N_saltelli=64, n_steps=2, seed=42,
        )
        direct_result = s.run()

        np.testing.assert_allclose(api_S1, direct_result.S1, rtol=1e-10)


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
