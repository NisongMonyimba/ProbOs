"""
python/tests/test_week8_examples.py

Tests for Week 8 cross-discipline particle-filtering examples,
following the same standard set for Week 4's forward-simulation
examples (test_week4_examples.py): shape contracts, physical/
statistical invariants, and validation against ground truth.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pytest

from python.examples.week4_ed_queue import EDQueueModel
from python.examples.week8_ed_queue_filter import (
    build_ed_queue_filter_priors,
    generate_synthetic_observations,
    queue_length_log_likelihood,
)
from python.src.particle_filter import ParticleFilter, PFResult


class FilterResultDict(TypedDict):
    """Typed structure returned by the filter_result fixture."""
    true_lambda: float
    posterior_mean: float
    posterior_std: float
    result: PFResult


# ---------------------------------------------------------------------------
# TestEDQueueFilterHelpers
# ---------------------------------------------------------------------------

class TestEDQueueFilterHelpers:

    def test_build_priors_returns_two(self) -> None:
        priors = build_ed_queue_filter_priors()
        assert len(priors) == 2

    def test_generate_synthetic_observations_shape(self) -> None:
        obs = generate_synthetic_observations(
            true_lambda=10.0, true_mu=16.0, n_steps=50, dt=0.05,
            sigma_obs=1.0, seed=1,
        )
        assert obs.shape == (50,)

    def test_generate_synthetic_observations_reproducible(self) -> None:
        obs1 = generate_synthetic_observations(
            true_lambda=10.0, true_mu=16.0, n_steps=50, dt=0.05,
            sigma_obs=1.0, seed=7,
        )
        obs2 = generate_synthetic_observations(
            true_lambda=10.0, true_mu=16.0, n_steps=50, dt=0.05,
            sigma_obs=1.0, seed=7,
        )
        np.testing.assert_array_equal(obs1, obs2)

    def test_generate_synthetic_observations_nonnegative_mean(self) -> None:
        """
        Queue lengths (before noise) are non-negative, so the mean of
        many noisy observations around a genuinely occupied queue
        should be positive.
        """
        obs = generate_synthetic_observations(
            true_lambda=10.0, true_mu=16.0, n_steps=200, dt=0.05,
            sigma_obs=1.0, seed=1,
        )
        assert np.mean(obs) > 0.0

    def test_loglik_returns_correct_shape(self) -> None:
        loglik = queue_length_log_likelihood(sigma_obs=1.5)
        state = np.array([[3.0], [5.0], [1.0]])
        obs = np.array([4.0])
        result = loglik(state, obs)
        assert result.shape == (3,)

    def test_loglik_higher_for_closer_match(self) -> None:
        """
        A particle whose state matches the observation more closely
        should have a HIGHER log-likelihood than one further away.
        """
        loglik = queue_length_log_likelihood(sigma_obs=1.5)
        state = np.array([[4.0], [10.0]])
        obs = np.array([4.0])
        result = loglik(state, obs)
        assert result[0] > result[1]


# ---------------------------------------------------------------------------
# TestEDQueueParticleFilter
#
# The critical validation: does the particle filter genuinely recover
# a known true arrival rate from noisy observations?
# ---------------------------------------------------------------------------

class TestEDQueueParticleFilter:

    @classmethod
    @pytest.fixture(scope="class")
    def filter_result(cls) -> FilterResultDict:
        """
        Runs the full ED queue filtering pipeline once, shared across
        tests in this class, matching the pattern already used for
        expensive shared fixtures elsewhere in this test suite
        (e.g. test_sensitivity.py's small_result).
        """
        true_lambda = 10.0
        true_mu = 16.0
        n_steps = 200
        dt = 0.05
        sigma_obs = 1.5

        observations = generate_synthetic_observations(
            true_lambda=true_lambda, true_mu=true_mu,
            n_steps=n_steps, dt=dt, sigma_obs=sigma_obs, seed=42,
        )

        model = EDQueueModel(Q0=0.0)
        priors = build_ed_queue_filter_priors(
            lambda_prior_low=5.0, lambda_prior_high=15.0, mu_fixed=true_mu,
        )
        pf = ParticleFilter(
            model, priors, N=2000, dt=dt, resample_threshold=0.5, seed=42,
        )
        loglik = queue_length_log_likelihood(sigma_obs=sigma_obs)
        result = pf.run(observations.reshape(-1, 1), loglik)

        final_lambda_values = pf.params[:, 0]
        final_weights = result.final_weights
        posterior_mean = float(
            np.average(final_lambda_values, weights=final_weights)
        )
        posterior_std = float(np.sqrt(
            np.average(
                (final_lambda_values - posterior_mean) ** 2,
                weights=final_weights,
            )
        ))

        return {
            "true_lambda": true_lambda,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "result": result,
        }

    def test_posterior_mean_close_to_true_lambda(
        self, filter_result: FilterResultDict
    ) -> None:
        """
        THE key validation: starting from a wide Uniform(5, 15) prior
        (deliberately less informative than Week 4's Uniform(8, 12)
        forward-simulation prior), the filter's posterior mean should
        land close to the true lambda=10.0, in terms of the filter's
        own reported posterior uncertainty -- not just "close in
        absolute terms by luck."
        """
        true_lambda = filter_result["true_lambda"]
        posterior_mean = filter_result["posterior_mean"]
        posterior_std = filter_result["posterior_std"]
        assert isinstance(true_lambda, float)
        assert isinstance(posterior_mean, float)
        assert isinstance(posterior_std, float)

        error_in_std_units = abs(posterior_mean - true_lambda) / posterior_std
        assert error_in_std_units < 3.0, (
            f"Posterior mean {posterior_mean:.3f} is "
            f"{error_in_std_units:.2f} std away from true lambda "
            f"{true_lambda} -- too far"
        )

    def test_posterior_std_is_positive_and_bounded(
        self, filter_result: FilterResultDict
    ) -> None:
        posterior_std = filter_result["posterior_std"]
        assert isinstance(posterior_std, float)
        assert posterior_std > 0.0
        # Should have shrunk well below the prior's own spread
        # (Uniform(5, 15) has std = (15-5)/sqrt(12) ~= 2.89)
        assert posterior_std < 2.89

    def test_ess_never_exceeds_N(
        self, filter_result: FilterResultDict
    ) -> None:
        result = filter_result["result"]
        assert hasattr(result, "ess_history")
        assert np.all(result.ess_history <= 2000.0 + 1e-6)
        assert np.all(result.ess_history >= 1.0)

    def test_result_shapes(self, filter_result: FilterResultDict) -> None:
        result = filter_result["result"]
        assert result.n_particles == 2000
        assert result.n_steps == 200
        assert result.means.shape == (200, 1)
        assert result.stds.shape == (200, 1)



# ---------------------------------------------------------------------------
# TestClinicalTrialFilterHelpers
# ---------------------------------------------------------------------------

from python.examples.week4_clinical_trial import (  # noqa: E402
    build_clinical_trial_priors,
)
from python.examples.week8_clinical_trial_filter import (  # noqa: E402
    ClinicalTrialFilterModel,
    build_trial_filter_log_likelihood,
    generate_synthetic_trial_data,
)


class TestClinicalTrialFilterHelpers:

    def test_generate_synthetic_trial_data_shapes(self) -> None:
        arm, outcome = generate_synthetic_trial_data(
            n_patients=50, true_p_treatment=0.45, true_p_control=0.30,
            randomisation_ratio=0.5, seed=1,
        )
        assert arm.shape == (50,)
        assert outcome.shape == (50,)

    def test_generate_synthetic_trial_data_reproducible(self) -> None:
        arm1, outcome1 = generate_synthetic_trial_data(
            n_patients=50, true_p_treatment=0.45, true_p_control=0.30,
            randomisation_ratio=0.5, seed=7,
        )
        arm2, outcome2 = generate_synthetic_trial_data(
            n_patients=50, true_p_treatment=0.45, true_p_control=0.30,
            randomisation_ratio=0.5, seed=7,
        )
        np.testing.assert_array_equal(arm1, arm2)
        np.testing.assert_array_equal(outcome1, outcome2)

    def test_generate_synthetic_trial_data_values_are_binary(self) -> None:
        arm, outcome = generate_synthetic_trial_data(
            n_patients=100, true_p_treatment=0.45, true_p_control=0.30,
            randomisation_ratio=0.5, seed=1,
        )
        assert set(np.unique(arm)).issubset({0.0, 1.0})
        assert set(np.unique(outcome)).issubset({0.0, 1.0})

    def test_clinical_trial_filter_model_state_dim(self) -> None:
        arm, outcome = generate_synthetic_trial_data(
            n_patients=10, true_p_treatment=0.45, true_p_control=0.30,
            randomisation_ratio=0.5, seed=1,
        )
        model = ClinicalTrialFilterModel(arm, outcome)
        assert model.state_dim == 4
        assert model.param_dim == 2

    def test_clinical_trial_filter_model_forward_batch_deterministic(
        self,
    ) -> None:
        """
        Unlike Week 4's ClinicalTrialModel (stochastic per particle),
        this model's forward_batch() must be fully DETERMINISTIC --
        all particles advance identically since real trial data is
        fully observed, not simulated.
        """
        arm, outcome = generate_synthetic_trial_data(
            n_patients=10, true_p_treatment=0.45, true_p_control=0.30,
            randomisation_ratio=0.5, seed=1,
        )
        model = ClinicalTrialFilterModel(arm, outcome)
        N = 50
        state = np.tile(model.initial_state(), (N, 1))
        params = np.zeros((N, 2))  # unused by this model's forward_batch
        new_state = model.forward_batch(state, params, dt=1.0)
        # Every particle's row should be identical.
        assert np.all(new_state == new_state[0])

    def test_clinical_trial_filter_model_counts_increment_by_one(
        self,
    ) -> None:
        arm, outcome = generate_synthetic_trial_data(
            n_patients=10, true_p_treatment=0.45, true_p_control=0.30,
            randomisation_ratio=0.5, seed=1,
        )
        model = ClinicalTrialFilterModel(arm, outcome)
        state = np.tile(model.initial_state(), (5, 1))
        params = np.zeros((5, 2))
        new_state = model.forward_batch(state, params, dt=1.0)
        total_before = state[0, 0] + state[0, 2]
        total_after = new_state[0, 0] + new_state[0, 2]
        assert total_after - total_before == 1.0


class TestClinicalTrialParticleFilter:

    @classmethod
    @pytest.fixture(scope="class")
    def trial_filter_result(cls) -> dict[str, float]:
        """
        Runs the full clinical trial filtering pipeline once, shared
        across tests in this class.
        """
        n_patients = 200
        true_p_treatment = 0.45
        true_p_control = 0.30

        arm, outcome = generate_synthetic_trial_data(
            n_patients=n_patients, true_p_treatment=true_p_treatment,
            true_p_control=true_p_control, randomisation_ratio=0.5,
            seed=42,
        )

        model = ClinicalTrialFilterModel(arm, outcome)
        priors = build_clinical_trial_priors(
            p_treatment_guess=0.45, p_control_guess=0.30,
        )
        pf = ParticleFilter(model, priors, N=2000, dt=1.0, seed=42)
        loglik = build_trial_filter_log_likelihood(pf, arm, outcome)

        observations = np.arange(n_patients, dtype=np.float64).reshape(-1, 1)
        result = pf.run(observations, loglik)

        final_p_treatment = pf.params[:, 0]
        final_p_control = pf.params[:, 1]
        final_weights = result.final_weights

        pf_prob = float(
            np.average(
                final_p_treatment > final_p_control, weights=final_weights
            )
        )

        final_state = result.final_state[0]
        final_n_treat, final_s_treat, final_n_ctrl, final_s_ctrl = final_state

        treat_alpha0, treat_beta0 = 0.45 * 20.0, (1.0 - 0.45) * 20.0
        ctrl_alpha0, ctrl_beta0 = 0.30 * 20.0, (1.0 - 0.30) * 20.0
        exact_rng = np.random.default_rng(123)
        n_mc = 20_000
        treat_samples = exact_rng.beta(
            treat_alpha0 + final_s_treat,
            treat_beta0 + (final_n_treat - final_s_treat),
            n_mc,
        )
        ctrl_samples = exact_rng.beta(
            ctrl_alpha0 + final_s_ctrl,
            ctrl_beta0 + (final_n_ctrl - final_s_ctrl),
            n_mc,
        )
        exact_prob = float(np.mean(treat_samples > ctrl_samples))

        return {
            "pf_prob": pf_prob,
            "exact_prob": exact_prob,
            "n_patients": float(n_patients),
        }

    def test_pf_matches_exact_batch_calculation(
        self, trial_filter_result: dict[str, float]
    ) -> None:
        """
        THE key validation: ParticleFilter's sequential posterior at
        the final patient should closely agree with the EXACT
        Beta-Binomial batch calculation on the same trial data, using
        the SAME informative per-arm priors (a common bug is
        comparing against mismatched priors, which was caught and
        fixed during this script's own development -- see the
        detailed comment in week8_clinical_trial_filter.py).
        """
        pf_prob = trial_filter_result["pf_prob"]
        exact_prob = trial_filter_result["exact_prob"]
        diff = abs(pf_prob - exact_prob)
        assert diff < 0.05, (
            f"PF prob {pf_prob:.4f} and exact prob {exact_prob:.4f} "
            f"differ by {diff:.4f} -- expected < 0.05"
        )

    def test_probabilities_are_valid(
        self, trial_filter_result: dict[str, float]
    ) -> None:
        assert 0.0 <= trial_filter_result["pf_prob"] <= 1.0
        assert 0.0 <= trial_filter_result["exact_prob"] <= 1.0
