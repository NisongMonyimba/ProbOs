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
