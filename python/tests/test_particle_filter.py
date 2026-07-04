"""
python/tests/test_particle_filter.py

Validates ParticleFilter against ground truth in two stages:

  1. UNIT TESTS: construction, predict/update/resample mechanics,
     weight normalisation, ESS computation -- verify the machinery
     itself is correct in isolation.

  2. VALIDATION: a 1D linear-Gaussian random walk with Gaussian
     observation noise has an EXACT closed-form solution via the
     Kalman filter. We run the particle filter on synthetic data from
     this model and assert its posterior mean/variance converge to
     the Kalman filter's exact values as N grows. This is the same
     "validate against a known analytical case before touching the
     real model" discipline used for BatteryModel2Cell against Kim
     2007 ARC data in Month 1.

Kalman filter recursion (textbook, e.g. Sarkka 2013 "Bayesian
Filtering and Smoothing" Ch 4), used here purely as ground truth,
not imported from any library -- implemented directly and minimally:

    Predict:  m_pred = m_{t-1}
              P_pred = P_{t-1} + Q
    Update:   S = P_pred + R
              K = P_pred / S
              m_t = m_pred + K * (y_t - m_pred)
              P_t = (1 - K) * P_pred
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import numpy as np
import pytest

from python.src.distributions import Normal
from python.src.particle_filter import ParticleFilter
from python.src.state import FloatArray, Model

# ---------------------------------------------------------------------------
# A minimal linear-Gaussian random walk Model, used ONLY for validation.
# State: [x] -- a single scalar. No uncertain parameters (param_dim=0 is
# not allowed by the existing param_dim>=1 conventions elsewhere in the
# codebase, so we use one fixed "process noise std" parameter with a
# near-degenerate prior, keeping the interface consistent with every
# other Model subclass in the project).
# ---------------------------------------------------------------------------

class SyntheticData(TypedDict):
    """Typed structure returned by the synthetic_data fixture."""
    T: int
    Q: float
    R: float
    x0: float
    true_states: FloatArray
    observations: FloatArray
    kf_means: FloatArray
    kf_variances: FloatArray


class RandomWalkModel(Model):
    """
    x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)

    STATE VECTOR (state_dim=1): state[:, 0] = x
    PARAMETER VECTOR (param_dim=1): params[:, 0] = sqrt(Q), the process
    noise standard deviation. Held essentially fixed via a tight prior
    (see build_random_walk_priors) since this model exists purely to
    validate the particle filter machinery against a known Kalman
    filter solution, not to demonstrate parameter-uncertainty
    propagation (that role is already covered by the Week 4 examples).
    """

    def __init__(self, x0: float = 0.0, seed: int = 0) -> None:
        self._x0 = x0
        self._rng = np.random.default_rng(seed)

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def param_dim(self) -> int:
        return 1

    def param_names(self) -> list[str]:
        return ["process_noise_std"]

    def initial_state(self) -> FloatArray:
        return np.array([self._x0], dtype=np.float64)

    def forward_batch(
        self, state: FloatArray, params: FloatArray, dt: float
    ) -> FloatArray:
        x = state[:, 0]
        q_std = params[:, 0]
        N = x.shape[0]
        w = self._rng.standard_normal(N) * q_std
        new_x = x + w
        return new_x.reshape(N, 1)


def build_random_walk_priors(q_std: float = 1.0) -> list[Normal]:
    """Near-degenerate prior on process noise std -- fixed at q_std."""
    return [Normal(mu=q_std, sigma=1e-9)]


def kalman_filter_1d(
    observations: FloatArray,
    x0: float,
    P0: float,
    Q: float,
    R: float,
) -> tuple[FloatArray, FloatArray]:
    """
    Exact closed-form Kalman filter for the 1D random walk + Gaussian
    observation model. Used ONLY as ground truth for validating
    ParticleFilter -- not part of the ProbOS kernel itself.

    Parameters
    ----------
    observations : shape (T,)
    x0 : prior mean at t=0
    P0 : prior variance at t=0
    Q  : process noise variance
    R  : observation noise variance

    Returns
    -------
    means : shape (T,) -- exact posterior mean at each step
    variances : shape (T,) -- exact posterior variance at each step
    """
    T = observations.shape[0]
    means = np.empty(T, dtype=np.float64)
    variances = np.empty(T, dtype=np.float64)

    m, P = x0, P0
    for t in range(T):
        # Predict
        m_pred = m
        P_pred = P + Q
        # Update
        S = P_pred + R
        K = P_pred / S
        m = m_pred + K * (observations[t] - m_pred)
        P = (1.0 - K) * P_pred

        means[t] = m
        variances[t] = P

    return means, variances


def gaussian_log_likelihood(
    sigma_obs: float,
) -> Callable[[FloatArray, FloatArray], FloatArray]:
    """
    Builds a log_likelihood_fn for ParticleFilter.update(): Gaussian
    observation noise with fixed known standard deviation sigma_obs on
    the state's first (and only) component.
    """
    def _fn(state: FloatArray, obs: FloatArray) -> FloatArray:
        x = state[:, 0]
        log_lik: FloatArray = -0.5 * ((x - obs[0]) / sigma_obs) ** 2 - np.log(
            sigma_obs * np.sqrt(2.0 * np.pi)
        )
        return log_lik
    return _fn


# ---------------------------------------------------------------------------
# TestParticleFilterConstruction
# ---------------------------------------------------------------------------

class TestParticleFilterConstruction:

    def test_valid_construction(self) -> None:
        model = RandomWalkModel()
        priors = build_random_walk_priors()
        pf = ParticleFilter(model, priors, N=100)
        assert pf.N == 100

    def test_wrong_prior_count_raises(self) -> None:
        model = RandomWalkModel()
        with pytest.raises(ValueError, match="param_dim"):
            ParticleFilter(model, priors=[], N=100)

    def test_N_zero_raises(self) -> None:
        model = RandomWalkModel()
        priors = build_random_walk_priors()
        with pytest.raises(ValueError, match="N must be"):
            ParticleFilter(model, priors, N=0)

    def test_dt_zero_raises(self) -> None:
        model = RandomWalkModel()
        priors = build_random_walk_priors()
        with pytest.raises(ValueError, match="dt must be"):
            ParticleFilter(model, priors, N=10, dt=0.0)

    def test_bad_resample_threshold_raises(self) -> None:
        model = RandomWalkModel()
        priors = build_random_walk_priors()
        with pytest.raises(ValueError, match="resample_threshold"):
            ParticleFilter(model, priors, N=10, resample_threshold=1.5)

    def test_initial_state_shape(self) -> None:
        model = RandomWalkModel()
        priors = build_random_walk_priors()
        pf = ParticleFilter(model, priors, N=50)
        assert pf.state.shape == (50, 1)

    def test_initial_weights_uniform_and_normalised(self) -> None:
        model = RandomWalkModel()
        priors = build_random_walk_priors()
        pf = ParticleFilter(model, priors, N=50)
        w = pf.weights
        np.testing.assert_allclose(w, np.full(50, 1.0 / 50), rtol=1e-10)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_initial_ess_equals_N(self) -> None:
        model = RandomWalkModel()
        priors = build_random_walk_priors()
        pf = ParticleFilter(model, priors, N=50)
        assert abs(pf.effective_sample_size() - 50.0) < 1e-6


# ---------------------------------------------------------------------------
# TestParticleFilterMechanics
# ---------------------------------------------------------------------------

class TestParticleFilterMechanics:

    def test_predict_changes_state(self) -> None:
        model = RandomWalkModel(x0=0.0)
        priors = build_random_walk_priors(q_std=1.0)
        pf = ParticleFilter(model, priors, N=200, seed=1)
        before = pf.state.copy()
        pf.predict()
        assert not np.allclose(before, pf.state)

    def test_update_reduces_ess_when_informative(self) -> None:
        """
        A tight observation likelihood should discriminate between
        particles, reducing ESS below N.
        """
        model = RandomWalkModel(x0=0.0)
        priors = build_random_walk_priors(q_std=5.0)
        pf = ParticleFilter(model, priors, N=500, seed=2)
        pf.predict()
        loglik = gaussian_log_likelihood(sigma_obs=0.5)
        pf.update(np.array([10.0]), loglik)
        assert pf.effective_sample_size() < 500

    def test_resample_restores_ess_to_N(self) -> None:
        model = RandomWalkModel(x0=0.0)
        priors = build_random_walk_priors(q_std=5.0)
        pf = ParticleFilter(model, priors, N=300, seed=3)
        pf.predict()
        loglik = gaussian_log_likelihood(sigma_obs=0.5)
        pf.update(np.array([10.0]), loglik)
        pf.resample()
        assert abs(pf.effective_sample_size() - 300.0) < 1e-6

    def test_resample_preserves_weight_normalisation(self) -> None:
        model = RandomWalkModel(x0=0.0)
        priors = build_random_walk_priors(q_std=5.0)
        pf = ParticleFilter(model, priors, N=300, seed=4)
        pf.predict()
        loglik = gaussian_log_likelihood(sigma_obs=0.5)
        pf.update(np.array([10.0]), loglik)
        pf.resample()
        assert abs(pf.weights.sum() - 1.0) < 1e-10

    def test_posterior_mean_shape(self) -> None:
        model = RandomWalkModel()
        priors = build_random_walk_priors()
        pf = ParticleFilter(model, priors, N=100)
        mean = pf.posterior_mean()
        assert mean.shape == (1,)

    def test_posterior_std_nonnegative(self) -> None:
        model = RandomWalkModel(x0=0.0)
        priors = build_random_walk_priors(q_std=2.0)
        pf = ParticleFilter(model, priors, N=200, seed=5)
        pf.predict()
        assert np.all(pf.posterior_std() >= 0.0)


# ---------------------------------------------------------------------------
# TestKalmanFilterGroundTruth
#
# The critical validation: does ParticleFilter converge to the exact
# analytical solution on a problem where that solution is known?
# ---------------------------------------------------------------------------

class TestKalmanFilterGroundTruth:

    @classmethod
    @pytest.fixture(scope="class")
    def synthetic_data(cls) -> SyntheticData:
        """
        Generate one fixed synthetic trajectory + noisy observations
        from the true random-walk model, used by all tests in this
        class so they validate against the SAME ground truth.
        """
        rng = np.random.default_rng(123)
        T = 50
        Q_true = 1.0   # process noise variance
        R_true = 4.0   # observation noise variance
        x0 = 0.0

        true_states = np.empty(T)
        x = x0
        for t in range(T):
            x = x + rng.standard_normal() * np.sqrt(Q_true)
            true_states[t] = x

        observations = true_states + rng.standard_normal(T) * np.sqrt(R_true)

        kf_means, kf_variances = kalman_filter_1d(
            observations, x0=x0, P0=1e-6, Q=Q_true, R=R_true
        )

        result: SyntheticData = {
            "T": T, "Q": Q_true, "R": R_true, "x0": x0,
            "true_states": true_states,
            "observations": observations,
            "kf_means": kf_means,
            "kf_variances": kf_variances,
        }
        return result

    def test_kalman_filter_tracks_true_state(
        self, synthetic_data: SyntheticData
    ) -> None:
        """Sanity check on the ground-truth Kalman filter itself."""
        rmse = np.sqrt(np.mean(
            (synthetic_data["kf_means"] - synthetic_data["true_states"]) ** 2
        ))
        # With R=4 (obs std=2), RMSE should be well under the raw
        # observation noise std, since filtering reduces uncertainty.
        assert rmse < 2.0

    def test_particle_filter_mean_matches_kalman_at_N1000(
        self, synthetic_data: SyntheticData
    ) -> None:
        """
        The core validation: with a large particle count, the SIR
        particle filter's posterior mean trajectory should closely
        track the EXACT Kalman filter posterior mean trajectory.
        """
        model = RandomWalkModel(x0=synthetic_data["x0"], seed=999)
        priors = build_random_walk_priors(
            q_std=float(np.sqrt(synthetic_data["Q"]))
        )
        pf = ParticleFilter(
            model, priors, N=1000, dt=1.0,
            resample_threshold=0.5, seed=999,
        )
        loglik = gaussian_log_likelihood(
            sigma_obs=float(np.sqrt(synthetic_data["R"]))
        )
        result = pf.run(
            synthetic_data["observations"].reshape(-1, 1), loglik
        )

        pf_means = result.means[:, 0]
        kf_means = synthetic_data["kf_means"]

        # RMSE between PF and exact KF posterior means should be small
        # relative to the KF's own posterior uncertainty (sqrt of its
        # variance) -- i.e. the particle filter's Monte Carlo error is
        # small compared to the genuine statistical uncertainty in the
        # problem, not just "close in absolute terms by luck."
        kf_std = np.sqrt(synthetic_data["kf_variances"])
        rmse = np.sqrt(np.mean((pf_means - kf_means) ** 2))
        mean_kf_std = float(np.mean(kf_std))

        assert rmse < 0.5 * mean_kf_std, (
            f"PF-KF RMSE={rmse:.4f} not small relative to "
            f"mean KF posterior std={mean_kf_std:.4f}"
        )

    def test_particle_filter_std_matches_kalman_at_N1000(
        self, synthetic_data: SyntheticData
    ) -> None:
        """
        The particle filter's own reported posterior std should be in
        the right ballpark of the exact Kalman filter posterior std --
        confirms the filter is not just tracking the mean correctly
        but also reporting sane uncertainty.
        """
        model = RandomWalkModel(x0=synthetic_data["x0"], seed=999)
        priors = build_random_walk_priors(
            q_std=float(np.sqrt(synthetic_data["Q"]))
        )
        pf = ParticleFilter(
            model, priors, N=1000, dt=1.0,
            resample_threshold=0.5, seed=999,
        )
        loglik = gaussian_log_likelihood(
            sigma_obs=float(np.sqrt(synthetic_data["R"]))
        )
        result = pf.run(
            synthetic_data["observations"].reshape(-1, 1), loglik
        )

        pf_stds = result.stds[:, 0]
        kf_stds = np.sqrt(synthetic_data["kf_variances"])

        # Compare only the second half of the trajectory (after the
        # filter has "warmed up" past initial transients) using a
        # generous relative tolerance -- PF variance estimation is
        # inherently noisier than its mean estimation.
        half = len(pf_stds) // 2
        ratio = pf_stds[half:] / kf_stds[half:]
        assert np.all(ratio > 0.3), "PF std collapsed far below KF std"
        assert np.all(ratio < 3.0), "PF std far exceeds KF std"

    def test_particle_filter_converges_as_N_increases(
        self, synthetic_data: SyntheticData
    ) -> None:
        """
        Monte Carlo error should shrink as N grows: RMSE against the
        exact Kalman filter mean at N=100 should exceed RMSE at
        N=2000, following the general 1/sqrt(N) Monte Carlo
        convergence behaviour established for MonteCarloEngine in
        Month 1 Week 3.
        """
        def rmse_at_N(N: int, seed: int) -> float:
            model = RandomWalkModel(x0=synthetic_data["x0"], seed=seed)
            priors = build_random_walk_priors(
                q_std=float(np.sqrt(synthetic_data["Q"]))
            )
            pf = ParticleFilter(
                model, priors, N=N, dt=1.0,
                resample_threshold=0.5, seed=seed,
            )
            loglik = gaussian_log_likelihood(
                sigma_obs=float(np.sqrt(synthetic_data["R"]))
            )
            result = pf.run(
                synthetic_data["observations"].reshape(-1, 1), loglik
            )
            return float(np.sqrt(np.mean(
                (result.means[:, 0] - synthetic_data["kf_means"]) ** 2
            )))

        rmse_small = rmse_at_N(N=100, seed=42)
        rmse_large = rmse_at_N(N=2000, seed=42)

        assert rmse_large < rmse_small, (
            f"RMSE did not shrink with N: N=100 RMSE={rmse_small:.4f}, "
            f"N=2000 RMSE={rmse_large:.4f}"
        )

    def test_ess_history_never_exceeds_N(
        self, synthetic_data: SyntheticData
    ) -> None:
        model = RandomWalkModel(x0=synthetic_data["x0"], seed=7)
        priors = build_random_walk_priors(
            q_std=float(np.sqrt(synthetic_data["Q"]))
        )
        pf = ParticleFilter(model, priors, N=500, seed=7)
        loglik = gaussian_log_likelihood(
            sigma_obs=float(np.sqrt(synthetic_data["R"]))
        )
        result = pf.run(
            synthetic_data["observations"].reshape(-1, 1), loglik
        )
        assert np.all(result.ess_history <= 500.0 + 1e-6)
        assert np.all(result.ess_history >= 1.0)

    def test_result_shapes(
        self, synthetic_data: SyntheticData
    ) -> None:
        model = RandomWalkModel(x0=synthetic_data["x0"], seed=8)
        priors = build_random_walk_priors(
            q_std=float(np.sqrt(synthetic_data["Q"]))
        )
        pf = ParticleFilter(model, priors, N=100, seed=8)
        loglik = gaussian_log_likelihood(
            sigma_obs=float(np.sqrt(synthetic_data["R"]))
        )
        result = pf.run(
            synthetic_data["observations"].reshape(-1, 1), loglik
        )
        T = synthetic_data["T"]
        assert result.means.shape == (T, 1)
        assert result.stds.shape == (T, 1)
        assert result.ess_history.shape == (T,)
        assert result.final_state.shape == (100, 1)
        assert result.final_weights.shape == (100,)
        assert result.n_particles == 100
        assert result.n_steps == T
