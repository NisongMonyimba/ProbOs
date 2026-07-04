"""
python/tests/test_distributions_properties.py

Property-based tests (Hypothesis) for Distribution subclasses and
ParticleFilter weight normalisation.

WHY PROPERTY-BASED TESTS
--------------------------
The existing test_distributions.py checks specific example-based cases
(e.g. "Normal(0, 1).pdf(0) equals 1/sqrt(2*pi)"). Property-based testing
instead checks that a MATHEMATICAL INVARIANT holds for hundreds of
randomly generated inputs per run, which catches edge cases a human
would never think to write by hand -- extreme parameter values, values
very close to distribution boundaries, very large or very small
magnitudes.

Per docs/standards/quality_standards.md Section 1: property-based tests
exist for mathematically load-bearing invariants -- distribution
log_pdf correctness, particle weight normalisation, percentile
ordering.

Hypothesis deliberately searches for the SMALLEST failing example when
it finds one (shrinking), which makes failures much easier to debug
than a randomly-seeded example-based test would produce.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from python.src.battery_model import BatteryModel2Cell
from python.src.distributions import Beta, LogNormal, Normal, Uniform
from python.src.parameter_priors import build_battery_priors
from python.src.particle_filter import ParticleFilter

# ---------------------------------------------------------------------------
# Hypothesis strategies for valid distribution parameters.
#
# Bounds are chosen to avoid numerically degenerate regions (e.g. sigma
# so close to 0 that log_pdf legitimately overflows, which is a
# floating-point limitation, not a correctness bug) while still
# covering a wide range including small and large magnitudes.
# ---------------------------------------------------------------------------

_finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_positive_floats = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_sample_sizes = st.integers(min_value=1, max_value=500)


# ---------------------------------------------------------------------------
# TestLogPdfMatchesLogOfPdf
#
# The core invariant every Distribution.log_pdf() implementation must
# satisfy: log_pdf(x) == log(pdf(x)) for any x where pdf(x) > 0.
# Distribution.log_pdf() exists specifically to be numerically stable
# where log(pdf(x)) would underflow -- but for x values where pdf(x)
# does NOT underflow, the two must agree.
# ---------------------------------------------------------------------------

class TestLogPdfMatchesLogOfPdf:

    @given(mu=_finite_floats, sigma=_positive_floats, x=_finite_floats)
    @settings(max_examples=200)
    def test_normal_log_pdf_matches_log_of_pdf(
        self, mu: float, sigma: float, x: float
    ) -> None:
        dist = Normal(mu=mu, sigma=sigma)
        pdf_val = dist.pdf(np.array([x]))[0]
        if pdf_val > 1e-300:  # avoid the underflow region log_pdf exists for
            log_pdf_val = dist.log_pdf(np.array([x]))[0]
            np.testing.assert_allclose(
                log_pdf_val, np.log(pdf_val), rtol=1e-6, atol=1e-6
            )

    @given(
        mu=st.floats(min_value=-10, max_value=10, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=5, allow_nan=False),
        x=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_lognormal_log_pdf_matches_log_of_pdf(
        self, mu: float, sigma: float, x: float
    ) -> None:
        dist = LogNormal(mu=mu, sigma=sigma)
        pdf_val = dist.pdf(np.array([x]))[0]
        if pdf_val > 1e-300:
            log_pdf_val = dist.log_pdf(np.array([x]))[0]
            np.testing.assert_allclose(
                log_pdf_val, np.log(pdf_val), rtol=1e-6, atol=1e-6
            )

    @given(
        low=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
        span=st.floats(min_value=1e-3, max_value=1e4, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_uniform_log_pdf_matches_log_of_pdf(
        self, low: float, span: float
    ) -> None:
        high = low + span
        dist = Uniform(low=low, high=high)
        x = low + span / 2.0  # a point guaranteed inside [low, high]
        pdf_val = dist.pdf(np.array([x]))[0]
        if pdf_val > 1e-300:
            log_pdf_val = dist.log_pdf(np.array([x]))[0]
            np.testing.assert_allclose(
                log_pdf_val, np.log(pdf_val), rtol=1e-6, atol=1e-6
            )

    @given(
        alpha=st.floats(min_value=0.1, max_value=50, allow_nan=False),
        beta=st.floats(min_value=0.1, max_value=50, allow_nan=False),
        x=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_beta_log_pdf_matches_log_of_pdf(
        self, alpha: float, beta: float, x: float
    ) -> None:
        dist = Beta(alpha=alpha, beta=beta)
        pdf_val = dist.pdf(np.array([x]))[0]
        if pdf_val > 1e-300:
            log_pdf_val = dist.log_pdf(np.array([x]))[0]
            np.testing.assert_allclose(
                log_pdf_val, np.log(pdf_val), rtol=1e-6, atol=1e-6
            )


# ---------------------------------------------------------------------------
# TestSampleWithinSupport
#
# Every sample drawn from a distribution must lie within that
# distribution's mathematical support -- Normal/LogNormal have no
# bound checks (support is (-inf, inf) or (0, inf)), but Uniform and
# Beta have hard bounds that sampling must never violate regardless of
# the random seed.
# ---------------------------------------------------------------------------

class TestSampleWithinSupport:

    @given(
        low=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
        span=st.floats(min_value=1e-3, max_value=1e4, allow_nan=False),
        n=_sample_sizes,
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_uniform_samples_within_bounds(
        self, low: float, span: float, n: int, seed: int
    ) -> None:
        high = low + span
        dist = Uniform(low=low, high=high)
        rng = np.random.default_rng(seed)
        samples = dist.sample(n, rng=rng)
        assert np.all(samples >= low)
        assert np.all(samples <= high)

    @given(
        alpha=st.floats(min_value=0.1, max_value=50, allow_nan=False),
        beta=st.floats(min_value=0.1, max_value=50, allow_nan=False),
        n=_sample_sizes,
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_beta_samples_within_unit_interval(
        self, alpha: float, beta: float, n: int, seed: int
    ) -> None:
        dist = Beta(alpha=alpha, beta=beta)
        rng = np.random.default_rng(seed)
        samples = dist.sample(n, rng=rng)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    @given(
        mu=st.floats(min_value=-10, max_value=10, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=5, allow_nan=False),
        n=_sample_sizes,
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_lognormal_samples_always_positive(
        self, mu: float, sigma: float, n: int, seed: int
    ) -> None:
        dist = LogNormal(mu=mu, sigma=sigma)
        rng = np.random.default_rng(seed)
        samples = dist.sample(n, rng=rng)
        assert np.all(samples > 0.0)


# ---------------------------------------------------------------------------
# TestConstructorValidation
#
# Invalid parameters must always raise ValueError, for any input in
# the invalid region -- not just the specific examples in
# test_distributions.py.
# ---------------------------------------------------------------------------

class TestConstructorValidation:

    @given(mu=_finite_floats, sigma=st.floats(max_value=0.0, allow_nan=False))
    @settings(max_examples=100)
    def test_normal_nonpositive_sigma_always_raises(
        self, mu: float, sigma: float
    ) -> None:
        with pytest.raises(ValueError):
            Normal(mu=mu, sigma=sigma)

    @given(
        low=_finite_floats,
        high=_finite_floats,
    )
    @settings(max_examples=100)
    def test_uniform_low_ge_high_always_raises(
        self, low: float, high: float
    ) -> None:
        if low >= high:
            with pytest.raises(ValueError):
                Uniform(low=low, high=high)

    @given(
        alpha=st.floats(max_value=0.0, allow_nan=False),
        beta=_positive_floats,
    )
    @settings(max_examples=100)
    def test_beta_nonpositive_alpha_always_raises(
        self, alpha: float, beta: float
    ) -> None:
        with pytest.raises(ValueError):
            Beta(alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# TestParticleFilterWeightNormalisation
#
# Per docs/standards/quality_standards.md: particle weight
# normalisation is a mathematically load-bearing invariant. Weights
# must sum to 1.0 after ANY sequence of update() calls, for any
# reasonable number of particles and any (finite) observation values.
# ---------------------------------------------------------------------------

class TestParticleFilterWeightNormalisation:

    @given(
        N=st.integers(min_value=2, max_value=300),
        obs_value=st.floats(
            min_value=350.0, max_value=500.0, allow_nan=False
        ),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50, deadline=None)
    def test_weights_sum_to_one_after_update(
        self, N: int, obs_value: float, seed: int
    ) -> None:
        model = BatteryModel2Cell()
        priors = build_battery_priors()
        pf = ParticleFilter(model, priors, N=N, dt=1.0, seed=seed)
        pf.predict()

        def loglik(state: np.ndarray, obs: np.ndarray) -> np.ndarray:
            T1 = state[:, 0]
            sigma_obs = 5.0
            result: np.ndarray = -0.5 * ((T1 - obs[0]) / sigma_obs) ** 2
            return result

        pf.update(np.array([obs_value]), loglik)
        assert abs(pf.weights.sum() - 1.0) < 1e-8

    @given(
        N=st.integers(min_value=2, max_value=300),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50, deadline=None)
    def test_ess_always_between_1_and_N(self, N: int, seed: int) -> None:
        model = BatteryModel2Cell()
        priors = build_battery_priors()
        pf = ParticleFilter(model, priors, N=N, dt=1.0, seed=seed)
        pf.predict()

        def loglik(state: np.ndarray, obs: np.ndarray) -> np.ndarray:
            T1 = state[:, 0]
            sigma_obs = 5.0
            result: np.ndarray = -0.5 * ((T1 - obs[0]) / sigma_obs) ** 2
            return result

        pf.update(np.array([420.0]), loglik)
        ess = pf.effective_sample_size()
        assert 1.0 - 1e-6 <= ess <= float(N) + 1e-6

    @given(
        N=st.integers(min_value=2, max_value=200),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50, deadline=None)
    def test_resample_always_restores_uniform_weights(
        self, N: int, seed: int
    ) -> None:
        model = BatteryModel2Cell()
        priors = build_battery_priors()
        pf = ParticleFilter(model, priors, N=N, dt=1.0, seed=seed)
        pf.predict()

        def loglik(state: np.ndarray, obs: np.ndarray) -> np.ndarray:
            T1 = state[:, 0]
            sigma_obs = 5.0
            result: np.ndarray = -0.5 * ((T1 - obs[0]) / sigma_obs) ** 2
            return result

        pf.update(np.array([420.0]), loglik)
        pf.resample()
        np.testing.assert_allclose(
            pf.weights, np.full(N, 1.0 / N), rtol=1e-8
        )


# ---------------------------------------------------------------------------
# TestMonteCarloPercentileOrdering
#
# P05 <= P50 <= P95 must hold pointwise for every state variable, at
# every timestep, for any valid N and seed -- this was verified
# example-based in test_monte_carlo.py; here it is verified as a
# property across many (N, seed) combinations.
# ---------------------------------------------------------------------------

class TestMonteCarloPercentileOrdering:

    @given(
        N=st.integers(min_value=10, max_value=500),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_percentile_ordering_holds_for_any_seed(
        self, N: int, seed: int
    ) -> None:
        from python.src.monte_carlo import MonteCarloEngine

        model = BatteryModel2Cell()
        priors = build_battery_priors()
        engine = MonteCarloEngine(
            model, priors, N=N, n_steps=5, dt=1.0, seed=seed
        )
        result = engine.run()
        p05 = result.percentiles[0]
        p50 = result.percentiles[1]
        p95 = result.percentiles[2]
        assert np.all(p05 <= p50 + 1e-9)
        assert np.all(p50 <= p95 + 1e-9)
