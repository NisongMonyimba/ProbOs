"""
================================================================================
python/tests/test_distributions.py — pytest test suite for distributions.py
================================================================================

HOW TO RUN THESE TESTS:
-----------------------
  pytest python/tests/test_distributions.py -v

  # Or run all tests with coverage report:
  pytest python/tests/ -v --cov=python/src --cov-report=term-missing

TESTING PHILOSOPHY:
-------------------
We test MATHEMATICAL PROPERTIES, not just "does the code run without error".

Bad test:
    def test_normal_samples():
        d = Normal(0.0, 1.0)
        s = d.sample(10)
        assert len(s) == 10  # this tells us almost nothing

Good test:
    def test_normal_mean_converges():
        d = Normal(mu=5.0, sigma=2.0)
        s = d.sample(100_000, rng=np.random.default_rng(42))
        tol = 3 * 2.0 / np.sqrt(100_000)  # 3-sigma LLN bound
        assert abs(s.mean() - 5.0) < tol  # tests the Law of Large Numbers

The second test is checking a real mathematical theorem (LLN). If it fails,
something is fundamentally wrong with the implementation.

WHAT IS pytest?
---------------
pytest is a test runner. It:
  1. Finds all files named test_*.py
  2. Finds all functions named test_*
  3. Runs each function
  4. Reports PASSED if the function completes without error
  5. Reports FAILED if the function raises an exception or an assertion fails

WHAT IS assert?
---------------
assert condition  raises AssertionError if condition is False.
pytest catches AssertionError and reports it as a test failure.
We also use np.testing.assert_allclose and pytest.raises for richer checks.
================================================================================
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# The module under test
from python.src.distributions import (
    Beta,
    Distribution,
    Empirical,
    LogNormal,
    Normal,
    Uniform,
)

# ==============================================================================
# TESTS FOR: Normal distribution
# ==============================================================================

class TestNormalConstruction:
    """Tests for Normal.__init__() — verifies constructor validation."""

    def test_valid_parameters_do_not_raise(self) -> None:
        """Normal distributions with valid parameters should not raise."""
        Normal(mu=0.0,    sigma=1.0)    # standard normal
        Normal(mu=-100.0, sigma=0.001)  # small sigma, negative mu
        Normal(mu=1.35e5, sigma=5.0e3) # battery Ea_SEI typical values

    def test_negative_sigma_raises_ValueError(self) -> None:
        """sigma must be positive — negative sigma should raise ValueError."""
        with pytest.raises(ValueError, match="sigma"):
            Normal(mu=0.0, sigma=-1.0)

    def test_zero_sigma_raises_ValueError(self) -> None:
        """sigma = 0 is not valid (degenerate distribution)."""
        with pytest.raises(ValueError, match="sigma"):
            Normal(mu=0.0, sigma=0.0)

    def test_accessors_return_constructor_arguments(self) -> None:
        """After construction, mu and sigma should match what we passed in."""
        d = Normal(mu=3.7, sigma=1.2)
        assert d.mu    == 3.7, f"mu should be 3.7, got {d.mu}"
        assert d.sigma == 1.2, f"sigma should be 1.2, got {d.sigma}"

    def test_mean_equals_mu(self) -> None:
        """The analytical mean of Normal(mu, sigma^2) is mu."""
        d = Normal(mu=7.5, sigma=2.0)
        assert d.mean() == 7.5

    def test_variance_equals_sigma_squared(self) -> None:
        """The variance of Normal(mu, sigma^2) is sigma^2."""
        d = Normal(mu=0.0, sigma=3.0)
        assert d.variance() == 9.0


class TestNormalSampling:
    """Tests for Normal.sample() — verifies statistical properties."""

    def test_sample_returns_correct_shape(self) -> None:
        """sample(n) must return exactly n values."""
        d = Normal(mu=0.0, sigma=1.0)
        for n in [1, 10, 100, 1000]:
            result = d.sample(n)
            assert result.shape == (n,), f"Expected shape ({n},), got {result.shape}"

    def test_sample_mean_obeys_law_of_large_numbers(self) -> None:
        """
        LAW OF LARGE NUMBERS:
        The sample mean converges to the true mean at rate 1/sqrt(N).
        The 3-sigma tolerance is: 3 * sigma / sqrt(N).
        """
        d   = Normal(mu=5.0, sigma=2.0)
        rng = np.random.default_rng(42)
        s   = d.sample(100_000, rng=rng)

        # The tolerance is the 3-sigma bound on |x_bar - mu|
        # With N=100,000 and sigma=2.0: tol = 3 * 2.0 / sqrt(100000) ≈ 0.019
        tolerance = 3.0 * 2.0 / np.sqrt(100_000)
        emp_mean  = s.mean()

        assert abs(emp_mean - 5.0) < tolerance, (
            f"Empirical mean {emp_mean:.4f} is too far from true mean 5.0. "
            f"Tolerance (3-sigma LLN): {tolerance:.4f}"
        )

    def test_sample_std_converges_to_sigma(self) -> None:
        """The empirical standard deviation must converge to sigma."""
        d   = Normal(mu=0.0, sigma=3.0)
        rng = np.random.default_rng(99)
        s   = d.sample(100_000, rng=rng)

        emp_std = s.std(ddof=1)
        assert abs(emp_std - 3.0) < 0.05, (
            f"Empirical std {emp_std:.4f} is too far from true sigma 3.0"
        )

    def test_sample_reproducible_with_same_seed(self) -> None:
        """Same seed must produce identical samples every time."""
        d  = Normal(mu=0.0, sigma=1.0)
        s1 = d.sample(50, rng=np.random.default_rng(77))
        s2 = d.sample(50, rng=np.random.default_rng(77))
        assert_allclose(s1, s2, rtol=0, atol=0,
                        err_msg="Different seeds produce different results")

    def test_sample_dtype_is_float64(self) -> None:
        """Samples must be float64 arrays for numerical precision."""
        d      = Normal(mu=0.0, sigma=1.0)
        result = d.sample(100)
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"


class TestNormalDensity:
    """Tests for Normal.pdf() and Normal.log_pdf()."""

    def test_pdf_integrates_to_one(self) -> None:
        """
        PROBABILITY AXIOM: Total probability must be 1.
        We verify this numerically using the trapezoidal rule.
        The Normal distribution is effectively zero beyond 8 sigma from the mean.
        """
        from scipy.integrate import quad

        d        = Normal(mu=2.0, sigma=1.5)
        integral, error = quad(
            lambda x: float(d.pdf(np.array([x]))[0]),
            d.mu - 8 * d.sigma,
            d.mu + 8 * d.sigma,
        )
        assert abs(integral - 1.0) < 1e-6, (
            f"PDF should integrate to 1.0, got {integral:.8f}"
        )

    def test_pdf_is_positive_everywhere(self) -> None:
        """The Normal PDF must be positive on the entire real line."""
        d = Normal(mu=0.0, sigma=1.0)
        x = np.linspace(-10.0, 10.0, 100)
        assert np.all(d.pdf(x) > 0.0), "pdf should be positive everywhere"

    def test_pdf_maximum_is_at_mean(self) -> None:
        """The Normal PDF is maximised at x = mu."""
        d        = Normal(mu=3.0, sigma=2.0)
        peak     = d.pdf(np.array([d.mu]))[0]
        offsets  = np.array([0.5, 1.0, 2.0, 3.0])
        for offset in offsets:
            assert peak > d.pdf(np.array([d.mu + offset]))[0]
            assert peak > d.pdf(np.array([d.mu - offset]))[0]

    def test_pdf_is_symmetric(self) -> None:
        """Normal PDF is symmetric: f(mu + d) == f(mu - d) for all d > 0."""
        d = Normal(mu=2.0, sigma=1.5)
        offsets = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
        above = d.pdf(d.mu + offsets)
        below = d.pdf(d.mu - offsets)
        assert_allclose(above, below, rtol=1e-15,
                        err_msg="PDF must be symmetric around the mean")

    def test_log_pdf_is_analytically_consistent_with_pdf(self) -> None:
        """
        exp(log_pdf(x)) must equal pdf(x) to machine precision.
        This verifies the analytical formula is correct — NOT log(pdf(x)).
        """
        d = Normal(mu=1.0, sigma=2.0)
        x = np.linspace(-5.0, 7.0, 50)

        via_log = np.exp(d.log_pdf(x))
        direct  = d.pdf(x)

        assert_allclose(via_log, direct, rtol=1e-10,
                        err_msg="exp(log_pdf(x)) must equal pdf(x)")

    def test_log_pdf_is_finite_at_extreme_values(self) -> None:
        """
        NUMERICAL STABILITY TEST (the most important test in this file):
        For x very far from the mean, pdf(x) underflows to 0.0.
        But log_pdf(x) must remain FINITE (a large negative number).
        This is why we implement log_pdf analytically instead of as log(pdf(x)).
        """
        d = Normal(mu=0.0, sigma=1.0)

        # x = 50 sigma from the mean — pdf(50) should underflow to 0.0
        x_extreme = np.array([50.0])

        pdf_val    = d.pdf(x_extreme)[0]
        logpdf_val = d.log_pdf(x_extreme)[0]

        # pdf SHOULD underflow (this is expected and correct)
        assert pdf_val == 0.0, (
            f"pdf(50*sigma) should underflow to 0.0, got {pdf_val}"
        )

        # log_pdf MUST remain finite (this is what we are testing)
        assert np.isfinite(logpdf_val), (
            f"log_pdf(50*sigma) must be finite, got {logpdf_val}. "
            f"If this fails, you used log(pdf(x)) instead of the analytical form."
        )

        # log_pdf should be a large negative number
        assert logpdf_val < -100.0, (
            f"log_pdf(50*sigma) should be very negative, got {logpdf_val}"
        )


class TestNormalPPF:
    """Tests for Normal.ppf() — the inverse CDF."""

    def test_ppf_is_inverse_of_cdf(self) -> None:
        """
        ppf(u) must satisfy: P(X <= ppf(u)) = u.
        We verify this using scipy's Normal CDF.
        """
        from scipy.stats import norm

        d = Normal(mu=2.0, sigma=1.5)
        u = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])

        quantiles    = d.ppf(u)
        cdf_at_quant = norm.cdf(quantiles, loc=2.0, scale=1.5)

        assert_allclose(cdf_at_quant, u, atol=1e-10,
                        err_msg="ppf must be the inverse of the CDF")

    def test_ppf_median_equals_mean(self) -> None:
        """For a symmetric distribution, the 50th percentile equals the mean."""
        d = Normal(mu=7.0, sigma=3.0)
        median = d.ppf(np.array([0.5]))[0]
        assert abs(median - 7.0) < 1e-10


# ==============================================================================
# TESTS FOR: LogNormal distribution
# ==============================================================================

class TestLogNormal:

    def test_all_samples_positive(self) -> None:
        """LogNormal samples must ALL be strictly positive."""
        d       = LogNormal(mu=0.0, sigma=1.0)
        samples = d.sample(10_000, rng=np.random.default_rng(0))
        assert np.all(samples > 0.0), (
            "ALL LogNormal samples must be positive"
        )

    def test_mean_matches_analytical_formula(self) -> None:
        """E[X] = exp(mu + sigma^2/2)"""
        mu, sigma = 1.0, 0.5
        d         = LogNormal(mu=mu, sigma=sigma)
        samples   = d.sample(200_000, rng=np.random.default_rng(1))

        theoretical = np.exp(mu + 0.5 * sigma ** 2)
        empirical   = samples.mean()

        # Allow 2% relative tolerance
        assert abs(empirical - theoretical) / theoretical < 0.02, (
            f"LogNormal mean: empirical={empirical:.4f}, "
            f"theoretical={theoretical:.4f}"
        )

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            LogNormal(mu=0.0, sigma=-1.0)

    def test_pdf_zero_for_negative_x(self) -> None:
        """LogNormal density must be zero for x <= 0."""
        d = LogNormal(mu=0.0, sigma=1.0)
        x = np.array([-5.0, -1.0, -0.001, 0.0])
        assert np.all(d.pdf(x) == 0.0), (
            "LogNormal PDF must be 0 for x <= 0"
        )

    def test_log_pdf_minus_inf_for_nonpositive_x(self) -> None:
        """log_pdf must return -inf for x <= 0."""
        d = LogNormal(mu=0.0, sigma=1.0)
        x = np.array([-1.0, 0.0])
        assert np.all(d.log_pdf(x) == -np.inf)

    def test_log_pdf_consistent_with_pdf(self) -> None:
        """exp(log_pdf(x)) must equal pdf(x) for positive x."""
        d = LogNormal(mu=1.0, sigma=0.5)
        x = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        assert_allclose(np.exp(d.log_pdf(x)), d.pdf(x), rtol=1e-10)


# ==============================================================================
# TESTS FOR: Uniform distribution
# ==============================================================================

class TestUniform:

    def test_all_samples_within_bounds(self) -> None:
        """All Uniform samples must be in [low, high]."""
        d       = Uniform(low=2.0, high=5.0)
        samples = d.sample(10_000, rng=np.random.default_rng(3))
        assert np.all(samples >= 2.0), "Some samples below lower bound"
        assert np.all(samples <= 5.0), "Some samples above upper bound"

    def test_pdf_constant_within_support(self) -> None:
        """Uniform PDF = 1/(high-low) everywhere in [low, high]."""
        d     = Uniform(low=0.0, high=4.0)
        x     = np.array([0.5, 1.0, 2.0, 3.0, 3.9])
        expected = 1.0 / (4.0 - 0.0)  # = 0.25
        assert_allclose(d.pdf(x), expected, rtol=1e-10)

    def test_pdf_zero_outside_support(self) -> None:
        """Uniform PDF = 0 outside [low, high]."""
        d = Uniform(low=1.0, high=3.0)
        x = np.array([-1.0, 0.9, 3.1, 10.0])
        assert np.all(d.pdf(x) == 0.0)

    def test_low_ge_high_raises(self) -> None:
        with pytest.raises(ValueError, match="low < high"):
            Uniform(low=5.0, high=2.0)

    def test_equal_low_high_raises(self) -> None:
        with pytest.raises(ValueError):
            Uniform(low=3.0, high=3.0)

    def test_mean_is_midpoint(self) -> None:
        """Mean of Uniform(low, high) = (low + high) / 2."""
        d = Uniform(low=1.0, high=5.0)
        assert d.mean() == 3.0

    def test_variance_formula(self) -> None:
        """Variance of Uniform(low, high) = (high - low)^2 / 12."""
        d = Uniform(low=0.0, high=6.0)
        assert abs(d.variance() - 36.0 / 12.0) < 1e-10


# ==============================================================================
# TESTS FOR: Beta distribution
# ==============================================================================

class TestBeta:

    def test_all_samples_in_unit_interval(self) -> None:
        """Beta samples must all be in [0, 1]."""
        d       = Beta(alpha=2.0, beta=5.0)
        samples = d.sample(10_000, rng=np.random.default_rng(4))
        assert np.all(samples >= 0.0), "Beta samples below 0"
        assert np.all(samples <= 1.0), "Beta samples above 1"

    def test_mean_formula(self) -> None:
        """E[X] = alpha / (alpha + beta)."""
        d         = Beta(alpha=3.0, beta=7.0)
        theoretical = 3.0 / (3.0 + 7.0)  # = 0.3
        samples   = d.sample(100_000, rng=np.random.default_rng(5))
        assert abs(samples.mean() - theoretical) < 0.01

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            Beta(alpha=0.0, beta=1.0)

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError):
            Beta(alpha=1.0, beta=-1.0)

    def test_log_pdf_consistent_with_pdf(self) -> None:
        """exp(log_pdf(x)) must equal pdf(x) in the interior (0, 1)."""
        d = Beta(alpha=2.0, beta=3.0)
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert_allclose(np.exp(d.log_pdf(x)), d.pdf(x), rtol=1e-10)


# ==============================================================================
# TESTS FOR: Empirical distribution
# ==============================================================================

class TestEmpirical:

    def test_sample_values_come_from_data(self) -> None:
        """All bootstrap samples must be values that appear in the input data."""
        data    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d       = Empirical(data)
        samples = d.sample(1000, rng=np.random.default_rng(6))
        for s in samples:
            assert s in data, f"Sample {s} not in original data"

    def test_mean_close_to_data_mean(self) -> None:
        """Bootstrap samples should have the same mean as the original data."""
        rng     = np.random.default_rng(7)
        data    = rng.normal(10.0, 2.0, size=500)
        d       = Empirical(data)
        samples = d.sample(100_000, rng=np.random.default_rng(8))
        assert abs(samples.mean() - data.mean()) < 0.1

    def test_pdf_positive_near_data(self) -> None:
        """KDE should be positive near the data values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d    = Empirical(data)
        x    = np.array([2.0, 3.0, 4.0])
        assert np.all(d.pdf(x) > 0.0)

    def test_too_few_data_points_raises(self) -> None:
        with pytest.raises(ValueError, match="2"):
            Empirical(np.array([1.0]))

    def test_log_pdf_finite_near_data(self) -> None:
        """log_pdf should be finite (not -inf) near data values."""
        data = np.array([1.0, 2.0, 3.0])
        d    = Empirical(data)
        x    = np.array([1.5, 2.0, 2.5])
        assert np.all(np.isfinite(d.log_pdf(x)))


# ==============================================================================
# CROSS-CUTTING TEST: Distribution ABC enforcement
# ==============================================================================

class TestDistributionABC:
    """Tests that the abstract base class actually enforces the interface."""

    def test_cannot_instantiate_Distribution_directly(self) -> None:
        """
        Distribution is abstract — it cannot be instantiated directly.
        Python raises TypeError if you try.
        """
        with pytest.raises(TypeError, match="abstract"):
            Distribution()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_be_instantiated(self) -> None:
        """
        A subclass that does not implement all abstract methods should also
        be uninstantiable.
        """
        class IncompleteDistribution(Distribution):
            # Forgot to implement pdf, log_pdf, ppf
            def sample(
                self, n: int, rng: np.random.Generator | None = None
            ) -> np.ndarray:
                return np.zeros(n)

        with pytest.raises(TypeError):
            IncompleteDistribution()  # type: ignore[abstract]
