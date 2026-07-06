// =============================================================================
// cpp/tests/test_lognormal.cpp
// Native GoogleTest tests for probos::distributions::LogNormal
// =============================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <limits>
#include <vector>
#include "distributions/lognormal.hpp"

using probos::distributions::LogNormal;

TEST(LogNormalConstructor, ValidParametersDoNotThrow) {
    EXPECT_NO_THROW(LogNormal(0.0, 1.0));
    EXPECT_NO_THROW(LogNormal(34.7447, 0.09975));
}

TEST(LogNormalConstructor, NegativeSigmaThrows) {
    EXPECT_THROW(LogNormal(0.0, -1.0), std::invalid_argument);
}

TEST(LogNormalConstructor, ZeroSigmaThrows) {
    EXPECT_THROW(LogNormal(0.0, 0.0), std::invalid_argument);
}

TEST(LogNormalConstructor, AccessorsMatchConstructorArguments) {
    LogNormal d(2.5, 0.3);
    EXPECT_DOUBLE_EQ(d.mu(), 2.5);
    EXPECT_DOUBLE_EQ(d.sigma(), 0.3);
}

TEST(LogNormalSampling, EmpiricalMeanConvergesToAnalyticalMean) {
    const double mu = 34.7447;
    const double sigma = 0.09975;
    LogNormal d(mu, sigma);

    std::mt19937_64 rng(42);
    const int N = 200000;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += d.sample(rng);
    }
    double empirical_mean = sum / N;
    double analytical_mean = d.mean();

    double relative_error = std::abs(empirical_mean - analytical_mean) / analytical_mean;
    EXPECT_LT(relative_error, 0.05);
}

TEST(LogNormalSampling, SameSeedProducesSameSequence) {
    // Reproducibility test: same seed -> same sample sequence.
    //
    // IMPORTANT (matching test_normal.cpp's established, correct
    // pattern): we run the FULL sequence with one RNG, then
    // create a FRESH LogNormal object (empty internal cache) and
    // run the FULL sequence again with a matching-seed RNG, THEN
    // compare. std::lognormal_distribution caches an internal
    // value between calls (like std::normal_distribution) --
    // interleaving calls to a SINGLE shared object with two
    // different RNGs on alternating iterations is invalid and
    // was confirmed to fail for exactly this reason.
    LogNormal d(0.0, 1.0);
    std::mt19937_64 rng1(123);
    std::vector<double> run1(50);
    for (int i = 0; i < 50; ++i) run1[i] = d.sample(rng1);

    d = LogNormal(0.0, 1.0);  // fresh object = empty cache
    std::mt19937_64 rng2(123);  // same seed
    for (int i = 0; i < 50; ++i) {
        double x2 = d.sample(rng2);
        EXPECT_DOUBLE_EQ(run1[i], x2)
            << "Sample " << i << " differs between two runs with same seed";
    }
}

TEST(LogNormalSampling, AllSamplesArePositive) {
    LogNormal d(0.0, 2.0);
    std::mt19937_64 rng(7);
    for (int i = 0; i < 10000; ++i) {
        EXPECT_GT(d.sample(rng), 0.0);
    }
}

TEST(LogNormalDensity, PdfIsZeroOrNegativeSupportAndPositiveOnPositiveSupport) {
    LogNormal d(0.0, 1.0);
    EXPECT_DOUBLE_EQ(d.pdf(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(d.pdf(0.0), 0.0);
    EXPECT_GT(d.pdf(1.0), 0.0);
}

TEST(LogNormalDensity, LogPdfIsConsistentWithPdf) {
    LogNormal d(1.0, 0.5);
    for (double x : {0.5, 1.0, 2.0, 5.0}) {
        EXPECT_NEAR(std::log(d.pdf(x)), d.log_pdf(x), 1e-9);
    }
}

TEST(LogNormalDensity, LogPdfIsNegativeInfinityForNonPositiveX) {
    LogNormal d(0.0, 1.0);
    EXPECT_EQ(d.log_pdf(0.0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(d.log_pdf(-5.0), -std::numeric_limits<double>::infinity());
}

TEST(LogNormalDensity, LogPdfRemainsFiniteAtExtremeValues) {
    LogNormal d(0.0, 1.0);
    EXPECT_TRUE(std::isfinite(d.log_pdf(1e10)));
    EXPECT_TRUE(std::isfinite(d.log_pdf(1e-10)));
}

TEST(LogNormalMoments, MeanAndVarianceMatchAnalyticalFormula) {
    const double mu = 1.0;
    const double sigma = 0.5;
    LogNormal d(mu, sigma);

    double expected_mean = std::exp(mu + 0.5 * sigma * sigma);
    double expected_var = (std::exp(sigma * sigma) - 1.0)
                           * std::exp(2.0 * mu + sigma * sigma);

    EXPECT_NEAR(d.mean(), expected_mean, 1e-9);
    EXPECT_NEAR(d.var(), expected_var, 1e-9);
}
