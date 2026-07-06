// =============================================================================
// cpp/tests/test_uniform.cpp
// Native GoogleTest tests for probos::distributions::Uniform
// =============================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <limits>
#include "distributions/uniform.hpp"

using probos::distributions::Uniform;

TEST(UniformConstructor, ValidParametersDoNotThrow) {
    EXPECT_NO_THROW(Uniform(0.0, 1.0));
    EXPECT_NO_THROW(Uniform(2.0, 10.0));
}

TEST(UniformConstructor, LowGreaterThanHighThrows) {
    EXPECT_THROW(Uniform(10.0, 2.0), std::invalid_argument);
}

TEST(UniformConstructor, LowEqualsHighThrows) {
    EXPECT_THROW(Uniform(5.0, 5.0), std::invalid_argument);
}

TEST(UniformConstructor, AccessorsMatchConstructorArguments) {
    Uniform d(2.0, 10.0);
    EXPECT_DOUBLE_EQ(d.low(), 2.0);
    EXPECT_DOUBLE_EQ(d.high(), 10.0);
}

TEST(UniformSampling, AllSamplesWithinBounds) {
    Uniform d(2.0, 10.0);
    std::mt19937_64 rng(42);
    for (int i = 0; i < 10000; ++i) {
        double x = d.sample(rng);
        EXPECT_GE(x, 2.0);
        EXPECT_LE(x, 10.0);
    }
}

TEST(UniformSampling, EmpiricalMeanConvergesToAnalyticalMean) {
    Uniform d(2.0, 10.0);
    std::mt19937_64 rng(42);
    const int N = 100000;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += d.sample(rng);
    }
    double empirical_mean = sum / N;
    EXPECT_NEAR(empirical_mean, d.mean(), 0.05);
}

TEST(UniformSampling, SameSeedProducesSameSequence) {
    Uniform d(0.0, 1.0);
    std::mt19937_64 rng1(123);
    std::mt19937_64 rng2(123);
    for (int i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(d.sample(rng1), d.sample(rng2));
    }
}

TEST(UniformDensity, PdfIsConstantWithinBoundsAndZeroOutside) {
    Uniform d(2.0, 10.0);
    EXPECT_DOUBLE_EQ(d.pdf(1.0), 0.0);
    EXPECT_DOUBLE_EQ(d.pdf(11.0), 0.0);
    EXPECT_NEAR(d.pdf(5.0), 1.0 / 8.0, 1e-12);
    EXPECT_NEAR(d.pdf(2.0), 1.0 / 8.0, 1e-12);
    EXPECT_NEAR(d.pdf(10.0), 1.0 / 8.0, 1e-12);
}

TEST(UniformDensity, LogPdfIsConsistentWithPdf) {
    Uniform d(2.0, 10.0);
    EXPECT_NEAR(std::log(d.pdf(5.0)), d.log_pdf(5.0), 1e-9);
}

TEST(UniformDensity, LogPdfIsNegativeInfinityOutsideBounds) {
    Uniform d(2.0, 10.0);
    EXPECT_EQ(d.log_pdf(1.0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(d.log_pdf(11.0), -std::numeric_limits<double>::infinity());
}

TEST(UniformMoments, MeanAndVarianceMatchAnalyticalFormula) {
    Uniform d(2.0, 10.0);
    EXPECT_DOUBLE_EQ(d.mean(), 6.0);
    EXPECT_NEAR(d.var(), 64.0 / 12.0, 1e-12);
}
