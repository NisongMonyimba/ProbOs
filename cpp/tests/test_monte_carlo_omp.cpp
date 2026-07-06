// =============================================================================
// cpp/tests/test_monte_carlo_omp.cpp
// Native GoogleTest tests for probos::kernel::MonteCarloEngineOMP
// =============================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include "kernel/monte_carlo_omp.hpp"

using namespace probos::kernel;

TEST(MonteCarloEngineOMPConstruction, AccessorsMatchConstructorArguments) {
    MonteCarloEngineOMP engine(500, 100, 1.0, 4);
    EXPECT_EQ(engine.N(), 500);
    EXPECT_EQ(engine.n_steps(), 100);
    EXPECT_EQ(engine.N_threads(), 4);
}

TEST(MonteCarloEngineOMPRun, ResultHasCorrectShapes) {
    MonteCarloEngineOMP engine(200, 20, 1.0);
    MCResult result = engine.run(42);

    EXPECT_EQ(result.final_state.size(), static_cast<size_t>(200 * STATE_DIM));
    EXPECT_EQ(result.percentiles.size(), static_cast<size_t>(3 * STATE_DIM));
    EXPECT_EQ(result.convergence.size(), static_cast<size_t>(STATE_DIM));
    EXPECT_EQ(result.n_particles, 200);
    EXPECT_EQ(result.n_steps, 20);
}

TEST(MonteCarloEngineOMPRun, WallTimeIsGenuinelyMeasuredAndPositive) {
    MonteCarloEngineOMP engine(200, 20, 1.0);
    MCResult result = engine.run(42);
    EXPECT_GT(result.wall_time_ms, 0.0);
}

TEST(MonteCarloEngineOMPRun, NoNaNOrInfInFinalState) {
    MonteCarloEngineOMP engine(500, 50, 1.0);
    MCResult result = engine.run(42);
    for (double x : result.final_state) {
        EXPECT_TRUE(std::isfinite(x));
    }
}

TEST(MonteCarloEngineOMPRun, PercentileOrderingP05LeP50LeP95) {
    MonteCarloEngineOMP engine(500, 50, 1.0);
    MCResult result = engine.run(42);
    for (int k = 0; k < STATE_DIM; ++k) {
        double p05 = result.percentiles[0 * STATE_DIM + k];
        double p50 = result.percentiles[1 * STATE_DIM + k];
        double p95 = result.percentiles[2 * STATE_DIM + k];
        EXPECT_LE(p05, p50 + 1e-9);
        EXPECT_LE(p50, p95 + 1e-9);
    }
}

TEST(MonteCarloEngineOMPRun, ConvergenceShrinksWithLargerN) {
    MonteCarloEngineOMP small(200, 30, 1.0);
    MonteCarloEngineOMP large(20000, 30, 1.0);

    MCResult result_small = small.run(42);
    MCResult result_large = large.run(42);

    EXPECT_LT(result_large.convergence[T1], result_small.convergence[T1]);
}

TEST(MonteCarloEngineOMPRun, SameSeedProducesIdenticalResults) {
    MonteCarloEngineOMP engine1(500, 30, 1.0);
    MonteCarloEngineOMP engine2(500, 30, 1.0);

    MCResult result1 = engine1.run(123);
    MCResult result2 = engine2.run(123);

    ASSERT_EQ(result1.final_state.size(), result2.final_state.size());
    for (size_t i = 0; i < result1.final_state.size(); ++i) {
        EXPECT_DOUBLE_EQ(result1.final_state[i], result2.final_state[i]);
    }
}

TEST(MonteCarloEngineOMPRun, SameSeedReproducibleAcrossDifferentThreadCounts) {
    MonteCarloEngineOMP engine_1thread(500, 30, 1.0, 1);
    MonteCarloEngineOMP engine_4thread(500, 30, 1.0, 4);

    MCResult result_1 = engine_1thread.run(77);
    MCResult result_4 = engine_4thread.run(77);

    ASSERT_EQ(result_1.final_state.size(), result_4.final_state.size());
    for (size_t i = 0; i < result_1.final_state.size(); ++i) {
        EXPECT_DOUBLE_EQ(result_1.final_state[i], result_4.final_state[i]);
    }
}

TEST(MonteCarloEngineOMPRun, DifferentSeedsProduceDifferentResults) {
    MonteCarloEngineOMP engine1(500, 30, 1.0);
    MonteCarloEngineOMP engine2(500, 30, 1.0);

    MCResult result1 = engine1.run(1);
    MCResult result2 = engine2.run(2);

    bool any_different = false;
    for (size_t i = 0; i < result1.final_state.size(); ++i) {
        if (result1.final_state[i] != result2.final_state[i]) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

TEST(MonteCarloEngineOMPRun, ThermalRunawayObservedAtDefaultConditions) {
    MonteCarloEngineOMP engine(1000, 300, 1.0);
    MCResult result = engine.run(42);

    double p50_final_T1 = result.percentiles[1 * STATE_DIM + T1];
    EXPECT_GT(p50_final_T1, 403.15);
}
