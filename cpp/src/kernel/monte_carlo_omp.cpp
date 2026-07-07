// =============================================================================
// cpp/src/kernel/monte_carlo_omp.cpp
// ProbOS Week 4 -- OpenMP Monte Carlo Engine Implementation
// =============================================================================

#include "kernel/monte_carlo_omp.hpp"
#include "kernel/battery_priors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace probos {
namespace kernel {

// ----------------------------------------------------------------
// Simple PCG-style hash for per-particle seed from global seed
// ----------------------------------------------------------------
static inline uint64_t hash64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// ----------------------------------------------------------------
// MCResult MonteCarloEngineOMP::run()
// ----------------------------------------------------------------
MCResult MonteCarloEngineOMP::run(uint64_t seed) const {

    auto t_start = std::chrono::high_resolution_clock::now();

    const int N  = N_;
    const int S  = STATE_DIM;
    const int P  = PARAM_DIM;

#ifdef _OPENMP
    if (N_threads_ > 0) omp_set_num_threads(N_threads_);
#endif

    // ------------------------------------------------------------------
    // Draw REAL parameter sets from the actual 15-distribution battery
    // priors (see kernel/battery_priors.hpp, a faithful C++ port of
    // python/src/parameter_priors.py's build_battery_priors()).
    //
    // Previously this used a simplified 'nominal * (1 + 0.05*U(-1,1))'
    // perturbation applied uniformly to ALL 15 parameters -- a real,
    // documented discrepancy, now fixed at the root.
    //
    // Each particle gets its own seed derived from global seed + i.
    // (P is unused now since sample_battery_params() knows its own
    // PARAM_DIM internally.)
    static_cast<void>(P);

    std::vector<Param> params(N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        uint64_t pseed = hash64(seed ^ static_cast<uint64_t>(i) * 2654435761ULL);
        std::mt19937_64 rng(pseed);
        params[i] = sample_battery_params(rng);
    }

    // ------------------------------------------------------------------
    // Allocate state array: shape (N, STATE_DIM) flat row-major
    // ------------------------------------------------------------------
    State init = BatteryCell::initial_state();
    std::vector<double> state(N * S);

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < S; ++k) {
            state[i * S + k] = init[k];
        }
    }

    // ------------------------------------------------------------------
    // Main loop: advance n_steps timesteps
    // OpenMP parallelises the particle dimension.
    // Each particle is independent -- no race conditions.
    // ------------------------------------------------------------------
    for (int t = 0; t < n_steps_; ++t) {

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            // Extract state for particle i
            State s;
            for (int k = 0; k < S; ++k) s[k] = state[i * S + k];

            // Advance one step
            State ns = BatteryCell::forward_step(s, params[i], dt_);

            // Write back
            for (int k = 0; k < S; ++k) state[i * S + k] = ns[k];
        }
    }

    // ------------------------------------------------------------------
    // Compute percentiles (P05, P50, P95) per state variable
    // ------------------------------------------------------------------
    std::vector<double> percentiles(3 * S, 0.0);
    std::vector<double> col(N);

    for (int k = 0; k < S; ++k) {
        for (int i = 0; i < N; ++i) col[i] = state[i * S + k];
        std::sort(col.begin(), col.end());

        auto pct = [&](double p) -> double {
            double idx = p * (N - 1);
            int lo = static_cast<int>(idx);
            int hi = std::min(lo + 1, N - 1);
            return col[lo] + (idx - lo) * (col[hi] - col[lo]);
        };

        percentiles[0 * S + k] = pct(0.05);   // P05
        percentiles[1 * S + k] = pct(0.50);   // P50
        percentiles[2 * S + k] = pct(0.95);   // P95
    }

    // ------------------------------------------------------------------
    // Compute convergence: sigma/sqrt(N) per state variable
    // ------------------------------------------------------------------
    std::vector<double> convergence(S, 0.0);

    for (int k = 0; k < S; ++k) {
        double mean = 0.0;
        for (int i = 0; i < N; ++i) mean += state[i * S + k];
        mean /= N;

        double var = 0.0;
        for (int i = 0; i < N; ++i) {
            double d = state[i * S + k] - mean;
            var += d * d;
        }
        var /= (N - 1);
        convergence[k] = std::sqrt(var) / std::sqrt(static_cast<double>(N));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    return MCResult{
        state,
        percentiles,
        convergence,
        N,
        n_steps_,
        dt_,
        wall_ms
    };
}

} // namespace kernel
} // namespace probos
