// =============================================================================
// cpp/include/kernel/monte_carlo_omp.hpp
// ProbOS Week 4 -- OpenMP Monte Carlo Engine
//
// Parallelises the particle loop using OpenMP.
// Each thread advances an independent subset of particles.
// No shared mutable state -- thread-safe by design.
//
// Design:
//   Serial Python:  for t in steps: forward_batch(N particles)  -- 1 thread
//   OpenMP C++:     for t in steps: #pragma omp parallel for    -- N_threads
//
// Memory layout: flat arrays (row-major, N * STATE_DIM doubles)
//   state[i * STATE_DIM + k] = state variable k of particle i
// =============================================================================

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include "kernel/battery_cell.hpp"

namespace probos {
namespace kernel {

// ============================================================
// MCResult: output container
// ============================================================
struct MCResult {
    std::vector<double> final_state;   // shape (N, STATE_DIM) flat
    std::vector<double> percentiles;   // shape (3, STATE_DIM) flat [P05,P50,P95]
    std::vector<double> convergence;   // shape (STATE_DIM,) sigma/sqrt(N)
    int    n_particles;
    int    n_steps;
    double dt;
    double wall_time_ms;               // measured wall time in milliseconds
};

// ============================================================
// MonteCarloEngineOMP
// ============================================================
class MonteCarloEngineOMP {
public:
    // N_threads=0 means use all available cores (OMP_NUM_THREADS)
    MonteCarloEngineOMP(
        int    N,
        int    n_steps,
        double dt       = 1.0,
        int    N_threads = 0
    ) : N_(N), n_steps_(n_steps), dt_(dt), N_threads_(N_threads) {}

    // Run: draw params from the REAL battery priors (see
    // kernel/battery_priors.hpp -- a faithful C++ port of
    // python/src/parameter_priors.py's build_battery_priors(), 9
    // Normal + 3 LogNormal + 1 Uniform + 2 more Normal), advance N
    // particles for n_steps, return MCResult.
    MCResult run(uint64_t seed = 42) const;

    int N()        const { return N_; }
    int n_steps()  const { return n_steps_; }
    int N_threads() const { return N_threads_; }

private:
    int    N_;
    int    n_steps_;
    double dt_;
    int    N_threads_;
};

} // namespace kernel
} // namespace probos
