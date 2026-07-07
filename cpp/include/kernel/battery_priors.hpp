#pragma once
// =============================================================================
// cpp/include/kernel/battery_priors.hpp
// =============================================================================
//
// C++ port of python/src/parameter_priors.py's build_battery_priors().
// This is the REAL 15-parameter prior specification for
// BatteryModel2Cell/BatteryCell -- previously, MonteCarloEngineOMP used
// a simplified "nominal * (1 + 0.05*U(-1,1))" perturbation applied
// uniformly to ALL 15 parameters, which did NOT match Python's actual
// priors (a mix of Normal, LogNormal, and Uniform distributions with
// widely varying coefficients of variation).
//
// DESIGN NOTE: mu_log/sigma_log for LogNormal parameters are computed
// PROGRAMMATICALLY here (via std::log/std::sqrt), mirroring Python's
// lognormal_cv() helper's exact formula, rather than hand-transcribed
// decimal constants -- this avoids manual arithmetic errors entirely
// and is directly auditable line-by-line against the Python source.
// =============================================================================

#include <array>
#include <cmath>
#include "kernel/battery_cell.hpp"
#include "distributions/normal.hpp"
#include "distributions/lognormal.hpp"
#include "distributions/uniform.hpp"

namespace probos {
namespace kernel {

inline Param sample_battery_params(std::mt19937_64& rng) {
    using probos::distributions::Normal;
    using probos::distributions::LogNormal;
    using probos::distributions::Uniform;

    const Param nom = BatteryCell::nominal_params();

    auto normal_cv = [&](int index, double cv) -> double {
        double mu = nom[index];
        double sigma = cv * mu;
        Normal d(mu, sigma);
        return d.sample(rng);
    };

    auto lognormal_cv = [&](int index, double cv) -> double {
        double mean = nom[index];
        double sigma_log = std::sqrt(std::log(1.0 + cv * cv));
        double mu_log = std::log(mean) - 0.5 * sigma_log * sigma_log;
        LogNormal d(mu_log, sigma_log);
        return d.sample(rng);
    };

    Param p;
    p[P_EA_SEI] = normal_cv(P_EA_SEI, 0.05);
    p[P_A_SEI]  = lognormal_cv(P_A_SEI, 0.10);
    p[P_H_SEI]  = normal_cv(P_H_SEI, 0.05);
    p[P_EA_AN]  = normal_cv(P_EA_AN, 0.05);
    p[P_A_AN]   = lognormal_cv(P_A_AN, 0.10);
    p[P_H_AN]   = normal_cv(P_H_AN, 0.05);
    p[P_EA_CA]  = normal_cv(P_EA_CA, 0.05);
    p[P_A_CA]   = lognormal_cv(P_A_CA, 0.10);
    p[P_H_CA]   = normal_cv(P_H_CA, 0.05);

    { Normal d(0.045, 0.002); p[P_M_CELL] = d.sample(rng); }
    { Normal d(800.0, 40.0); p[P_CP] = d.sample(rng); }
    { Uniform d(2.0, 10.0); p[P_H_CONV] = d.sample(rng); }
    { Normal d(3.5e-3, 1e-4); p[P_A_SURF] = d.sample(rng); }
    { Normal d(298.15, 5.0); p[P_T_AMB] = d.sample(rng); }
    { Normal d(403.15, 5.0); p[P_T_ONSET] = d.sample(rng); }

    return p;
}

} // namespace kernel
} // namespace probos
