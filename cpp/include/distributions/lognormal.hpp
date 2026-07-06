#pragma once
// =============================================================================
// probos/distributions/lognormal.hpp
// =============================================================================
//
// The LogNormal distribution -- if X ~ LogNormal(mu, sigma), then
// ln(X) ~ Normal(mu, sigma). Used in python/src/parameter_priors.py for
// pre-exponential Arrhenius factors (A_SEI, A_anode, A_cathode), which
// span many orders of magnitude -- multiplicative uncertainty is more
// appropriate than additive (Normal) uncertainty for such parameters.
//
// mu and sigma here are the LOG-SPACE parameters (the mean and std of
// ln(X)), NOT the mean/std of X itself -- matching the exact convention
// used by python/src/distributions.py's LogNormal class and by
// std::lognormal_distribution.
// =============================================================================

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <numbers>
#include <limits>

namespace probos {
namespace distributions {

class LogNormal {
public:
    // Parameters:
    //   mu    (double) : mean of ln(X)
    //   sigma (double) : std of ln(X), MUST be > 0
    //
    // Throws: std::invalid_argument if sigma <= 0
    explicit LogNormal(double mu, double sigma);

    // Draw ONE sample from LogNormal(mu, sigma).
    double sample(std::mt19937_64& rng) const;

    // PDF of the LogNormal distribution, defined for x > 0:
    //   f(x) = exp( -(ln(x)-mu)^2 / (2*sigma^2) ) / (x * sigma * sqrt(2*pi))
    // Returns 0.0 for x <= 0 (LogNormal has no support there).
    [[nodiscard]] double pdf(double x) const noexcept;

    // Log-PDF, numerically stable. Returns -infinity for x <= 0.
    [[nodiscard]] double log_pdf(double x) const noexcept;

    [[nodiscard]] double mu()    const noexcept { return mu_; }
    [[nodiscard]] double sigma() const noexcept { return sigma_; }

    // mean/var are of X itself (NOT ln(X)):
    //   E[X]   = exp(mu + sigma^2/2)
    //   Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    [[nodiscard]] double mean() const noexcept;
    [[nodiscard]] double var()  const noexcept;

private:
    double mu_;
    double sigma_;
    mutable std::lognormal_distribution<double> dist_;
};

} // namespace distributions
} // namespace probos
