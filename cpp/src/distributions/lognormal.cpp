// =============================================================================
// probos/distributions/lognormal.cpp
// =============================================================================

#include "distributions/lognormal.hpp"

namespace probos {
namespace distributions {

LogNormal::LogNormal(double mu, double sigma)
    : mu_(mu)
    , sigma_(sigma)
    , dist_(mu, sigma)
{
    if (sigma <= 0.0) {
        throw std::invalid_argument(
            "probos::LogNormal: sigma must be strictly positive, got "
            + std::to_string(sigma)
            + ". A LogNormal distribution with sigma <= 0 is mathematically "
              "undefined."
        );
    }
}

double LogNormal::sample(std::mt19937_64& rng) const {
    return dist_(rng);
}

double LogNormal::pdf(double x) const noexcept {
    if (x <= 0.0) {
        return 0.0;
    }
    const double z = (std::log(x) - mu_) / sigma_;
    return std::exp(-0.5 * z * z)
           / (x * sigma_ * std::sqrt(2.0 * std::numbers::pi));
}

double LogNormal::log_pdf(double x) const noexcept {
    if (x <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    const double z = (std::log(x) - mu_) / sigma_;
    const double log_normaliser =
        0.5 * std::log(2.0 * std::numbers::pi * sigma_ * sigma_);
    return -log_normaliser - 0.5 * z * z - std::log(x);
}

double LogNormal::mean() const noexcept {
    return std::exp(mu_ + 0.5 * sigma_ * sigma_);
}

double LogNormal::var() const noexcept {
    const double s2 = sigma_ * sigma_;
    return (std::exp(s2) - 1.0) * std::exp(2.0 * mu_ + s2);
}

} // namespace distributions
} // namespace probos
