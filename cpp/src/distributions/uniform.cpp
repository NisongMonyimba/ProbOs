// =============================================================================
// probos/distributions/uniform.cpp
// =============================================================================

#include "distributions/uniform.hpp"

namespace probos {
namespace distributions {

Uniform::Uniform(double low, double high)
    : low_(low)
    , high_(high)
    , dist_(low, high)
{
    if (low >= high) {
        throw std::invalid_argument(
            "probos::Uniform: low must be strictly less than high, got low="
            + std::to_string(low) + ", high=" + std::to_string(high)
        );
    }
}

double Uniform::sample(std::mt19937_64& rng) const {
    return dist_(rng);
}

double Uniform::pdf(double x) const noexcept {
    if (x < low_ || x > high_) {
        return 0.0;
    }
    return 1.0 / (high_ - low_);
}

double Uniform::log_pdf(double x) const noexcept {
    if (x < low_ || x > high_) {
        return -std::numeric_limits<double>::infinity();
    }
    return -std::log(high_ - low_);
}

} // namespace distributions
} // namespace probos
