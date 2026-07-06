#pragma once
// =============================================================================
// probos/distributions/uniform.hpp
// =============================================================================
//
// The continuous Uniform distribution on [low, high]. Used in
// python/src/parameter_priors.py for the convective heat-transfer
// coefficient h_conv, which is highly geometry-dependent and poorly
// known -- Uniform is the "maximum ignorance within bounds" prior.
// =============================================================================

#include <random>
#include <stdexcept>
#include <string>
#include <cmath>
#include <limits>

namespace probos {
namespace distributions {

class Uniform {
public:
    // Parameters:
    //   low, high (double) : the support bounds, MUST have low < high
    //
    // Throws: std::invalid_argument if low >= high
    explicit Uniform(double low, double high);

    double sample(std::mt19937_64& rng) const;

    // PDF: 1/(high-low) for low <= x <= high, 0.0 otherwise.
    [[nodiscard]] double pdf(double x) const noexcept;

    // Log-PDF: log(1/(high-low)) in range, -infinity outside.
    [[nodiscard]] double log_pdf(double x) const noexcept;

    [[nodiscard]] double low()  const noexcept { return low_; }
    [[nodiscard]] double high() const noexcept { return high_; }
    [[nodiscard]] double mean() const noexcept { return 0.5 * (low_ + high_); }
    [[nodiscard]] double var()  const noexcept {
        const double range = high_ - low_;
        return (range * range) / 12.0;
    }

private:
    double low_;
    double high_;
    mutable std::uniform_real_distribution<double> dist_;
};

} // namespace distributions
} // namespace probos
