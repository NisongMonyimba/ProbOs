"""
Week 2 Example: Central Limit Theorem convergence demo.

WHAT THIS SCRIPT DEMONSTRATES:
  The Monte Carlo estimation error shrinks like 1/sqrt(N).
  This is the Central Limit Theorem (CLT) in action.

WHY THIS MATTERS FOR PROBOS:
  Every Week 3 simulation result comes with an error bound.
  If we run N=5000 particles, the error in the mean is:
    error ~ sigma / sqrt(N) = sigma / 70.7
  Doubling accuracy requires 4x more particles (not 2x).
  This is the fundamental cost of Monte Carlo methods.

WHAT THE SCRIPT DOES:
  1. Draws samples from Normal(mu=135080, sigma=5000) -- battery Ea_SEI
  2. Estimates the mean using N=10, 100, 1000, 10000, 100000 samples
  3. Plots the absolute error vs N on a log-log scale
  4. Overlays the theoretical 1/sqrt(N) reference line
  5. Saves the figure as week2_clt_demo.png

HOW TO READ THE LOG-LOG PLOT:
  If the error falls along a line with slope -0.5 on the log-log plot,
  the CLT is confirmed: error ~ N^(-0.5) = 1/sqrt(N).

HOW TO RUN:
  cd /home/nison/ProbOsWeek1
  source .venv/bin/activate
  python python/examples/week2_clt_demo.py
"""

import os


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from python.src.distributions import Normal

os.makedirs("outputs/figures", exist_ok=True)


def estimate_clt_convergence(
    dist: Normal,
    n_values: list[int],
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For each sample size N in n_values, estimate the mean n_trials times
    and record the average absolute error.

    Parameters
    ----------
    dist     : the distribution to sample from
    n_values : list of sample sizes, e.g. [10, 100, 1000, ...]
    n_trials : how many independent estimates to average per N
    rng      : random number generator for reproducibility

    Returns
    -------
    mean_errors : np.ndarray of shape (len(n_values),)
        mean_errors[i] = average |estimated_mean - true_mean| over n_trials
        at sample size n_values[i]
    """
    true_mean = dist.mean()
    mean_errors = np.zeros(len(n_values))

    for i, N in enumerate(n_values):
        errors = np.zeros(n_trials)
        for trial in range(n_trials):
            # Draw N samples and compute their mean
            samples = dist.sample(N, rng=rng)
            estimated_mean = float(np.mean(samples))
            errors[trial] = abs(estimated_mean - true_mean)

        # Average error over all trials
        mean_errors[i] = np.mean(errors)
        print(f"  N={N:>7,}  avg error = {mean_errors[i]:.2f} J/mol")

    return mean_errors


def plot_clt(
    n_values: list[int],
    mean_errors: np.ndarray,
    sigma: float,
    save_path: str,
) -> None:
    """
    Plot error vs N on a log-log scale with the theoretical 1/sqrt(N) line.

    Parameters
    ----------
    n_values   : list of sample sizes
    mean_errors: measured average absolute errors
    sigma      : true standard deviation of the distribution
    save_path  : where to save the PNG
    """
    n_arr = np.array(n_values, dtype=float)

    # Theoretical CLT bound: E[|mean - mu|] ~ sigma / sqrt(N)
    # We scale it to pass through the first data point for a clean overlay
    theoretical = sigma / np.sqrt(n_arr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Central Limit Theorem: Monte Carlo Error vs Sample Size\n"
        "Distribution: Normal(mu=135080, sigma=5000) -- Battery Ea_SEI",
        fontsize=12, fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Panel 1: Log-log plot
    # On this scale, 1/sqrt(N) appears as a straight line with slope -0.5
    # If the measured errors follow this line, CLT is confirmed.
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.loglog(n_arr, mean_errors,   "o-", color="tab:blue",
              lw=2, ms=8, label="Measured error")
    ax.loglog(n_arr, theoretical,   "k--",
              lw=1.5, label="Theoretical sigma/sqrt(N)")
    ax.set_xlabel("Number of samples N", fontsize=11)
    ax.set_ylabel("Mean absolute error [J/mol]", fontsize=11)
    ax.set_title("Log-log scale (slope = -0.5 confirms CLT)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Annotate the slope
    ax.annotate(
        "slope = -0.5\n(error halves\nwhen N quadruples)",
        xy=(1000, theoretical[2]),
        xytext=(200, theoretical[2] * 3),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray"),
        color="gray",
    )

    # ------------------------------------------------------------------
    # Panel 2: Linear scale showing the same data
    # This makes it easy to see the absolute improvement in error.
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.plot(n_arr, mean_errors,  "o-", color="tab:blue",
            lw=2, ms=8, label="Measured error")
    ax.plot(n_arr, theoretical,  "k--",
            lw=1.5, label="Theoretical sigma/sqrt(N)")
    ax.set_xlabel("Number of samples N", fontsize=11)
    ax.set_ylabel("Mean absolute error [J/mol]", fontsize=11)
    ax.set_title("Linear scale", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Mark N=5000 (production batch size) with a vertical line
    ax.axvline(5000, color="tab:orange", ls=":", lw=1.5,
               label="N=5000 (production)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {save_path}")


def print_summary(n_values: list[int], mean_errors: np.ndarray, sigma: float) -> None:
    """Print a table comparing measured error to theoretical CLT bound."""
    print()
    print("=" * 65)
    print("  CLT CONVERGENCE TABLE")
    print("  Distribution: Normal(mu=135080, sigma=5000) -- Battery Ea_SEI")
    print("=" * 65)
    header = f"  {'N':>8}  {'Measured error':>16}"
    header += f"  {'Theory sigma/sqrt(N)':>22}  {'Ratio':>6}"
    print(header)
    print("-" * 65)
    for i, N in enumerate(n_values):
        theory = sigma / (N ** 0.5)
        ratio  = mean_errors[i] / theory
        print(f"  {N:>8,}  {mean_errors[i]:>16.2f}  {theory:>22.2f}  {ratio:>6.3f}")
    print("=" * 65)
    print()
    print("  INTERPRETATION:")
    print("  Ratio close to 1.0 confirms the CLT bound is tight.")
    print("  At N=5000 (production), error ~ sigma/sqrt(5000) =",
          f"{sigma/(5000**0.5):.1f} J/mol")
    print("  This is the uncertainty floor for Week 3 Monte Carlo results.")
    print("=" * 65)
    print()


if __name__ == "__main__":
    print("Running CLT convergence demo...")
    print("Distribution: Normal(mu=135080, sigma=5000) -- Battery Ea_SEI")
    print()

    # Reproducible random number generator
    rng = np.random.default_rng(seed=42)

    # Battery SEI activation energy distribution (Kim 2007)
    dist  = Normal(mu=135080.0, sigma=5000.0)
    sigma = 5000.0

    # Sample sizes to test: 10, 100, 1000, 10000, 100000
    n_values = [10, 100, 1_000, 5_000, 10_000, 100_000]
    n_trials = 200   # average over 200 independent estimates per N

    print(f"  Sample sizes : {n_values}")
    print(f"  Trials per N : {n_trials}")
    print()

    mean_errors = estimate_clt_convergence(dist, n_values, n_trials, rng)

    print_summary(n_values, mean_errors, sigma)

    save_path = "outputs/figures/week2_clt_demo.png"
    print("Generating figure...")
    plot_clt(n_values, mean_errors, sigma, save_path)

    print()
    print("Done.")
