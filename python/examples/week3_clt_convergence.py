"""
Week 3 Tuesday: CLT convergence demo with +/-2 sigma band.

Verifies that Monte Carlo estimation error shrinks at 1/sqrt(N) rate.

Key insight:
    The MC estimator of the MEAN is: mu_hat = (1/N) * sum(x_i)
    Its error is: |mu_hat - mu_true| ~ sigma / sqrt(N)
    where sigma = std of the distribution being sampled.

    We estimate mu_true by running one very large N=100000 reference run.
    Then for each N in N_VALUES, we run N_TRIALS independent engines
    and measure mean(|mu_hat - mu_true|).

    This should decay as sigma/sqrt(N) with slope -0.5 on log-log.

Saves: week3_clt_convergence.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from python.src.battery_model import BatteryModel2Cell
from python.src.monte_carlo import MonteCarloEngine
from python.src.parameter_priors import build_battery_priors

# -----------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------
N_VALUES  = [10, 50, 100, 500, 1000, 5000, 10000]
N_TRIALS  = 30       # independent runs per N
N_STEPS   = 5        # short run for speed
DT        = 1.0
STATE_IDX = 0        # T1

model  = BatteryModel2Cell()
priors = build_battery_priors()

# -----------------------------------------------------------------------
# Step 1: estimate mu_true and sigma_true from large reference run
# -----------------------------------------------------------------------
print("Computing reference (N=100000, may take ~10s)...")
ref = MonteCarloEngine(
    model, priors, N=100_000, n_steps=N_STEPS, dt=DT, seed=9999
).run()

# mu_true = mean of T1 at final step across all 100k particles
final_T1_ref = ref.trajectories[:, -1, STATE_IDX]
mu_true      = float(np.mean(final_T1_ref))
sigma_true   = float(np.std(final_T1_ref, ddof=1))

print(f"mu_true    = {mu_true:.4f} K")
print(f"sigma_true = {sigma_true:.4f} K")
print()

# -----------------------------------------------------------------------
# Step 2: for each N, run N_TRIALS engines and measure |mu_hat - mu_true|
# -----------------------------------------------------------------------
measured_errors: list[float] = []

for N in N_VALUES:
    errors = []
    for trial in range(N_TRIALS):
        eng = MonteCarloEngine(
            model, priors, N=N, n_steps=N_STEPS, dt=DT,
            seed=trial * 7919 + N   # distinct seed per (N, trial)
        )
        res = eng.run()
        mu_hat = float(np.mean(res.trajectories[:, -1, STATE_IDX]))
        errors.append(abs(mu_hat - mu_true))
    measured_errors.append(float(np.mean(errors)))
    print(f"  N={N:>6}: mean|error| = {measured_errors[-1]:.4f} K")

errors_arr = np.array(measured_errors)

# -----------------------------------------------------------------------
# Step 3: theoretical bound and +/-2 sigma band
# -----------------------------------------------------------------------
N_arr  = np.array(N_VALUES, dtype=float)
theory = sigma_true / np.sqrt(N_arr)

# Standard error on mean|error| over N_TRIALS trials
# E[|X|] = sigma * sqrt(2/pi) for X ~ N(0, sigma^2/N)
# Var[|X|] ~ sigma^2/N * (1 - 2/pi)
# SE of mean(|error|) ~ sqrt(Var[|X|] / N_TRIALS)
var_abs_error = (sigma_true**2 / N_arr) * (1.0 - 2.0 / np.pi)
band_half     = 2.0 * np.sqrt(var_abs_error / N_TRIALS)

# -----------------------------------------------------------------------
# Step 4: print table
# -----------------------------------------------------------------------
print()
print("=" * 72)
print(f"  CLT Convergence  |  T1  |  sigma_true={sigma_true:.2f} K  |"
      f"  {N_TRIALS} trials/N")
print("=" * 72)
print(f"  {'N':>8}  {'Measured':>12}  {'Theory':>12}  "
      f"{'Ratio':>7}  {'In +/-2sig?':>11}")
print("-" * 72)
within = 0
for i, N in enumerate(N_VALUES):
    ratio   = errors_arr[i] / theory[i] if theory[i] > 0 else 0
    lo      = theory[i] - band_half[i]
    hi      = theory[i] + band_half[i]
    in_band = lo <= errors_arr[i] <= hi
    if in_band:
        within += 1
    flag = "YES" if in_band else "NO "
    print(f"  {N:>8}  {errors_arr[i]:>12.4f}  {theory[i]:>12.4f}  "
          f"{ratio:>7.3f}  {flag:>11}")
print("=" * 72)
print(f"  Within +/-2 sigma band: {within}/{len(N_VALUES)}")

# Log-log slope
log_N    = np.log(N_arr)
log_err  = np.log(errors_arr)
slope, _ = np.polyfit(log_N, log_err, 1)
print(f"  Log-log slope: {slope:.3f}  (theory: -0.500)")
print("=" * 72)

# -----------------------------------------------------------------------
# Step 5: plot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "ProbOS — CLT Convergence Verification\n"
    f"State: T1  |  sigma_true={sigma_true:.1f} K  |  "
    f"{N_TRIALS} trials/N  |  n_steps={N_STEPS}",
    fontsize=12, fontweight="bold",
)

for ax, xscale, yscale, title_suffix in [
    (axes[0], "log", "log", "Log-log scale"),
    (axes[1], "linear", "linear", "Linear scale"),
]:
    ax.fill_between(
        N_arr, theory - band_half, theory + band_half,
        alpha=0.25, color="steelblue",
        label=r"Theory $\pm 2\sigma$ band",
    )
    ax.plot(N_arr, theory, "b--", linewidth=1.5,
            label=r"Theory $\sigma/\sqrt{N}$")
    ax.plot(N_arr, errors_arr, "ro", markersize=7,
            label="Measured mean |error|")
    if xscale == "log":
        fit_line = np.exp(_) * N_arr ** slope
        ax.plot(N_arr, fit_line, "k:", linewidth=1.2,
                label=f"Fitted slope = {slope:.3f}")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("Mean absolute error (K)")
    ax.set_title(title_suffix)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
out = "week3_clt_convergence.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
