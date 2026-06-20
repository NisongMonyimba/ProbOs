"""
Week 4 Monday: CLT convergence demo using DP subset trick.

DP insight:
    The original week3_clt_convergence.py ran N_TRIALS=30 independent
    engines at each N in [10, 50, 100, 500, 1000, 5000, 10000].
    Total: 7 * 30 = 210 full MonteCarloEngine.run() calls.

    DP observation: A run at N=10000 contains runs at N=5000, N=1000, etc.
    as subsets -- if we run once at N_max and subsample the first N particles,
    we get the same statistical estimate with 1 run instead of 210.

    This is tabulation (bottom-up DP):
      1. Run once at N_max=10000 (the largest subproblem)
      2. For each N in N_VALUES: subsample trajectories[:N, -1, STATE_IDX]
      3. Compute mean error against mu_true

    Speedup: 210 engine runs -> 1 engine run + 7 subsamples = ~200x

Saves: week4_clt_convergence_dp.png
"""

from __future__ import annotations

import time

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from python.src.battery_model import BatteryModel2Cell
from python.src.monte_carlo import MonteCarloEngine
from python.src.parameter_priors import build_battery_priors

os.makedirs("outputs/figures", exist_ok=True)

# -----------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------
N_VALUES  = [10, 50, 100, 500, 1000, 5000, 10000]
N_MAX     = 10000
N_STEPS   = 5
DT        = 1.0
STATE_IDX = 0        # T1
N_TRIALS  = 30       # for comparison with old approach

model  = BatteryModel2Cell()
priors = build_battery_priors()

# -----------------------------------------------------------------------
# Reference mean (large N)
# -----------------------------------------------------------------------
print("Computing reference (N=100000)...")
ref = MonteCarloEngine(
    model, priors, N=100_000, n_steps=N_STEPS, dt=DT, seed=9999
).run()
mu_true    = float(np.mean(ref.trajectories[:, -1, STATE_IDX]))
sigma_true = float(np.std(ref.trajectories[:, -1, STATE_IDX], ddof=1))
print(f"mu_true    = {mu_true:.4f} K")
print(f"sigma_true = {sigma_true:.4f} K")

# -----------------------------------------------------------------------
# OLD approach: 210 full engine runs
# -----------------------------------------------------------------------
print(f"\nOLD approach: {len(N_VALUES)} x {N_TRIALS} = "
      f"{len(N_VALUES)*N_TRIALS} engine runs...")
t0 = time.perf_counter()
old_errors: list[float] = []
for N in N_VALUES:
    errs = []
    for trial in range(N_TRIALS):
        eng = MonteCarloEngine(
            model, priors, N=N, n_steps=N_STEPS, dt=DT,
            seed=trial * 7919 + N
        )
        res = eng.run()
        mu_hat = float(np.mean(res.trajectories[:, -1, STATE_IDX]))
        errs.append(abs(mu_hat - mu_true))
    old_errors.append(float(np.mean(errs)))
old_time = time.perf_counter() - t0
print(f"OLD time: {old_time:.3f}s")

# -----------------------------------------------------------------------
# NEW approach: 1 engine run + 7 subsamples (DP subset trick)
# -----------------------------------------------------------------------
print(f"\nNEW approach (DP): 1 engine run at N={N_MAX} + subsamples...")
t0 = time.perf_counter()

# Step 1: run once at N_max -- this is the single "large subproblem"
big = MonteCarloEngine(
    model, priors, N=N_MAX, n_steps=N_STEPS, dt=DT, seed=42
).run()
big_finals = big.trajectories[:, -1, STATE_IDX]   # shape (N_MAX,)

# Step 2: subsample -- O(1) per N value
new_errors: list[float] = []
for N in N_VALUES:
    mu_hat = float(np.mean(big_finals[:N]))
    new_errors.append(abs(mu_hat - mu_true))

new_time = time.perf_counter() - t0
print(f"NEW time: {new_time:.3f}s")

# -----------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------
speedup = old_time / new_time
theory  = sigma_true / np.sqrt(np.array(N_VALUES, dtype=float))

print()
print("=" * 72)
print("  DP Subset Trick Results")
print(f"  OLD: {len(N_VALUES)*N_TRIALS} engine runs  -> {old_time:.3f}s")
print(f"  NEW: 1 engine run + subsamples -> {new_time:.3f}s")
print(f"  Speedup: {speedup:.0f}x")
print("=" * 72)
print(f"  {'N':>8}  {'OLD error':>12}  {'NEW error':>12}  "
      f"{'Theory':>12}  {'Match?':>8}")
print("-" * 72)
for i, N in enumerate(N_VALUES):
    tolerance = sigma_true * 2 / N**0.5 + 1
    match = "YES" if abs(old_errors[i] - new_errors[i]) < tolerance else "NO"
    print(f"  {N:>8}  {old_errors[i]:>12.4f}  {new_errors[i]:>12.4f}  "
          f"{theory[i]:>12.4f}  {match:>8}")
print("=" * 72)

# Log-log slopes
N_arr      = np.array(N_VALUES, dtype=float)
slope_old  = np.polyfit(np.log(N_arr), np.log(np.array(old_errors)), 1)[0]
slope_new  = np.polyfit(np.log(N_arr), np.log(np.array(new_errors)), 1)[0]
print(f"  Log-log slope OLD: {slope_old:.3f}  NEW: {slope_new:.3f}  "
      f"theory: -0.500")
print("=" * 72)

# -----------------------------------------------------------------------
# Plot: side-by-side comparison
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"ProbOS Week 4 — CLT Convergence: DP Subset Trick\n"
    f"OLD: {len(N_VALUES)*N_TRIALS} engine runs ({old_time:.2f}s)  |  "
    f"NEW: 1 run + subsamples ({new_time:.3f}s)  |  "
    f"Speedup: {speedup:.0f}x",
    fontsize=11, fontweight="bold",
)

for ax, errors, label, color in [
    (axes[0], old_errors, f"OLD ({len(N_VALUES)*N_TRIALS} runs)", "red"),
    (axes[1], new_errors, "NEW (DP subset, 1 run)", "green"),
]:
    ax.fill_between(
        N_arr,
        theory * 0.5, theory * 2.0,
        alpha=0.2, color="steelblue", label=r"Theory $\pm$ 2x band"
    )
    ax.loglog(N_arr, theory, "b--", linewidth=1.5,
              label=r"Theory $\sigma/\sqrt{N}$")
    ax.loglog(N_arr, errors, "o-", color=color, markersize=7,
              linewidth=1.5, label=label)
    ax.set_xlabel("N (particles)")
    ax.set_ylabel("Mean absolute error (K)")
    slope = np.polyfit(np.log(N_arr), np.log(np.array(errors)), 1)[0]
    ax.set_title(f"Log-log slope: {slope:.3f} (theory −0.500)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
out = "outputs/figures/week4_clt_convergence_dp.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
