"""
Week 4 Monday: DP optimisation benchmark summary.

Documents the profiling results from Week 4 Monday:
  1. inv_RT precompute -- no speedup in Python (exp dominates, SIMD)
  2. nominal_params cache -- marginal (rarely called)
  3. SobolSensitivity._build_problem() cache -- 1040x speedup
  4. CLT convergence DP subset trick -- 45x speedup

Saves: week4_dp_benchmark.png
"""

from __future__ import annotations

import os

import matplotlib

from python.src.battery_model import BatteryModel2Cell
from python.src.monte_carlo import MonteCarloEngine
from python.src.parameter_priors import build_battery_priors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("outputs/figures", exist_ok=True)

# -----------------------------------------------------------------------
# Benchmark results (measured)
# -----------------------------------------------------------------------
techniques = [
    "inv_RT\nprecompute\n(forward_batch)",
    "nominal_params\ncache\n(BatteryModel2Cell)",
    "SALib problem\ncache\n(SobolSensitivity)",
    "CLT subset\nDP trick\n(convergence demo)",
]
speedups   = [0.99, 1.05, 1040.0, 45.0]
colors     = ["#d62728", "#ff7f0e", "#2ca02c", "#2ca02c"]
notes      = [
    "NumPy exp() SIMD\ndominates; division\ncost negligible",
    "Rarely called;\nmarginal gain\nin practice",
    "1040x: bounds\nbuilt once at init\nvs per run()",
    "45x: 1 engine run\nvs 210 runs\n(DP tabulation)",
]

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "ProbOS Week 4 Monday — DP Optimisation Profiling Results\n"
    "Lesson: profile before optimising (item 33). "
    "Only caching + tabulation gave real speedups.",
    fontsize=11, fontweight="bold",
)

# ---- Panel 1: bar chart of speedups (log scale) ----
ax = axes[0]
x = np.arange(len(techniques))
bars = ax.bar(x, speedups, color=colors, alpha=0.85, edgecolor="white", width=0.6)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="No speedup (1x)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(techniques, fontsize=8)
ax.set_ylabel("Speedup (log scale)")
ax.set_title("Speedup by optimisation technique")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, speedups, strict=True):
    label = f"{val:.0f}x" if val >= 2 else f"{val:.2f}x"
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.1,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold")

# ---- Panel 2: CLT convergence comparison ----
ax = axes[1]

# Reproduce results (model/priors imported at top of file)

model  = BatteryModel2Cell()
priors = build_battery_priors()

N_VALUES  = [10, 50, 100, 500, 1000, 5000, 10000]
N_STEPS   = 5
N_TRIALS  = 30

ref = MonteCarloEngine(model, priors, N=100_000, n_steps=N_STEPS, seed=9999).run()
mu_true    = float(np.mean(ref.trajectories[:, -1, 0]))
sigma_true = float(np.std(ref.trajectories[:, -1, 0], ddof=1))

# OLD: 210 runs
old_errors: list[float] = []
for N in N_VALUES:
    errs = []
    for trial in range(N_TRIALS):
        eng = MonteCarloEngine(model, priors, N=N, n_steps=N_STEPS, seed=trial*7919+N)
        res = eng.run()
        errs.append(abs(float(np.mean(res.trajectories[:, -1, 0])) - mu_true))
    old_errors.append(float(np.mean(errs)))

# NEW: 1 run + subsamples
big      = MonteCarloEngine(model, priors, N=10000, n_steps=N_STEPS, seed=42).run()
finals   = big.trajectories[:, -1, 0]
new_errors = [abs(float(np.mean(finals[:N])) - mu_true) for N in N_VALUES]

N_arr  = np.array(N_VALUES, dtype=float)
theory = sigma_true / np.sqrt(N_arr)

ax.fill_between(N_arr, theory*0.5, theory*2.0,
                alpha=0.15, color="steelblue", label=r"Theory $\pm$2x band")
ax.loglog(N_arr, theory,     "b--",  linewidth=1.5, label=r"Theory $\sigma/\sqrt{N}$")
ax.loglog(N_arr, old_errors, "r-o",  markersize=6,  linewidth=1.5,
          label="OLD: 210 engine runs")
ax.loglog(N_arr, new_errors, "g-s",  markersize=6,  linewidth=1.5,
          label="NEW: 1 run + DP subsamples")
ax.set_xlabel("N (particles)")
ax.set_ylabel("Mean absolute error (K)")
ax.set_title(
    "CLT Convergence: OLD vs NEW (DP subset trick)\n"
    "45x speedup, same statistical result"
)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
out = "outputs/figures/week4_dp_benchmark.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Summary table
print()
print("=" * 60)
print("  Week 4 Monday DP Optimisation Summary")
print("=" * 60)
rows = zip(
    ["inv_RT precompute", "nominal_params cache",
     "SALib problem cache", "CLT subset trick"],
    speedups, notes, strict=True,
)
for name, sp, _note in rows:
    flag = "REAL WIN" if sp >= 10 else ("marginal" if sp >= 1 else "SLOWER")
    print(f"  {name:<22} {sp:>8.0f}x  {flag}")
print("=" * 60)
print("  Key lesson: profiling revealed only caching (memoization)")
print("  and tabulation (DP) give real Python speedups.")
print("  True parallelism requires C++/OpenMP (Tuesday).")
print("=" * 60)
