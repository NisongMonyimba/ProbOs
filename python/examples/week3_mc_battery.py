"""
Week 3 Tuesday: P05/P50/P95 battery thermal runaway fan plot.

4-panel figure:
  Panel 1: T1 temperature fan (100 grey trajectories + P05/P50/P95)
  Panel 2: c_SEI_1 concentration fan (SEI depletion spread)
  Panel 3: Convergence certificate bar chart (sigma/sqrt(N) per state)
  Panel 4: T1 distribution at final timestep (histogram + fitted Normal)

Saves: week3_mc_battery.png
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as sp_norm

from python.src.battery_model import BatteryModel2Cell
from python.src.monte_carlo import MonteCarloEngine
from python.src.parameter_priors import build_battery_priors

os.makedirs("outputs/figures", exist_ok=True)

# -----------------------------------------------------------------------
# Run Monte Carlo
# -----------------------------------------------------------------------
N       = 5000
N_STEPS = 300
DT      = 1.0
SEED    = 42

model  = BatteryModel2Cell()
priors = build_battery_priors()
engine = MonteCarloEngine(model, priors, N=N, n_steps=N_STEPS, dt=DT, seed=SEED)
result = engine.run()

time_min = np.arange(N_STEPS + 1) * DT / 60.0   # seconds -> minutes

# Indices into state vector
T1_IDX    = BatteryModel2Cell.T1
C_SEI_IDX = BatteryModel2Cell.C_SEI1

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "ProbOS — Battery Thermal Runaway: Monte Carlo Uncertainty Quantification\n"
    f"N={N} particles | dt={DT}s | {N_STEPS} steps "
    f"({N_STEPS/60:.0f} min) | seed={SEED}",
    fontsize=13, fontweight="bold",
)

GREY   = "#cccccc"
RED    = "#d62728"
BLACK  = "#111111"
BLUE   = "#1f77b4"

# ---- Panel 1: Temperature T1 fan ----
ax = axes[0, 0]
rng_plot = np.random.default_rng(0)
sample_idx = rng_plot.choice(N, size=100, replace=False)
for i in sample_idx:
    ax.plot(time_min, result.trajectories[i, :, T1_IDX] - 273.15,
            color=GREY, alpha=0.3, linewidth=0.5)
ax.plot(time_min, result.percentiles[0, :, T1_IDX] - 273.15,
        color=RED,   linewidth=2.0, label="P05 (coolest 5%)")
ax.plot(time_min, result.percentiles[1, :, T1_IDX] - 273.15,
        color=BLACK, linewidth=2.0, label="P50 (median)")
ax.plot(time_min, result.percentiles[2, :, T1_IDX] - 273.15,
        color=BLUE,  linewidth=2.0, label="P95 (hottest 5%)")
spread = (result.percentiles[2, -1, T1_IDX]
          - result.percentiles[0, -1, T1_IDX])
ax.set_xlabel("Time (min)")
ax.set_ylabel("Temperature T1 (°C)")
ax.set_title(f"Cell 1 Temperature\nP95−P05 spread at t=300min: {spread:.1f} K")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ---- Panel 2: SEI concentration fan ----
ax = axes[0, 1]
for i in sample_idx:
    ax.plot(time_min, result.trajectories[i, :, C_SEI_IDX],
            color=GREY, alpha=0.3, linewidth=0.5)
ax.plot(time_min, result.percentiles[0, :, C_SEI_IDX],
        color=RED,   linewidth=2.0, label="P05")
ax.plot(time_min, result.percentiles[1, :, C_SEI_IDX],
        color=BLACK, linewidth=2.0, label="P50")
ax.plot(time_min, result.percentiles[2, :, C_SEI_IDX],
        color=BLUE,  linewidth=2.0, label="P95")
ax.axhline(0.1, color="orange", linestyle="--", linewidth=1.5,
           label="Near-depletion (c=0.1)")
ax.set_xlabel("Time (min)")
ax.set_ylabel("SEI Reactant Concentration c_SEI_1")
ax.set_title("SEI Reactant Depletion\nHigher temperature -> faster depletion")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ---- Panel 3: Convergence certificate bar chart ----
ax = axes[1, 0]
state_labels = [
    "T1", "T2",
    "c_SEI_1", "c_SEI_2",
    "c_an_1",  "c_an_2",
    "c_ca_1",  "c_ca_2",
]
x = np.arange(model.state_dim)
bars = ax.bar(x, result.convergence, color=BLUE, alpha=0.8, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(state_labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("sigma / sqrt(N)")
ax.set_title(
    f"Monte Carlo Convergence Certificate\n"
    f"sigma/sqrt(N) per state variable at t={N_STEPS}s  (N={N})"
)
for bar, val in zip(bars, result.convergence):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# ---- Panel 4: T1 distribution at final timestep ----
ax = axes[1, 1]
final_T1_C = result.trajectories[:, -1, T1_IDX] - 273.15
ax.hist(final_T1_C, bins=60, color=BLUE, alpha=0.7,
        edgecolor="white", density=True, label="Particle histogram")
# Fitted Normal overlay
mu_fit  = np.mean(final_T1_C)
std_fit = np.std(final_T1_C, ddof=1)
x_fit   = np.linspace(final_T1_C.min(), final_T1_C.max(), 300)
ax.plot(x_fit, sp_norm.pdf(x_fit, mu_fit, std_fit),
        color=BLACK, linewidth=2.0, label=f"Fitted N({mu_fit:.1f}, {std_fit:.1f}²)")
p05_val = result.percentiles[0, -1, T1_IDX] - 273.15
p50_val = result.percentiles[1, -1, T1_IDX] - 273.15
p95_val = result.percentiles[2, -1, T1_IDX] - 273.15
ax.axvline(p05_val, color=RED,   linestyle="--", linewidth=1.5,
           label=f"P05 = {p05_val:.1f}°C")
ax.axvline(p50_val, color=BLACK, linestyle="--", linewidth=1.5,
           label=f"P50 = {p50_val:.1f}°C")
ax.axvline(p95_val, color=BLUE,  linestyle="--", linewidth=1.5,
           label=f"P95 = {p95_val:.1f}°C")
ax.set_xlabel("Temperature T1 at t=300min (°C)")
ax.set_ylabel("Probability density")
ax.set_title("Temperature Distribution at t=300min")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "outputs/figures/week3_mc_battery.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# -----------------------------------------------------------------------
# Print convergence certificate
# -----------------------------------------------------------------------
print()
print(engine.convergence_certificate())
