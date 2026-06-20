"""
Week 3 Wednesday: Sobol sensitivity analysis for battery thermal runaway.

2-panel figure:
  Panel 1: First-order S1 indices for T1 at final timestep (sorted desc)
  Panel 2: Total-effect ST indices for T1 at final timestep (same order)

Saves: week3_sobol_battery.png
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from python.src.battery_model import BatteryModel2Cell
from python.src.parameter_priors import build_battery_priors
from python.src.sensitivity import SobolSensitivity

os.makedirs("outputs/figures", exist_ok=True)

# -----------------------------------------------------------------------
# Run Sobol analysis
# N_saltelli=1024: 1024*(15+2) = 17408 evaluations (~10s on CPU)
# -----------------------------------------------------------------------
model  = BatteryModel2Cell()
priors = build_battery_priors()

print("Running Sobol sensitivity analysis...")
print("N_saltelli=1024 -> 17,408 model evaluations")
sobol = SobolSensitivity(
    model, priors, N_saltelli=1024, n_steps=10, seed=42
)
result = sobol.run()
print(f"Done. n_evaluations = {result.n_evaluations}")
print(f"Dominant parameter for T1: {result.dominant_param}")
print()
print(result.summary())

# -----------------------------------------------------------------------
# Sort by S1 for T1 (state index 0)
# -----------------------------------------------------------------------
order  = np.argsort(result.S1[:, 0])[::-1]
names  = [result.param_names[i] for i in order]
s1     = result.S1[order, 0]
s1_c   = result.S1_conf[order, 0]
st     = result.ST[order, 0]
st_c   = result.ST_conf[order, 0]

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "ProbOS — Sobol Sensitivity Analysis: Battery Thermal Runaway\n"
    f"State variable: T1 (Cell 1 Temperature) | "
    f"N_saltelli={1024} | n_steps=10 | "
    f"Dominant: {result.dominant_param}",
    fontsize=12, fontweight="bold",
)

x      = np.arange(len(names))
BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
THRESH = 0.05

# ---- Panel 1: S1 ----
ax = axes[0]
bars = ax.bar(x, np.maximum(s1, 0), color=BLUE, alpha=0.8,
              edgecolor="white", label="S1")
ax.errorbar(x, np.maximum(s1, 0), yerr=s1_c, fmt="none",
            color="black", capsize=3, linewidth=1)
ax.axhline(THRESH, color="red", linestyle="--", linewidth=1.2,
           label=f"Significance threshold (S1={THRESH})")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("First-order Sobol index S1")
ax.set_title("First-order effects (S1)\nDirect contribution of each parameter")
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars, s1):
    if val > 0.02:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(val, 0) + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7,
        )

# ---- Panel 2: ST ----
ax = axes[1]
bars = ax.bar(x, np.maximum(st, 0), color=ORANGE, alpha=0.8,
              edgecolor="white", label="ST")
ax.errorbar(x, np.maximum(st, 0), yerr=st_c, fmt="none",
            color="black", capsize=3, linewidth=1)
ax.axhline(THRESH, color="red", linestyle="--", linewidth=1.2,
           label=f"Significance threshold (ST={THRESH})")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Total-effect Sobol index ST")
ax.set_title(
    "Total effects (ST)\nDirect + all interaction effects"
)
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars, st):
    if val > 0.02:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(val, 0) + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7,
        )

plt.tight_layout()
out = "outputs/figures/week3_sobol_battery.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
