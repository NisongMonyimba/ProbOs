import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from python.src.distributions import Normal

R_GAS: float = 8.314462
T_ONSET: float = 403.15


def arrhenius_rate(A: float, Ea: float, T: float) -> float:
    return float(A * np.exp(-Ea / (R_GAS * T)))


def main() -> None:
    print()
    print("=" * 65)
    print("  ProbOS Week 1 -- Example 2: Normal Distribution")
    print("  Battery SEI Activation Energy Uncertainty")
    print("=" * 65)
    print()

    Ea_SEI = Normal(mu=1.3508e5, sigma=5.0e3)
    print("DISTRIBUTION: SEI Decomposition Activation Energy")
    print(f"  Model:     {Ea_SEI}")
    print(f"  Mean:      {Ea_SEI.mean():>12.1f} J/mol")
    print(f"  Std dev:   {Ea_SEI.sigma:>12.1f} J/mol")
    print(f"  CV: {100 * Ea_SEI.sigma / Ea_SEI.mean():.1f}%")
    print()

    rng = np.random.default_rng(42)
    N = 5_000
    samples = Ea_SEI.sample(N, rng=rng)
    print(f"SAMPLING: {N:,} batteries")
    print(
        f"  Empirical mean: {samples.mean():>10.1f} J/mol"
        f"  (true: {Ea_SEI.mean():.1f})"
    )
    print(
        f"  Empirical std:  {samples.std(ddof=1):>10.1f} J/mol"
        f"  (true: {Ea_SEI.sigma:.1f})"
    )
    print(f"  Min: {samples.min():.1f}  Max: {samples.max():.1f}")
    print()

    print("PERCENTILES: Which batteries are dangerous?")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        q = float(Ea_SEI.ppf(np.array([p / 100.0]))[0])
        rate_this = arrhenius_rate(1.667e15, q, T_ONSET)
        rate_mean = arrhenius_rate(1.667e15, Ea_SEI.mean(), T_ONSET)
        ratio = rate_this / rate_mean
        label = ""
        if p <= 5:
            label = " <- FASTEST (highest risk)"
        elif p >= 95:
            label = " <- SLOWEST (lowest risk)"
        elif p == 50:
            label = " <- MEDIAN"
        print(f"  P{p:02d}: Ea={q:>10.1f} J/mol  ratio={ratio:.3f}x{label}")
    print()

    p05 = float(Ea_SEI.ppf(np.array([0.05]))[0])
    p95 = float(Ea_SEI.ppf(np.array([0.95]))[0])
    rate_p05 = arrhenius_rate(1.667e15, p05, T_ONSET)
    rate_mean = arrhenius_rate(1.667e15, Ea_SEI.mean(), T_ONSET)
    rate_p95 = arrhenius_rate(1.667e15, p95, T_ONSET)
    deg = T_ONSET - 273.15

    print("THE PHYSICAL INSIGHT:")
    print(f"  P05 battery (Ea={p05:.0f} J/mol):")
    rate_str = f"{rate_p05:.4e} s^-1"
    ratio_str = f"{rate_p05 / rate_mean:.2f}x FASTER"
    print(f"    Rate at {deg:.0f}C = {rate_str}  ({ratio_str})")
    print()
    print(f"  Mean battery (Ea={Ea_SEI.mean():.0f} J/mol):")
    print(f"    Rate at {deg:.0f}C = {rate_mean:.4e} s^-1")
    print("    Deterministic model uses this for ALL batteries.")
    print()
    print(f"  P95 battery (Ea={p95:.0f} J/mol):")
    rate_str2 = f"{rate_p95:.4e} s^-1"
    ratio_str2 = f"{rate_p95 / rate_mean:.2f}x SLOWER"
    print(f"    Rate at {deg:.0f}C = {rate_str2}  ({ratio_str2})")
    print()
    print("  Deterministic model sees ONLY the mean battery.")
    print("  ProbOS sees ALL 5,000 batteries -- finds the P05 tail.")
    print()

    print("NUMERICAL STABILITY: Why log_pdf matters")
    x_ext = np.array([Ea_SEI.mu + 50 * Ea_SEI.sigma])
    pdf_v = Ea_SEI.pdf(x_ext)[0]
    logpdf_v = Ea_SEI.log_pdf(x_ext)[0]
    print(f"  x = mu + 50*sigma = {x_ext[0]:.0f} J/mol")
    print(f"  pdf(x)      = {pdf_v}  (underflows to 0)")
    print(f"  log_pdf(x)  = {logpdf_v:.4f}  (finite -- analytical)")
    bad = float(np.log(pdf_v + 1e-300))
    print(f"  log(pdf(x)) = {bad:.4f}  (WRONG: -inf)")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = (
        f"Battery SEI Activation Energy | "
        f"Normal(mu={Ea_SEI.mu:.0f}, sigma={Ea_SEI.sigma:.0f}) | "
        f"N={N:,}"
    )
    fig.suptitle(title, fontsize=12)

    ax = axes[0]
    ax.hist(
        samples, bins=50, density=True, alpha=0.6,
        color="#4C9ED9", edgecolor="white", linewidth=0.5,
        label=f"Sampled ({N:,} batteries)",
    )
    x_lo = Ea_SEI.mu - 4.5 * Ea_SEI.sigma
    x_hi = Ea_SEI.mu + 4.5 * Ea_SEI.sigma
    x_r = np.linspace(x_lo, x_hi, 500)
    ax.plot(x_r, Ea_SEI.pdf(x_r), "k-", lw=2, label="True density")
    ax.axvline(
        p05, color="#E74C3C", ls="--", lw=1.5,
        label=f"P05={p05:.0f}",
    )
    ax.axvline(
        Ea_SEI.mu, color="#2ECC71", ls="-", lw=1.5,
        label=f"mu={Ea_SEI.mu:.0f}",
    )
    ax.set_xlabel("Ea_SEI [J/mol]", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title("SEI Activation Energy", fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15)

    ax2 = axes[1]
    A_SEI = 1.667e15
    rates = A_SEI * np.exp(-samples / (R_GAS * T_ONSET))
    ax2.hist(
        rates, bins=50, density=True, alpha=0.6,
        color="#E67E22", edgecolor="white", linewidth=0.5,
        label=f"Rate at {deg:.0f}C",
    )
    ax2.axvline(
        A_SEI * np.exp(-p05 / (R_GAS * T_ONSET)),
        color="#E74C3C", ls="--", lw=1.5, label="P05 rate",
    )
    ax2.axvline(
        A_SEI * np.exp(-Ea_SEI.mu / (R_GAS * T_ONSET)),
        color="#2ECC71", ls="-", lw=1.5, label="Mean rate",
    )
    ax2.set_xlabel("SEI decomposition rate [s^-1]", fontsize=11)
    ax2.set_ylabel("Probability density", fontsize=11)
    ax2.set_title(f"Rate at {deg:.0f}C", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig(
        "outputs/figures/week1_battery_Ea_distribution.png",
        dpi=150, bbox_inches="tight",
    )
    print("Figure saved: week1_battery_Ea_distribution.png")
    print()
    print("=" * 65)
    print("  Week 1 Example 2 complete.")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
