"""
================================================================================
probos.parameter_priors  --  Prior distributions for BatteryModel2Cell
================================================================================

WHAT IS THIS FILE?
------------------
Before running the Monte Carlo engine, we need a probability distribution
for each of the 15 uncertain parameters in BatteryModel2Cell.

These distributions represent our PRIOR BELIEFS about each parameter --
what we know from the literature before seeing any experimental data.

In Week 3, the Monte Carlo engine will:
  1. Draw N=5000 samples from each prior distribution
  2. Stack them into a params array of shape (N, 15)
  3. Pass the array to BatteryModel2Cell.forward_batch()

HOW TO USE THIS FILE:
---------------------
    from python.src.parameter_priors import build_battery_priors
    from python.src.battery_model import BatteryModel2Cell

    model  = BatteryModel2Cell()
    priors = build_battery_priors()   # list of 15 Distribution objects

    # Sample N=5000 parameter sets
    rng    = np.random.default_rng(seed=42)
    N      = 5000
    params = np.column_stack([
        d.sample(N, rng=rng) for d in priors
    ])   # shape (N, 15)

WHERE DO THE UNCERTAINTY RANGES COME FROM?
------------------------------------------
Kim et al. (2007) report nominal values.
We model uncertainty as:
  - Activation energies Ea:  Normal(mu=nominal, sigma=5% of nominal)
    Justification: manufacturing variability in electrode chemistry
    is typically 3-7% for activation energies (literature consensus).

  - Pre-exponential factors A:  LogNormal with 10% coefficient of variation
    Justification: A spans many orders of magnitude; multiplicative
    uncertainty (LogNormal) is more appropriate than additive (Normal).

  - Heats of reaction H:  Normal(mu=nominal, sigma=5% of nominal)

  - Cell mass m_cell:  Normal(mu=0.045, sigma=0.002)
    Justification: 18650 cell mass varies by about +/- 2g in practice.

  - Specific heat Cp:  Normal(mu=800, sigma=40)
    Justification: +/- 5% is typical for battery calorimetry measurements.

  - Convective coefficient h_conv:  Uniform(low=2.0, high=10.0)
    Justification: h_conv is highly geometry-dependent and poorly known.

  - Surface area A_surf:  Normal(mu=3.5e-3, sigma=1e-4)
    Justification: small manufacturing variation in cell dimensions.

  - Ambient temperature T_amb:  Normal(mu=298.15, sigma=5.0)
    Justification: test environment varies by a few degrees.

  - Onset temperature T_onset:  Normal(mu=403.15, sigma=5.0)
    Justification: ARC onset is repeatable to within 5 C (Kim 2007).

REFERENCE:
  Kim, G-H., Pesaran, A., Spotnitz, R. (2007).
  J. Power Sources 170(2), 476-489.
================================================================================
"""

from __future__ import annotations

import numpy as np

from python.src.distributions import Distribution, Normal, LogNormal, Uniform
from python.src.battery_model import BatteryModel2Cell


def build_battery_priors() -> list[Distribution]:
    """
    Build the list of 15 prior distributions for BatteryModel2Cell.

    The list is ordered to match BatteryModel2Cell.param_names():
      Index 0  -> Ea_SEI
      Index 1  -> A_SEI
      ...
      Index 14 -> T_onset

    Returns
    -------
    list of Distribution objects, length 15.
    Each element is one of: Normal, LogNormal, Uniform.

    Usage
    -----
    priors = build_battery_priors()
    rng    = np.random.default_rng(seed=42)
    N      = 5000

    # Sample all 15 parameters for all N particles at once
    params = np.column_stack([d.sample(N, rng=rng) for d in priors])
    # params has shape (N, 15)
    """
    # Nominal values from Kim 2007 (same as BatteryModel2Cell.nominal_params)
    nom = BatteryModel2Cell().nominal_params()

    # ------------------------------------------------------------------
    # Helper: Normal distribution with coefficient of variation cv
    # cv = sigma / mu, so sigma = cv * mu
    # Example: cv=0.05 means 5% standard deviation relative to the mean
    # ------------------------------------------------------------------
    def normal_cv(index: int, cv: float) -> Normal:
        mu = float(nom[index])
        sigma = cv * mu
        return Normal(mu=mu, sigma=sigma)

    # ------------------------------------------------------------------
    # Helper: LogNormal distribution with coefficient of variation cv
    # For LogNormal: if X ~ LN(mu_log, sigma_log), then
    #   E[X] = exp(mu_log + sigma_log^2 / 2)
    #   CV^2 = exp(sigma_log^2) - 1
    # We parameterise by the desired mean and CV for clarity.
    # ------------------------------------------------------------------
    def lognormal_cv(index: int, cv: float) -> LogNormal:
        mean = float(nom[index])
        # Derive log-space parameters from mean and CV
        # sigma_log^2 = log(1 + CV^2)
        # mu_log = log(mean) - sigma_log^2 / 2
        sigma_log = float(np.sqrt(np.log(1.0 + cv ** 2)))
        mu_log    = float(np.log(mean) - 0.5 * sigma_log ** 2)
        return LogNormal(mu=mu_log, sigma=sigma_log)

    # ------------------------------------------------------------------
    # Build priors in the same order as BatteryModel2Cell.param_names()
    # ------------------------------------------------------------------
    priors: list[Distribution] = [
        # Index  Name        Distribution   Justification
        normal_cv(0,  0.05),   # Ea_SEI   Normal, 5% CV  -- electrode chemistry
        lognormal_cv(1, 0.10), # A_SEI    LogNormal, 10% CV -- spans orders of mag
        normal_cv(2,  0.05),   # H_SEI    Normal, 5% CV  -- calorimetry
        normal_cv(3,  0.05),   # Ea_anode Normal, 5% CV
        lognormal_cv(4, 0.10), # A_anode  LogNormal, 10% CV
        normal_cv(5,  0.05),   # H_anode  Normal, 5% CV
        normal_cv(6,  0.05),   # Ea_cath  Normal, 5% CV
        lognormal_cv(7, 0.10), # A_cath   LogNormal, 10% CV
        normal_cv(8,  0.05),   # H_cath   Normal, 5% CV
        Normal(mu=0.045,   sigma=0.002),   # m_cell  +/- 2g
        Normal(mu=800.0,   sigma=40.0),    # Cp      +/- 5%
        Uniform(low=2.0,   high=10.0),     # h_conv  poorly known
        Normal(mu=3.5e-3,  sigma=1e-4),    # A_surf  small mfg variation
        Normal(mu=298.15,  sigma=5.0),     # T_amb   +/- 5 C
        Normal(mu=403.15,  sigma=5.0),     # T_onset +/- 5 C (Kim 2007)
    ]

    # Sanity check: list length must equal param_dim
    model = BatteryModel2Cell()
    assert len(priors) == model.param_dim, (
        f"build_battery_priors() returned {len(priors)} distributions "
        f"but param_dim = {model.param_dim}"
    )

    return priors


def sample_params(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw N parameter sets from the battery priors.

    This is a convenience wrapper around build_battery_priors().
    The Monte Carlo engine (Week 3) will call this directly.

    Parameters
    ----------
    N   : number of particles (Monte Carlo sample size)
    rng : numpy random generator (use np.random.default_rng(seed=42))

    Returns
    -------
    params : np.ndarray of shape (N, 15)
        params[i, j] is the j-th parameter value for particle i.
        Each column is drawn from a different prior distribution.

    Example
    -------
    rng    = np.random.default_rng(seed=42)
    params = sample_params(N=5000, rng=rng)
    # params.shape == (5000, 15)
    """
    priors = build_battery_priors()
    return np.column_stack([d.sample(N, rng=rng) for d in priors])


def print_prior_summary() -> None:
    """
    Print a human-readable summary of all 15 prior distributions.
    Useful for debugging and for the research paper.
    """
    model  = BatteryModel2Cell()
    priors = build_battery_priors()
    names  = model.param_names()
    nom    = model.nominal_params()

    print()
    print("=" * 72)
    print("  BATTERY PARAMETER PRIOR DISTRIBUTIONS")
    print("  Reference: Kim et al. (2007), J. Power Sources 170(2), 476-489")
    print("=" * 72)
    print(f"  {'#':>2}  {'Name':<12}  {'Type':<12}  {'Mean':>14}  {'Std':>12}")
    print("-" * 72)

    for i, (name, dist, nominal) in enumerate(zip(names, priors, nom)):
        dist_type = type(dist).__name__
        try:
            mean = dist.mean()
            std  = float(np.sqrt(dist.variance()))
        except (NotImplementedError, AttributeError, ValueError):
            # Some Distribution subclasses may not implement mean()/
            # variance() (e.g. Empirical without enough data), or a
            # distribution parameterisation may make variance()
            # momentarily undefined. This only affects the printed
            # summary table below, never any actual computation --
            # fall back to the nominal value for display purposes.
            mean = nominal
            std  = float("nan")
        print(f"  {i:>2}  {name:<12}  {dist_type:<12}  {mean:>14.4g}  {std:>12.4g}")

    print("=" * 72)
    print()


if __name__ == "__main__":
    print_prior_summary()

    # Demonstrate sampling
    print("Sampling N=5 parameter sets to verify shapes:")
    rng    = np.random.default_rng(seed=42)
    params = sample_params(N=5, rng=rng)
    print(f"  params.shape = {params.shape}")
    print(f"  params dtype = {params.dtype}")
    print()
    print("  First row (particle 0):")
    model = BatteryModel2Cell()
    for j, name in enumerate(model.param_names()):
        print(f"    {name:<12} = {params[0, j]:.4g}")


__all__ = ["build_battery_priors", "sample_params", "print_prior_summary"]
