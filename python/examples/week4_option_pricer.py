"""
python/examples/week4_option_pricer.py

Week 4 Saturday: European option pricer using the ProbOS Monte Carlo kernel.

PURPOSE
-------
This example demonstrates that the SAME kernel built for battery thermal
runaway (MonteCarloEngine + Distribution ABC) generalises directly to
quantitative finance with zero changes to the kernel itself. Only a new
Model subclass is needed.

This is the core architectural claim of ProbOS: the kernel is domain
agnostic. Everything domain-specific (the asset price process, the option
payoff, the battery ODE) lives in a Model subclass; the kernel
(MonteCarloEngine, Distribution, SobolSensitivity, ProvenanceTracker)
never needs to know what domain it is being used for.

FINANCIAL MODEL
----------------
We price a European call option under the Black-Scholes-Merton
geometric Brownian motion (GBM) assumption:

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

where:
    S_t   = asset price at time t
    mu    = drift (under the risk-neutral measure, mu = r, the risk-free rate)
    sigma = volatility
    W_t   = standard Brownian motion

Under risk-neutral pricing, the exact discrete-time update (which is
what we implement -- not an Euler approximation, because GBM has a known
closed-form solution) is:

    S_{t+dt} = S_t * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )

where Z ~ N(0,1) is a standard normal random variable drawn once per
particle per timestep.

We do NOT use forward_batch() with explicit Euler here, because GBM has
an exact solution and using Euler would introduce discretisation bias
that does not exist in the true process. This double duty as a teaching
point: forward_batch() can implement either an Euler step OR an exact
update rule, as long as it advances state by one dt and is vectorised
over all N particles. The Model ABC does not care which.

PAYOFF AND PRICING
-------------------
A European call option can only be exercised at maturity T. Its payoff is:

    payoff = max(S_T - K, 0)

where K is the strike price. The option's fair price today is the
risk-neutral expected discounted payoff:

    price = exp(-r*T) * E[max(S_T - K, 0)]

The Monte Carlo estimator of this expectation is exactly what
MonteCarloEngine.run() already computes: the P50 (median) is NOT what we
want here -- we want the MEAN, which is available from the trajectories
directly, not from the percentiles. We extract result.trajectories and
take the mean ourselves, since MCResult does not currently expose a mean()
helper (a possible Month 2 addition).

We also have an analytical closed-form benchmark: the Black-Scholes
formula. Comparing our Monte Carlo price against the closed-form price
validates that OptionPricerModel and MonteCarloEngine are working
correctly together -- this is the same "validate against known physics"
discipline we used for BatteryModel2Cell against Kim 2007 ARC data.

UNCERTAINTY QUANTIFICATION ANGLE
----------------------------------
Unlike a textbook Black-Scholes calculator, ProbOS treats volatility
itself as UNCERTAIN (drawn from a prior distribution) rather than a
single fixed number. This mirrors exactly what we did for the battery's
activation energy Ea_SEI: in real markets, volatility is never known
with certainty (it is itself estimated from historical data, which has
sampling error). By placing a prior on sigma, we get not just a point
price estimate but a full P05/P50/P95 PRICE DISTRIBUTION reflecting
parameter uncertainty -- something a standard Black-Scholes calculator
cannot give you.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as scipy_norm

from python.src.distributions import Distribution, Normal
from python.src.monte_carlo import MonteCarloEngine
from python.src.state import FloatArray, Model

# ===========================================================================
# MODEL DEFINITION
# ===========================================================================


class OptionPricerModel(Model):
    """
    European call option pricer under Black-Scholes-Merton GBM dynamics.

    STATE VECTOR (state_dim = 1):
        state[:, 0] = S, the underlying asset price for each particle

    PARAMETER VECTOR (param_dim = 2):
        params[:, 0] = mu    -- drift (set to risk-free rate r under
                                 risk-neutral pricing)
        params[:, 1] = sigma -- volatility (the UNCERTAIN parameter we
                                 place a prior on)

    Unlike BatteryModel2Cell, this model has only ONE state variable
    (the asset price) and uses the EXACT GBM solution rather than an
    Euler-discretised ODE, because GBM admits a closed form. This shows
    forward_batch() is flexible: any vectorised one-step state update
    is valid, not just explicit Euler integration of an ODE.

    Parameters
    ----------
    S0 : float
        Initial asset price at t=0. Stored at construction time and
        returned by initial_state().
    K : float
        Strike price of the option. Stored for use by price_option().
    T : float
        Time to maturity in years. Stored for use by price_option().
    r : float
        Risk-free interest rate (annualised, continuously compounded).
        Used both as the GBM drift (risk-neutral pricing) and as the
        discount rate when computing the option price.
    """

    def __init__(self, S0: float = 100.0, K: float = 100.0,
                 T: float = 1.0, r: float = 0.05) -> None:
        self._S0 = S0
        self._K  = K
        self._T  = T
        self._r  = r

    # ------------------------------------------------------------------
    # Model ABC required properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Single state variable: the asset price S."""
        return 1

    @property
    def param_dim(self) -> int:
        """Two parameters: drift (mu) and volatility (sigma)."""
        return 2

    def param_names(self) -> list[str]:
        """Human-readable names matching param column order."""
        return ["mu", "sigma"]

    def initial_state(self) -> FloatArray:
        """
        Every particle starts at the same known initial price S0.
        Unlike the battery model (where initial state is also fixed
        but state_dim=8), here state_dim=1 so this is a length-1 array.
        """
        return np.array([self._S0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Core dynamics: exact GBM update (NOT Euler -- see module docstring)
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        state:  FloatArray,   # shape (N, 1) -- current asset price per particle
        params: FloatArray,   # shape (N, 2) -- [mu, sigma] per particle
        dt:     float,        # time step in years
    ) -> FloatArray:
        """
        Advance all N particles by one EXACT GBM step of dt years.

        We do not use np.random.default_rng() with a stored seed here --
        the random draws come from a fresh standard normal sample each
        call, which is the correct behaviour for a stochastic process
        simulation (we want a NEW random shock each timestep, not a
        deterministic Euler-style drift+diffusion split). This is
        different from BatteryModel2Cell.forward_batch(), which is
        purely deterministic given state and params (no randomness
        inside forward_batch itself -- all randomness in the battery
        model comes from the PARAMETER priors, sampled once at the
        start of the MC run, not re-sampled every step).

        Here, by contrast, GBM is a genuinely STOCHASTIC differential
        equation: every timestep needs a fresh Brownian increment. This
        is an important conceptual distinction documented for future
        readers of this codebase:
            - Battery model: parameter uncertainty propagated through
              a deterministic ODE (epistemic uncertainty in fixed
              physical constants).
            - Option pricer: genuine stochastic process simulation
              (aleatoric uncertainty in the asset's future path itself).
        MonteCarloEngine handles both correctly because forward_batch()
        is allowed to use np.random internally if the underlying
        process truly is stochastic.

        Parameters
        ----------
        state  : shape (N, 1) -- current S for each particle
        params : shape (N, 2) -- [mu, sigma] for each particle
        dt     : float        -- time step in years

        Returns
        -------
        FloatArray of shape (N, 1) -- updated S for each particle
        """
        S     = state[:, 0]      # shape (N,)
        mu    = params[:, 0]     # shape (N,)
        sigma = params[:, 1]     # shape (N,)

        N = S.shape[0]

        # Draw one fresh standard normal shock per particle for this step.
        # Using the module-level np.random call (not a stored Generator)
        # is intentional here: each call to forward_batch() represents
        # genuinely new randomness in the asset's future path, not a
        # re-sampling of fixed physical parameters.
        Z: FloatArray = np.random.standard_normal(N)

        # Exact GBM solution (NOT an Euler approximation):
        #   S_{t+dt} = S_t * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
        # This is exact because log(S_t) follows a Brownian motion with
        # drift, and exponentiating a Gaussian increment gives the exact
        # one-step transition density of GBM -- no discretisation error.
        drift_term     = (mu - 0.5 * sigma ** 2) * dt
        diffusion_term = sigma * np.sqrt(dt) * Z

        new_S: FloatArray = S * np.exp(drift_term + diffusion_term)

        # Reshape to (N, 1) to match the (N, state_dim) contract.
        return new_S.reshape(N, 1)

    # ------------------------------------------------------------------
    # Domain-specific helper: NOT part of the Model ABC, but useful for
    # this example to turn raw trajectories into an option price.
    # ------------------------------------------------------------------

    def price_option(self, final_prices: FloatArray) -> tuple[float, float]:
        """
        Compute the Monte Carlo European call option price and its
        standard error from a batch of simulated terminal asset prices.

        Parameters
        ----------
        final_prices : shape (N,)
            Simulated S_T values for N particles, taken at the FINAL
            timestep of a completed MonteCarloEngine run.

        Returns
        -------
        price : float
            Risk-neutral discounted expected payoff -- the MC option price.
        std_error : float
            Standard error of the MC price estimate (sigma_payoff / sqrt(N)),
            i.e. the same sigma/sqrt(N) convergence quantity that
            MCResult.convergence reports for the battery model, but
            computed here on the PAYOFF rather than on the raw state.
        """
        payoff = np.maximum(final_prices - self._K, 0.0)   # shape (N,)
        discounted_payoff = np.exp(-self._r * self._T) * payoff

        price     = float(np.mean(discounted_payoff))
        n = len(discounted_payoff)
        std_error = float(np.std(discounted_payoff, ddof=1) / np.sqrt(n))
        return price, std_error

    def black_scholes_price(self, sigma: float) -> float:
        """
        Closed-form Black-Scholes price for a European call, used ONLY
        as a validation benchmark against our Monte Carlo price -- the
        same role Kim 2007 ARC data plays for validating BatteryModel2Cell.

        This formula assumes sigma is a SINGLE KNOWN constant, which is
        why it gives one number rather than a distribution. Our Monte
        Carlo approach instead places a PRIOR on sigma (see priors below)
        and propagates that uncertainty through to a full price
        distribution -- the key UQ advantage over textbook Black-Scholes.

        Parameters
        ----------
        sigma : float
            A single volatility value (e.g. the prior mean) at which to
            evaluate the closed-form price for comparison purposes.

        Returns
        -------
        float
            Black-Scholes call price at the given sigma.
        """
        S0, K, T, r = self._S0, self._K, self._T, self._r
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(
            S0 * scipy_norm.cdf(d1) - K * np.exp(-r * T) * scipy_norm.cdf(d2)
        )


def build_option_priors(r: float = 0.05) -> list[Distribution]:
    """
    Prior distributions for the option pricer's 2 parameters.

    The DRIFT (mu) is set to the risk-free rate r with essentially zero
    uncertainty (a tight Normal), because under risk-neutral pricing the
    drift is a MODELLING CHOICE, not an empirically estimated quantity --
    we are not uncertain about it the way we are about volatility.

    The VOLATILITY (sigma) is the genuinely uncertain parameter: in
    practice, sigma is estimated from historical returns and carries
    real sampling uncertainty. We use a Normal prior centred at a
    plausible annualised volatility (20%) with a modest spread (2%),
    analogous to how Ea_SEI in the battery model is a physical constant
    known only approximately from calorimetry experiments.

    Parameters
    ----------
    r : float
        Risk-free rate to centre the (near-deterministic) mu prior on.

    Returns
    -------
    list[Distribution]
        [mu_prior, sigma_prior] matching OptionPricerModel.param_names().
    """
    return [
        Normal(mu=r, sigma=1e-6),      # mu: risk-free rate, ~fixed by assumption
        Normal(mu=0.20, sigma=0.02),   # sigma: 20% annualised vol, 2% prior uncertainty
    ]


# ===========================================================================
# DEMO / VALIDATION SCRIPT
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  ProbOS Week 4 Saturday -- European Option Pricer")
    print("  Demonstrates kernel generalisation: battery -> finance")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: set up the model and priors
    # ------------------------------------------------------------------
    model  = OptionPricerModel(S0=100.0, K=100.0, T=1.0, r=0.05)
    priors = build_option_priors(r=0.05)

    # ------------------------------------------------------------------
    # Step 2: run the Monte Carlo engine
    #   N=20000 particles for a tight standard error
    #   n_steps=252 -- one simulation step per trading day over 1 year,
    #                   a standard convention in quantitative finance
    #   dt = T / n_steps in years
    # ------------------------------------------------------------------
    N_STEPS = 252
    DT      = 1.0 / N_STEPS

    engine = MonteCarloEngine(
        model, priors, N=20_000, n_steps=N_STEPS, dt=DT, seed=42
    )
    result = engine.run()

    # ------------------------------------------------------------------
    # Step 3: extract terminal prices and compute the option price
    # ------------------------------------------------------------------
    final_prices: FloatArray = result.trajectories[:, -1, 0]  # shape (N,)
    mc_price, mc_std_err = model.price_option(final_prices)

    # ------------------------------------------------------------------
    # Step 4: compute the closed-form benchmark at the prior MEAN sigma
    # ------------------------------------------------------------------
    bs_price = model.black_scholes_price(sigma=0.20)

    # ------------------------------------------------------------------
    # Step 5: report results
    # ------------------------------------------------------------------
    print()
    print(f"  Underlying S0      : {model._S0:.2f}")
    print(f"  Strike K           : {model._K:.2f}")
    print(f"  Maturity T         : {model._T:.2f} years")
    print(f"  Risk-free rate r   : {model._r:.4f}")
    print("  Volatility prior   : Normal(mu=0.20, sigma=0.02)")
    print(f"  Particles N        : {result.n_particles}")
    print(f"  Steps              : {result.n_steps} (daily, dt={DT:.6f})")
    print()
    print(f"  Black-Scholes price (sigma=0.20 fixed) : {bs_price:.4f}")
    print(f"  Monte Carlo price (sigma uncertain)    : {mc_price:.4f}")
    print(f"  Monte Carlo standard error             : {mc_std_err:.4f}")
    print(f"  Relative difference                    : "
          f"{abs(mc_price - bs_price) / bs_price * 100:.2f}%")
    print()

    # ------------------------------------------------------------------
    # Step 6: validation assertion -- MC price should be close to the
    # closed-form price (within a few standard errors), confirming the
    # OptionPricerModel + MonteCarloEngine combination is correct.
    # ------------------------------------------------------------------
    diff_in_std_errors = abs(mc_price - bs_price) / mc_std_err
    print(f"  Difference in standard errors: {diff_in_std_errors:.2f}")
    if diff_in_std_errors < 5.0:
        print("  VALIDATION: PASS (within 5 standard errors of closed form)")
    else:
        print("  VALIDATION: WARNING -- check model implementation")

    print()
    print("=" * 70)
    print("  Key takeaway: identical MonteCarloEngine, Distribution,")
    print("  and Model ABC used for BatteryModel2Cell now price options.")
    print("  Only a new Model subclass was required -- zero kernel changes.")
    print("=" * 70)
