"""
python/examples/week4_ed_queue.py

Week 4 Saturday: Emergency Department M/M/1 queue simulator using the
ProbOS Monte Carlo kernel.

PURPOSE
-------
A second demonstration that the kernel (MonteCarloEngine + Distribution
ABC + Model ABC) generalises beyond physics/finance into discrete-event
operations research. This is the "hospital operations" pillar of the
Month 1 plan.

Unlike the battery model (continuous ODE state) and the option pricer
(continuous stochastic process), an M/M/1 queue is naturally a discrete
COUNT process: the number of patients waiting. We represent it as a
continuous-valued state purely for vectorisation convenience (so we can
reuse the exact same forward_batch / MonteCarloEngine machinery), then
round to the nearest non-negative integer when reporting results.

QUEUEING THEORY MODEL
-----------------------
M/M/1 notation (Kendall's notation) means:
    M -- arrivals follow a Markov (Poisson) process, rate lambda
    M -- service times are Markov (exponential), rate mu
    1 -- a single server (e.g. one triage nurse / one ED bay)

For a stable queue we require lambda < mu (arrival rate less than
service rate), otherwise the queue grows without bound.

Exact M/M/1 theory gives closed-form steady-state results we use as a
validation benchmark (the same role Kim 2007 plays for the battery model
and Black-Scholes plays for the option pricer):

    rho (utilisation)        = lambda / mu
    L   (mean number in system, steady state) = rho / (1 - rho)
    W   (mean time in system, steady state)   = L / lambda  (Little's Law)

DISCRETE-EVENT vs OUR CONTINUOUS-STATE APPROXIMATION
------------------------------------------------------
A textbook M/M/1 simulation is event-driven: you track the exact instant
of each arrival and each departure. ProbOS's kernel instead advances all
particles by a FIXED dt each step (this is what forward_batch expects).
We approximate the discrete birth-death queue process by a stochastic
difference equation over small dt:

    Q_{t+dt} = max(0, Q_t + dN_arrival - dN_departure)

where, over a short interval dt:
    dN_arrival   ~ Poisson(lambda * dt)   (number of new arrivals)
    dN_departure ~ Poisson(mu * dt) if Q_t > 0, else 0
                   (departures can only happen if someone is being served)

This is the discrete-time analogue of a continuous-time birth-death
Markov chain, and converges to the exact M/M/1 process as dt -> 0. We
validate this convergence empirically below by comparing simulated
steady-state L against the closed-form L = rho/(1-rho).

UNCERTAINTY QUANTIFICATION ANGLE
-----------------------------------
As with the other two examples, we do NOT treat lambda and mu as fixed
known constants. Real EDs experience day-to-day variability in arrival
rate (different triage acuity mixes, time-of-day effects averaged out,
seasonal flu surges) and service rate (staffing variability, case-mix
variability). We place priors on both lambda and mu, mirroring exactly
how we placed a prior on Ea_SEI for the battery and on sigma for the
option pricer. The result is a full DISTRIBUTION over steady-state queue
length and waiting time, not a single number -- directly useful for
capacity planning ("what is the P95 number of patients waiting, not just
the average?").
"""

from __future__ import annotations

import numpy as np

from python.src.distributions import Distribution, Uniform
from python.src.monte_carlo import MonteCarloEngine
from python.src.state import FloatArray, Model

# ===========================================================================
# MODEL DEFINITION
# ===========================================================================


class EDQueueModel(Model):
    """
    M/M/1 Emergency Department queue, simulated as a continuous-state
    approximation of a discrete birth-death Markov chain.

    STATE VECTOR (state_dim = 1):
        state[:, 0] = Q, the number of patients in the ED (waiting +
                          being served) for each particle

    PARAMETER VECTOR (param_dim = 2):
        params[:, 0] = lam  -- patient arrival rate (patients per hour)
        params[:, 1] = mu   -- service completion rate (patients per
                               hour, per single server/triage bay)

    Parameters
    ----------
    Q0 : float
        Initial queue length at t=0 (number of patients already in
        the ED when the simulation starts). Defaults to 0 (empty ED
        at simulation start).
    """

    def __init__(self, Q0: float = 0.0, seed: int = 42) -> None:
        self._Q0 = Q0
        # Per-instance seeded Generator (Month 3 Week 9 fix --
        # see OptionPricerModel's __init__ for the full
        # rationale). This also lets Week 8's
        # week8_ed_queue_filter.py drop its earlier explicit
        # np.random.seed() doctrine exception entirely.
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Model ABC required properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Single state variable: the number of patients in the ED."""
        return 1

    @property
    def param_dim(self) -> int:
        """Two parameters: arrival rate lambda and service rate mu."""
        return 2

    def param_names(self) -> list[str]:
        """Human-readable names matching param column order."""
        return ["lambda", "mu"]

    def initial_state(self) -> FloatArray:
        """Every particle starts with the same initial queue length."""
        return np.array([self._Q0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Core dynamics: discrete-time birth-death approximation
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        state:  FloatArray,   # shape (N, 1) -- current queue length per particle
        params: FloatArray,   # shape (N, 2) -- [lambda, mu] per particle
        dt:     float,        # time step in hours
    ) -> FloatArray:
        """
        Advance all N particles by one discrete-time birth-death step
        of dt hours.

        We draw Poisson-distributed arrival and departure counts using
        NumPy's vectorised np.random.poisson, which accepts an array of
        rate parameters and returns one draw per element -- exactly the
        same vectorisation pattern used for the Gaussian shock in
        OptionPricerModel.forward_batch(), just with a different
        distribution because arrivals/departures are COUNTS, not
        continuous price changes.

        IMPORTANT: like the option pricer (and unlike the battery model),
        this forward_batch() is genuinely stochastic -- it draws new
        random numbers every call, simulating the inherent randomness of
        patient arrivals and service completions, not just propagating
        uncertainty in a few fixed parameters through a deterministic ODE.

        Departures are masked to occur ONLY where Q_t > 0 (you cannot
        discharge a patient from an empty queue -- the server is idle
        when there is nobody to serve). This masking is the vectorised
        equivalent of the standard M/M/1 birth-death chain's boundary
        condition at state 0.

        Parameters
        ----------
        state  : shape (N, 1) -- current Q for each particle
        params : shape (N, 2) -- [lambda, mu] for each particle
        dt     : float        -- time step in hours

        Returns
        -------
        FloatArray of shape (N, 1) -- updated Q for each particle
        (non-negative by construction via np.maximum)
        """
        Q   = state[:, 0]      # shape (N,)
        lam = params[:, 0]     # shape (N,)
        mu  = params[:, 1]     # shape (N,)

        N = Q.shape[0]

        # Arrivals: Poisson(lambda * dt) -- always possible regardless
        # of current queue length (patients keep arriving even when the
        # ED is busy; they just wait longer).
        arrivals = self._rng.poisson(lam * dt).astype(np.float64)

        # Departures: Poisson(mu * dt), but ONLY where Q > 0. If the
        # queue is empty, the server is idle and no departure can occur
        # this step -- we zero out the departure draw in that case using
        # np.where, the vectorised equivalent of an if/else per particle.
        raw_departures = self._rng.poisson(mu * dt).astype(np.float64)
        departures = np.where(Q > 0, raw_departures, 0.0)

        # Update and clip at zero: the queue can never go negative.
        # This clip is the discrete-time analogue of the M/M/1 chain's
        # reflecting boundary at state 0 (you cannot have -1 patients).
        new_Q: FloatArray = np.maximum(Q + arrivals - departures, 0.0)

        return new_Q.reshape(N, 1)

    # ------------------------------------------------------------------
    # Domain-specific helper: NOT part of Model ABC. Computes the
    # closed-form M/M/1 steady-state benchmark for validation.
    # ------------------------------------------------------------------

    @staticmethod
    def steady_state_theory(lam: float, mu: float) -> tuple[float, float, float]:
        """
        Closed-form M/M/1 steady-state quantities, used to validate the
        Monte Carlo simulation -- the same validation role Kim 2007 ARC
        data plays for BatteryModel2Cell and the Black-Scholes formula
        plays for OptionPricerModel.

        Parameters
        ----------
        lam : float
            Arrival rate (patients/hour).
        mu : float
            Service rate (patients/hour).

        Returns
        -------
        rho : float
            Server utilisation, lambda/mu. Must be < 1 for a stable
            (non-exploding) queue.
        L : float
            Mean number of patients in the system at steady state.
        W : float
            Mean time a patient spends in the system at steady state
            (waiting + being served), via Little's Law: W = L / lambda.

        Raises
        ------
        ValueError
            If rho >= 1 (unstable queue -- arrivals exceed service
            capacity, so steady state does not exist; the queue grows
            without bound).
        """
        rho = lam / mu
        if rho >= 1.0:
            raise ValueError(
                f"Unstable queue: rho={rho:.4f} >= 1. "
                f"Arrival rate must be strictly less than service rate."
            )
        L = rho / (1.0 - rho)
        W = L / lam
        return rho, L, W


def build_ed_queue_priors() -> list[Distribution]:
    """
    Prior distributions for the ED queue's 2 parameters.

    We use Uniform priors (rather than Normal, as used for the battery
    and option pricer) to reflect GENUINE RANGE UNCERTAINTY rather than
    a best-guess-plus-noise: hospital administrators typically know
    arrival/service rates only within a plausible operating RANGE (e.g.
    "between 8 and 12 patients per hour during a typical shift"), not as
    a precise estimate with a symmetric error bar. This demonstrates
    that ProbOS's kernel is agnostic to WHICH Distribution subclass is
    used for a prior -- any of the 5 concrete distributions can be
    swapped in per parameter as the domain expert judges appropriate.

    Returns
    -------
    list[Distribution]
        [lambda_prior, mu_prior] matching EDQueueModel.param_names().
        Bounds are chosen so that mu's lower bound (12) safely exceeds
        lambda's upper bound (12) -- wait, we choose mu's range strictly
        above lambda's range to guarantee rho < 1 (a stable queue) for
        every possible draw, avoiding the ValueError in
        steady_state_theory for any sampled parameter combination.
    """
    return [
        Uniform(low=8.0,  high=12.0),   # lambda: 8-12 patients/hour
        Uniform(low=14.0, high=18.0),   # mu:     14-18 patients/hour (single bay)
    ]


# ===========================================================================
# DEMO / VALIDATION SCRIPT
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  ProbOS Week 4 Saturday -- Emergency Department M/M/1 Queue")
    print("  Demonstrates kernel generalisation: physics/finance -> ops")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: set up the model and priors
    # ------------------------------------------------------------------
    model  = EDQueueModel(Q0=0.0)
    priors = build_ed_queue_priors()

    # ------------------------------------------------------------------
    # Step 2: run the Monte Carlo engine
    #   N=5000 particles, each an independent "parallel universe" ED
    #   n_steps -- enough hours of simulated time to reach steady state
    #   dt -- small enough that at most ~1 arrival/departure occurs per
    #         step on average (keeps the Poisson approximation accurate)
    # ------------------------------------------------------------------
    DT          = 0.05    # hours per step (3 minutes)
    N_STEPS     = 2000    # 2000 * 0.05h = 100 simulated hours -- long
                           # enough to reach steady state from Q0=0
    BURN_IN_IDX = 1000     # discard the first 50 simulated hours
                           # (transient from starting empty) before
                           # estimating steady-state quantities, the
                           # same "warm-up period" concept used in any
                           # discrete-event simulation textbook

    engine = MonteCarloEngine(
        model, priors, N=5000, n_steps=N_STEPS, dt=DT, seed=42
    )
    result = engine.run()

    # ------------------------------------------------------------------
    # Step 3: estimate steady-state mean queue length from the
    # post-burn-in portion of every particle's trajectory
    # ------------------------------------------------------------------
    # shape (N, n_steps - BURN_IN_IDX + 1)
    post_burn_in = result.trajectories[:, BURN_IN_IDX:, 0]
    L_simulated  = float(np.mean(post_burn_in))
    L_p05        = float(np.percentile(post_burn_in, 5))
    L_p50        = float(np.percentile(post_burn_in, 50))
    L_p95        = float(np.percentile(post_burn_in, 95))

    # ------------------------------------------------------------------
    # Step 4: closed-form benchmark at the prior MEANS
    # (lambda_mean = 10, mu_mean = 16 -- midpoints of the Uniform priors)
    # ------------------------------------------------------------------
    lam_mean = 10.0
    mu_mean  = 16.0
    rho_theory, L_theory, W_theory = model.steady_state_theory(lam_mean, mu_mean)

    # ------------------------------------------------------------------
    # Step 5: report results
    # ------------------------------------------------------------------
    print()
    print(f"  Initial queue length Q0      : {model._Q0:.1f}")
    print("  Arrival rate prior (lambda)  : Uniform(8, 12) patients/hour")
    print("  Service rate prior (mu)      : Uniform(14, 18) patients/hour")
    print(f"  Particles N                  : {result.n_particles}")
    print(f"  Simulated time               : {N_STEPS * DT:.1f} hours "
          f"(burn-in: {BURN_IN_IDX * DT:.1f} hours)")
    print()
    print(f"  Theory  (lambda={lam_mean}, mu={mu_mean}):")
    print(f"    rho (utilisation)         : {rho_theory:.4f}")
    print(f"    L (mean # in system)      : {L_theory:.4f}")
    print(f"    W (mean time in system)   : {W_theory:.4f} hours "
          f"({W_theory * 60:.1f} minutes)")
    print()
    print("  Simulated (uncertain lambda, mu via priors):")
    print(f"    L mean across particles  : {L_simulated:.4f}")
    print(f"    L P05 / P50 / P95         : "
          f"{L_p05:.2f} / {L_p50:.2f} / {L_p95:.2f}")
    print()

    # ------------------------------------------------------------------
    # Step 6: validation -- simulated mean L should be in the right
    # ballpark of the theoretical L at the prior means. We do not expect
    # an exact match because the simulated distribution AVERAGES OVER
    # the lambda/mu priors (a mixture over many different rho values),
    # whereas L_theory is evaluated at a single fixed (lambda, mu) pair.
    # By Jensen's inequality, since L(rho) = rho/(1-rho) is CONVEX in
    # rho, the mixture-averaged L should be slightly HIGHER than
    # L evaluated at the mean rho -- this is a real, expected, and
    # informative effect of parameter uncertainty, not a bug.
    # ------------------------------------------------------------------
    print(f"  Relative difference (sim vs theory): "
          f"{abs(L_simulated - L_theory) / L_theory * 100:.1f}%")
    if L_simulated > L_theory:
        print("  NOTE: simulated L > theory L, as EXPECTED -- L(rho) is")
        print("  convex, so parameter uncertainty inflates mean queue")
        print("  length above the deterministic point estimate (Jensen's")
        print("  inequality). This is a genuine UQ insight, not noise.")

    sanity_ok = 0.5 * L_theory < L_simulated < 3.0 * L_theory
    print()
    if sanity_ok:
        print("  VALIDATION: PASS (simulated L within sane range of theory)")
    else:
        print("  VALIDATION: WARNING -- check model implementation")

    print()
    print("=" * 70)
    print("  Key takeaway: identical MonteCarloEngine, Distribution,")
    print("  and Model ABC used for battery/finance now model hospital")
    print("  operations. Only a new Model subclass was required.")
    print("=" * 70)
