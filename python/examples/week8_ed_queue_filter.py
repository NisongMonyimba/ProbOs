"""
python/examples/week8_ed_queue_filter.py

Week 8 Day 1-2: Sequential inference of the true ED patient arrival
rate from observed queue-length data, using ParticleFilter.

WHY THIS EXTENDS WEEK 4's EDQueueModel
-----------------------------------------
Week 4's week4_ed_queue.py used MonteCarloEngine for FORWARD
simulation: given a prior belief about lambda (Uniform(8, 12)),
propagate many "parallel universe" EDs forward and summarise the
resulting queue-length distribution.

This script asks the OPPOSITE question: given an OBSERVED sequence
of queue-length measurements from a real (or simulated) ED, what do
we now believe the TRUE arrival rate lambda is? This is exactly the
sequential Bayesian inference problem ParticleFilter (Week 5) was
built for, and EDQueueModel is reused here completely UNMODIFIED --
zero changes to the Week 4 model, demonstrating the same
domain-agnostic-kernel property already shown for MonteCarloEngine
in Month 1 Week 4.

VALIDATION STRATEGY
----------------------
We generate SYNTHETIC observations from a model run at a KNOWN true
lambda (so we have ground truth), add Gaussian observation noise
(simulating imperfect queue-length measurement, e.g. a nurse
periodically counting patients rather than an exact real-time
sensor), then run ParticleFilter on those observations and confirm
its posterior mean for lambda converges toward the true value as more
observations arrive -- the same "validate against a known ground
truth" discipline used everywhere else in this project (Kim 2007 for
BatteryModel2Cell, the exact Kalman filter for ParticleFilter itself
in Week 5).

WHY WE FILTER ON lambda SPECIFICALLY, NOT mu
------------------------------------------------
EDQueueModel has two parameters: lambda (arrival rate) and mu
(service rate). We hold mu FIXED at a known, tightly-informative
prior here (representing a well-characterised, unchanging service
process -- e.g. a fixed number of triage nurses working at a
consistent pace) and focus inference on lambda (representing the
genuinely time-varying, uncertain quantity -- patient arrival rates
vary with time of day, season, and local events in ways that are
harder to know in advance). This mirrors the same "hold one thing
fixed, infer the other" pattern already used in
python/tests/test_particle_filter.py's RandomWalkModel validation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from python.examples.week4_ed_queue import EDQueueModel
from python.src.distributions import Distribution, Normal, Uniform
from python.src.particle_filter import ParticleFilter
from python.src.state import FloatArray


def build_ed_queue_filter_priors(
    lambda_prior_low: float = 5.0,
    lambda_prior_high: float = 15.0,
    mu_fixed: float = 16.0,
) -> list[Distribution]:
    """
    Prior distributions for ED queue filtering: a WIDE Uniform prior
    on lambda (representing genuine pre-observation uncertainty about
    the true arrival rate -- deliberately wider than Week 4's
    Uniform(8, 12) forward-simulation prior, since here we want to
    demonstrate the filter can recover lambda even starting from a
    less-informative prior) and a tight, near-degenerate prior on mu
    (representing a well-characterised, fixed service process).

    Parameters
    ----------
    lambda_prior_low, lambda_prior_high : float
        Bounds of the Uniform prior on the arrival rate.
    mu_fixed : float
        The (assumed known) service rate, held fixed via a
        near-degenerate Normal prior.

    Returns
    -------
    list[Distribution]
        [lambda_prior, mu_prior] matching EDQueueModel.param_names().
    """
    return [
        Uniform(low=lambda_prior_low, high=lambda_prior_high),
        Normal(mu=mu_fixed, sigma=1e-6),
    ]


def queue_length_log_likelihood(
    sigma_obs: float,
) -> Callable[[FloatArray, FloatArray], FloatArray]:
    """
    Builds a log_likelihood_fn for ParticleFilter.update(): Gaussian
    observation noise on the queue length (state[:, 0]), representing
    imperfect measurement of the true queue length (e.g. periodic
    manual counts rather than a perfect real-time sensor).

    Parameters
    ----------
    sigma_obs : float
        Assumed standard deviation of the observation noise, in
        units of patients.
    """
    def _fn(state: FloatArray, obs: FloatArray) -> FloatArray:
        Q = state[:, 0]
        result: FloatArray = -0.5 * ((Q - obs[0]) / sigma_obs) ** 2
        return result
    return _fn


def generate_synthetic_observations(
    true_lambda: float,
    true_mu: float,
    n_steps: int,
    dt: float,
    sigma_obs: float,
    seed: int,
) -> FloatArray:
    """
    Generate a synthetic queue-length observation sequence from
    EDQueueModel run at a KNOWN true (lambda, mu), with added
    Gaussian observation noise -- the ground truth against which we
    validate the particle filter's recovered posterior.

    Uses a SINGLE particle (N=1) run of EDQueueModel.forward_batch()
    directly, not the full ParticleFilter or MonteCarloEngine
    machinery, since we want one specific realised trajectory to
    treat as "the real world," not a distribution over trajectories.

    Returns
    -------
    FloatArray of shape (n_steps,) -- noisy queue-length observations,
    one per timestep.
    """
    # DELIBERATE, NARROWLY-SCOPED EXCEPTION to project doctrine
    # (see week1_coin_flip.py: "NEVER use np.random.seed()
    # (global state, not thread-safe, legacy API)"). This
    # exception is unavoidable, not an oversight: EDQueueModel.
    # forward_batch() itself uses the legacy global
    # np.random.poisson() internally, by deliberate Week 4
    # design (genuine stochastic process, fresh randomness
    # every step -- the same pattern used in OptionPricerModel
    # and ClinicalTrialModel). Given that, seeding the GLOBAL
    # state is the ONLY way to get a genuinely reproducible
    # fixed ground-truth realization out of this specific
    # model. The prior global state is saved and restored
    # afterward via try/finally so this cannot leak into
    # anything run later in the same Python process.
    prior_state = np.random.get_state()
    try:
        np.random.seed(seed)
        model = EDQueueModel(Q0=0.0)

        # A separate, locally-seeded Generator for the
        # OBSERVATION noise specifically (distinct from the
        # model's own internal randomness), derived from the
        # same seed so the whole function remains
        # deterministic end-to-end.
        obs_rng = np.random.default_rng(seed + 1)

        state = model.initial_state().reshape(1, 1)
        params = np.array([[true_lambda, true_mu]])

        observations = np.empty(n_steps, dtype=np.float64)
        for t in range(n_steps):
            state = model.forward_batch(state, params, dt)
            true_Q = state[0, 0]
            observations[t] = (
                true_Q + obs_rng.normal(0.0, sigma_obs)
            )
    finally:
        np.random.set_state(prior_state)

    return observations


if __name__ == "__main__":
    print("=" * 70)
    print("  ProbOS Week 8 Day 1-2 -- ED Queue Sequential Filtering")
    print("  Infers the true arrival rate lambda from observed queue data")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: generate synthetic ground-truth observations
    # ------------------------------------------------------------------
    TRUE_LAMBDA = 10.0
    TRUE_MU     = 16.0
    N_STEPS     = 200
    DT          = 0.05
    SIGMA_OBS   = 1.5

    observations = generate_synthetic_observations(
        true_lambda=TRUE_LAMBDA, true_mu=TRUE_MU,
        n_steps=N_STEPS, dt=DT, sigma_obs=SIGMA_OBS, seed=42,
    )

    print()
    print(f"  True arrival rate (lambda)   : {TRUE_LAMBDA}")
    print(f"  Fixed service rate (mu)      : {TRUE_MU}")
    print(f"  Simulated observations       : {N_STEPS} "
          f"(dt={DT}h, sigma_obs={SIGMA_OBS} patients)")

    # ------------------------------------------------------------------
    # Step 2: run ParticleFilter with a WIDE prior on lambda
    # ------------------------------------------------------------------
    model  = EDQueueModel(Q0=0.0)
    priors = build_ed_queue_filter_priors(
        lambda_prior_low=5.0, lambda_prior_high=15.0, mu_fixed=TRUE_MU,
    )

    pf = ParticleFilter(
        model, priors, N=2000, dt=DT, resample_threshold=0.5, seed=42,
    )

    loglik = queue_length_log_likelihood(sigma_obs=SIGMA_OBS)
    result = pf.run(observations.reshape(-1, 1), loglik)

    # ------------------------------------------------------------------
    # Step 3: extract the posterior lambda distribution
    #
    # NOTE: ParticleFilter's posterior mean/std (PFResult.means/stds)
    # report STATE posterior (queue length), not PARAMETER posterior.
    # To get the posterior over lambda specifically, we read the
    # particles' own final parameter values via the public
    # `params` property, weighted by their final importance weights --
    # lambda is drawn once per particle at ParticleFilter construction
    # and never resampled independently of the particle's state, so
    # weighting by final_weights gives the correct posterior over
    # lambda.
    # ------------------------------------------------------------------
    final_lambda_values = pf.params[:, 0]
    final_weights = result.final_weights

    posterior_mean_lambda = float(
        np.average(final_lambda_values, weights=final_weights)
    )
    posterior_std_lambda = float(np.sqrt(
        np.average(
            (final_lambda_values - posterior_mean_lambda) ** 2,
            weights=final_weights,
        )
    ))

    print()
    print("  Prior on lambda               : Uniform(5, 15)")
    print(f"  Posterior mean lambda          : {posterior_mean_lambda:.3f}")
    print(f"  Posterior std lambda           : {posterior_std_lambda:.3f}")
    print(f"  True lambda                    : {TRUE_LAMBDA}")
    print(f"  Error                          : "
          f"{abs(posterior_mean_lambda - TRUE_LAMBDA):.3f}")
    print(f"  Error in posterior std units    : "
          f"{abs(posterior_mean_lambda - TRUE_LAMBDA) / posterior_std_lambda:.2f}")

    # ------------------------------------------------------------------
    # Step 4: validation
    # ------------------------------------------------------------------
    error_in_std_units = abs(posterior_mean_lambda - TRUE_LAMBDA) / posterior_std_lambda
    print()
    if error_in_std_units < 3.0:
        print("  VALIDATION: PASS (posterior mean within 3 std of true lambda)")
    else:
        print("  VALIDATION: WARNING -- posterior mean far from true lambda")

    print()
    print("=" * 70)
    print("  Key takeaway: identical ParticleFilter used for validation")
    print("  against the exact Kalman filter (Week 5) now recovers a")
    print("  hospital operations parameter from noisy observational")
    print("  data. EDQueueModel required ZERO changes from Week 4.")
    print("=" * 70)
