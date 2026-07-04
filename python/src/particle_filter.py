"""
python/src/particle_filter.py

Bootstrap (SIR) particle filter -- ProbOS Month 2 Week 5.

WHY THIS EXISTS
----------------
Everything built in Month 1 answers "given uncertain parameters, what
does the system do?" (forward simulation). This module answers the
opposite question: "given OBSERVED DATA, what do we now believe about
the system's hidden state?" (inference). That is a fundamentally
different problem and needs a different algorithm -- sequential Monte
Carlo, not batch Monte Carlo.

ALGORITHM (bootstrap / SIR particle filter)
---------------------------------------------
Following the unifying template in Naesseth, Lindsten & Schon (2019),
"Elements of Sequential Monte Carlo," and the textbook treatment in
Chopin & Papaspiliopoulos (2020), "An Introduction to Sequential Monte
Carlo," Ch 8-10:

At each observation time t:
  1. PREDICT: advance all N particles one step using the model's own
     forward_batch() (reused directly from the Model ABC -- no new
     dynamics code needed here, which is the point of building on top
     of a domain-agnostic kernel).
  2. UPDATE (reweight): multiply each particle's importance weight by
     the likelihood of the observation given that particle's state,
     p(y_t | x_t^(i)). Particles whose state poorly explains the
     observation get down-weighted; particles that explain it well
     get up-weighted.
  3. RESAMPLE: when weights become too concentrated on a few particles
     (measured by effective sample size, ESS), draw a fresh set of N
     particles proportional to their weights, then reset weights to
     uniform. This is the step that distinguishes SEQUENTIAL Monte
     Carlo from just re-running batch Monte Carlo at every timestep --
     without it, weight degeneracy makes the estimate collapse onto a
     single particle after enough steps.

WHY LOG-SPACE WEIGHTS
------------------------
Likelihoods multiply across many timesteps and can underflow to exactly
0.0 in floating point after only a few dozen steps if computed directly
in probability space. We keep weights in LOG space throughout
(log_weights) and only exponentiate right before resampling/reporting,
using the standard log-sum-exp trick for the normalisation constant.
This is the exact same numerical-stability discipline that motivated
Distribution.log_pdf() in Month 1 (Proposition 1 in the manuscript) --
the lesson generalises directly to weighted particle sets.

WHY SYSTEMATIC RESAMPLING
-----------------------------
Naive multinomial resampling (draw N indices i.i.d. from the weight
distribution) has higher variance than necessary. Systematic
resampling draws a SINGLE uniform random offset and then takes N
equally-spaced points along the cumulative weight distribution -- this
is the standard low-variance choice used throughout the SMC literature
(Chopin & Papaspiliopoulos 2020, Ch 9).

RELATIONSHIP TO THE REST OF THE KERNEL
------------------------------------------
ParticleFilter takes the SAME (Model, list[Distribution]) construction
signature as MonteCarloEngine and SobolSensitivity. It reuses
Model.forward_batch() for prediction with zero modification to the
Model ABC. The only new concept it introduces is the observation
likelihood function, which the caller supplies directly (not baked
into Model) -- this keeps Month 1's domain-agnostic Model ABC
untouched while still allowing inference on any existing model
(BatteryModel2Cell, the Week 4 cross-discipline examples, etc.)
without editing their source.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from python.src.distributions import Distribution
from python.src.state import FloatArray, Model

# Type alias: a function mapping (state, obs) -> log-likelihood per particle.
# state : shape (N, state_dim)
# obs   : shape (obs_dim,) -- a single observation vector, shared across
#         all N particles (the same real-world measurement is compared
#         against every particle's hypothesis of the hidden state)
# returns: shape (N,) -- log p(obs | state_i) for each particle i
LogLikelihoodFn = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass
class PFResult:
    """
    Output of a completed particle filter run.

    Attributes
    ----------
    means : shape (T, state_dim)
        Posterior mean state estimate at each observation time,
        weighted by the (post-update, pre-resample) particle weights.
    stds : shape (T, state_dim)
        Posterior standard deviation at each observation time --
        the filter's own uncertainty about the hidden state, analogous
        to MCResult.convergence but tracking POSTERIOR uncertainty
        (shrinks as informative data arrives) rather than MONTE CARLO
        estimation error (shrinks with more particles).
    ess_history : shape (T,)
        Effective sample size after each update step, before any
        resampling triggered by that step. ESS = 1 / sum(w_i^2) for
        normalised weights w_i. ESS = N means all particles equally
        informative; ESS = 1 means weight has collapsed onto one
        particle (degeneracy).
    n_resamples : int
        Total number of times resampling was triggered across the run.
        A healthy filter resamples occasionally, not every step (too
        frequent = losing particle diversity unnecessarily) and not
        never (too rare = weight degeneracy).
    final_state : shape (N, state_dim)
        The particle states after the final predict/update step.
    final_weights : shape (N,)
        Normalised (probability-space, not log) weights after the
        final update, before any final resampling.
    n_particles : int
    n_steps : int
    """

    means:         FloatArray
    stds:          FloatArray
    ess_history:   FloatArray
    n_resamples:   int
    final_state:   FloatArray
    final_weights: FloatArray
    n_particles:   int
    n_steps:       int


class ParticleFilter:
    """
    Bootstrap (SIR) particle filter for sequential Bayesian inference.

    Parameters
    ----------
    model : Model
        Any Model ABC subclass. forward_batch() is reused unmodified
        for the predict step.
    priors : list[Distribution]
        One Distribution per model parameter, sampled ONCE at
        initialisation (particle parameters are held fixed across the
        filter run -- only the STATE evolves per-particle via
        forward_batch's own internal stochasticity, if any; parameter
        uncertainty here plays the same role as in MonteCarloEngine:
        each particle represents one hypothesis about the unknown
        parameter values, carried forward and reweighted by how well
        it explains the incoming data).
    N : int
        Number of particles.
    dt : float
        Time step passed to forward_batch() at each predict step.
    resample_threshold : float
        Resample when ESS / N falls below this fraction. Default 0.5
        is the standard choice (Chopin & Papaspiliopoulos 2020, Ch 9):
        resample once effective sample size drops below half of N.
    seed : int
        Random seed for the initial particle draw and resampling.

    Raises
    ------
    ValueError
        If len(priors) != model.param_dim, N < 1, or dt <= 0.
    """

    def __init__(
        self,
        model: Model,
        priors: Sequence[Distribution],
        N: int = 1000,
        dt: float = 1.0,
        resample_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        if len(priors) != model.param_dim:
            raise ValueError(
                f"len(priors)={len(priors)} != model.param_dim={model.param_dim}"
            )
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if not (0.0 < resample_threshold <= 1.0):
            raise ValueError(
                f"resample_threshold must be in (0, 1], got {resample_threshold}"
            )

        self._model  = model
        self._priors = list(priors)
        self._N      = N
        self._dt     = dt
        self._resample_threshold = resample_threshold
        self._seed   = seed

        self._rng = np.random.default_rng(seed)

        # Draw N parameter sets once, held fixed for the whole run.
        pd = model.param_dim
        self._params: FloatArray = np.empty((N, pd), dtype=np.float64)
        for j, prior in enumerate(priors):
            self._params[:, j] = prior.sample(N, rng=self._rng)

        # All particles start at the model's known initial state.
        self._state: FloatArray = np.tile(
            model.initial_state(), (N, 1)
        ).astype(np.float64)

        # Validate shape once at construction time, same discipline
        # as MonteCarloEngine.run(). Catches a malformed Model
        # subclass immediately with a clear ValueError, rather than
        # a confusing broadcasting error deep inside predict().
        model.validate_state(self._state)

        # Start with uniform weights in LOG space: log(1/N) for all particles.
        self._log_weights: FloatArray = np.full(N, -np.log(N), dtype=np.float64)

        self._n_resamples = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return self._N

    @property
    def state(self) -> FloatArray:
        """Current particle states, shape (N, state_dim)."""
        return self._state

    @property
    def params(self) -> FloatArray:
        """
        Current particle parameter values, shape (N, param_dim).

        Each particle's parameters are drawn ONCE at construction
        (see __init__) and held fixed for the life of the filter --
        only the STATE evolves per-particle via predict(). This
        means the posterior over any parameter can be recovered by
        weighting these fixed per-particle values by the filter's
        current importance weights (see the `weights` property),
        exactly as the posterior over state is recovered via
        posterior_mean()/posterior_std().
        """
        return self._params

    @property
    def weights(self) -> FloatArray:
        """
        Current NORMALISED weights in probability space, shape (N,).
        Converted from internal log-space storage on access.
        """
        return self._normalised_weights()

    # ------------------------------------------------------------------
    # Core SMC steps
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """
        Advance all N particles by one forward_batch() step.

        This is a direct reuse of the Model ABC's existing contract --
        no new dynamics code. Any stochasticity in forward_batch()
        (e.g. OptionPricerModel's fresh Brownian shock each call, or
        EDQueueModel's Poisson arrival/departure draws) naturally
        diversifies the particle cloud from one predict() call to the
        next, which is exactly what a particle filter needs.
        """
        self._state = self._model.forward_batch(
            self._state, self._params, self._dt
        )

    def update(self, obs: FloatArray, log_likelihood_fn: LogLikelihoodFn) -> None:
        """
        Reweight particles by the likelihood of the observation.

        Weights are updated multiplicatively in LOG space:
            log_w_i <- log_w_i + log p(obs | state_i)
        then renormalised (log-sum-exp) so weights sum to 1 in
        probability space. Working in log space avoids the underflow
        that would occur from directly multiplying many small
        probabilities together across timesteps.

        Parameters
        ----------
        obs : shape (obs_dim,)
            A single observation, compared against every particle's
            current state via log_likelihood_fn.
        log_likelihood_fn : LogLikelihoodFn
            Function (state, obs) -> log p(obs | state), shape (N,).
        """
        log_lik = log_likelihood_fn(self._state, obs)
        self._log_weights = self._log_weights + log_lik
        # Renormalise via log-sum-exp for numerical stability.
        log_norm = self._log_sum_exp(self._log_weights)
        self._log_weights = self._log_weights - log_norm

    def effective_sample_size(self) -> float:
        """
        ESS = 1 / sum(w_i^2) for normalised weights w_i.

        Returns
        -------
        float
            Ranges from 1 (weight collapsed onto one particle,
            maximally degenerate) to N (all particles equally
            weighted, maximally diverse).
        """
        w = self._normalised_weights()
        return float(1.0 / np.sum(w ** 2))

    def resample(self) -> None:
        """
        Systematic resampling: draw N new particle indices with
        probability proportional to their current weights, replace
        the particle set with the resampled copies, and reset weights
        to uniform.

        Uses the low-variance systematic scheme (a single random
        offset, then N equally-spaced points along the cumulative
        weight distribution) rather than naive multinomial resampling,
        following Chopin & Papaspiliopoulos (2020) Ch 9.
        """
        w = self._normalised_weights()
        cumsum = np.cumsum(w)
        cumsum[-1] = 1.0  # guard against floating-point drift

        # Single random offset in [0, 1/N), then N equally-spaced points.
        u0 = self._rng.uniform(0.0, 1.0 / self._N)
        points = u0 + np.arange(self._N) / self._N

        indices = np.searchsorted(cumsum, points)
        indices = np.clip(indices, 0, self._N - 1)

        self._state  = self._state[indices]
        self._params = self._params[indices]
        self._log_weights = np.full(
            self._N, -np.log(self._N), dtype=np.float64
        )
        self._n_resamples += 1

    def posterior_mean(self) -> FloatArray:
        """Weighted mean of current particle states, shape (state_dim,)."""
        w = self._normalised_weights()
        return np.average(self._state, axis=0, weights=w)

    def posterior_std(self) -> FloatArray:
        """Weighted standard deviation of current particle states."""
        w = self._normalised_weights()
        mean = self.posterior_mean()
        variance = np.average((self._state - mean) ** 2, axis=0, weights=w)
        return np.sqrt(variance)

    # ------------------------------------------------------------------
    # Full run over an observation sequence
    # ------------------------------------------------------------------

    def run(
        self,
        observations: FloatArray,
        log_likelihood_fn: LogLikelihoodFn,
    ) -> PFResult:
        """
        Run the full predict/update/resample loop over a sequence of
        observations.

        Parameters
        ----------
        observations : shape (T, obs_dim)
            One observation per timestep.
        log_likelihood_fn : LogLikelihoodFn
            Passed through to update() at every step.

        Returns
        -------
        PFResult
        """
        T = observations.shape[0]
        sd = self._model.state_dim

        means       = np.empty((T, sd), dtype=np.float64)
        stds        = np.empty((T, sd), dtype=np.float64)
        ess_history = np.empty(T, dtype=np.float64)

        for t in range(T):
            self.predict()
            self.update(observations[t], log_likelihood_fn)

            means[t]       = self.posterior_mean()
            stds[t]        = self.posterior_std()
            ess_history[t] = self.effective_sample_size()

            if ess_history[t] / self._N < self._resample_threshold:
                self.resample()

        return PFResult(
            means=means,
            stds=stds,
            ess_history=ess_history,
            n_resamples=self._n_resamples,
            final_state=self._state,
            final_weights=self._normalised_weights(),
            n_particles=self._N,
            n_steps=T,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalised_weights(self) -> FloatArray:
        """Convert internal log-weights to normalised probability-space weights."""
        log_norm = self._log_sum_exp(self._log_weights)
        return np.exp(self._log_weights - log_norm)

    @staticmethod
    def _log_sum_exp(log_values: FloatArray) -> float:
        """
        Numerically stable log(sum(exp(log_values))).

        Standard log-sum-exp trick: subtract the max before
        exponentiating, then add it back after taking the log, so the
        largest term never underflows/overflows during the intermediate
        exp() computation. Same underflow-avoidance discipline as
        Distribution.log_pdf() in Month 1.
        """
        max_val = np.max(log_values)
        return float(max_val + np.log(np.sum(np.exp(log_values - max_val))))
