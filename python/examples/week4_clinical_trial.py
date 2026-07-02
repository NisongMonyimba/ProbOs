"""
python/examples/week4_clinical_trial.py

Week 4 Saturday: Adaptive Bayesian clinical trial simulator using the
ProbOS Monte Carlo kernel.

PURPOSE
-------
The third and final Week 4 cross-discipline example. Together with
week4_option_pricer.py (finance) and week4_ed_queue.py (hospital
operations), this completes the demonstration that ProbOS's kernel
(MonteCarloEngine + Distribution ABC + Model ABC) generalises across
physics, finance, operations research, AND clinical medicine without
any change to the kernel itself.

This example is explicitly informed by the FDA's 2019 guidance document
on adaptive trial designs (see docs/study/study_guide.md, Week 4 entry),
which is the current U.S. regulatory standard for exactly this kind of
simulation-based trial design evidence.

CLINICAL TRIAL MODEL
----------------------
We simulate a two-arm randomised controlled trial comparing a new
treatment against a control (e.g. placebo or standard of care), using
a BETA-BINOMIAL conjugate Bayesian model -- the simplest and most
widely used adaptive trial design in the literature (Berry 2011; FDA
2019 guidance, Section V).

Each patient's outcome is a binary success/failure (e.g. "responded to
treatment" vs "did not respond"). For each arm:

    Prior:      p_arm ~ Beta(alpha_0, beta_0)
    Likelihood: outcome_i ~ Bernoulli(p_arm)  for each patient i
    Posterior:  p_arm | data ~ Beta(alpha_0 + successes, beta_0 + failures)

This conjugacy is the key mathematical fact that makes adaptive Bayesian
trials computationally tractable: after each patient's outcome is
observed, the posterior can be updated in closed form (no MCMC needed)
by simply incrementing alpha or beta by 1.

WHY THIS FITS THE ProbOS KERNEL PATTERN
------------------------------------------
Unlike the battery model (continuous ODE state) or the option pricer
(continuous stochastic process), a clinical trial's "state" is a pair
of DISCRETE COUNTS: (successes_treatment, successes_control) or,
equivalently, the number of patients enrolled and responding in each
arm so far. We represent this as a continuous-valued state vector for
the same reason as the ED queue: it lets us reuse forward_batch's
vectorised (N, state_dim) contract exactly as-is.

STATE VECTOR (state_dim = 4):
    state[:, 0] = n_treatment      (patients enrolled, treatment arm)
    state[:, 1] = s_treatment      (successes so far, treatment arm)
    state[:, 2] = n_control        (patients enrolled, control arm)
    state[:, 3] = s_control        (successes so far, control arm)

PARAMETER VECTOR (param_dim = 2):
    params[:, 0] = p_treatment_true  (TRUE, unknown response rate,
                                       treatment arm -- what we are
                                       trying to learn)
    params[:, 1] = p_control_true    (TRUE, unknown response rate,
                                       control arm)

Each forward_batch() call enrols ONE new patient (chosen uniformly at
random into either arm, standard 1:1 randomisation) and simulates
whether they respond, using the TRUE (but experimentally unknown)
p_treatment_true / p_control_true parameters. This mirrors exactly how
the battery model propagates uncertain Ea_SEI through the ODE, and how
the option pricer propagates uncertain sigma through GBM: here we are
propagating uncertainty in the TRUE (unobservable) response rates
through the trial enrollment process.

THE ADAPTIVE STOPPING RULE
-----------------------------
A trial "succeeds early" (stops enrolling and declares the treatment
effective) when the POSTERIOR probability that treatment is better than
control exceeds a pre-specified threshold (we use 0.975, a common
one-sided efficacy boundary). This is computed AFTER the Monte Carlo
run completes, from the final state, using the same conjugate Beta
posterior update as above -- NOT inside forward_batch(), because the
stopping decision is a POST-HOC analysis of the trial trajectory, not
part of the per-patient dynamics.

UNCERTAINTY QUANTIFICATION ANGLE
------------------------------------
As with the battery, option, and queue examples, we do not assume we
know the TRUE response rates in advance -- that is precisely the point
of running the trial. We place priors on p_treatment_true and
p_control_true representing our best PRE-TRIAL guess (e.g. from a
Phase II trial or literature review), then use the Monte Carlo kernel
to simulate MANY possible "parallel universe" trials, each with a
different draw of the true (unknown) response rates. This gives us the
POWER of the trial design: what fraction of simulated trials correctly
detect a real treatment effect, across the full range of plausible true
effect sizes -- exactly the quantity a trial biostatistician needs to
report in an FDA submission (FDA 2019 guidance, Section IV.B).
"""

from __future__ import annotations

import os

import numpy as np

from python.src.distributions import Beta, Distribution
from python.src.monte_carlo import MonteCarloEngine
from python.src.state import FloatArray, Model

os.makedirs("outputs/figures", exist_ok=True)

# ===========================================================================
# MODEL DEFINITION
# ===========================================================================


class ClinicalTrialModel(Model):
    """
    Two-arm adaptive Bayesian clinical trial with Beta-Binomial conjugacy.

    STATE VECTOR (state_dim = 4):
        state[:, 0] = n_treatment  -- patients enrolled, treatment arm
        state[:, 1] = s_treatment  -- successes so far, treatment arm
        state[:, 2] = n_control    -- patients enrolled, control arm
        state[:, 3] = s_control    -- successes so far, control arm

    PARAMETER VECTOR (param_dim = 2):
        params[:, 0] = p_treatment_true -- true (unknown) response rate,
                                            treatment arm
        params[:, 1] = p_control_true   -- true (unknown) response rate,
                                            control arm

    Parameters
    ----------
    randomisation_ratio : float
        Probability a newly enrolled patient is assigned to the
        TREATMENT arm (vs control). Defaults to 0.5 for standard 1:1
        randomisation, the most common design in Phase III trials.
    """

    def __init__(self, randomisation_ratio: float = 0.5) -> None:
        self._rand_ratio = randomisation_ratio

    # ------------------------------------------------------------------
    # Model ABC required properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Four state variables: n and s for each of two arms."""
        return 4

    @property
    def param_dim(self) -> int:
        """Two parameters: true response rate for each arm."""
        return 2

    def param_names(self) -> list[str]:
        """Human-readable names matching param column order."""
        return ["p_treatment_true", "p_control_true"]

    def initial_state(self) -> FloatArray:
        """Trial starts with zero patients enrolled in either arm."""
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Core dynamics: enrol one patient per step, simulate their outcome
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        state:  FloatArray,   # shape (N, 4) -- current trial state per particle
        params: FloatArray,   # shape (N, 2) -- [p_treat_true, p_ctrl_true]
        dt:     float,        # UNUSED -- see note below
    ) -> FloatArray:
        """
        Advance all N particles by enrolling exactly ONE new patient
        each, in each parallel-universe trial simulation.

        NOTE ON dt: unlike the battery ODE (continuous time in seconds)
        or the option pricer (continuous time in years), a clinical
        trial's natural "clock" is PATIENTS ENROLLED, not wall-clock
        time. We ignore dt entirely here (it is required by the Model
        ABC signature for interface consistency across all domains, but
        has no meaning in this discrete-enrollment context). Each call
        to forward_batch() = exactly one new patient enrolled, and
        MonteCarloEngine's n_steps parameter therefore represents the
        MAXIMUM TRIAL SIZE (total patients to enrol), not a duration.
        This is a deliberate, documented reuse of the Model ABC's
        generic step-based interface for a non-time-indexed process --
        the same flexibility already demonstrated by the option pricer
        using genuine SDE steps and the ED queue using birth-death steps.

        Two independent Bernoulli-style random draws happen per particle
        per step:
          1. WHICH ARM does this patient get randomised to? (treatment
             with probability self._rand_ratio, else control)
          2. DOES the patient respond? (Bernoulli draw using the TRUE,
             unknown response rate for whichever arm they landed in)

        Both draws are vectorised across all N particles using NumPy's
        np.random.uniform and elementwise comparison -- the same
        "compare a uniform draw against a rate" vectorisation pattern
        used nowhere else in this codebase but standard practice for
        simulating Bernoulli trials at scale without a Python loop.

        Parameters
        ----------
        state  : shape (N, 4) -- current trial state per particle
        params : shape (N, 2) -- [p_treatment_true, p_control_true]
        dt     : float        -- unused (see note above)

        Returns
        -------
        FloatArray of shape (N, 4) -- updated trial state per particle
        """
        n_treat = state[:, 0]
        s_treat = state[:, 1]
        n_ctrl  = state[:, 2]
        s_ctrl  = state[:, 3]

        p_treat_true = params[:, 0]
        p_ctrl_true  = params[:, 1]

        N = n_treat.shape[0]

        # Step 1: randomise each new patient to treatment or control.
        # goes_to_treatment[i] = True means particle i's new patient
        # was assigned to the treatment arm this step.
        goes_to_treatment = np.random.uniform(0.0, 1.0, N) < self._rand_ratio

        # Step 2: simulate whether the new patient responds, using the
        # TRUE response rate of whichever arm they were assigned to.
        # np.where selects p_treat_true or p_ctrl_true per particle
        # based on their arm assignment, then a single vectorised
        # uniform draw determines "success" for all N particles at once.
        effective_p = np.where(goes_to_treatment, p_treat_true, p_ctrl_true)
        responded   = np.random.uniform(0.0, 1.0, N) < effective_p

        # Step 3: update counts. Only the arm the patient was assigned
        # to gets its n incremented; only responders get s incremented.
        new_n_treat = n_treat + np.where(goes_to_treatment, 1.0, 0.0)
        new_n_ctrl  = n_ctrl  + np.where(goes_to_treatment, 0.0, 1.0)
        new_s_treat = s_treat + np.where(
            goes_to_treatment & responded, 1.0, 0.0
        )
        new_s_ctrl = s_ctrl + np.where(
            (~goes_to_treatment) & responded, 1.0, 0.0
        )

        return np.column_stack([
            new_n_treat, new_s_treat, new_n_ctrl, new_s_ctrl,
        ])

    # ------------------------------------------------------------------
    # Domain-specific helpers: NOT part of Model ABC.
    # ------------------------------------------------------------------

    @staticmethod
    def posterior_prob_treatment_better(
        s_treat: FloatArray,
        n_treat: FloatArray,
        s_ctrl:  FloatArray,
        n_ctrl:  FloatArray,
        alpha_0: float = 1.0,
        beta_0:  float = 1.0,
        n_mc_samples: int = 10_000,
    ) -> FloatArray:
        """
        Compute, for EACH particle (parallel-universe trial), the
        posterior probability that the treatment arm's TRUE response
        rate exceeds the control arm's TRUE response rate, given the
        observed trial data (s_treat successes out of n_treat patients,
        etc.), under a Beta(alpha_0, beta_0) prior on each arm.

        This is NOT available in closed form for a difference of two
        Beta random variables in general, so we use a nested Monte
        Carlo estimate: draw n_mc_samples samples from each arm's
        POSTERIOR Beta distribution (not the trial's own N particles --
        a separate, smaller inner MC loop per particle, standard
        practice in Bayesian trial analysis) and compute the fraction
        of draws where treatment > control.

        Parameters
        ----------
        s_treat, n_treat, s_ctrl, n_ctrl : shape (N,)
            Successes and enrollments for each arm, one value per
            simulated trial (particle).
        alpha_0, beta_0 : float
            Beta prior hyperparameters, shared across both arms.
            Beta(1,1) is the uniform prior (default), representing
            genuine pre-trial uncertainty about the response rate.
        n_mc_samples : int
            Number of posterior samples to draw per particle for the
            nested Monte Carlo comparison. 10,000 gives a standard
            error on the probability estimate of roughly 0.5%.

        Returns
        -------
        FloatArray of shape (N,)
            Posterior probability treatment is better than control,
            one value per particle (simulated trial).
        """
        N = s_treat.shape[0]
        rng = np.random.default_rng(seed=123)

        # Posterior parameters via Beta-Binomial conjugacy
        alpha_treat = alpha_0 + s_treat
        beta_treat  = beta_0 + (n_treat - s_treat)
        alpha_ctrl  = alpha_0 + s_ctrl
        beta_ctrl   = beta_0 + (n_ctrl - s_ctrl)

        prob_better = np.zeros(N, dtype=np.float64)
        for i in range(N):
            treat_samples = rng.beta(alpha_treat[i], beta_treat[i], n_mc_samples)
            ctrl_samples  = rng.beta(alpha_ctrl[i], beta_ctrl[i], n_mc_samples)
            prob_better[i] = float(np.mean(treat_samples > ctrl_samples))

        return prob_better


def build_clinical_trial_priors(
    p_treatment_guess: float = 0.45,
    p_control_guess:   float = 0.30,
) -> list[Distribution]:
    """
    Prior distributions for the clinical trial's 2 parameters: the TRUE
    (unknown) response rates in each arm.

    We use Beta priors -- the natural choice for probabilities in
    [0, 1], and conjugate to the Bernoulli outcome model used in
    forward_batch(), mirroring the mathematical structure of the trial
    itself (though these Beta priors represent our uncertainty about
    the true rate BEFORE the trial, distinct from the posterior_...
    calculation above which represents uncertainty AFTER observing
    trial data).

    Parameters
    ----------
    p_treatment_guess : float
        Best pre-trial guess for the treatment arm's true response
        rate (e.g. from Phase II data). Defaults to 0.45 -- a
        plausible response rate for many oncology/immunology trials.
    p_control_guess : float
        Best pre-trial guess for the control arm's true response
        rate (e.g. historical standard-of-care response rate).
        Defaults to 0.30.

    Returns
    -------
    list[Distribution]
        [p_treatment_prior, p_control_prior], each a Beta distribution
        with mean equal to the given guess and a moderate concentration
        (alpha+beta=20) representing realistic pre-trial uncertainty --
        not so tight that the trial teaches us nothing, not so loose
        that the prior is uninformative.
    """
    def _beta_from_mean(mean: float, concentration: float = 20.0) -> Beta:
        alpha = mean * concentration
        beta  = (1.0 - mean) * concentration
        return Beta(alpha=alpha, beta=beta)

    return [
        _beta_from_mean(p_treatment_guess),
        _beta_from_mean(p_control_guess),
    ]


# ===========================================================================
# DEMO / VALIDATION SCRIPT
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  ProbOS Week 4 Saturday -- Adaptive Bayesian Clinical Trial")
    print("  Demonstrates kernel generalisation: physics/finance/ops -> medicine")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: set up the model and priors
    # ------------------------------------------------------------------
    model  = ClinicalTrialModel(randomisation_ratio=0.5)
    priors = build_clinical_trial_priors(
        p_treatment_guess=0.45, p_control_guess=0.30
    )

    # ------------------------------------------------------------------
    # Step 2: run the Monte Carlo engine
    #   N=2000 simulated parallel-universe trials
    #   n_steps=200 -- maximum trial size of 200 patients (see dt note
    #                  in forward_batch docstring: n_steps = patients,
    #                  not time)
    # ------------------------------------------------------------------
    N_PATIENTS_MAX = 200

    engine = MonteCarloEngine(
        model, priors, N=2000, n_steps=N_PATIENTS_MAX, dt=1.0, seed=42
    )
    result = engine.run()

    # ------------------------------------------------------------------
    # Step 3: extract final trial states (after all 200 patients enrolled)
    # ------------------------------------------------------------------
    final_n_treat = result.trajectories[:, -1, 0]
    final_s_treat = result.trajectories[:, -1, 1]
    final_n_ctrl  = result.trajectories[:, -1, 2]
    final_s_ctrl  = result.trajectories[:, -1, 3]

    # ------------------------------------------------------------------
    # Step 4: compute posterior probability treatment > control for
    # each simulated trial (this is the "did the trial succeed?" metric)
    # ------------------------------------------------------------------
    prob_better = ClinicalTrialModel.posterior_prob_treatment_better(
        final_s_treat, final_n_treat, final_s_ctrl, final_n_ctrl,
        n_mc_samples=2000,   # smaller inner MC for demo speed
    )

    # ------------------------------------------------------------------
    # Step 5: apply the efficacy stopping rule (posterior prob > 0.975)
    # and report trial-level summary statistics
    # ------------------------------------------------------------------
    EFFICACY_THRESHOLD = 0.975
    trial_succeeded = prob_better > EFFICACY_THRESHOLD
    power_estimate  = float(np.mean(trial_succeeded))

    observed_rate_treat = float(np.mean(final_s_treat / final_n_treat))
    observed_rate_ctrl  = float(np.mean(final_s_ctrl / final_n_ctrl))

    print()
    print("  Prior guess p_treatment      : 0.45")
    print("  Prior guess p_control        : 0.30")
    print("  Randomisation ratio          : 50/50")
    print(f"  Max trial size (patients)    : {N_PATIENTS_MAX}")
    print(f"  Simulated trials N           : {result.n_particles}")
    print(f"  Efficacy threshold           : {EFFICACY_THRESHOLD}")
    print()
    print(f"  Mean patients enrolled/arm   : "
          f"treatment={np.mean(final_n_treat):.1f}, "
          f"control={np.mean(final_n_ctrl):.1f}")
    print(f"  Mean observed response rate  : "
          f"treatment={observed_rate_treat:.4f}, "
          f"control={observed_rate_ctrl:.4f}")
    print()
    print(f"  Trial power estimate         : {power_estimate:.4f}")
    print("    (fraction of simulated trials that correctly detected")
    print("     the treatment effect at the pre-specified threshold)")
    print()

    # ------------------------------------------------------------------
    # Step 6: validation -- with a genuine 15-percentage-point true
    # effect size (0.45 vs 0.30) and N=200 patients, a well-powered
    # trial design should detect the effect in the clear majority of
    # simulated trials. We check power is meaningfully above the false
    # positive rate we'd see under the null (no true effect), which is
    # approximately 1 - EFFICACY_THRESHOLD by construction of the
    # one-sided stopping boundary.
    # ------------------------------------------------------------------
    print(f"  Difference from null false-positive rate: "
          f"{power_estimate - (1.0 - EFFICACY_THRESHOLD):.4f}")
    if power_estimate > 0.5:
        print("  VALIDATION: PASS (trial design has adequate power "
              "to detect the assumed effect size)")
    else:
        print("  VALIDATION: WARNING -- power lower than expected, "
              "check model implementation")

    print()
    print("=" * 70)
    print("  Key takeaway: identical MonteCarloEngine, Distribution,")
    print("  and Model ABC used for battery/finance/hospital-ops now")
    print("  power adaptive clinical trial design -- the FDA's 2019")
    print("  guidance describes exactly this kind of simulation-based")
    print("  trial design evidence. Only a new Model subclass was")
    print("  required across all four Week 4 examples.")
    print("=" * 70)
