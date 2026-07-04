"""
python/examples/week8_clinical_trial_filter.py

Week 8 Days 3-4: Sequential (patient-by-patient) posterior monitoring
of an adaptive clinical trial, using ParticleFilter instead of Week
4's batch nested-Monte-Carlo calculation.

WHY THIS DIFFERS FROM WEEK 8's ED QUEUE FILTERING
-----------------------------------------------------
The ED queue case (week8_ed_queue_filter.py) had NOISY observations
of a partially-hidden true state: queue length was measured
imperfectly, and the model's own forward_batch() genuinely simulated
plausible-but-uncertain dynamics.

A clinical trial is different: patient outcomes are FULLY OBSERVED.
We know exactly who was randomised to which arm and whether they
responded -- there is no measurement noise on the (n, s) counts
themselves. What is uncertain is the TRUE underlying response rates
(p_treatment_true, p_control_true) that generated those outcomes.

This means the correct particle-filter design here is:
  - STATE UPDATE (predict): DETERMINISTIC. All particles share the
    IDENTICAL, real (n_treatment, s_treatment, n_control, s_control)
    counts as the trial progresses -- this is just bookkeeping against
    a known, externally-supplied patient sequence, not a per-particle
    simulation.
  - LIKELIHOOD (update): each particle's own guessed
    (p_treatment_true, p_control_true) -- accessible via
    ParticleFilter.params (added in Week 8 Days 1-2) -- is scored
    against the REAL observed outcome using the exact Bernoulli
    likelihood: p if the patient responded, (1-p) if not. This is a
    smooth, non-degenerate likelihood (not a discrete match/no-match
    check against a randomly re-simulated outcome), because we are
    evaluating an analytical probability, not comparing two
    independent random draws.

VALIDATION STRATEGY
----------------------
Generate a REAL synthetic patient sequence from KNOWN true response
rates. Run ParticleFilter sequentially over that sequence. At the
FINAL patient, compare the particle filter's own (weighted) estimate
of P(treatment better than control) against Week 4's EXACT nested
Monte Carlo calculation
(ClinicalTrialModel.posterior_prob_treatment_better) computed on the
SAME final (n, s) counts -- both should closely agree, since with a
Beta(1,1) prior and a Bernoulli likelihood, the exact conjugate
posterior and a well-converged SMC approximation of it should match
closely. This is the same "validate against a known correct answer"
discipline used in Week 5 (exact Kalman filter) and Week 8 Days 1-2
(known true arrival rate).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from python.examples.week4_clinical_trial import (
    build_clinical_trial_priors,
)
from python.src.distributions import Distribution
from python.src.particle_filter import ParticleFilter
from python.src.state import FloatArray, Model


class ClinicalTrialFilterModel(Model):
    """
    A thin state-update wrapper around ClinicalTrialModel's (n, s)
    count representation, purpose-built for sequential filtering over
    a KNOWN, externally-supplied real patient sequence.

    Unlike ClinicalTrialModel.forward_batch() (Week 4), which
    randomly simulates a NEW patient's arm assignment and outcome
    using each particle's own guessed true rates, this model's
    forward_batch() advances ALL particles IDENTICALLY using the
    REAL, already-known (arm, outcome) pair for the current step --
    reflecting that real trial data is fully observed, not something
    to be simulated. The uncertain quantity (true response rates)
    lives entirely in the particle PARAMETERS, scored by the
    log-likelihood function at update() time, not in this
    deterministic state transition.
    """

    def __init__(
        self,
        real_treatment_arm: FloatArray,
        real_outcome: FloatArray,
    ) -> None:
        """
        Parameters
        ----------
        real_treatment_arm : shape (n_patients,)
            1.0 if patient i was randomised to treatment, 0.0 if control.
        real_outcome : shape (n_patients,)
            1.0 if patient i responded, 0.0 if not.
        """
        self._real_treatment_arm = real_treatment_arm
        self._real_outcome = real_outcome
        self._step = 0

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def param_dim(self) -> int:
        return 2

    def param_names(self) -> list[str]:
        return ["p_treatment_true", "p_control_true"]

    def initial_state(self) -> FloatArray:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def forward_batch(
        self, state: FloatArray, params: FloatArray, dt: float
    ) -> FloatArray:
        """
        Deterministically advance ALL particles by the SAME real,
        already-known next patient. params is intentionally unused
        here -- the uncertain quantity is scored at update() time via
        the log-likelihood function, not baked into this transition.
        """
        n_treat = state[:, 0]
        s_treat = state[:, 1]
        n_ctrl  = state[:, 2]
        s_ctrl  = state[:, 3]

        is_treatment = self._real_treatment_arm[self._step]
        responded    = self._real_outcome[self._step]
        self._step += 1

        if is_treatment > 0.5:
            n_treat = n_treat + 1.0
            s_treat = s_treat + responded
        else:
            n_ctrl = n_ctrl + 1.0
            s_ctrl = s_ctrl + responded

        return np.column_stack([n_treat, s_treat, n_ctrl, s_ctrl])


def generate_synthetic_trial_data(
    n_patients: int,
    true_p_treatment: float,
    true_p_control: float,
    randomisation_ratio: float,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """
    Generate a synthetic, fully-observed patient sequence from KNOWN
    true response rates -- the ground truth against which we validate
    both Week 4's batch calculation and this script's sequential
    filter.

    Uses a LOCAL, properly-seeded np.random.Generator throughout (no
    global-state doctrine exception needed here, unlike Week 8 Days
    1-2's EDQueueModel case -- this generation logic is entirely
    self-contained and does not depend on any model's own internal
    global-random-state usage).

    Returns
    -------
    treatment_arm : shape (n_patients,) -- 1.0 if treatment, 0.0 if control
    outcome : shape (n_patients,) -- 1.0 if responded, 0.0 if not
    """
    rng = np.random.default_rng(seed)

    treatment_arm = (rng.uniform(0.0, 1.0, n_patients) < randomisation_ratio).astype(
        np.float64
    )
    true_rate = np.where(treatment_arm > 0.5, true_p_treatment, true_p_control)
    outcome = (rng.uniform(0.0, 1.0, n_patients) < true_rate).astype(np.float64)

    return treatment_arm, outcome


def build_trial_filter_log_likelihood(
    pf: ParticleFilter,
    treatment_arm: FloatArray,
    outcome: FloatArray,
) -> Callable[[FloatArray, FloatArray], FloatArray]:
    """
    Builds a log_likelihood_fn scoring each particle's own guessed
    (p_treatment_true, p_control_true) -- read via the ParticleFilter's
    public `params` property (Week 8 Days 1-2) -- against the REAL
    observed outcome for the current step, using the exact Bernoulli
    likelihood.

    Closes over `pf` itself: since particle parameters are drawn ONCE
    at construction and held fixed for the filter's lifetime (per
    ParticleFilter's own docstring), reading pf.params at each
    update() call always reflects the correct, unchanged per-particle
    parameter values.

    Parameters
    ----------
    pf : ParticleFilter
        The filter instance whose .params will be read at each call.
    treatment_arm, outcome : shape (n_patients,)
        The full real patient sequence -- indexed by step count via
        `obs`, which this function expects to carry the step index.
    """
    def _fn(state: FloatArray, obs: FloatArray) -> FloatArray:
        step = int(obs[0])
        p_treatment = pf.params[:, 0]
        p_control = pf.params[:, 1]

        is_treatment = treatment_arm[step] > 0.5
        responded = outcome[step] > 0.5

        p = p_treatment if is_treatment else p_control
        # Bernoulli log-likelihood: log(p) if responded, log(1-p) if not.
        # Clip to avoid log(0) if a particle's prior draw is extremely
        # close to 0 or 1 (Beta priors have full support on (0, 1) but
        # can draw values arbitrarily close to the boundary).
        p_clipped = np.clip(p, 1e-10, 1.0 - 1e-10)
        log_lik: FloatArray = np.where(
            responded, np.log(p_clipped), np.log(1.0 - p_clipped)
        )
        return log_lik
    return _fn


if __name__ == "__main__":
    print("=" * 70)
    print("  ProbOS Week 8 Days 3-4 -- Clinical Trial Sequential Filtering")
    print("  Patient-by-patient posterior updating via ParticleFilter")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: generate synthetic ground-truth trial data
    # ------------------------------------------------------------------
    N_PATIENTS = 200
    TRUE_P_TREATMENT = 0.45
    TRUE_P_CONTROL   = 0.30

    treatment_arm, outcome = generate_synthetic_trial_data(
        n_patients=N_PATIENTS,
        true_p_treatment=TRUE_P_TREATMENT,
        true_p_control=TRUE_P_CONTROL,
        randomisation_ratio=0.5,
        seed=42,
    )

    print()
    print(f"  True p_treatment              : {TRUE_P_TREATMENT}")
    print(f"  True p_control                : {TRUE_P_CONTROL}")
    print(f"  Patients                      : {N_PATIENTS}")

    # ------------------------------------------------------------------
    # Step 2: run ParticleFilter sequentially, one patient at a time
    # ------------------------------------------------------------------
    model = ClinicalTrialFilterModel(treatment_arm, outcome)
    priors: list[Distribution] = build_clinical_trial_priors(
        p_treatment_guess=0.45, p_control_guess=0.30,
    )

    pf = ParticleFilter(model, priors, N=2000, dt=1.0, seed=42)
    loglik = build_trial_filter_log_likelihood(pf, treatment_arm, outcome)

    # obs at step t just needs to carry the step index -- the actual
    # arm/outcome data is read directly from the closed-over arrays.
    observations = np.arange(N_PATIENTS, dtype=np.float64).reshape(-1, 1)
    result = pf.run(observations, loglik)

    # ------------------------------------------------------------------
    # Step 3: compute the filter's own posterior P(treatment > control)
    # at the FINAL step, using its final weighted particles
    # ------------------------------------------------------------------
    final_p_treatment = pf.params[:, 0]
    final_p_control = pf.params[:, 1]
    final_weights = result.final_weights

    pf_prob_treatment_better = float(
        np.average(final_p_treatment > final_p_control, weights=final_weights)
    )

    # ------------------------------------------------------------------
    # Step 4: compute Week 4's EXACT batch calculation on the SAME
    # final (n, s) counts, as the ground-truth consistency check
    # ------------------------------------------------------------------
    final_state = result.final_state[0]  # all particles share identical state
    final_n_treat, final_s_treat, final_n_ctrl, final_s_ctrl = final_state

    # IMPORTANT: build_clinical_trial_priors(0.45, 0.30) returns
    # INFORMATIVE per-arm priors (Beta(9, 11) for treatment,
    # Beta(6, 14) for control -- see _beta_from_mean's
    # concentration=20 in week4_clinical_trial.py), NOT the
    # uniform Beta(1, 1) that
    # ClinicalTrialModel.posterior_prob_treatment_better()
    # defaults to. Using the DEFAULT would compare two DIFFERENT
    # posteriors (different priors necessarily give different
    # answers) -- not a genuine 'same problem, two methods'
    # validation. We therefore compute the exact posterior
    # manually here using the SAME per-arm informative priors
    # the particles were actually drawn from, via the identical
    # Beta-Binomial conjugate nested-MC approach
    # posterior_prob_treatment_better() itself uses internally,
    # just with the correct DIFFERENT alpha/beta per arm rather
    # than one shared pair.
    treat_alpha0, treat_beta0 = 0.45 * 20.0, (1.0 - 0.45) * 20.0
    ctrl_alpha0, ctrl_beta0 = 0.30 * 20.0, (1.0 - 0.30) * 20.0

    exact_rng = np.random.default_rng(123)
    n_mc = 20_000
    treat_samples = exact_rng.beta(
        treat_alpha0 + final_s_treat,
        treat_beta0 + (final_n_treat - final_s_treat),
        n_mc,
    )
    ctrl_samples = exact_rng.beta(
        ctrl_alpha0 + final_s_ctrl,
        ctrl_beta0 + (final_n_ctrl - final_s_ctrl),
        n_mc,
    )
    exact_prob = float(np.mean(treat_samples > ctrl_samples))

    print()
    print(f"  Final enrollment              : "
          f"treatment={final_n_treat:.0f}, control={final_n_ctrl:.0f}")
    print(f"  Final responses                : "
          f"treatment={final_s_treat:.0f}, control={final_s_ctrl:.0f}")
    print()
    print(f"  ParticleFilter P(treat>ctrl)   : {pf_prob_treatment_better:.4f}")
    print(f"  Week 4 exact batch calc        : {exact_prob:.4f}")
    print(f"  Absolute difference             : "
          f"{abs(pf_prob_treatment_better - exact_prob):.4f}")

    # ------------------------------------------------------------------
    # Step 5: validation
    # ------------------------------------------------------------------
    diff = abs(pf_prob_treatment_better - exact_prob)
    print()
    if diff < 0.05:
        print("  VALIDATION: PASS (PF and exact batch calc agree within 0.05)")
    else:
        print("  VALIDATION: WARNING -- PF and exact calc disagree substantially")

    print()
    print("=" * 70)
    print("  Key takeaway: identical ParticleFilter machinery used for")
    print("  ED queue filtering now performs sequential (patient-by-")
    print("  patient) clinical trial monitoring, and its final posterior")
    print("  agrees closely with Week 4's exact batch nested-MC")
    print("  calculation on the same trial data.")
    print("=" * 70)
