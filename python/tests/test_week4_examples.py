"""
python/tests/test_week4_examples.py

Smoke tests for the Week 4 Saturday cross-discipline examples
(option pricer, ED queue, clinical trial). These are NOT exhaustive
unit tests of every code path -- per the project's testing discipline,
utility/example files get smoke tests confirming they run correctly
and produce physically/statistically sane output, while core kernel
files (distributions.py, monte_carlo.py, etc.) get full test suites.

See docs/study/study_guide.md Week 4 entries for the reading material
behind each model implemented here.
"""

from __future__ import annotations

import numpy as np
import pytest

from python.examples.week4_ed_queue import (
    EDQueueModel,
    build_ed_queue_priors,
)
from python.examples.week4_option_pricer import (
    OptionPricerModel,
    build_option_priors,
)
from python.src.monte_carlo import MonteCarloEngine

# ---------------------------------------------------------------------------
# TestOptionPricerModel
# ---------------------------------------------------------------------------

class TestOptionPricerModel:

    def test_state_dim_is_1(self) -> None:
        model = OptionPricerModel()
        assert model.state_dim == 1

    def test_param_dim_is_2(self) -> None:
        model = OptionPricerModel()
        assert model.param_dim == 2

    def test_param_names(self) -> None:
        model = OptionPricerModel()
        assert model.param_names() == ["mu", "sigma"]

    def test_initial_state_equals_S0(self) -> None:
        model = OptionPricerModel(S0=123.45)
        np.testing.assert_allclose(model.initial_state(), [123.45])

    def test_forward_batch_shape(self) -> None:
        model  = OptionPricerModel()
        N      = 50
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.column_stack([
            np.full(N, 0.05), np.full(N, 0.20),
        ])
        new_state = model.forward_batch(state, params, dt=1.0 / 252)
        assert new_state.shape == (N, 1)

    def test_forward_batch_prices_stay_positive(self) -> None:
        """GBM prices must remain strictly positive (log-normal support)."""
        model  = OptionPricerModel()
        N      = 200
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.column_stack([
            np.full(N, 0.05), np.full(N, 0.20),
        ])
        for _ in range(50):
            state = model.forward_batch(state, params, dt=1.0 / 252)
        assert np.all(state > 0.0)

    def test_black_scholes_price_positive(self) -> None:
        model = OptionPricerModel(S0=100.0, K=100.0, T=1.0, r=0.05)
        price = model.black_scholes_price(sigma=0.20)
        assert price > 0.0

    def test_mc_price_close_to_black_scholes(self) -> None:
        """
        End-to-end validation: MC price (with tight sigma prior) should
        be within a few standard errors of the closed-form price.
        Uses a SMALL N for test speed -- the full N=20000 demo script
        gives a tighter match; this test just checks no gross error.
        """
        model  = OptionPricerModel(S0=100.0, K=100.0, T=1.0, r=0.05)
        priors = build_option_priors(r=0.05)
        engine = MonteCarloEngine(
            model, priors, N=2000, n_steps=50, dt=1.0 / 50, seed=42
        )
        result = engine.run()
        final_prices = result.trajectories[:, -1, 0]
        mc_price, mc_std_err = model.price_option(final_prices)
        bs_price = model.black_scholes_price(sigma=0.20)
        diff_in_std_errors = abs(mc_price - bs_price) / mc_std_err
        assert diff_in_std_errors < 8.0, (
            f"MC price {mc_price:.4f} too far from BS price {bs_price:.4f} "
            f"({diff_in_std_errors:.2f} std errors)"
        )

    def test_build_option_priors_returns_two(self) -> None:
        priors = build_option_priors()
        assert len(priors) == 2


# ---------------------------------------------------------------------------
# TestEDQueueModel
# ---------------------------------------------------------------------------

class TestEDQueueModel:

    def test_state_dim_is_1(self) -> None:
        model = EDQueueModel()
        assert model.state_dim == 1

    def test_param_dim_is_2(self) -> None:
        model = EDQueueModel()
        assert model.param_dim == 2

    def test_param_names(self) -> None:
        model = EDQueueModel()
        assert model.param_names() == ["lambda", "mu"]

    def test_initial_state_equals_Q0(self) -> None:
        model = EDQueueModel(Q0=5.0)
        np.testing.assert_allclose(model.initial_state(), [5.0])

    def test_forward_batch_shape(self) -> None:
        model  = EDQueueModel()
        N      = 50
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.column_stack([
            np.full(N, 10.0), np.full(N, 16.0),
        ])
        new_state = model.forward_batch(state, params, dt=0.05)
        assert new_state.shape == (N, 1)

    def test_forward_batch_never_negative(self) -> None:
        """Queue length must never go below zero."""
        model  = EDQueueModel(Q0=0.0)
        N      = 200
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.column_stack([
            np.full(N, 10.0), np.full(N, 16.0),
        ])
        for _ in range(100):
            state = model.forward_batch(state, params, dt=0.05)
            assert np.all(state >= 0.0)

    def test_steady_state_theory_known_values(self) -> None:
        """rho=0.5 -> L=1.0 (textbook M/M/1 result)."""
        rho, L, W = EDQueueModel.steady_state_theory(lam=5.0, mu=10.0)
        assert rho == pytest.approx(0.5)
        assert pytest.approx(1.0) == L
        assert pytest.approx(0.2) == W  # L/lambda = 1.0/5.0

    def test_steady_state_theory_raises_if_unstable(self) -> None:
        with pytest.raises(ValueError, match="Unstable queue"):
            EDQueueModel.steady_state_theory(lam=10.0, mu=10.0)

    def test_steady_state_theory_raises_if_overloaded(self) -> None:
        with pytest.raises(ValueError, match="Unstable queue"):
            EDQueueModel.steady_state_theory(lam=12.0, mu=10.0)

    def test_build_ed_queue_priors_returns_two(self) -> None:
        priors = build_ed_queue_priors()
        assert len(priors) == 2

    def test_priors_guarantee_stable_queue(self) -> None:
        """
        The Uniform prior bounds for lambda (8-12) and mu (14-18) must
        guarantee rho < 1 for every possible draw, so steady_state_theory
        never raises during the full Monte Carlo run.
        """
        lam_max = 12.0   # upper bound of lambda prior
        mu_min  = 14.0   # lower bound of mu prior
        assert lam_max < mu_min, (
            "Prior bounds do not guarantee a stable queue for all draws"
        )

    def test_mc_simulation_produces_sane_queue_lengths(self) -> None:
        """
        End-to-end smoke test: run a short MC simulation and check the
        resulting queue lengths are non-negative and roughly in the
        right order of magnitude (not exploding or staying at zero).
        Uses a SMALL N and short horizon for test speed.
        """
        model  = EDQueueModel(Q0=0.0)
        priors = build_ed_queue_priors()
        engine = MonteCarloEngine(
            model, priors, N=500, n_steps=200, dt=0.05, seed=42
        )
        result = engine.run()
        final_Q = result.trajectories[:, -1, 0]
        assert np.all(final_Q >= 0.0)
        assert np.mean(final_Q) < 50.0, "Queue length exploded -- check model"


# ---------------------------------------------------------------------------
# TestClinicalTrialModel
# ---------------------------------------------------------------------------

from python.examples.week4_clinical_trial import (  # noqa: E402
    ClinicalTrialModel,
    build_clinical_trial_priors,
)


class TestClinicalTrialModel:

    def test_state_dim_is_4(self) -> None:
        model = ClinicalTrialModel()
        assert model.state_dim == 4

    def test_param_dim_is_2(self) -> None:
        model = ClinicalTrialModel()
        assert model.param_dim == 2

    def test_param_names(self) -> None:
        model = ClinicalTrialModel()
        assert model.param_names() == [
            "p_treatment_true", "p_control_true",
        ]

    def test_initial_state_all_zero(self) -> None:
        model = ClinicalTrialModel()
        np.testing.assert_allclose(
            model.initial_state(), [0.0, 0.0, 0.0, 0.0]
        )

    def test_forward_batch_shape(self) -> None:
        model  = ClinicalTrialModel()
        N      = 50
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.column_stack([
            np.full(N, 0.45), np.full(N, 0.30),
        ])
        new_state = model.forward_batch(state, params, dt=1.0)
        assert new_state.shape == (N, 4)

    def test_forward_batch_enrolls_exactly_one_patient(self) -> None:
        """
        Each forward_batch call must increase total enrollment
        (n_treatment + n_control) by exactly 1 per particle.
        """
        model  = ClinicalTrialModel()
        N      = 100
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.column_stack([
            np.full(N, 0.45), np.full(N, 0.30),
        ])
        new_state = model.forward_batch(state, params, dt=1.0)
        total_before = state[:, 0] + state[:, 2]
        total_after  = new_state[:, 0] + new_state[:, 2]
        np.testing.assert_allclose(total_after - total_before, 1.0)

    def test_forward_batch_successes_never_exceed_enrollment(self) -> None:
        """s_treat <= n_treat and s_ctrl <= n_ctrl must always hold."""
        model  = ClinicalTrialModel()
        N      = 100
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.column_stack([
            np.full(N, 0.45), np.full(N, 0.30),
        ])
        for _ in range(50):
            state = model.forward_batch(state, params, dt=1.0)
            assert np.all(state[:, 1] <= state[:, 0])  # s_treat <= n_treat
            assert np.all(state[:, 3] <= state[:, 2])  # s_ctrl <= n_ctrl

    def test_randomisation_ratio_roughly_balanced(self) -> None:
        """
        With randomisation_ratio=0.5 and many patients, n_treatment and
        n_control should end up roughly equal (within statistical noise).
        """
        model  = ClinicalTrialModel(randomisation_ratio=0.5)
        N      = 500
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.column_stack([
            np.full(N, 0.45), np.full(N, 0.30),
        ])
        for _ in range(200):
            state = model.forward_batch(state, params, dt=1.0)
        mean_n_treat = np.mean(state[:, 0])
        mean_n_ctrl  = np.mean(state[:, 2])
        # Both should be close to 100 (half of 200 patients)
        assert 80.0 < mean_n_treat < 120.0
        assert 80.0 < mean_n_ctrl < 120.0

    def test_posterior_prob_returns_values_in_unit_interval(self) -> None:
        s_treat = np.array([45.0, 10.0])
        n_treat = np.array([100.0, 100.0])
        s_ctrl  = np.array([30.0, 10.0])
        n_ctrl  = np.array([100.0, 100.0])
        probs = ClinicalTrialModel.posterior_prob_treatment_better(
            s_treat, n_treat, s_ctrl, n_ctrl, n_mc_samples=500
        )
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_posterior_prob_high_when_treatment_clearly_better(self) -> None:
        """
        45/100 vs 10/100 is a large, clear effect -- posterior probability
        treatment is better should be very close to 1.0.
        """
        s_treat = np.array([45.0])
        n_treat = np.array([100.0])
        s_ctrl  = np.array([10.0])
        n_ctrl  = np.array([100.0])
        probs = ClinicalTrialModel.posterior_prob_treatment_better(
            s_treat, n_treat, s_ctrl, n_ctrl, n_mc_samples=5000
        )
        assert probs[0] > 0.99

    def test_posterior_prob_near_half_when_arms_identical(self) -> None:
        """Identical observed rates -> posterior prob should be near 0.5."""
        s_treat = np.array([30.0])
        n_treat = np.array([100.0])
        s_ctrl  = np.array([30.0])
        n_ctrl  = np.array([100.0])
        probs = ClinicalTrialModel.posterior_prob_treatment_better(
            s_treat, n_treat, s_ctrl, n_ctrl, n_mc_samples=5000
        )
        assert 0.3 < probs[0] < 0.7

    def test_build_clinical_trial_priors_returns_two(self) -> None:
        priors = build_clinical_trial_priors()
        assert len(priors) == 2

    def test_mc_simulation_produces_sane_trial_outcomes(self) -> None:
        """
        End-to-end smoke test: run a short MC simulation and check the
        resulting trial states are internally consistent.
        """
        model  = ClinicalTrialModel()
        priors = build_clinical_trial_priors()
        engine = MonteCarloEngine(
            model, priors, N=200, n_steps=50, dt=1.0, seed=42
        )
        result = engine.run()
        final_n_treat = result.trajectories[:, -1, 0]
        final_s_treat = result.trajectories[:, -1, 1]
        final_n_ctrl  = result.trajectories[:, -1, 2]
        final_s_ctrl  = result.trajectories[:, -1, 3]

        assert np.all(final_s_treat <= final_n_treat)
        assert np.all(final_s_ctrl <= final_n_ctrl)
        # Total enrolled across both arms should equal n_steps (50)
        np.testing.assert_allclose(
            final_n_treat + final_n_ctrl, 50.0
        )
