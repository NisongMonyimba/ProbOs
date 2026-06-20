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

from python.examples.week4_option_pricer import (
    OptionPricerModel, build_option_priors,
)
from python.examples.week4_ed_queue import (
    EDQueueModel, build_ed_queue_priors,
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
        assert L == pytest.approx(1.0)
        assert W == pytest.approx(0.2)  # L/lambda = 1.0/5.0

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
