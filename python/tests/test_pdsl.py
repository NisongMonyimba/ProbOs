"""
Tests for PDSL parser, codegen, and compiler.

25 tests covering:
  - Grammar parsing (states, params, drifts, run config)
  - Expression parsing (arithmetic, unary minus, function calls)
  - Codegen correctness (analytical match for simple ODEs)
  - Compiler end-to-end (compile -> forward_batch -> MonteCarloEngine)
"""

from __future__ import annotations

import numpy as np
import pytest

from python.pdsl.ast_nodes import (
    BinOp,
    Program,
    UnaryOp,
)
from python.pdsl.codegen import generate
from python.pdsl.compiler import compile_pdsl
from python.pdsl.parser import parse
from python.src.distributions import Normal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DECAY_SRC = """
model decay {
  state x = 100.0
  param k ~ Normal(0.1, 0.01)
  drift x = -(k * x)
  run N=200 steps=50 dt=1.0 seed=42
}
"""

BATTERY_SRC = """
model battery {
  state T1    = 403.15
  state c_SEI = 1.0
  param Ea_SEI ~ Normal(135080.0, 5000.0)
  param A_SEI  ~ Normal(1.667e15, 1.0e13)
  drift T1    = (A_SEI * exp(-Ea_SEI / (8.314 * T1)) * c_SEI * 250000.0)
  drift c_SEI = -(A_SEI * exp(-Ea_SEI / (8.314 * T1)) * c_SEI)
  run N=500 steps=10 dt=0.01 seed=42
}
"""


# ---------------------------------------------------------------------------
# TestGrammarParsing
# ---------------------------------------------------------------------------

class TestGrammarParsing:

    def test_parse_returns_program(self) -> None:
        prog = parse(DECAY_SRC)
        assert isinstance(prog, Program)

    def test_parse_one_model(self) -> None:
        prog = parse(DECAY_SRC)
        assert len(prog.models) == 1

    def test_model_name(self) -> None:
        prog = parse(DECAY_SRC)
        assert prog.models[0].name == "decay"

    def test_one_state(self) -> None:
        prog = parse(DECAY_SRC)
        assert len(prog.models[0].states) == 1
        assert prog.models[0].states[0].name == "x"
        assert prog.models[0].states[0].init == 100.0

    def test_one_param(self) -> None:
        prog = parse(DECAY_SRC)
        params = prog.models[0].params
        assert len(params) == 1
        assert params[0].name == "k"
        assert params[0].distribution.name == "Normal"
        assert params[0].distribution.args == [0.1, 0.01]

    def test_one_drift(self) -> None:
        prog = parse(DECAY_SRC)
        drifts = prog.models[0].drifts
        assert len(drifts) == 1
        assert drifts[0].state_name == "x"

    def test_run_config(self) -> None:
        prog = parse(DECAY_SRC)
        run = prog.models[0].run
        assert run.N == 200
        assert run.steps == 50
        assert run.dt == 1.0
        assert run.seed == 42

    def test_battery_two_states(self) -> None:
        prog = parse(BATTERY_SRC)
        assert len(prog.models[0].states) == 2

    def test_battery_two_params(self) -> None:
        prog = parse(BATTERY_SRC)
        assert len(prog.models[0].params) == 2

    def test_battery_two_drifts(self) -> None:
        prog = parse(BATTERY_SRC)
        assert len(prog.models[0].drifts) == 2


# ---------------------------------------------------------------------------
# TestExpressionParsing
# ---------------------------------------------------------------------------

class TestExpressionParsing:

    def test_unary_minus_in_func_call(self) -> None:
        """exp(-Ea_SEI / (...)) must parse without error."""
        prog = parse(BATTERY_SRC)
        drift = prog.models[0].drifts[0]
        assert drift.expr is not None  # parsed successfully

    def test_negated_expression_drift(self) -> None:
        """drift x = -(k * x) must parse as UnaryOp(BinOp)."""
        prog = parse(DECAY_SRC)
        expr = prog.models[0].drifts[0].expr
        assert isinstance(expr, UnaryOp)
        assert expr.op == "-"
        assert isinstance(expr.operand, BinOp)
        assert expr.operand.op == "*"

    def test_func_call_parses(self) -> None:
        prog = parse(BATTERY_SRC)
        # drift T1 contains exp(...) -- verify no parse error occurred
        drift_T1 = next(
            d for d in prog.models[0].drifts if d.state_name == "T1"
        )
        assert drift_T1.expr is not None


# ---------------------------------------------------------------------------
# TestCodegen
# ---------------------------------------------------------------------------

class TestCodegen:

    def test_generate_returns_string(self) -> None:
        prog = parse(DECAY_SRC)
        code = generate(prog)
        assert isinstance(code, str)

    def test_generate_contains_class_def(self) -> None:
        prog = parse(DECAY_SRC)
        code = generate(prog)
        assert "class PDSL_DecayModel" in code

    def test_generate_contains_forward_batch(self) -> None:
        prog = parse(DECAY_SRC)
        code = generate(prog)
        assert "def forward_batch" in code

    def test_generate_contains_priors_function(self) -> None:
        prog = parse(DECAY_SRC)
        code = generate(prog)
        assert "def build_decay_priors" in code

    def test_generate_is_valid_python(self) -> None:
        """Generated code must be exec()-able without SyntaxError."""
        prog = parse(DECAY_SRC)
        code = generate(prog)
        namespace: dict[str, object] = {}
        exec(code, namespace)  # noqa: S102
        assert "PDSL_DecayModel" in namespace


# ---------------------------------------------------------------------------
# TestCompilerEndToEnd
# ---------------------------------------------------------------------------

class TestCompilerEndToEnd:

    def test_compile_returns_tuple(self) -> None:
        result = compile_pdsl(DECAY_SRC)
        assert len(result) == 3

    def test_compile_model_state_dim(self) -> None:
        model, _, _ = compile_pdsl(DECAY_SRC)
        assert model.state_dim == 1

    def test_compile_model_param_dim(self) -> None:
        model, _, _ = compile_pdsl(DECAY_SRC)
        assert model.param_dim == 1

    def test_compile_priors_are_distributions(self) -> None:
        _, priors, _ = compile_pdsl(DECAY_SRC)
        assert len(priors) == 1
        assert isinstance(priors[0], Normal)

    def test_compile_run_config(self) -> None:
        _, _, run_cfg = compile_pdsl(DECAY_SRC)
        assert run_cfg["N"] == 200
        assert run_cfg["steps"] == 50

    def test_forward_batch_analytical_match(self) -> None:
        """
        drift x = -(k*x), Euler step:
        new_x = x + dt*(-k*x) = x*(1 - dt*k)
        """
        model, _, _ = compile_pdsl(DECAY_SRC)
        N = 10
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.full((N, 1), 0.1)
        new_state = model.forward_batch(state, params, dt=1.0)
        expected = 100.0 * (1.0 - 1.0 * 0.1)
        np.testing.assert_allclose(new_state[:, 0], expected, rtol=1e-10)

    def test_forward_batch_decays_over_time(self) -> None:
        model, _, _ = compile_pdsl(DECAY_SRC)
        N = 10
        state  = np.tile(model.initial_state(), (N, 1))
        params = np.full((N, 1), 0.1)
        for _ in range(10):
            state = model.forward_batch(state, params, dt=1.0)
        assert np.all(state[:, 0] < 100.0), "x should decay"
        assert np.all(state[:, 0] > 0.0), "x should stay positive"

    def test_compile_through_monte_carlo_engine(self) -> None:
        from python.src.monte_carlo import MonteCarloEngine

        model, priors, run_cfg = compile_pdsl(DECAY_SRC)
        N_val     = run_cfg["N"]
        steps_val = run_cfg["steps"]
        dt_val    = run_cfg["dt"]
        seed_val  = run_cfg["seed"]
        assert isinstance(N_val, int)
        assert isinstance(steps_val, int)
        assert isinstance(dt_val, float)
        assert isinstance(seed_val, int)
        engine = MonteCarloEngine(
            model, priors,
            N=N_val, n_steps=steps_val,
            dt=dt_val, seed=seed_val,
        )
        result = engine.run()
        assert result.trajectories.shape == (200, 51, 1)
        # Decay -- P50 at final step should be less than initial value
        assert result.percentiles[1, -1, 0] < 100.0

    def test_unknown_model_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            compile_pdsl(DECAY_SRC, model_name="nonexistent")

    def test_empty_program_raises(self) -> None:
        """
        Empty source fails at the grammar level (start: model+ requires
        at least one model), not at the semantic compile_pdsl() level.
        Lark raises UnexpectedEOF before compile_pdsl's own ValueError
        check is ever reached.
        """
        from lark.exceptions import UnexpectedEOF
        with pytest.raises(UnexpectedEOF):
            compile_pdsl("")
