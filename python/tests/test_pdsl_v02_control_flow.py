"""
python/tests/test_pdsl_v02_control_flow.py

Tests for PDSL v0.2's control-flow constructs (Comparison,
Conditional) -- added Month 3 Week 13. Complements test_pdsl.py
(v0.1's arithmetic-only grammar), which remains valid and unchanged.
"""

from __future__ import annotations

import numpy as np

from python.pdsl.ast_nodes import Comparison, Conditional, NumberLit, UnaryOp, Var
from python.pdsl.codegen import generate
from python.pdsl.parser import parse


class TestGrammarLevel:

    def test_basic_conditional_parses(self) -> None:
        source = """
        model Test {
            state x = 1.0
            param p ~ Normal(0.0, 1.0)
            drift x = if x > 5.0 then 1.0 else -1.0
            run N=100 steps=10 dt=1.0 seed=42
        }
        """
        program = parse(source)
        assert len(program.models) == 1

    def test_all_six_comparison_operators_parse(self) -> None:
        for op in ["<", ">", "<=", ">=", "==", "!="]:
            source = f"""
            model Test {{
                state x = 1.0
                param p ~ Normal(0.0, 1.0)
                drift x = if x {op} 1.0 then 1.0 else 0.0
                run N=100 steps=10 dt=1.0 seed=42
            }}
            """
            program = parse(source)
            assert len(program.models) == 1, f"failed to parse operator {op}"

    def test_nested_arithmetic_inside_condition_and_branches(self) -> None:
        source = """
        model Test {
            state x = 1.0
            param p ~ Normal(0.0, 1.0)
            drift x = if x + p < 10.0 then x * 2.0 else x / 2.0
            run N=100 steps=10 dt=1.0 seed=42
        }
        """
        program = parse(source)
        assert len(program.models) == 1


class TestASTLevel:

    def test_conditional_has_correct_structure(self) -> None:
        source = """
        model Test {
            state x = 1.0
            param p ~ Normal(0.0, 1.0)
            drift x = if x > 5.0 then 1.0 else -1.0
            run N=100 steps=10 dt=1.0 seed=42
        }
        """
        program = parse(source)
        drift = program.models[0].drifts[0]

        assert isinstance(drift.expr, Conditional)
        assert isinstance(drift.expr.cond, Comparison)
        assert drift.expr.cond.op == ">"
        assert isinstance(drift.expr.cond.left, Var)
        assert drift.expr.cond.left.name == "x"
        assert isinstance(drift.expr.cond.right, NumberLit)
        assert drift.expr.cond.right.value == 5.0
        assert isinstance(drift.expr.then_expr, NumberLit)
        assert drift.expr.then_expr.value == 1.0
        assert isinstance(drift.expr.else_expr, UnaryOp)
        assert drift.expr.else_expr.op == "-"

    def test_all_comparison_ops_map_correctly(self) -> None:
        expected_ops = ["<", ">", "<=", ">=", "==", "!="]
        for op in expected_ops:
            source = f"""
            model Test {{
                state x = 1.0
                param p ~ Normal(0.0, 1.0)
                drift x = if x {op} 1.0 then 1.0 else 0.0
                run N=100 steps=10 dt=1.0 seed=42
            }}
            """
            program = parse(source)
            drift = program.models[0].drifts[0]
            assert isinstance(drift.expr, Conditional)
            assert drift.expr.cond.op == op


class TestCodegenLevel:

    def test_generated_code_contains_np_where(self) -> None:
        source = """
        model Test {
            state x = 1.0
            param p ~ Normal(0.0, 1.0)
            drift x = if x > 5.0 then 1.0 else -1.0
            run N=100 steps=10 dt=1.0 seed=42
        }
        """
        program = parse(source)
        python_src = generate(program)
        assert "np.where(" in python_src

    def test_generated_code_is_syntactically_valid(self) -> None:
        source = """
        model Test {
            state x = 1.0
            param p ~ Normal(0.0, 1.0)
            drift x = if x > 5.0 then 1.0 else -1.0
            run N=100 steps=10 dt=1.0 seed=42
        }
        """
        program = parse(source)
        python_src = generate(program)
        compile(python_src, "<generated>", "exec")


class TestEndToEndExecution:

    def test_conditional_vectorises_correctly_across_particles(self) -> None:
        source = """
        model Threshold {
            state x = 10.0
            param p ~ Normal(0.0, 1.0)
            drift x = if x > 5.0 then -1.0 else 1.0
            run N=100 steps=10 dt=1.0 seed=42
        }
        """
        program = parse(source)
        python_src = generate(program)
        namespace: dict[str, object] = {}
        exec(python_src, namespace)
        model_cls = namespace["PDSL_ThresholdModel"]
        model = model_cls()  # type: ignore[operator]

        state = np.array([
            [15.0], [15.0], [15.0], [15.0], [15.0],
            [2.0],  [2.0],  [2.0],  [2.0],  [2.0],
        ])
        params = np.zeros((10, 1))
        new_state = model.forward_batch(state, params, dt=1.0)

        assert np.allclose(new_state[:5, 0], 14.0)
        assert np.allclose(new_state[5:, 0], 3.0)

    def test_conditional_at_exact_boundary(self) -> None:
        source = """
        model Boundary {
            state x = 5.0
            param p ~ Normal(0.0, 1.0)
            drift x = if x > 5.0 then -1.0 else 1.0
            run N=10 steps=1 dt=1.0 seed=42
        }
        """
        program = parse(source)
        python_src = generate(program)
        namespace: dict[str, object] = {}
        exec(python_src, namespace)
        model_cls = namespace["PDSL_BoundaryModel"]
        model = model_cls()  # type: ignore[operator]

        state = np.array([[5.0]])
        params = np.zeros((1, 1))
        new_state = model.forward_batch(state, params, dt=1.0)

        assert np.allclose(new_state[0, 0], 6.0)


class TestRegressionV01StillWorks:

    def test_v01_model_still_parses_and_generates(self) -> None:
        source = """
        model OldStyle {
            state x = 1.0
            param p ~ Normal(135080.0, 6754.0)
            drift x = -p * exp(-x / (8.314 * 298.15))
            run N=5000 steps=300 dt=1.0 seed=42
        }
        """
        program = parse(source)
        python_src = generate(program)
        namespace: dict[str, object] = {}
        exec(python_src, namespace)
        model_cls = namespace["PDSL_OldstyleModel"]
        model = model_cls()  # type: ignore[operator]
        assert model.state_dim == 1
        assert model.param_dim == 1
