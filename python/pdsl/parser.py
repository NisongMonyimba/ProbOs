"""
python/pdsl/parser.py
PDSL parser: Lark parse tree -> PDSL AST.
"""

from __future__ import annotations

import pathlib
from typing import Any, cast

from lark import Lark, Token, Transformer

from python.pdsl.ast_nodes import (
    BinOp,
    Comparison,
    Conditional,
    DistributionNode,
    DriftDecl,
    Expr,
    FuncCall,
    ModelNode,
    NumberLit,
    ParamDecl,
    Pow,
    Program,
    RunConfig,
    StateDecl,
    UnaryOp,
    Var,
)

_GRAMMAR_PATH = pathlib.Path(__file__).parent / "grammar.lark"
_GRAMMAR      = _GRAMMAR_PATH.read_text()
_PARSER       = Lark(_GRAMMAR, parser="earley", ambiguity="resolve")


class _PDSLTransformer(Transformer[Token, Any]):

    # ------------------------------------------------------------------
    # Terminals
    # ------------------------------------------------------------------
    def NUMBER(self, tok: Token) -> float:
        return float(tok)

    def INTEGER(self, tok: Token) -> int:
        return int(tok)

    def NAME(self, tok: Token) -> str:
        return str(tok)

    # ------------------------------------------------------------------
    # number rule
    # ------------------------------------------------------------------
    def number(self, items: list[Any]) -> float:
        return float(cast(float, items[0]))

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------
    def number_lit(self, items: list[Any]) -> NumberLit:
        return NumberLit(float(cast(float, items[0])))

    def var(self, items: list[Any]) -> Var:
        return Var(cast(str, items[0]))

    def add(self, items: list[Any]) -> BinOp:
        return BinOp("+", cast(Expr, items[0]), cast(Expr, items[1]))

    def sub(self, items: list[Any]) -> BinOp:
        return BinOp("-", cast(Expr, items[0]), cast(Expr, items[1]))

    def mul(self, items: list[Any]) -> BinOp:
        return BinOp("*", cast(Expr, items[0]), cast(Expr, items[1]))

    def div(self, items: list[Any]) -> BinOp:
        return BinOp("/", cast(Expr, items[0]), cast(Expr, items[1]))

    def pow(self, items: list[Any]) -> Pow:
        return Pow(cast(Expr, items[0]), cast(Expr, items[1]))

    def pos(self, items: list[Any]) -> Expr:
        return cast(Expr, items[0])

    def neg(self, items: list[Any]) -> UnaryOp:
        return UnaryOp("-", cast(Expr, items[0]))

    # ------------------------------------------------------------------
    # Comparisons + conditional (PDSL v0.2, Week 13)
    # ------------------------------------------------------------------
    def lt(self, items: list[Any]) -> Comparison:
        return Comparison("<", cast(Expr, items[0]), cast(Expr, items[1]))

    def gt(self, items: list[Any]) -> Comparison:
        return Comparison(">", cast(Expr, items[0]), cast(Expr, items[1]))

    def le(self, items: list[Any]) -> Comparison:
        return Comparison("<=", cast(Expr, items[0]), cast(Expr, items[1]))

    def ge(self, items: list[Any]) -> Comparison:
        return Comparison(">=", cast(Expr, items[0]), cast(Expr, items[1]))

    def eq(self, items: list[Any]) -> Comparison:
        return Comparison("==", cast(Expr, items[0]), cast(Expr, items[1]))

    def ne(self, items: list[Any]) -> Comparison:
        return Comparison("!=", cast(Expr, items[0]), cast(Expr, items[1]))

    def conditional(self, items: list[Any]) -> Conditional:
        return Conditional(
            cond=cast(Comparison, items[0]),
            then_expr=cast(Expr, items[1]),
            else_expr=cast(Expr, items[2]),
        )

    def exprlist(self, items: list[Any]) -> list[Expr]:
        return [cast(Expr, x) for x in items]

    def func_call(self, items: list[Any]) -> FuncCall:
        name: str = cast(str, items[0])
        args: list[Expr] = cast(list[Expr], items[1]) if len(items) > 1 else []
        return FuncCall(name, args)

    # ------------------------------------------------------------------
    # Distributions
    # ------------------------------------------------------------------
    def numlist(self, items: list[Any]) -> list[float]:
        return [float(cast(float, x)) for x in items]

    def distribution(self, items: list[Any]) -> DistributionNode:
        name: str = cast(str, items[0])
        args: list[float] = cast(list[float], items[1]) if len(items) > 1 else []
        return DistributionNode(name, [float(a) for a in args])

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------
    def state_decl(self, items: list[Any]) -> StateDecl:
        return StateDecl(name=cast(str, items[0]), init=float(cast(float, items[1])))

    def param_decl(self, items: list[Any]) -> ParamDecl:
        return ParamDecl(
            name=cast(str, items[0]),
            distribution=cast(DistributionNode, items[1]),
        )

    def drift_decl(self, items: list[Any]) -> DriftDecl:
        return DriftDecl(state_name=cast(str, items[0]), expr=cast(Expr, items[1]))

    def run_n(self, items: list[Any]) -> tuple[str, int]:
        return ("N", int(cast(int, items[0])))

    def run_steps(self, items: list[Any]) -> tuple[str, int]:
        return ("steps", int(cast(int, items[0])))

    def run_dt(self, items: list[Any]) -> tuple[str, float]:
        return ("dt", float(cast(float, items[0])))

    def run_seed(self, items: list[Any]) -> tuple[str, int]:
        return ("seed", int(cast(int, items[0])))

    def run_decl(self, items: list[Any]) -> RunConfig:
        cfg: dict[str, object] = {}
        for item in items:
            if isinstance(item, tuple):
                key, val = item
                cfg[str(key)] = val
        return RunConfig(
            N     = int(cast(int, cfg.get("N",     1000))),
            steps = int(cast(int, cfg.get("steps", 300))),
            dt    = float(cast(float, cfg.get("dt",  1.0))),
            seed  = int(cast(int, cfg.get("seed",  42))),
        )

    def stmt(self, items: list[Any]) -> object:
        return items[0]

    # ------------------------------------------------------------------
    # Model + Program
    # ------------------------------------------------------------------
    def model(self, items: list[Any]) -> ModelNode:
        name  = cast(str, items[0])
        stmts = list(items[1:])
        node  = ModelNode(name=name)
        for s in stmts:
            if isinstance(s, StateDecl):
                node.states.append(s)
            elif isinstance(s, ParamDecl):
                node.params.append(s)
            elif isinstance(s, DriftDecl):
                node.drifts.append(s)
            elif isinstance(s, RunConfig):
                node.run = s
        return node

    def start(self, items: list[Any]) -> Program:
        models = [m for m in items if isinstance(m, ModelNode)]
        return Program(models=models)


def parse(source: str) -> Program:
    """Parse a PDSL source string and return a Program AST."""
    tree = _PARSER.parse(source)
    result = _PDSLTransformer().transform(tree)
    assert isinstance(result, Program)
    return result
