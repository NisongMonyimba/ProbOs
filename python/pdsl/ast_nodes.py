"""
python/pdsl/ast_nodes.py
PDSL Abstract Syntax Tree node definitions.

Every construct in a PDSL program maps to one of these dataclasses.
The AST is the bridge between the Lark parse tree and the code generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class NumberLit:
    value: float

@dataclass
class Var:
    name: str

@dataclass
class BinOp:
    op:    str   # '+' '-' '*' '/'
    left:  Expr
    right: Expr

@dataclass
class UnaryOp:
    op:      str   # '+' '-'
    operand: Expr

@dataclass
class Pow:
    base: Expr
    exp:  Expr

@dataclass
class FuncCall:
    name: str
    args: list[Expr]

Expr = NumberLit | Var | BinOp | UnaryOp | Pow | FuncCall


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass
class StateDecl:
    name:  str
    init:  float

@dataclass
class DistributionNode:
    name: str          # "Normal", "LogNormal", "Uniform", "Beta"
    args: list[float]

@dataclass
class ParamDecl:
    name:         str
    distribution: DistributionNode

@dataclass
class DriftDecl:
    state_name: str
    expr:       Expr

@dataclass
class RunConfig:
    N:     int   = 1000
    steps: int   = 300
    dt:    float = 1.0
    seed:  int   = 42


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@dataclass
class ModelNode:
    name:   str
    states: list[StateDecl]    = field(default_factory=list)
    params: list[ParamDecl]    = field(default_factory=list)
    drifts: list[DriftDecl]    = field(default_factory=list)
    run:    RunConfig           = field(default_factory=RunConfig)

@dataclass
class Program:
    models: list[ModelNode]
