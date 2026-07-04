"""
python/pdsl/compiler.py
PDSL top-level compiler: source string -> runnable Python objects.

Usage
-----
    from python.pdsl.compiler import compile_pdsl

    src = open("battery.pdsl").read()
    model, priors, run_cfg = compile_pdsl(src)

    from python.src.monte_carlo import MonteCarloEngine
    engine = MonteCarloEngine(model, priors, **run_cfg)
    result = engine.run()
"""

from __future__ import annotations

from python.pdsl.codegen import generate
from python.pdsl.parser import parse
from python.src.distributions import Distribution
from python.src.state import Model


def compile_pdsl(
    source:     str,
    model_name: str | None = None,
) -> tuple[Model, list[Distribution], dict[str, object]]:
    """
    Compile a PDSL source string to a runnable (model, priors, run_config).

    Parameters
    ----------
    source : str
        PDSL program text.
    model_name : str | None
        Which model to return if the program defines multiple models.
        Defaults to the first model.

    Returns
    -------
    model : Model
        An instantiated Model subclass generated from the PDSL program.
    priors : list[Distribution]
        One Distribution per parameter, in declaration order.
    run_config : dict
        {'N': int, 'steps': int, 'dt': float, 'seed': int}

    Raises
    ------
    ValueError
        If the program defines no models, or the named model is not found.
    lark.exceptions.UnexpectedInput
        If the source has a syntax error.
    """
    # Step 1: parse source -> AST
    program = parse(source)

    if not program.models:
        raise ValueError("PDSL program defines no models")

    # Step 2: select target model
    if model_name is None:
        target = program.models[0]
    else:
        matches = [m for m in program.models if m.name == model_name]
        if not matches:
            available = [m.name for m in program.models]
            raise ValueError(
                f"Model '{model_name}' not found. Available: {available}"
            )
        target = matches[0]

    # Step 3: generate Python source
    python_src = generate(program)

    # Step 4: execute generated code in a fresh namespace
    #
    # SECURITY BOUNDARY (verified, see docs/standards/quality_standards.md
    # Section 8): python_src here comes EXCLUSIVELY from generate(program)
    # at Step 3 above -- our own AST-to-Python codegen function. It is
    # never the raw PDSL `source` string or any other unvalidated input
    # passed through unmodified. Re-verify this boundary explicitly
    # whenever compile_pdsl() or generate() changes.
    namespace: dict[str, object] = {}
    exec(python_src, namespace)  # nosec B102 -- see security boundary note above  # noqa: S102

    # Step 5: extract model instance, priors, run config
    class_name    = f"PDSL_{target.name.capitalize()}Model"
    priors_fn     = f"build_{target.name}_priors"
    run_cfg_name  = f"{target.name}_run_config"

    model_cls  = namespace[class_name]
    priors_fn_ = namespace[priors_fn]
    run_cfg    = namespace[run_cfg_name]

    model  = model_cls()      # type: ignore[operator]
    priors = priors_fn_()     # type: ignore[operator]

    return model, priors, run_cfg  # type: ignore[return-value]
