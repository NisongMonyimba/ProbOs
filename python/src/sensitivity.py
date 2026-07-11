"""
Sobol sensitivity analysis for ProbOS models.

Computational cost note (IMPORTANT for contributors):
    Saltelli sampling requires N_saltelli * (2 * param_dim + 2) model evaluations.
    With N_saltelli=1024, param_dim=15:
        total evaluations = 1024 * (2*15 + 2) = 1024 * 32 = 32,768 forward passes.
    This takes ~5-10 seconds on CPU for BatteryModel2Cell.
    On GPU/C++ (Month 4), N_saltelli can be scaled to 16384+ without issue.
    Always set N_saltelli to a power of 2 (SALib requirement).

    Quick smoke test : N_saltelli=64   ->  2,048 evaluations  (~0.1s)
    Standard run     : N_saltelli=1024 -> 32,768 evaluations  (~5-10s)
    Publication      : N_saltelli=2048 -> 65,536 evaluations  (~20s)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as saltelli

from python.src.distributions import Distribution
from python.src.state import Model

FloatArray = NDArray[np.float64]


@dataclass
class SobolResult:
    """
    Container for Sobol sensitivity analysis output.

    Attributes
    ----------
    param_names : list[str]
        Human-readable name for each parameter. Length = param_dim.
    S1 : shape (param_dim, state_dim)
        First-order Sobol indices.
        S1[i, k] = fraction of variance in state k explained by param i alone.
    ST : shape (param_dim, state_dim)
        Total-effect Sobol indices.
        ST[i, k] = fraction of variance in state k involving param i
                   (including all interactions).
    S1_conf : shape (param_dim, state_dim)
        95% confidence interval half-width for S1.
    ST_conf : shape (param_dim, state_dim)
        95% confidence interval half-width for ST.
    dominant_param : str
        Name of the parameter with highest S1 for state variable 0 (T1).
    n_evaluations : int
        Total number of model forward passes used.
    """

    param_names:    list[str]
    S1:             FloatArray
    ST:             FloatArray
    S1_conf:        FloatArray
    ST_conf:        FloatArray
    dominant_param: str
    n_evaluations:  int

    def summary(self) -> str:
        """
        Return a human-readable table of S1 and ST for state variable 0 (T1).
        Sorted by S1 descending.
        """
        order = np.argsort(self.S1[:, 0])[::-1]

        lines = [
            "=" * 72,
            "  ProbOS Sobol Sensitivity Analysis -- State variable: T1",
            f"  Total model evaluations: {self.n_evaluations}",
            "=" * 72,
            f"  {'Parameter':<16} {'S1':>8} {'S1 ±':>8} "
            f"{'ST':>8} {'ST ±':>8}  {'S1<=ST?':>8}",
            "-" * 72,
        ]
        for i in order:
            s1    = self.S1[i, 0]
            s1c   = self.S1_conf[i, 0]
            st    = self.ST[i, 0]
            stc   = self.ST_conf[i, 0]
            check = "YES" if st >= s1 - 1e-6 else "NO "
            lines.append(
                f"  {self.param_names[i]:<16} {s1:>8.4f} {s1c:>8.4f} "
                f"{st:>8.4f} {stc:>8.4f}  {check:>8}"
            )
        lines.append("=" * 72)
        lines.append(f"  Dominant parameter for T1: {self.dominant_param}")
        lines.append("=" * 72)
        return "\n".join(lines)


class SobolSensitivity:
    """
    Computes first-order (S1) and total-effect (ST) Sobol indices
    for a Model ABC using SALib Saltelli sampling.

    Parameters
    ----------
    model : Model
        Any class implementing the Model ABC.
    priors : list[Distribution]
        One Distribution per model parameter. Used for bounds.
    N_saltelli : int
        Base sample size (must be power of 2). Default 1024.
        Total evaluations = N_saltelli * (2 * param_dim + 2).
    n_steps : int
        Number of forward steps per evaluation. Default 5 for speed.
        Increase for publication results.
    dt : float
        Time step in seconds. Default 1.0.
    seed : int
        Random seed. Default 42.

    Raises
    ------
    ValueError
        If N_saltelli is not a power of 2, or < 64.
    """

    def __init__(
        self,
        model: Model,
        priors: Sequence[Distribution],
        N_saltelli: int = 1024,
        n_steps: int = 5,
        dt: float = 1.0,
        seed: int = 42,
    ) -> None:
        if N_saltelli < 64:
            raise ValueError(f"N_saltelli must be >= 64, got {N_saltelli}")
        if N_saltelli & (N_saltelli - 1) != 0:
            raise ValueError(
                f"N_saltelli must be a power of 2, got {N_saltelli}"
            )
        if len(priors) != model.param_dim:
            raise ValueError(
                f"len(priors)={len(priors)} != model.param_dim={model.param_dim}"
            )

        self._model      = model
        self._priors     = priors
        self._N_saltelli = N_saltelli
        self._n_steps    = n_steps
        self._dt         = dt
        self._seed       = seed
        # DP: cache SALib problem dict once at init, not per run()
        self._problem: dict[str, object] = self._build_problem()

    @property
    def N_saltelli(self) -> int:
        return self._N_saltelli

    def _build_problem(self) -> dict[str, object]:
        """Build SALib problem dict from prior supports. Cached at __init__."""
        pd     = self._model.param_dim
        pnames = self._model.param_names()
        bounds: list[list[float]] = []
        for prior in self._priors:
            lo, hi = prior.support()
            if not np.isfinite(lo):
                lo = float(prior.mean() - 4.0 * float(np.sqrt(prior.variance())))
            if not np.isfinite(hi):
                hi = float(prior.mean() + 4.0 * float(np.sqrt(prior.variance())))
            bounds.append([lo, hi])
        return {
            "num_vars": pd,
            "names":    pnames,
            "bounds":   bounds,
        }

    def run(self) -> SobolResult:
        """
        Run Saltelli sampling, evaluate the model, compute Sobol indices.

        Algorithm
        ---------
        1. Build SALib problem dict from prior supports.
        2. Generate Saltelli sample matrix: shape (N*(2d+2), d).
        3. Tile initial_state to shape (N_total, state_dim).
        4. Run forward_batch for n_steps -- one call per step, all samples at once.
        5. Extract final state values: shape (N_total, state_dim).
        6. For each state variable k: call SALib sobol.analyze on Y[:, k].
        7. Assemble SobolResult.
        """
        # Use cached problem dict (built once at __init__)
        problem = self._problem
        pd      = self._model.param_dim
        sd      = self._model.state_dim
        pnames  = self._model.param_names()

        # Step 2: Saltelli sample -- shape (N_saltelli*(2*pd+2), pd)
        #
        # BUG FIX: np.random.seed() does NOT control
        # saltelli.sample()'s output -- confirmed via direct
        # testing. saltelli.sample() takes its own explicit seed=
        # keyword argument, which must be passed directly -- the
        # same bug class already fixed for other stochastic models
        # (drawing from GLOBAL np.random state is not reproducible
        # via an external seed parameter alone).
        param_matrix = saltelli.sample(
            problem,
            self._N_saltelli,
            calc_second_order=False,
            seed=self._seed,
        )
        n_total = param_matrix.shape[0]

        # Step 3: tile initial state -- shape (n_total, state_dim)
        state = np.tile(
            self._model.initial_state(),
            (n_total, 1),
        ).astype(np.float64)

        # Validate shapes once, before the expensive n_steps loop --
        # same discipline as MonteCarloEngine.run() and
        # ParticleFilter.__init__(). Catches a malformed Model
        # subclass immediately with a clear ValueError.
        self._model.validate_params(param_matrix.astype(np.float64))
        self._model.validate_state(state)

        # Step 4: run forward_batch for n_steps
        # All n_total samples advanced simultaneously -- no Python for-loops
        for _ in range(self._n_steps):
            state = self._model.forward_batch(
                state, param_matrix.astype(np.float64), self._dt
            )

        # Step 5: final state -- shape (n_total, state_dim)
        Y = state   # shape (n_total, state_dim)

        # Step 6: compute Sobol indices per state variable
        S1      = np.zeros((pd, sd), dtype=np.float64)
        ST      = np.zeros((pd, sd), dtype=np.float64)
        S1_conf = np.zeros((pd, sd), dtype=np.float64)
        ST_conf = np.zeros((pd, sd), dtype=np.float64)

        for k in range(sd):
            si = sobol_analyze.analyze(
                problem,
                Y[:, k],
                calc_second_order=False,
                print_to_console=False,
            )
            S1[:, k]      = si["S1"]
            ST[:, k]      = si["ST"]
            S1_conf[:, k] = si["S1_conf"]
            ST_conf[:, k] = si["ST_conf"]

        # Step 7: dominant parameter for T1 (state index 0)
        dominant_idx   = int(np.argmax(S1[:, 0]))
        dominant_param = pnames[dominant_idx]

        return SobolResult(
            param_names=pnames,
            S1=S1,
            ST=ST,
            S1_conf=S1_conf,
            ST_conf=ST_conf,
            dominant_param=dominant_param,
            n_evaluations=n_total,
        )
