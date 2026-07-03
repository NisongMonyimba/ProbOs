"""
MonteCarloEngine: vectorised Monte Carlo simulation for ProbOS.

Design principles:
    - No Python for-loops over particles. All N particles are advanced
      simultaneously via model.forward_batch(state, params, dt).
    - P05/P50/P95 computed with numpy.percentile along the particle axis.
    - Convergence certificate: sigma/sqrt(N) per state variable at the
      final timestep, giving the 95% CI half-width on the mean estimate.

Computational cost:
    - Memory: O(N * n_steps * state_dim) float64
      Example: N=5000, n_steps=300, state_dim=8 -> ~96 MB
    - Time: O(N * n_steps) forward_batch calls, all vectorised over N.

Saltelli note (for SobolSensitivity in sensitivity.py):
    Sobol sampling requires N_saltelli * (2 * param_dim + 2) evaluations.
    With N_saltelli=1024, param_dim=15: 1024 * 32 = 32,768 forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from python.src.state import Model
from python.src.distributions import Distribution

FloatArray = NDArray[np.float64]


@dataclass
class MCResult:
    """
    Container for Monte Carlo simulation output.

    Attributes
    ----------
    trajectories : shape (N, n_steps+1, state_dim)
        Full state trajectory for every particle.
        trajectories[i, t, k] = state variable k of particle i at step t.
    params_used : shape (N, param_dim)
        Parameter set drawn for each particle.
    percentiles : shape (3, n_steps+1, state_dim)
        Percentile trajectories. Axis 0: [P05, P50, P95].
        percentiles[0] = P05, percentiles[1] = P50, percentiles[2] = P95.
    convergence : shape (state_dim,)
        sigma/sqrt(N) at the final timestep per state variable.
        Interpretation: 95% CI half-width on the mean trajectory estimate.
    n_particles : int
        Number of particles used.
    n_steps : int
        Number of forward steps taken.
    dt : float
        Time step in seconds.
    """

    trajectories: FloatArray
    params_used:  FloatArray
    percentiles:  FloatArray
    convergence:  FloatArray
    n_particles:  int
    n_steps:      int
    dt:           float


class MonteCarloEngine:
    """
    Runs N particles through a Model ABC using parameter sets drawn
    from a list of Distribution ABCs.

    Parameters
    ----------
    model : Model
        Any class implementing the Model ABC (e.g. BatteryModel2Cell).
    priors : list[Distribution]
        One Distribution per model parameter (len == model.param_dim).
    N : int
        Number of particles. Default 5000.
    dt : float
        Time step in seconds. Default 1.0.
    n_steps : int
        Number of forward Euler steps. Default 300.
    seed : int
        Random seed for full reproducibility. Default 42.

    Raises
    ------
    ValueError
        If len(priors) != model.param_dim, or N < 1, or dt <= 0.
    """

    def __init__(
        self,
        model: Model,
        priors: list[Distribution],
        N: int = 5000,
        dt: float = 1.0,
        n_steps: int = 300,
        seed: int = 42,
    ) -> None:
        if len(priors) != model.param_dim:
            raise ValueError(
                f"len(priors)={len(priors)} != model.param_dim={model.param_dim}"
            )
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        self._model   = model
        self._priors  = priors
        self._N       = N
        self._dt      = dt
        self._n_steps = n_steps
        self._seed    = seed

    @property
    def N(self) -> int:
        return self._N

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def seed(self) -> int:
        return self._seed

    def run(self) -> MCResult:
        """
        Draw N parameter sets, run forward_batch over n_steps, return MCResult.

        Algorithm
        ---------
        1. Draw params: shape (N, param_dim) -- one row per particle.
        2. Tile initial_state to shape (N, state_dim).
        3. Store state at t=0 in trajectories[:, 0, :].
        4. For t in 1..n_steps: advance all N particles in one
           forward_batch call. No Python for-loops over particles.
        5. Compute P05/P50/P95 along the particle axis (axis=0).
        6. Compute convergence = std(final_state, axis=0) / sqrt(N).
        """
        rng = np.random.default_rng(self._seed)
        sd  = self._model.state_dim
        pd  = self._model.param_dim

        # Step 1: draw params -- shape (N, param_dim)
        params = np.empty((self._N, pd), dtype=np.float64)
        for j, prior in enumerate(self._priors):
            params[:, j] = prior.sample(self._N, rng=rng)

        # Step 2: tile initial state -- shape (N, state_dim)
        state = np.tile(
            self._model.initial_state(),
            (self._N, 1),
        ).astype(np.float64)

        # Validate shapes once, before the expensive n_steps loop.
        # Catches a malformed Model subclass immediately with a
        # clear ValueError, rather than a confusing broadcasting
        # error deep inside forward_batch() on step 1.
        self._model.validate_params(params)
        self._model.validate_state(state)

        # Step 3: allocate trajectory array -- shape (N, n_steps+1, state_dim)
        trajectories = np.empty(
            (self._N, self._n_steps + 1, sd), dtype=np.float64
        )
        trajectories[:, 0, :] = state

        # Step 4: advance all N particles simultaneously
        for t in range(1, self._n_steps + 1):
            state = self._model.forward_batch(state, params, self._dt)
            trajectories[:, t, :] = state

        # Step 5: percentiles -- shape (3, n_steps+1, state_dim)
        percentiles = np.percentile(
            trajectories, [5.0, 50.0, 95.0], axis=0
        )

        # Step 6: convergence = sigma/sqrt(N) at final timestep
        final_state = trajectories[:, -1, :]
        convergence = np.std(final_state, axis=0, ddof=1) / np.sqrt(self._N)

        return MCResult(
            trajectories=trajectories,
            params_used=params,
            percentiles=percentiles,
            convergence=convergence,
            n_particles=self._N,
            n_steps=self._n_steps,
            dt=self._dt,
        )

    def convergence_certificate(self) -> str:
        """
        Run the engine and return a human-readable convergence table.
        One row per state variable: sigma/sqrt(N) and 95% CI half-width.
        """
        result = self.run()
        state_names = [f"state_{k}" for k in range(self._model.state_dim)]
        lines = [
            "=" * 60,
            "  ProbOS Monte Carlo Convergence Certificate",
            f"  N={self._N}  dt={self._dt}s  n_steps={self._n_steps}",
            "=" * 60,
            f"  {'State var':<16} {'sigma/sqrt(N)':>16} {'95% CI +/-':>12}",
            "-" * 60,
        ]
        for k, name in enumerate(state_names):
            sigma_n = result.convergence[k]
            ci95    = 1.96 * sigma_n
            lines.append(f"  {name:<16} {sigma_n:>16.6f} {ci95:>12.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)
