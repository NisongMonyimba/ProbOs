"""
python/src/gpu_monte_carlo.py

Month 3 Week 10: GPU-accelerated Monte Carlo engine, following
MonteCarloEngine's exact existing architecture but running the
forward-simulation loop with CuPy arrays on the GPU.

WHY A SEPARATE CLASS RATHER THAN A FLAG ON MonteCarloEngine
-----------------------------------------------------------------
Keeping GPUMonteCarloEngine separate (rather than adding an
on_gpu=True flag to the existing MonteCarloEngine) means:
  - MonteCarloEngine's own code and tests are completely unaffected
    -- zero risk of regressing the CPU path while adding GPU support
  - The class can raise a clear ImportError at construction time if
    CuPy is not installed, rather than MonteCarloEngine silently
    gaining a parameter that only works in some environments
  - Matches the same "prove the concept before generalising" pattern
    already used for the REST API model registry (Week 9): this
    initial version targets BatteryModel2Cell specifically, since
    that is the only model with GPU dispatch implemented so far
    (Week 10 Day 2). Extending to other models is a natural,
    low-risk follow-up.

CRITICAL LIMITATION, STATED HONESTLY
-----------------------------------------------------------------
This class requires an NVIDIA GPU with CuPy installed
(pip install "cupy-cuda12x[ctk]") -- it will raise ImportError at
construction time otherwise. It CANNOT be validated in CI (GitHub
Actions runners have no GPU access) -- see
python/tests/test_gpu_monte_carlo.py for how tests skip gracefully
in that environment. Only genuine local-machine testing validates
this code path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from python.src.distributions import Distribution
from python.src.state import Model

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

FloatArray = NDArray[np.float64]


@dataclass
class GPUMCResult:
    """
    Container for GPUMonteCarloEngine output. Mirrors MCResult's
    field set exactly, so downstream code (e.g. percentile/convergence
    consumers) does not need to know which engine produced a result.

    All array fields are returned as plain NumPy arrays (copied back
    from the GPU) -- callers never need to import CuPy themselves to
    use a GPUMCResult.
    """

    trajectories: FloatArray
    params_used:  FloatArray
    percentiles:  FloatArray
    convergence:  FloatArray
    n_particles:  int
    n_steps:      int
    dt:           float
    wall_time_ms: float


class GPUMonteCarloEngine:
    """
    GPU-accelerated Monte Carlo engine for BatteryModel2Cell,
    following MonteCarloEngine's exact constructor signature and
    algorithm structure (see python/src/monte_carlo.py), but running
    the particle-propagation loop on the GPU via CuPy.

    Raises
    ------
    ImportError
        If CuPy is not installed/available in this environment.
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
        if not CUPY_AVAILABLE:
            raise ImportError(
                "GPUMonteCarloEngine requires CuPy, which is not "
                "installed in this environment. Install it with: "
                'pip install "cupy-cuda12x[ctk]" '
                "(requires an NVIDIA GPU with CUDA 12.x-compatible "
                "drivers). See README.md 'GPU Setup' for details."
            )
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

    def run(self) -> GPUMCResult:
        """
        Identical algorithm to MonteCarloEngine.run() (see that
        docstring for the full 6-step description), except:
          - params and state are transferred to the GPU immediately
            after being drawn/tiled on the CPU (prior sampling itself
            stays on CPU -- Distribution.sample() is not GPU-aware,
            and prior sampling is a tiny fraction of total cost
            compared to the n_steps forward-simulation loop)
          - the forward_batch() loop runs entirely on the GPU via
            BatteryModel2Cell's CuPy dispatch (Week 10 Day 2)
          - results are transferred back to the CPU (as plain NumPy
            arrays) only once, at the end, not per-step
        """
        start_time = time.perf_counter()

        rng = np.random.default_rng(self._seed)
        sd  = self._model.state_dim
        pd  = self._model.param_dim

        params_cpu = np.empty((self._N, pd), dtype=np.float64)
        for j, prior in enumerate(self._priors):
            params_cpu[:, j] = prior.sample(self._N, rng=rng)

        state_cpu = np.tile(
            self._model.initial_state(), (self._N, 1)
        ).astype(np.float64)

        self._model.validate_params(params_cpu)
        self._model.validate_state(state_cpu)

        params_gpu = cp.asarray(params_cpu)
        state_gpu = cp.asarray(state_cpu)

        trajectories_gpu = cp.empty(
            (self._N, self._n_steps + 1, sd), dtype=cp.float64
        )
        trajectories_gpu[:, 0, :] = state_gpu

        for t in range(1, self._n_steps + 1):
            state_gpu = self._model.forward_batch(state_gpu, params_gpu, self._dt)
            trajectories_gpu[:, t, :] = state_gpu

        trajectories = cp.asnumpy(trajectories_gpu)
        params_used = params_cpu

        percentiles = np.percentile(trajectories, [5, 50, 95], axis=0)
        convergence = np.std(trajectories[:, -1, :], axis=0) / np.sqrt(self._N)

        wall_time_ms = (time.perf_counter() - start_time) * 1000.0

        return GPUMCResult(
            trajectories=trajectories,
            params_used=params_used,
            percentiles=percentiles,
            convergence=convergence,
            n_particles=self._N,
            n_steps=self._n_steps,
            dt=self._dt,
            wall_time_ms=wall_time_ms,
        )
