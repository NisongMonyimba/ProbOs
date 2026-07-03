"""
python/server/main.py

ProbOS FastAPI service layer -- Month 2 Week 7.

Exposes the kernel over HTTP:
    GET  /health       -- liveness check
    POST /simulate      -- MonteCarloEngine
    POST /sensitivity    -- SobolSensitivity
    POST /filter         -- ParticleFilter

MODEL REGISTRY
----------------
Only "battery" (BatteryModel2Cell + build_battery_priors) is wired up
in Week 7. This is a deliberate scope decision: the goal is proving the
service layer works correctly end-to-end against one well-validated
model, not registering all five existing models on day one. Extending
_MODEL_REGISTRY to other Week 4 models (option pricer, ED queue,
clinical trial) is a natural, low-risk follow-up once this pattern is
proven.

Run locally:
    uvicorn python.server.main:app --reload
    open http://localhost:8000/docs
"""

from __future__ import annotations


from fastapi import FastAPI, HTTPException

from python.server.schemas import (
    FilterRequest,
    FilterResponse,
    HealthResponse,
    SensitivityRequest,
    SensitivityResponse,
    SimulateRequest,
    SimulateResponse,
)
from python.src.battery_model import BatteryModel2Cell
from python.src.monte_carlo import MonteCarloEngine
from python.src.parameter_priors import build_battery_priors
from python.src.particle_filter import ParticleFilter
from python.src.distributions import Distribution
from python.src.sensitivity import SobolSensitivity
from python.src.state import FloatArray, Model
import numpy as np

__version__ = "0.2.0"

app = FastAPI(
    title="ProbOS API",
    description="Probabilistic execution runtime -- Month 2 Week 7 "
                 "service layer.",
    version=__version__,
)


def _build_model_and_priors(
    model_name: str,
) -> tuple[Model, list[Distribution]]:
    """
    Resolve a model_name string to a (Model instance, priors list) pair.

    Raises
    ------
    HTTPException(404)
        If model_name is not registered.
    """
    if model_name == "battery":
        return BatteryModel2Cell(), build_battery_priors()
    raise HTTPException(
        status_code=404,
        detail=f"Unknown model_name '{model_name}'. "
               f"Registered models: ['battery']",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check -- confirms the service is up and importable."""
    return HealthResponse(status="ok", version=__version__)


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest) -> SimulateResponse:
    """
    Run MonteCarloEngine and return percentiles + convergence.

    Directly mirrors:
        engine = MonteCarloEngine(model, priors, N=req.N,
                                   n_steps=req.n_steps, dt=req.dt,
                                   seed=req.seed)
        result = engine.run()
    """
    model, priors = _build_model_and_priors(req.model_name)

    engine = MonteCarloEngine(
        model, priors,
        N=req.N, n_steps=req.n_steps, dt=req.dt, seed=req.seed,
    )
    result = engine.run()

    return SimulateResponse(
        model_name=req.model_name,
        n_particles=result.n_particles,
        n_steps=result.n_steps,
        state_dim=model.state_dim,
        percentiles=result.percentiles.tolist(),
        convergence=result.convergence.tolist(),
        wall_time_ms=0.0,  # MCResult (Python engine) does not track
                            # wall time -- only probos_cpp.MCResult does.
                            # Left at 0.0 here rather than faked; a
                            # future week can add timing to the Python
                            # engine if needed.
    )


@app.post("/sensitivity", response_model=SensitivityResponse)
async def sensitivity(req: SensitivityRequest) -> SensitivityResponse:
    """
    Run SobolSensitivity and return S1/ST indices.

    Directly mirrors:
        s = SobolSensitivity(model, priors, N_saltelli=req.N_saltelli,
                              n_steps=req.n_steps, seed=req.seed)
        result = s.run()
    """
    model, priors = _build_model_and_priors(req.model_name)

    try:
        s = SobolSensitivity(
            model, priors,
            N_saltelli=req.N_saltelli, n_steps=req.n_steps,
            seed=req.seed,
        )
    except ValueError as e:
        # SobolSensitivity itself validates N_saltelli is a power of 2
        # -- surface that as a 422, not a 500.
        raise HTTPException(status_code=422, detail=str(e)) from e

    result = s.run()

    return SensitivityResponse(
        model_name=req.model_name,
        param_names=result.param_names,
        S1=result.S1.tolist(),
        ST=result.ST.tolist(),
        dominant_param=result.dominant_param,
        n_evaluations=result.n_evaluations,
    )


@app.post("/filter", response_model=FilterResponse)
async def filter_endpoint(req: FilterRequest) -> FilterResponse:
    """
    Run ParticleFilter against posted observation data.

    Uses the same Gaussian observation-noise log-likelihood pattern
    established in python/tests/test_particle_filter.py and
    python/tests/test_distributions_properties.py: noise on the
    model's FIRST state variable only (e.g. T1 for BatteryModel2Cell).
    """
    model, priors = _build_model_and_priors(req.model_name)

    pf = ParticleFilter(
        model, priors, N=req.N, dt=req.dt, seed=req.seed,
    )

    sigma_obs = req.sigma_obs

    def loglik(state: FloatArray, obs: FloatArray) -> FloatArray:
        x = state[:, 0]
        result: FloatArray = -0.5 * ((x - obs[0]) / sigma_obs) ** 2
        return result

    observations = np.array(req.observations).reshape(-1, 1)
    result = pf.run(observations, loglik)

    return FilterResponse(
        model_name=req.model_name,
        n_particles=result.n_particles,
        n_steps=result.n_steps,
        means=result.means.tolist(),
        stds=result.stds.tolist(),
        ess_history=result.ess_history.tolist(),
        n_resamples=result.n_resamples,
    )
