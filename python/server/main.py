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
All four existing models are registered: "battery"
(BatteryModel2Cell), "option_pricer" (OptionPricerModel),
"ed_queue" (EDQueueModel), "clinical_trial" (ClinicalTrialModel).
Week 7 deliberately started with just "battery" to prove the
service layer worked end-to-end against one well-validated model
before extending; Week 9 completes that extension. All three
additional models use their own constructor defaults (each has
sensible defaults -- confirmed by direct inspection before this
change), so no new request-schema fields were needed.

Run locally:
    uvicorn python.server.main:app --reload
    open http://localhost:8000/docs
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException

from python.examples.week4_clinical_trial import (
    ClinicalTrialModel,
    build_clinical_trial_priors,
)
from python.examples.week4_ed_queue import (
    EDQueueModel,
    build_ed_queue_priors,
)
from python.examples.week4_option_pricer import (
    OptionPricerModel,
    build_option_priors,
)
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
from python.src.distributions import Distribution
from python.src.monte_carlo import MonteCarloEngine
from python.src.parameter_priors import build_battery_priors
from python.src.particle_filter import ParticleFilter
from python.src.sensitivity import SobolSensitivity
from python.src.state import FloatArray, Model

__version__ = "0.2.0"

app = FastAPI(
    title="ProbOS API",
    description="Probabilistic execution runtime -- Month 2 Week 7 "
                 "service layer.",
    version=__version__,
)


_REGISTERED_MODEL_NAMES = [
    "battery", "option_pricer", "ed_queue", "clinical_trial",
]


def _build_model_and_priors(
    model_name: str,
    seed: int = 42,
) -> tuple[Model, list[Distribution]]:
    """
    Resolve a model_name string to a (Model instance, priors list) pair.

    seed is passed through to the three genuinely stochastic
    models (OptionPricerModel, EDQueueModel, ClinicalTrialModel),
    which each carry their own per-instance seeded Generator
    (Month 3 Week 9 fix) -- this makes them genuinely
    reproducible via the API, matching BatteryModel2Cell's
    existing reproducibility (which comes from
    MonteCarloEngine/ParticleFilter/SobolSensitivity seeding
    prior sampling, since BatteryModel2Cell.forward_batch()
    itself has no internal randomness).

    Raises
    ------
    HTTPException(404)
        If model_name is not registered.
    """
    if model_name == "battery":
        return BatteryModel2Cell(), build_battery_priors()
    if model_name == "option_pricer":
        return OptionPricerModel(seed=seed), build_option_priors()
    if model_name == "ed_queue":
        return EDQueueModel(seed=seed), build_ed_queue_priors()
    if model_name == "clinical_trial":
        return ClinicalTrialModel(seed=seed), build_clinical_trial_priors()
    raise HTTPException(
        status_code=404,
        detail=f"Unknown model_name '{model_name}'. "
               f"Registered models: {_REGISTERED_MODEL_NAMES}",
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
    model, priors = _build_model_and_priors(req.model_name, req.seed)

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
        wall_time_ms=result.wall_time_ms,
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
    model, priors = _build_model_and_priors(req.model_name, req.seed)

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

    NOT SUPPORTED for model_name='clinical_trial': that model's
    first state variable (n_treatment) is an integer enrollment
    count, not a continuous quantity with genuine observation
    noise -- applying this endpoint's generic Gaussian-noise
    likelihood to it produces a mechanically valid but
    MEANINGLESS result (confirmed by direct investigation during
    Month 3 Week 9). Week 8's week8_clinical_trial_filter.py
    demonstrates the CORRECT design for this model: deterministic
    state transitions against known real trial data, with an
    informative Bernoulli likelihood scored via particle
    parameters -- a fundamentally different approach from what
    this generic endpoint offers, not yet exposed over HTTP.
    """
    if req.model_name == "clinical_trial":
        raise HTTPException(
            status_code=422,
            detail=(
                "model_name='clinical_trial' is not supported by "
                "/filter: this endpoint's generic Gaussian-"
                "observation-noise likelihood does not apply meaningfully "
                "to this model's count-based state (see "
                "python/examples/week8_clinical_trial_filter.py "
                "for the correct, purpose-built sequential "
                "filtering design for clinical trials, not yet "
                "exposed over HTTP)."
            ),
        )

    model, priors = _build_model_and_priors(req.model_name, req.seed)

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
