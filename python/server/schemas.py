"""
python/server/schemas.py

Pydantic request/response schemas for the ProbOS FastAPI service layer.

WHY PYDANTIC SCHEMAS MATTER HERE
-----------------------------------
Per docs/standards/quality_standards.md Section 8 (Security), any
endpoint accepting N/n_steps must validate against unbounded resource
consumption BEFORE the request reaches kernel code. A malicious or
mistaken client could otherwise request N=10_000_000, n_steps=1_000_000
and exhaust server memory/CPU. Pydantic's Field(le=..., ge=...)
constraints enforce sane upper bounds at the HTTP boundary, not deep
inside MonteCarloEngine.

These schemas deliberately mirror the constructor signatures of
Model/Distribution/MonteCarloEngine/SobolSensitivity/ParticleFilter
established in Month 1-2, so a JSON request maps directly onto the
existing kernel objects with minimal translation logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# /simulate
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    """
    Request body for POST /simulate.

    model_name is currently restricted to "battery" (BatteryModel2Cell)
    -- the only Model with a matching prior-builder function
    (build_battery_priors) wired into the server layer so far. Other
    Week 4 models (option pricer, ED queue, clinical trial) can be
    added the same way in a future week without changing this schema.
    """

    model_name: str = Field(
        default="battery",
        description="Which registered model to simulate.",
    )
    N: int = Field(
        default=1000, ge=1, le=100_000,
        description="Number of Monte Carlo particles. Bounded to "
                     "prevent resource exhaustion (quality_standards.md "
                     "Section 8).",
    )
    n_steps: int = Field(
        default=300, ge=1, le=10_000,
        description="Number of forward_batch() steps.",
    )
    dt: float = Field(default=1.0, gt=0.0, le=1000.0)
    seed: int = Field(default=42, ge=0)


class SimulateResponse(BaseModel):
    model_name:    str
    n_particles:   int
    n_steps:       int
    state_dim:     int
    percentiles:   list[list[list[float]]]  # shape [3, n_steps+1, state_dim]
    convergence:   list[float]              # shape [state_dim]
    wall_time_ms:  float


# ---------------------------------------------------------------------------
# /sensitivity
# ---------------------------------------------------------------------------

class SensitivityRequest(BaseModel):
    model_name: str = Field(default="battery")
    N_saltelli: int = Field(
        default=256, ge=64, le=4096,
        description="Saltelli base sample size. Must be a power of 2 "
                     "(enforced by SobolSensitivity itself). Upper "
                     "bound of 4096 keeps total evaluations "
                     "(N_saltelli * (param_dim+2)) bounded even for "
                     "models with many parameters.",
    )
    n_steps: int = Field(default=3, ge=1, le=1000)
    seed: int = Field(default=42, ge=0)


class SensitivityResponse(BaseModel):
    model_name:      str
    param_names:     list[str]
    S1:              list[list[float]]   # shape [param_dim, state_dim]
    ST:              list[list[float]]   # shape [param_dim, state_dim]
    dominant_param:  str
    n_evaluations:   int


# ---------------------------------------------------------------------------
# /filter
# ---------------------------------------------------------------------------

class FilterRequest(BaseModel):
    model_name:  str = Field(default="battery")
    N:           int = Field(default=1000, ge=1, le=50_000)
    dt:          float = Field(default=1.0, gt=0.0, le=1000.0)
    seed:        int = Field(default=42, ge=0)
    sigma_obs:   float = Field(
        default=5.0, gt=0.0,
        description="Assumed Gaussian observation noise std on the "
                     "first state variable (matches the pattern used "
                     "in test_particle_filter.py and "
                     "test_distributions_properties.py).",
    )
    observations: list[float] = Field(
        min_length=1, max_length=10_000,
        description="Sequence of scalar observations, one per "
                     "timestep, on the model's first state variable.",
    )


class FilterResponse(BaseModel):
    model_name:   str
    n_particles:  int
    n_steps:      int
    means:        list[list[float]]  # shape [T, state_dim]
    stds:         list[list[float]]  # shape [T, state_dim]
    ess_history:  list[float]        # shape [T]
    n_resamples:  int
