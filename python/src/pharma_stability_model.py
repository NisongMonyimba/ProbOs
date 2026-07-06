"""
python/src/pharma_stability_model.py

Month 3 Week 11: pharmaceutical stability model using Avrami kinetics,
the direct pharma analogue of BatteryModel2Cell's Arrhenius
thermal-runaway model. Reuses the same architectural pattern (Model
ABC, xp dispatch for CPU/GPU, vectorised-over-N forward_batch()) but
with a different reaction kinetics law, since the real, best-validated
pharmaceutical stability dataset found for this model follows Avrami
kinetics, not the simple first-order depletion used by the battery
model's SEI/anode/cathode reactions.

VALIDATION SOURCE
------------------
Gonzalez-Gonzalez, O.; Ballesteros, M.P.; Torrado, J.J.; Serrano, D.R.
"Application of Accelerated Predictive Stability Studies in
Extemporaneously Compounded Formulations of Chlorhexidine to Assess
the Shelf Life." Molecules 2023, 28(23), 7925.
DOI: 10.3390/molecules28237925

Real, reported values used:
  - Ea = 18.52 +/- 2.61 kcal/mol for the DCHX (chlorhexidine solution)
    formulation's Avrami-kinetics fit (R^2 = 0.941), independently
    cross-checked in the same paper against an independent 1987 PhD
    thesis reporting Ea in the range 16.5-22.9 kcal/mol for related
    chlorhexidine degradation -- our value falls inside that range.
  - Real long-term experimental degradation at 365 days: 3.1% (5C),
    17.4% (25C), 25.9% (30C) -- used as the Day 3 validation target.

HONEST METHODOLOGICAL LIMITATION
------------------------------------
Unlike BatteryModel2Cell's validation against Kim (2007) (a true
forward replication using literature parameter values against an
independent experimental curve), this model's pre-exponential factor
A and Avrami exponent n are FIT to reproduce the three degradation
percentages above, since the source paper does not report A and n
numerically in isolation. Only Ea is held fixed at its independently-
cited value. This is a real but WEAKER validation standard than
Kim (2007) -- a parameter fit to a single paper's own summary results,
not an independent replication -- and must not be presented as
equivalent rigor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

from python.src.state import Model

FloatArray = NDArray[np.float64]

# Universal gas constant [kcal / (mol * K)] -- kcal, not J, to match
# this paper's reported Ea units directly without a conversion step
# that could introduce a transcription error.
R_GAS_KCAL: float = 1.987204e-3


class PharmaStabilityModel(Model):
    """
    Pharmaceutical stability model using Avrami degradation kinetics:

        potency(t) = exp(-k * t^n)
        k = A * exp(-Ea / (R * T))

    where t is elapsed time (days), T is storage temperature (Kelvin),
    and A, n, Ea are formulation-specific kinetic parameters.

    Because Avrami kinetics has an exact closed-form solution (unlike
    BatteryModel2Cell's coupled multi-reaction system, which does
    not), forward_batch() evaluates the closed form directly at each
    new elapsed time rather than using an approximate Euler
    integration step -- more accurate, and avoids introducing
    artificial numerical error into an otherwise exact model.

    State vector (state_dim = 2):
        state[:, POTENCY]      -- fraction of original potency remaining, [0, 1]
        state[:, ELAPSED_TIME] -- elapsed time since t=0, in days

    Parameter vector (param_dim = 4):
        params[:, P_EA]      -- activation energy [kcal/mol]
        params[:, P_A]       -- Avrami pre-exponential factor
        params[:, P_N]       -- Avrami exponent (shape parameter)
        params[:, P_T]       -- storage temperature [Kelvin]
    """

    POTENCY = 0
    ELAPSED_TIME = 1

    P_EA = 0
    P_A = 1
    P_N = 2
    P_T = 3

    def __init__(self, initial_potency: float = 1.0) -> None:
        self._initial_potency = initial_potency

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def param_dim(self) -> int:
        return 4

    def param_names(self) -> list[str]:
        return ["Ea", "A", "n", "T"]

    def initial_state(self) -> FloatArray:
        return np.array([self._initial_potency, 0.0], dtype=np.float64)

    def validate_params(self, params: FloatArray) -> None:
        if params.shape[1] != self.param_dim:
            raise ValueError(
                f"params must have shape (N, {self.param_dim}), "
                f"got {params.shape}"
            )
        if np.any(params[:, self.P_EA] <= 0):
            raise ValueError("Ea must be positive")
        if np.any(params[:, self.P_A] <= 0):
            raise ValueError("A (pre-exponential factor) must be positive")
        if np.any(params[:, self.P_N] <= 0):
            raise ValueError("n (Avrami exponent) must be positive")
        if np.any(params[:, self.P_T] <= 0):
            raise ValueError("T (storage temperature) must be positive (Kelvin)")

    def validate_state(self, state: FloatArray) -> None:
        if state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape (N, {self.state_dim}), "
                f"got {state.shape}"
            )

    @staticmethod
    def nominal_parameters() -> FloatArray:
        """
        Validated parameter set for DCHX (chlorhexidine solution),
        Month 3 Week 11 Day 3:
          Ea = 18.52 kcal/mol -- independently cited from
               Gonzalez-Gonzalez et al. (2023), cross-checked
               against an independent 1987 PhD thesis range
               (16.5-22.9 kcal/mol).
          A  = 1.89e10 -- fit to the paper's 3 real reported
               365-day degradation percentages (5C/25C/30C),
               with n fixed at 1.0.
          n  = 1.0 -- EXPLICIT SIMPLIFYING ASSUMPTION, not a
               recovered value. The paper's own n was fit
               against multi-timepoint accelerated (50-80C)
               data not available to us numerically; fitting
               n directly against only 3 single-timepoint
               points is fundamentally underdetermined
               (confirmed directly: an unconstrained 2-parameter
               fit converged to a physically absurd
               A=1.5e255, n=-94.6). Fixing n=1 and fitting only
               A achieves an IDENTICAL residual error
               (max 1.18 percentage points) with a physically
               plausible A.
          T  = 298.15 K (25C) -- a representative long-term
               storage condition; vary this per-scenario.
        """
        return np.array([18.52, 1.89e10, 1.0, 298.15], dtype=np.float64)

    def forward_batch(
        self,
        state: FloatArray,
        params: FloatArray,
        dt: float,
    ) -> FloatArray:
        """
        Advance all N particles by dt (days), evaluating the exact
        Avrami closed-form solution at the new elapsed time.
        """
        xp = cp.get_array_module(state) if _CUPY_AVAILABLE else np

        elapsed_time = state[:, self.ELAPSED_TIME]
        new_elapsed_time = elapsed_time + dt

        Ea = params[:, self.P_EA]
        A = params[:, self.P_A]
        n = params[:, self.P_N]
        T = params[:, self.P_T]

        k = A * xp.exp(-Ea / (R_GAS_KCAL * T))
        new_potency = xp.exp(-k * xp.power(new_elapsed_time, n))
        new_potency = xp.clip(new_potency, 0.0, 1.0)

        new_state: FloatArray = xp.column_stack([new_potency, new_elapsed_time])
        return new_state
