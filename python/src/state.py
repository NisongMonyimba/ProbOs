"""
================================================================================
probos.state  --  the Model abstract base class
================================================================================

WHAT IS THIS FILE?
------------------
Every domain model in ProbOS must inherit from Model and implement
the four abstract methods below.  The Monte Carlo engine (Week 3)
depends ONLY on this interface -- it never knows which specific model
it is running.

EXAMPLES OF MODELS BUILT ON THIS ABC:
--------------------------------------
  Week 2  : BatteryModel2Cell  -- 8-state Arrhenius thermal ODE
  Month 2 : PharmacokineticsModel -- 4-state PK/PD ODE
  Month 3 : HestonModel -- 3-state stochastic volatility SDE
  Month 4 : SIRModel -- 3-state epidemic ODE

THE CONTRACT:
-------------
Every Model must provide:
  state_dim       int       how many state variables
  param_dim       int       how many uncertain parameters
  param_names()   list[str] human-readable name for each parameter
  initial_state() array     starting state for ONE particle, shape (state_dim,)
  forward_batch() array     advance N particles by dt, returns (N, state_dim)
================================================================================
"""

from __future__ import annotations

# ABC = the base class to inherit from
# abstractmethod = decorator that marks a method as "must be overridden"
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

# FloatArray: a numpy array of 64-bit floats
# Using a type alias means we only need to change precision in one place
FloatArray = NDArray[np.float64]


class Model(ABC):
    """
    Abstract base class for all domain models in ProbOS.

    HOW TO CREATE A NEW MODEL
    --------------------------
    class MyModel(Model):

        @property
        def state_dim(self) -> int:
            return 3    # e.g. (S, I, R) for an epidemic model

        @property
        def param_dim(self) -> int:
            return 4    # e.g. (beta, gamma, mu, nu)

        def param_names(self) -> list[str]:
            return ["beta", "gamma", "mu", "nu"]

        def initial_state(self) -> FloatArray:
            return np.array([0.99, 0.01, 0.0])   # shape (state_dim,)

        def forward_batch(self, state, params, dt):
            # vectorised ODE step over N particles
            # state:  shape (N, 3)
            # params: shape (N, 4)
            # return: shape (N, 3)
            ...
    """

    # =========================================================================
    # ABSTRACT PROPERTIES
    # Every subclass MUST define these two integers.
    # If they do not, Python raises TypeError when you try to instantiate.
    # =========================================================================

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """
        Number of state variables.

        The STATE is the current condition of the system at one moment in time.

        Examples:
          BatteryModel2Cell        : 8  (T1, T2, c_SEI_1, c_SEI_2, ...)
          PharmacokineticsModel    : 4  (C_central, C_periph, E_eff, E_tox)
          SIRModel                 : 3  (S, I, R)
        """
        ...

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """
        Number of uncertain parameters.

        Each of the N Monte Carlo particles has its own parameter vector
        drawn from the prior distributions.  This is what makes the
        simulation probabilistic -- different particles get different
        parameter values and therefore produce different trajectories.

        Examples:
          BatteryModel2Cell     : 15 (activation energies, pre-exponentials, ...)
          PharmacokineticsModel : 10 (CL, V_c, k_a, EC50, ...)
        """
        ...

    # =========================================================================
    # ABSTRACT METHODS
    # Every subclass MUST implement all four of these.
    # =========================================================================

    @abstractmethod
    def param_names(self) -> list[str]:
        """
        Human-readable names for each parameter, in order.

        These names appear on:
          - Sobol sensitivity bar charts (Week 4)
          - Provenance audit trails (Month 3)
          - Debug output and error messages

        Returns
        -------
        list of str, length == self.param_dim

        Example:
          ["Ea_SEI", "A_SEI", "Ea_anode", "A_anode", ...]
        """
        ...

    @abstractmethod
    def initial_state(self) -> FloatArray:
        """
        Return the starting state vector for ONE particle.

        The Monte Carlo engine calls this once, then tiles it to N particles:
          states = np.tile(model.initial_state(), (N, 1))   # shape (N, state_dim)

        The initial state is the same for all particles -- the uncertainty
        comes from the parameters, not the initial conditions (in Week 2).
        In Month 2 we add uncertain initial conditions.

        Returns
        -------
        FloatArray of shape (state_dim,)
        """
        ...

    @abstractmethod
    def forward_batch(
        self,
        state: FloatArray,   # shape (N, state_dim)
        params: FloatArray,  # shape (N, param_dim)
        dt: float,           # time step in model-specific units
    ) -> FloatArray:
        """
        Advance ALL N particles by one time step of size dt.

        This is the core computation of ProbOS.  Every other component
        (sensitivity analysis, particle filter, provenance graph) wraps
        this single method.

        Parameters
        ----------
        state : FloatArray, shape (N, state_dim)
            state[i, :] is the full state of particle i.
            state[:, j] is state variable j across all particles.

        params : FloatArray, shape (N, param_dim)
            params[i, :] is the parameter vector of particle i.
            Drawn once from priors and fixed for the whole simulation.

        dt : float
            Time step size. Units depend on the model:
              BatteryModel2Cell     : seconds
              PharmacokineticsModel : hours
              SIRModel              : days

        Returns
        -------
        FloatArray of shape (N, state_dim)
            The new state of all N particles after dt.

        IMPLEMENTATION RULES:
        ---------------------
        1. USE NUMPY BROADCASTING -- no Python for-loops over particles.
           BAD:  for i in range(N): new[i] = step(state[i], params[i])
           GOOD: new = state + dt * f(state, params)   # all N at once

        2. CLIP TO PHYSICAL BOUNDS at the end of each step.
           new_state[:, 0] = np.clip(new_state[:, 0], 273.15, 2000.0)

        3. USE EXPLICIT EULER for Week 2 (simpler, sufficient for validation).
           new_state = state + dt * f(state, params)
           Higher-order methods (RK4, implicit) come in Month 2.
        """
        ...

    # =========================================================================
    # CONCRETE METHODS
    # These are provided FREE to every subclass -- no need to override them.
    # =========================================================================

    def validate_params(self, params: FloatArray) -> None:
        """
        Check that params has the correct shape.  Raises ValueError if not.

        Called once by MonteCarloEngine.run() before the simulation
        loop begins. Subclasses can also call this at the start of
        forward_batch() for additional per-step safety if desired.

        Raises
        ------
        ValueError  if params.ndim != 2 or params.shape[1] != self.param_dim
        """
        if params.ndim != 2:
            raise ValueError(
                f"{type(self).__name__}.validate_params: "
                f"params must be 2-D (N, param_dim), got shape {params.shape}. "
                f"Did you forget to call params.reshape(N, param_dim)?"
            )
        if params.shape[1] != self.param_dim:
            raise ValueError(
                f"{type(self).__name__}.validate_params: "
                f"expected {self.param_dim} columns (param_dim), "
                f"got {params.shape[1]}. "
                f"Check that all {self.param_dim} parameters are present."
            )

    def validate_state(self, state: FloatArray) -> None:
        """
        Check that state has the correct shape.  Raises ValueError if not.

        Raises
        ------
        ValueError  if state.ndim != 2 or state.shape[1] != self.state_dim
        """
        if state.ndim != 2:
            raise ValueError(
                f"{type(self).__name__}.validate_state: "
                f"state must be 2-D (N, state_dim), got shape {state.shape}."
            )
        if state.shape[1] != self.state_dim:
            raise ValueError(
                f"{type(self).__name__}.validate_state: "
                f"expected {self.state_dim} columns (state_dim), "
                f"got {state.shape[1]}."
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"state_dim={self.state_dim}, "
            f"param_dim={self.param_dim})"
        )


__all__ = ["Model", "FloatArray"]
