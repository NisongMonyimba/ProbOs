"""
================================================================================
probos.battery_model  --  BatteryModel2Cell
================================================================================

A 2-cell lithium-ion battery pack model implementing the Model ABC.
Simulates thermal runaway via three sequential Arrhenius reactions.

State vector (8 variables per particle):
  [T1, T2, c_SEI_1, c_SEI_2, c_an_1, c_an_2, c_ca_1, c_ca_2]

Parameter vector (15 variables per particle):
  [Ea_SEI, A_SEI, H_SEI, Ea_an, A_an, H_an,
   Ea_ca, A_ca, H_ca, m_cell, Cp, h_conv, A_surf, T_amb, T_onset]

Reference: Kim et al. (2007), J. Power Sources 170(2), 476-489.
================================================================================
"""

from __future__ import annotations

import types

import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


from python.src.state import FloatArray, Model

# Universal gas constant [J / (mol * K)]
# This is a physical constant -- never changes
R_GAS: float = 8.314462


def _battery_step_unfused(
    T1: FloatArray, T2: FloatArray,
    c_sei1: FloatArray, c_sei2: FloatArray,
    c_an1: FloatArray, c_an2: FloatArray,
    c_ca1: FloatArray, c_ca2: FloatArray,
    Ea_SEI: FloatArray, A_SEI: FloatArray, H_SEI: FloatArray,
    Ea_an: FloatArray, A_an: FloatArray, H_an: FloatArray,
    Ea_ca: FloatArray, A_ca: FloatArray, H_ca: FloatArray,
    m: FloatArray, Cp: FloatArray, h: FloatArray,
    A_s: FloatArray, T_amb: FloatArray, dt: float,
    xp: types.ModuleType,
) -> tuple[
    FloatArray, FloatArray, FloatArray, FloatArray,
    FloatArray, FloatArray, FloatArray, FloatArray,
]:
    """
    The per-step battery physics update, factored out of
    forward_batch() so it can be wrapped with cp.fuse() below.
    """
    T1_safe = xp.clip(T1, 273.15, 5000.0)
    T2_safe = xp.clip(T2, 273.15, 5000.0)

    k_sei1 = A_SEI * xp.exp(-Ea_SEI / (R_GAS * T1_safe))
    k_an1  = A_an  * xp.exp(-Ea_an  / (R_GAS * T1_safe))
    k_ca1  = A_ca  * xp.exp(-Ea_ca  / (R_GAS * T1_safe))
    k_sei2 = A_SEI * xp.exp(-Ea_SEI / (R_GAS * T2_safe))
    k_an2  = A_an  * xp.exp(-Ea_an  / (R_GAS * T2_safe))
    k_ca2  = A_ca  * xp.exp(-Ea_ca  / (R_GAS * T2_safe))

    dc_sei1 = -k_sei1 * xp.clip(c_sei1, 0.0, 1.0)
    dc_sei2 = -k_sei2 * xp.clip(c_sei2, 0.0, 1.0)
    dc_an1  = -k_an1  * xp.clip(c_an1,  0.0, 1.0)
    dc_an2  = -k_an2  * xp.clip(c_an2,  0.0, 1.0)
    dc_ca1  = -k_ca1  * xp.clip(c_ca1,  0.0, 1.0)
    dc_ca2  = -k_ca2  * xp.clip(c_ca2,  0.0, 1.0)

    Q_sei1 = H_SEI * m * (-dc_sei1)
    Q_an1  = H_an  * m * (-dc_an1)
    Q_ca1  = H_ca  * m * (-dc_ca1)
    Q_sei2 = H_SEI * m * (-dc_sei2)
    Q_an2  = H_an  * m * (-dc_an2)
    Q_ca2  = H_ca  * m * (-dc_ca2)

    Q_loss1 = h * A_s * (T1 - T_amb)
    Q_loss2 = h * A_s * (T2 - T_amb)

    thermal_mass = m * Cp
    dT1_dt = (Q_sei1 + Q_an1 + Q_ca1 - Q_loss1) / thermal_mass
    dT2_dt = (Q_sei2 + Q_an2 + Q_ca2 - Q_loss2) / thermal_mass

    new_T1     = xp.clip(T1     + dt * dT1_dt, 273.15, 5000.0)
    new_T2     = xp.clip(T2     + dt * dT2_dt, 273.15, 5000.0)
    new_c_sei1 = xp.clip(c_sei1 + dt * dc_sei1, 0.0, 1.0)
    new_c_sei2 = xp.clip(c_sei2 + dt * dc_sei2, 0.0, 1.0)
    new_c_an1  = xp.clip(c_an1  + dt * dc_an1,  0.0, 1.0)
    new_c_an2  = xp.clip(c_an2  + dt * dc_an2,  0.0, 1.0)
    new_c_ca1  = xp.clip(c_ca1  + dt * dc_ca1,  0.0, 1.0)
    new_c_ca2  = xp.clip(c_ca2  + dt * dc_ca2,  0.0, 1.0)

    return (
        new_T1, new_T2, new_c_sei1, new_c_sei2,
        new_c_an1, new_c_an2, new_c_ca1, new_c_ca2,
    )


if _CUPY_AVAILABLE:
    @cp.fuse()  # type: ignore[untyped-decorator]
    def _battery_step_fused(
        T1: FloatArray, T2: FloatArray,
        c_sei1: FloatArray, c_sei2: FloatArray,
        c_an1: FloatArray, c_an2: FloatArray,
        c_ca1: FloatArray, c_ca2: FloatArray,
        Ea_SEI: FloatArray, A_SEI: FloatArray, H_SEI: FloatArray,
        Ea_an: FloatArray, A_an: FloatArray, H_an: FloatArray,
        Ea_ca: FloatArray, A_ca: FloatArray, H_ca: FloatArray,
        m: FloatArray, Cp: FloatArray, h: FloatArray,
        A_s: FloatArray, T_amb: FloatArray, dt: float,
    ) -> tuple[
        FloatArray, FloatArray, FloatArray, FloatArray,
        FloatArray, FloatArray, FloatArray, FloatArray,
    ]:
        """
        cp.fuse()-wrapped version of _battery_step_unfused, letting
        CuPy compile the ENTIRE per-step update into ONE GPU kernel
        launch instead of ~15 separate ones.

        Deliberately left WITHOUT type annotations: cp.fuse() wraps
        this into a cupy fusion callable at decoration time, not a
        plain Python function -- annotating here does not aid mypy.
        """
        return _battery_step_unfused(
            T1, T2, c_sei1, c_sei2, c_an1, c_an2, c_ca1, c_ca2,
            Ea_SEI, A_SEI, H_SEI, Ea_an, A_an, H_an, Ea_ca, A_ca, H_ca,
            m, Cp, h, A_s, T_amb, dt, cp,
        )




class BatteryModel2Cell(Model):
    """
    Two-cell lithium-ion battery thermal abuse model.

    Implements three sequential Arrhenius exothermic reactions:
      SEI decomposition -> anode reaction -> cathode reaction

    Each reaction depletes a normalised reactant concentration c in [0, 1]
    and releases heat proportional to the reaction enthalpy H [J/kg].

    USAGE EXAMPLE:
    --------------
    from python.src.battery_model import BatteryModel2Cell
    import numpy as np

    model = BatteryModel2Cell()
    print(model)   # BatteryModel2Cell(state_dim=8, param_dim=15)

    # Draw N=100 parameter sets from prior distributions
    # (Week 3 will do this automatically; here we use nominal values)
    N = 100
    params = np.tile(model.nominal_params(), (N, 1))   # shape (N, 15)

    # Tile initial state across N particles
    state = np.tile(model.initial_state(), (N, 1))     # shape (N, 8)

    # Step forward 1 second
    new_state = model.forward_batch(state, params, dt=1.0)
    """

    # =========================================================================
    # STATE VARIABLE INDICES
    # Using named constants prevents bugs from magic numbers.
    # Instead of state[:, 0] we write state[:, BatteryModel2Cell.T1]
    # which is self-documenting and safe against reordering.
    # =========================================================================
    T1     = 0   # Cell 1 temperature [K]
    T2     = 1   # Cell 2 temperature [K]
    C_SEI1 = 2   # Cell 1 SEI reactant remaining [0, 1]
    C_SEI2 = 3   # Cell 2 SEI reactant remaining [0, 1]
    C_AN1  = 4   # Cell 1 anode reactant remaining [0, 1]
    C_AN2  = 5   # Cell 2 anode reactant remaining [0, 1]
    C_CA1  = 6   # Cell 1 cathode reactant remaining [0, 1]
    C_CA2  = 7   # Cell 2 cathode reactant remaining [0, 1]

    # =========================================================================
    # PARAMETER INDICES
    # =========================================================================
    P_EA_SEI  = 0    # SEI activation energy [J/mol]
    P_A_SEI   = 1    # SEI pre-exponential [s^-1]
    P_H_SEI   = 2    # SEI heat of reaction [J/kg]
    P_EA_AN   = 3    # Anode activation energy [J/mol]
    P_A_AN    = 4    # Anode pre-exponential [s^-1]
    P_H_AN    = 5    # Anode heat of reaction [J/kg]
    P_EA_CA   = 6    # Cathode activation energy [J/mol]
    P_A_CA    = 7    # Cathode pre-exponential [s^-1]
    P_H_CA    = 8    # Cathode heat of reaction [J/kg]
    P_M_CELL  = 9    # Cell mass [kg]
    P_CP      = 10   # Specific heat capacity [J/(kg K)]
    P_H_CONV  = 11   # Convective heat transfer coefficient [W/(m^2 K)]
    P_A_SURF  = 12   # Cell surface area [m^2]
    P_T_AMB   = 13   # Ambient temperature [K]
    P_T_ONSET = 14   # ARC onset temperature [K]

    # =========================================================================
    # ABC PROPERTIES
    # =========================================================================

    @property
    def state_dim(self) -> int:
        """8 state variables: T1, T2, c_SEI x2, c_anode x2, c_cathode x2."""
        return 8

    @property
    def param_dim(self) -> int:
        """15 uncertain parameters from Kim et al. (2007)."""
        return 15

    # =========================================================================
    # ABC METHODS
    # =========================================================================

    def param_names(self) -> list[str]:
        """Human-readable names for all 15 parameters, in index order."""
        return [
            "Ea_SEI",    # 0
            "A_SEI",     # 1
            "H_SEI",     # 2
            "Ea_anode",  # 3
            "A_anode",   # 4
            "H_anode",   # 5
            "Ea_cath",   # 6
            "A_cath",    # 7
            "H_cath",    # 8
            "m_cell",    # 9
            "Cp",        # 10
            "h_conv",    # 11
            "A_surf",    # 12
            "T_amb",     # 13
            "T_onset",   # 14
        ]

    def initial_state(self) -> FloatArray:
        """
        Initial state for ONE particle at ARC test conditions.

        Temperatures are set to the ARC onset temperature (403.15 K = 130 C).
        All reactant concentrations start at 1.0 (fully charged / unreacted).

        Returns
        -------
        FloatArray of shape (8,)
        """
        T0 = 403.15   # ARC onset temperature [K] = 130 C
        return np.array([
            T0,   # T1  [K]
            T0,   # T2  [K]
            1.0,  # c_SEI_1  (fully unreacted)
            1.0,  # c_SEI_2
            1.0,  # c_an_1
            1.0,  # c_an_2
            1.0,  # c_ca_1
            1.0,  # c_ca_2
        ], dtype=np.float64)

    def forward_batch(
        self,
        state: FloatArray,   # shape (N, 8)
        params: FloatArray,  # shape (N, 15)
        dt: float,           # time step [seconds]
    ) -> FloatArray:
        """
        Advance all N particles by one explicit Euler step of dt seconds.

        All operations are vectorised over the N-particle dimension using
        NumPy broadcasting.  No Python for-loops.

        Parameters
        ----------
        state  : shape (N, 8)  -- current state of all particles
        params : shape (N, 15) -- parameter vector for each particle
        dt     : float         -- time step in seconds

        Returns
        -------
        FloatArray of shape (N, 8)  -- updated state
        """
        # Month 3 Week 10: dispatch to CuPy if state is a GPU
        # array, NumPy otherwise -- cp.get_array_module() is
        # CuPy's own official utility for exactly this pattern,
        # letting the SAME code below run correctly on either
        # backend with zero duplication. Falls back to plain
        # NumPy if CuPy is not installed at all (this model
        # continues to work with no GPU/CuPy present, unchanged
        # from Week 2).
        xp = cp.get_array_module(state) if _CUPY_AVAILABLE else np

        # ------------------------------------------------------------------
        # STEP 1: Extract state columns
        # ------------------------------------------------------------------
        T1 = state[:, self.T1]
        T2 = state[:, self.T2]
        c_sei1 = state[:, self.C_SEI1]
        c_sei2 = state[:, self.C_SEI2]
        c_an1  = state[:, self.C_AN1]
        c_an2  = state[:, self.C_AN2]
        c_ca1  = state[:, self.C_CA1]
        c_ca2  = state[:, self.C_CA2]

        # ------------------------------------------------------------------
        # STEP 2: Extract parameter columns
        # ------------------------------------------------------------------
        Ea_SEI = params[:, self.P_EA_SEI]
        A_SEI  = params[:, self.P_A_SEI]
        H_SEI  = params[:, self.P_H_SEI]
        Ea_an  = params[:, self.P_EA_AN]
        A_an   = params[:, self.P_A_AN]
        H_an   = params[:, self.P_H_AN]
        Ea_ca  = params[:, self.P_EA_CA]
        A_ca   = params[:, self.P_A_CA]
        H_ca   = params[:, self.P_H_CA]
        m      = params[:, self.P_M_CELL]
        Cp     = params[:, self.P_CP]
        h      = params[:, self.P_H_CONV]
        A_s    = params[:, self.P_A_SURF]
        T_amb  = params[:, self.P_T_AMB]

        # ------------------------------------------------------------------
        # STEP 3: Compute the full per-step update -- FUSED into a single
        # GPU kernel launch when running on CuPy arrays (Month 3 Week 10
        # kernel-fusion fix), or plain sequential NumPy calls on CPU.
        # ------------------------------------------------------------------
        if _CUPY_AVAILABLE and xp is cp:
            (
                new_T1, new_T2, new_c_sei1, new_c_sei2,
                new_c_an1, new_c_an2, new_c_ca1, new_c_ca2,
            ) = _battery_step_fused(
                T1, T2, c_sei1, c_sei2, c_an1, c_an2, c_ca1, c_ca2,
                Ea_SEI, A_SEI, H_SEI, Ea_an, A_an, H_an, Ea_ca, A_ca, H_ca,
                m, Cp, h, A_s, T_amb, dt,
            )
        else:
            (
                new_T1, new_T2, new_c_sei1, new_c_sei2,
                new_c_an1, new_c_an2, new_c_ca1, new_c_ca2,
            ) = _battery_step_unfused(
                T1, T2, c_sei1, c_sei2, c_an1, c_an2, c_ca1, c_ca2,
                Ea_SEI, A_SEI, H_SEI, Ea_an, A_an, H_an, Ea_ca, A_ca, H_ca,
                m, Cp, h, A_s, T_amb, dt, xp,
            )

        # ------------------------------------------------------------------
        # STEP 4: Pack the 8 updated state variables back into one array
        # ------------------------------------------------------------------
        new_state: FloatArray = xp.column_stack([
            new_T1,
            new_T2,
            new_c_sei1,
            new_c_sei2,
            new_c_an1,
            new_c_an2,
            new_c_ca1,
            new_c_ca2,
        ])
        return new_state

    # =========================================================================
    # HELPER: NOMINAL PARAMETERS
    # Returns the literature mean values for all 15 parameters.
    # Used for deterministic validation (Kim 2007 ARC test).
    # In the Monte Carlo run (Week 3), each particle draws its own
    # parameter vector from the prior distributions.
    # =========================================================================

    def nominal_params(self) -> FloatArray:
        """
        Nominal (literature mean) parameter values from Kim et al. (2007).

        Returns
        -------
        FloatArray of shape (15,) -- one value per parameter

        Usage
        -----
        params_1 = model.nominal_params()                     # shape (15,)
        params_N = np.tile(model.nominal_params(), (N, 1))    # shape (N, 15)
        """
        return np.array([
            1.3508e5,   # 0  Ea_SEI   [J/mol]
            1.667e15,   # 1  A_SEI    [s^-1]
            2.5e5,      # 2  H_SEI    [J/kg]
            1.3508e5,   # 3  Ea_anode [J/mol]
            2.5e13,     # 4  A_anode  [s^-1]
            1.714e6,    # 5  H_anode  [J/kg]
            1.396e5,    # 6  Ea_cath  [J/mol]
            1.0e9,      # 7  A_cath   [s^-1]
            3.14e5,     # 8  H_cath   [J/kg]
            0.045,      # 9  m_cell   [kg]   (45 g typical 18650 cell)
            800.0,      # 10 Cp       [J/(kg K)]
            5.0,        # 11 h_conv   [W/(m^2 K)]
            3.5e-3,     # 12 A_surf   [m^2]  (surface area of 18650 cell)
            298.15,     # 13 T_amb    [K]    (25 C)
            403.15,     # 14 T_onset  [K]    (130 C ARC onset)
        ], dtype=np.float64)
