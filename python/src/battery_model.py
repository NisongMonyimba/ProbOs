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

import numpy as np

from python.src.state import FloatArray, Model

# Universal gas constant [J / (mol * K)]
# This is a physical constant -- never changes
R_GAS: float = 8.314462


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
        # ------------------------------------------------------------------
        # STEP 1: Extract state columns
        # state[:, i] extracts column i for ALL N particles at once.
        # The result has shape (N,).
        # ------------------------------------------------------------------
        T1 = state[:, self.T1]      # Cell 1 temperature [K],  shape (N,)
        T2 = state[:, self.T2]      # Cell 2 temperature [K],  shape (N,)
        c_sei1 = state[:, self.C_SEI1]   # shape (N,)
        c_sei2 = state[:, self.C_SEI2]
        c_an1  = state[:, self.C_AN1]
        c_an2  = state[:, self.C_AN2]
        c_ca1  = state[:, self.C_CA1]
        c_ca2  = state[:, self.C_CA2]

        # ------------------------------------------------------------------
        # STEP 2: Extract parameter columns
        # params[:, i] extracts parameter i for ALL N particles at once.
        # ------------------------------------------------------------------
        Ea_SEI = params[:, self.P_EA_SEI]   # [J/mol], shape (N,)
        A_SEI  = params[:, self.P_A_SEI]    # [s^-1]
        H_SEI  = params[:, self.P_H_SEI]    # [J/kg]
        Ea_an  = params[:, self.P_EA_AN]
        A_an   = params[:, self.P_A_AN]
        H_an   = params[:, self.P_H_AN]
        Ea_ca  = params[:, self.P_EA_CA]
        A_ca   = params[:, self.P_A_CA]
        H_ca   = params[:, self.P_H_CA]
        m      = params[:, self.P_M_CELL]   # [kg]
        Cp     = params[:, self.P_CP]       # [J/(kg K)]
        h      = params[:, self.P_H_CONV]   # [W/(m^2 K)]
        A_s    = params[:, self.P_A_SURF]   # [m^2]
        T_amb  = params[:, self.P_T_AMB]    # [K]

        # ------------------------------------------------------------------
        # STEP 3: Arrhenius rates for Cell 1
        #
        # Arrhenius law:  k = A * exp(-Ea / (R * T))
        #
        # np.clip(T1, 273.15, 5000.0) prevents exp() overflow if temperature
        # goes negative (which should not happen physically but can occur in
        # the first few time steps of a poorly-initialised simulation).
        # ------------------------------------------------------------------
        T1_safe = np.clip(T1, 273.15, 5000.0)
        T2_safe = np.clip(T2, 273.15, 5000.0)

        # Arrhenius exponent: -Ea / (R * T)  for each reaction, Cell 1
        # Note (Week 4 profiling): NumPy exp() is SIMD-vectorised per call.
        # Batching exp calls or precomputing inv_RT showed no speedup in
        # Python (exp dominates; division cost is negligible).
        # Real parallelism gain deferred to C++ OpenMP kernel (Week 4 Tue).
        k_sei1 = A_SEI * np.exp(-Ea_SEI / (R_GAS * T1_safe))
        k_an1  = A_an  * np.exp(-Ea_an  / (R_GAS * T1_safe))
        k_ca1  = A_ca  * np.exp(-Ea_ca  / (R_GAS * T1_safe))

        k_sei2 = A_SEI * np.exp(-Ea_SEI / (R_GAS * T2_safe))
        k_an2  = A_an  * np.exp(-Ea_an  / (R_GAS * T2_safe))
        k_ca2  = A_ca  * np.exp(-Ea_ca  / (R_GAS * T2_safe))

        # ------------------------------------------------------------------
        # STEP 4: Reactant depletion rates
        #
        # dc/dt = -k * c
        # The reaction rate is proportional to the remaining reactant.
        # When c = 0, the reaction has stopped (no more fuel).
        # np.clip(c, 0.0, 1.0) prevents c from going below 0 due to
        # numerical error in large time steps.
        # ------------------------------------------------------------------
        dc_sei1 = -k_sei1 * np.clip(c_sei1, 0.0, 1.0)
        dc_sei2 = -k_sei2 * np.clip(c_sei2, 0.0, 1.0)
        dc_an1  = -k_an1  * np.clip(c_an1,  0.0, 1.0)
        dc_an2  = -k_an2  * np.clip(c_an2,  0.0, 1.0)
        dc_ca1  = -k_ca1  * np.clip(c_ca1,  0.0, 1.0)
        dc_ca2  = -k_ca2  * np.clip(c_ca2,  0.0, 1.0)

        # ------------------------------------------------------------------
        # STEP 5: Heat generation rates [W]
        #
        # Q = H * m_cell * (-dc/dt)
        #
        # H  [J/kg] * m [kg] * (-dc/dt) [s^-1] = [W]
        #
        # -dc/dt is positive (reactant is being consumed).
        # ------------------------------------------------------------------
        Q_sei1 = H_SEI * m * (-dc_sei1)
        Q_an1  = H_an  * m * (-dc_an1)
        Q_ca1  = H_ca  * m * (-dc_ca1)

        Q_sei2 = H_SEI * m * (-dc_sei2)
        Q_an2  = H_an  * m * (-dc_an2)
        Q_ca2  = H_ca  * m * (-dc_ca2)

        # ------------------------------------------------------------------
        # STEP 6: Heat loss by convection (Newton law of cooling) [W]
        #
        # Q_loss = h * A_surf * (T - T_amb)
        #
        # Positive when T > T_amb (cell is hotter than ambient -- losing heat).
        # ------------------------------------------------------------------
        Q_loss1 = h * A_s * (T1 - T_amb)
        Q_loss2 = h * A_s * (T2 - T_amb)

        # ------------------------------------------------------------------
        # STEP 7: Temperature rate of change [K/s]
        #
        # dT/dt = (Q_rxn_total - Q_loss) / (m * Cp)
        #
        # m * Cp [J/K] is the thermal mass of the cell.
        # Dividing by it converts power [W] to temperature rate [K/s].
        # ------------------------------------------------------------------
        thermal_mass = m * Cp   # [J/K], shape (N,)

        dT1_dt = (Q_sei1 + Q_an1 + Q_ca1 - Q_loss1) / thermal_mass
        dT2_dt = (Q_sei2 + Q_an2 + Q_ca2 - Q_loss2) / thermal_mass

        # ------------------------------------------------------------------
        # STEP 8: Explicit Euler integration
        #
        # new_value = old_value + dt * rate
        #
        # This is the simplest numerical integration method.
        # It is first-order accurate: error ~ O(dt).
        # For Week 2, dt = 1.0 s is fine for validation.
        # Month 2 will upgrade to RK4 for production accuracy.
        # ------------------------------------------------------------------
        new_T1    = T1     + dt * dT1_dt
        new_T2    = T2     + dt * dT2_dt
        new_c_sei1 = c_sei1 + dt * dc_sei1
        new_c_sei2 = c_sei2 + dt * dc_sei2
        new_c_an1  = c_an1  + dt * dc_an1
        new_c_an2  = c_an2  + dt * dc_an2
        new_c_ca1  = c_ca1  + dt * dc_ca1
        new_c_ca2  = c_ca2  + dt * dc_ca2

        # ------------------------------------------------------------------
        # STEP 9: Physical bounds enforcement
        #
        # Temperatures must be above absolute zero and below plasma.
        # Concentrations must stay in [0, 1] -- cannot go negative
        # or above the initial value.
        # ------------------------------------------------------------------
        new_T1    = np.clip(new_T1,     273.15, 5000.0)
        new_T2    = np.clip(new_T2,     273.15, 5000.0)
        new_c_sei1 = np.clip(new_c_sei1, 0.0, 1.0)
        new_c_sei2 = np.clip(new_c_sei2, 0.0, 1.0)
        new_c_an1  = np.clip(new_c_an1,  0.0, 1.0)
        new_c_an2  = np.clip(new_c_an2,  0.0, 1.0)
        new_c_ca1  = np.clip(new_c_ca1,  0.0, 1.0)
        new_c_ca2  = np.clip(new_c_ca2,  0.0, 1.0)

        # ------------------------------------------------------------------
        # STEP 10: Pack the 8 updated state variables back into one array
        #
        # np.column_stack([a, b, c, ...]) takes N-length 1-D arrays and
        # stacks them as columns to produce an (N, 8) array.
        # ------------------------------------------------------------------
        new_state: FloatArray = np.column_stack([
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
