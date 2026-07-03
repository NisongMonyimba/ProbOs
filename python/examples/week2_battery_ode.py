"""
Week 2 Example: Deterministic battery ODE simulation.

Runs BatteryModel2Cell forward using nominal Kim 2007 parameters.
Saves a 4-panel figure showing temperature and reactant concentrations
over time.

This is the DETERMINISTIC baseline.
Week 3 adds uncertainty: N=5000 particles with different parameter values.
"""

import os


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from python.src.battery_model import BatteryModel2Cell

os.makedirs("outputs/figures", exist_ok=True)


def run_simulation(
    model: BatteryModel2Cell,
    n_steps: int,
    dt: float,
) -> np.ndarray:
    """
    Run the battery ODE for n_steps time steps of size dt seconds.

    Parameters
    ----------
    model   : BatteryModel2Cell instance
    n_steps : number of time steps to run
    dt      : time step size in seconds

    Returns
    -------
    history : np.ndarray of shape (n_steps + 1, 8)
        Row 0 is the initial state.
        Row i is the state after i time steps.
    """
    # -----------------------------------------------------------------------
    # Set up initial conditions
    # N=1 because this is a DETERMINISTIC run (one particle, not 5000).
    # We use nominal (literature mean) parameter values.
    # -----------------------------------------------------------------------
    N = 1
    state  = np.tile(model.initial_state(), (N, 1))    # shape (1, 8)
    params = np.tile(model.nominal_params(), (N, 1))   # shape (1, 15)

    # Adiabatic conditions: no convective heat loss (ARC test protocol)
    # In a real ARC test, the calorimeter is perfectly insulated.
    params[:, BatteryModel2Cell.P_H_CONV] = 0.0

    # -----------------------------------------------------------------------
    # Allocate history array to store every state
    # Shape: (n_steps + 1, 8) -- one row per time step including t=0
    # -----------------------------------------------------------------------
    history = np.zeros((n_steps + 1, 8), dtype=np.float64)
    history[0, :] = state[0, :]   # store initial state

    # -----------------------------------------------------------------------
    # Time integration loop
    # Each call to forward_batch advances the state by dt seconds.
    # This is an explicit Euler method: z(t+dt) = z(t) + dt * f(z(t))
    # -----------------------------------------------------------------------
    for step in range(n_steps):
        state = model.forward_batch(state, params, dt=dt)
        history[step + 1, :] = state[0, :]

    return history


def plot_results(history: np.ndarray, dt: float, save_path: str) -> None:
    """
    Create a 4-panel figure from the simulation history and save it.

    Parameters
    ----------
    history   : shape (n_steps + 1, 8) -- full state trajectory
    dt        : time step in seconds (used to build the time axis)
    save_path : file path where the PNG will be saved
    """
    n_steps = history.shape[0] - 1

    # Build time axis in minutes (easier to read than seconds)
    # time_min[i] = time in minutes at step i
    time_min = np.arange(n_steps + 1) * dt / 60.0

    # -----------------------------------------------------------------------
    # Extract each state variable from the history array
    # history[:, i] gives column i across all time steps
    # -----------------------------------------------------------------------
    T1     = history[:, BatteryModel2Cell.T1]      # Cell 1 temperature [K]
    T2     = history[:, BatteryModel2Cell.T2]      # Cell 2 temperature [K]
    c_sei1 = history[:, BatteryModel2Cell.C_SEI1]  # SEI concentration, Cell 1
    c_sei2 = history[:, BatteryModel2Cell.C_SEI2]
    c_an1  = history[:, BatteryModel2Cell.C_AN1]   # Anode concentration, Cell 1
    c_an2  = history[:, BatteryModel2Cell.C_AN2]
    c_ca1  = history[:, BatteryModel2Cell.C_CA1]   # Cathode concentration, Cell 1
    c_ca2  = history[:, BatteryModel2Cell.C_CA2]

    # Convert temperatures from Kelvin to Celsius for readability
    T1_C = T1 - 273.15
    T2_C = T2 - 273.15

    # -----------------------------------------------------------------------
    # Create figure with 4 panels stacked vertically
    # figsize=(10, 12) means 10 inches wide, 12 inches tall
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        "BatteryModel2Cell -- Deterministic ARC Simulation\n"
        "Kim et al. (2007) nominal parameters, adiabatic conditions",
        fontsize=13, fontweight="bold",
    )

    # -----------------------------------------------------------------------
    # Panel 1: Temperature vs time
    # -----------------------------------------------------------------------
    ax = axes[0]
    ax.plot(time_min, T1_C, color="tab:red",  lw=2, label="Cell 1")
    ax.plot(time_min, T2_C, color="tab:orange", lw=2, ls="--", label="Cell 2")
    ax.axhline(130.0, color="gray", ls=":", lw=1, label="ARC onset (130 C)")
    ax.set_ylabel("Temperature [°C]", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Temperature", fontsize=11)

    # -----------------------------------------------------------------------
    # Panel 2: SEI concentration vs time
    # -----------------------------------------------------------------------
    ax = axes[1]
    ax.plot(time_min, c_sei1, color="tab:blue",  lw=2, label="Cell 1")
    ax.plot(time_min, c_sei2, color="tab:cyan",  lw=2, ls="--", label="Cell 2")
    ax.set_ylabel("SEI concentration [−]", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("SEI Reactant Remaining", fontsize=11)

    # -----------------------------------------------------------------------
    # Panel 3: Anode concentration vs time
    # -----------------------------------------------------------------------
    ax = axes[2]
    ax.plot(time_min, c_an1, color="tab:green",  lw=2, label="Cell 1")
    ax.plot(time_min, c_an2, color="tab:olive",  lw=2, ls="--", label="Cell 2")
    ax.set_ylabel("Anode concentration [−]", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Anode Reactant Remaining", fontsize=11)

    # -----------------------------------------------------------------------
    # Panel 4: Cathode concentration vs time
    # -----------------------------------------------------------------------
    ax = axes[3]
    ax.plot(time_min, c_ca1, color="tab:purple", lw=2, label="Cell 1")
    ax.plot(time_min, c_ca2, color="tab:pink",   lw=2, ls="--", label="Cell 2")
    ax.set_ylabel("Cathode concentration [−]", fontsize=11)
    ax.set_xlabel("Time [minutes]", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Cathode Reactant Remaining", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {save_path}")


def print_summary(history: np.ndarray, dt: float) -> None:
    """Print a text summary of key simulation results."""
    T1_C = history[:, BatteryModel2Cell.T1] - 273.15

    T_start  = T1_C[0]
    T_max    = T1_C.max()
    t_max_s  = T1_C.argmax() * dt
    t_max_m  = t_max_s / 60.0
    dT_total = T_max - T_start

    print()
    print("=" * 55)
    print("  SIMULATION SUMMARY")
    print("=" * 55)
    print(f"  Start temperature   : {T_start:.2f} C")
    print(f"  Peak temperature    : {T_max:.2f} C")
    print(f"  Total temperature   : +{dT_total:.2f} C")
    print(f"  Time to peak        : {t_max_m:.1f} minutes")
    print(f"  Time steps run      : {history.shape[0] - 1:,}")
    print(f"  Time step size      : {dt:.1f} s")
    print(f"  Total time simulated: {(history.shape[0]-1)*dt/60:.1f} minutes")
    print("=" * 55)
    print()


if __name__ == "__main__":
    print("Running BatteryModel2Cell deterministic ODE simulation...")
    print("Parameters: Kim et al. (2007) nominal values")
    print("Conditions: adiabatic ARC test (h_conv = 0)")
    print()

    # -----------------------------------------------------------------------
    # Simulation settings
    # n_steps = 18,000 steps of 1 second each = 300 minutes = 5 hours
    # This is long enough to see the full thermal runaway curve.
    # -----------------------------------------------------------------------
    model   = BatteryModel2Cell()
    n_steps = 18_000   # 300 minutes
    dt      = 1.0      # 1 second per step

    print(f"  Model      : {model}")
    print(f"  Steps      : {n_steps:,} x {dt:.0f} s = {n_steps*dt/60:.0f} minutes")
    print()

    # Run the simulation
    history = run_simulation(model, n_steps=n_steps, dt=dt)

    # Print text summary
    print_summary(history, dt)

    # Save plot
    save_path = "outputs/figures/week2_battery_ode.png"
    print("Generating 4-panel figure...")
    plot_results(history, dt=dt, save_path=save_path)

    print("Done.")
    print()
    print("Next step: Week 3 Monte Carlo engine runs N=5000 of these")
    print("simultaneously with different parameter values per particle.")
