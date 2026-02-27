import matplotlib.pyplot as plt

from src.simulate import simulate_to_steady_state
from src.model import rates


def main():
    params = {
        "PCO": 1.0,
        "PO2": 0.2,
        "PCO2": 0.0,

        "T": 600.0,      # K
        "A": 1e3,       # 1/s

        # Activation energies (eV) — illustrative placeholders (DFT-derived values plug in here) Activation energies (eV) — toy values for now
        "E1f": 0.35, "E1r": 0.60,   # CO adsorption/desorption
        "E2f": 0.55, "E2r": 1.00,   # O2 dissociation/recombination
        "E3f": 0.70, "E3r": 0.90,   # surface reaction
        "E4f": 0.70, "E4r": 1.20,   # CO2 desorption/readsorption
    }
    

    sol, theta_ss, theta_star_ss = simulate_to_steady_state(params)

    theta_CO, theta_O, theta_CO2 = sol.y
    theta_star = 1 - theta_CO - theta_O - theta_CO2

    # Steady-state TOF = net r4 at steady state
    r1, r2, r3, r4 = rates(theta_ss, params)
    tof_ss = r4

    print("Steady-state coverages [θ_CO, θ_O, θ_CO2] =", theta_ss)
    print("Steady-state θ_* =", theta_star_ss)
    print("Steady-state TOF =", tof_ss)

    # Figure: Coverages vs time
    plt.figure()
    plt.plot(sol.t, theta_CO, label="θ_CO")
    plt.plot(sol.t, theta_O, label="θ_O")
    plt.plot(sol.t, theta_CO2, label="θ_CO2")
    plt.plot(sol.t, theta_star, label="θ_*")
    plt.xlabel("time")
    plt.ylabel("coverage")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
