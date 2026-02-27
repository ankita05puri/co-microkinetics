import numpy as np
import matplotlib.pyplot as plt

from src.simulate import simulate_to_steady_state
from src.model import rates


def main():
    # Fixed conditions for this sweep
    base_params = {
        "PO2": 0.2,
        "PCO2": 0.0,
        "T": 600.0,   # K
        "A": 1e3,     # shared prefactor (demo)

        # Activation energies (eV) — keep consistent across scripts
        "E1f": 0.35, "E1r": 0.60,   # CO adsorption/desorption
        "E2f": 0.55, "E2r": 1.00,   # O2 dissociation/recombination
        "E3f": 0.70, "E3r": 0.90,   # surface reaction
        "E4f": 0.70, "E4r": 1.20,   # CO2 desorption/readsorption
    }

    pco_grid = np.linspace(0.01, 3.0, 40)

    tofs = []
    theta_co = []
    theta_o = []
    theta_star = []

    for pco in pco_grid:
        params = {**base_params, "PCO": float(pco)}

        _, theta_ss, theta_star_ss = simulate_to_steady_state(params, t_final=80.0)

        # TOF defined as net CO2 formation rate (r4)
        _, _, _, r4 = rates(theta_ss, params)

        tofs.append(float(r4))
        theta_co.append(float(theta_ss[0]))
        theta_o.append(float(theta_ss[1]))
        theta_star.append(float(theta_star_ss))

    # --- Plot 1: TOF vs PCO ---
    plt.figure()
    plt.plot(pco_grid, tofs, marker="o")
    plt.yscale("log")
    plt.xlabel("PCO (dimensionless)")
    plt.ylabel("Steady-state TOF (net r4)")
    plt.title("TOF vs CO Partial Pressure")
    plt.tight_layout()
    plt.savefig("figures/tof_vs_pco.png", dpi=300)
    plt.close()

    # --- Plot 2: Coverages vs PCO ---
    plt.figure()
    plt.plot(pco_grid, theta_co, label="θ_CO")
    plt.plot(pco_grid, theta_o, label="θ_O")
    plt.plot(pco_grid, theta_star, label="θ_*")
    plt.xlabel("PCO (dimensionless)")
    plt.ylabel("Steady-state coverage")
    plt.title("Surface Coverage vs CO Partial Pressure")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/coverage_vs_pco.png", dpi=300)
    plt.close()

    print("Saved figures/tof_vs_pco.png")
    print("Saved figures/coverage_vs_pco.png")
    print(f"TOF range: {min(tofs):.3e} to {max(tofs):.3e}")


if __name__ == "__main__":
    main()
