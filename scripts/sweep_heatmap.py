import numpy as np
import matplotlib.pyplot as plt

from src.simulate import simulate_to_steady_state
from src.model import rates


K_B_EV_PER_K = 8.617333262e-5


def tof_for_params(params, T_values, t_final=80.0):
    """Return TOF array (net r4) across a temperature sweep."""
    tofs = []
    for T in T_values:
        p = dict(params)
        p["T"] = float(T)

        _, theta_ss, _ = simulate_to_steady_state(p, t_final=t_final)
        _, _, _, r4 = rates(theta_ss, p)
        tofs.append(float(r4))

    return np.array(tofs, dtype=float)


def Ea_app_from_arrhenius(T_values, tofs):
    """Fit ln(TOF) vs 1/T and return apparent Ea (eV)."""
    T_values = np.asarray(T_values, dtype=float)
    tofs = np.asarray(tofs, dtype=float)

    mask = tofs > 0.0
    T_fit = T_values[mask]
    tof_fit = tofs[mask]

    if len(T_fit) < 3:
        return np.nan  # not enough points to fit

    invT = 1.0 / T_fit
    lnTOF = np.log(tof_fit)

    m, _ = np.polyfit(invT, lnTOF, 1)
    Ea_app_eV = -m * K_B_EV_PER_K
    return float(Ea_app_eV)


def Ea_app_for_params(params, T_values, t_final=80.0):
    tofs = tof_for_params(params, T_values, t_final=t_final)
    return Ea_app_from_arrhenius(T_values, tofs)


def main():
    base_params = {
        "PCO": 1.0,  # overridden in sweep
        "PO2": 0.2,
        "PCO2": 0.0,
        "A": 1e3,

        "E1f": 0.35, "E1r": 0.60,
        "E2f": 0.55, "E2r": 1.00,
        "E3f": 0.70, "E3r": 0.90,
        "E4f": 0.70, "E4r": 1.20,
    }

    T_values = np.linspace(400, 900, 26)
    pco_grid = np.linspace(0.05, 3.0, 20)
    delta = 0.05  # eV perturbation

    barrier_labels = ["E2f (O2 diss)", "E3f (surf rxn)", "E4f (CO2 des)"]
    dEa = np.zeros((len(barrier_labels), len(pco_grid)), dtype=float)
    Ea0_list = np.zeros(len(pco_grid), dtype=float)

    for j, pco in enumerate(pco_grid):
        params = {**base_params, "PCO": float(pco)}

        Ea0 = Ea_app_for_params(params, T_values)
        Ea0_list[j] = Ea0

        params_E2 = {**params, "E2f": params["E2f"] + delta}
        params_E3 = {**params, "E3f": params["E3f"] + delta}
        params_E4 = {**params, "E4f": params["E4f"] + delta}

        Ea2 = Ea_app_for_params(params_E2, T_values)
        Ea3 = Ea_app_for_params(params_E3, T_values)
        Ea4 = Ea_app_for_params(params_E4, T_values)

        dEa[0, j] = Ea2 - Ea0
        dEa[1, j] = Ea3 - Ea0
        dEa[2, j] = Ea4 - Ea0

    # --- Plot A: heatmap ΔEa_app vs PCO ---
    plt.figure()
    im = plt.imshow(
        dEa,
        aspect="auto",
        origin="lower",
        extent=[pco_grid.min(), pco_grid.max(), 0, len(barrier_labels)],
    )
    plt.colorbar(im, label=f"ΔEa_app (eV) for +{delta:.2f} eV perturbation")
    plt.yticks(np.arange(len(barrier_labels)) + 0.5, barrier_labels)
    plt.xlabel("PCO")
    plt.title("Regime map: barrier attribution vs CO partial pressure")
    plt.tight_layout()
    plt.savefig("figures/barrier_attribution_heatmap.png", dpi=300)
    plt.close()
    print("Saved figures/barrier_attribution_heatmap.png")

    # --- Plot B: baseline Ea_app vs PCO ---
    plt.figure()
    plt.plot(pco_grid, Ea0_list, marker="o")
    plt.xlabel("PCO")
    plt.ylabel("Ea_app (eV)")
    plt.title("Apparent activation energy vs CO partial pressure")
    plt.tight_layout()
    plt.savefig("figures/Ea_app_vs_PCO.png", dpi=300)
    plt.close()
    print("Saved figures/Ea_app_vs_PCO.png")


if __name__ == "__main__":
    main()
