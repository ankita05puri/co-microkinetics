import numpy as np
import matplotlib.pyplot as plt

from src.simulate import simulate_to_steady_state
from src.model import rates


def sweep_tof_vs_T(params_base, T_values, t_final=80.0):
    """Run TOF over a temperature sweep."""
    tofs = []
    for T in T_values:
        params = dict(params_base)
        params["T"] = float(T)

        _, theta_ss, _ = simulate_to_steady_state(params, t_final=t_final)
        _, _, _, r4 = rates(theta_ss, params)  # TOF = net r4
        tofs.append(float(r4))

    return np.array(tofs)


def fit_arrhenius(T_values, tofs):
    """Fit ln(TOF) vs 1/T and return Ea_app (eV, kJ/mol) + fit data."""
    T_values = np.array(T_values, dtype=float)
    tofs = np.array(tofs, dtype=float)

    mask = tofs > 0
    T_fit = T_values[mask]
    tof_fit = tofs[mask]

    if len(T_fit) < 3:
        raise ValueError("Not enough positive TOF points for Arrhenius fit.")

    invT = 1.0 / T_fit
    lnTOF = np.log(tof_fit)

    m, b = np.polyfit(invT, lnTOF, 1)

    kB_eV_per_K = 8.617333262e-5
    Ea_app_eV = -m * kB_eV_per_K
    Ea_app_kJmol = Ea_app_eV * 96.485

    return invT, lnTOF, m, b, Ea_app_eV, Ea_app_kJmol


def compare_Ea_app_at_pco(base_params, T_values, pco, delta_e=0.05):
    """Compare Ea_app when bumping key forward barriers at a fixed PCO."""
    params_regime = dict(base_params)
    params_regime["PCO"] = float(pco)

    cases = [
        ("baseline", {}),
        ("E2f +Δ (O2 dissociation)", {"E2f": params_regime["E2f"] + delta_e}),
        ("E3f +Δ (surface rxn)", {"E3f": params_regime["E3f"] + delta_e}),
        ("E4f +Δ (CO2 desorp)", {"E4f": params_regime["E4f"] + delta_e}),
    ]

    results = {}
    plt.figure()

    for label, overrides in cases:
        params = dict(params_regime)
        params.update(overrides)

        tofs = sweep_tof_vs_T(params, T_values)
        invT, lnTOF, m, b, Ea_eV, _ = fit_arrhenius(T_values, tofs)

        results[label] = Ea_eV

        # points + fit line
        plt.plot(invT, lnTOF, "o", label=f"{label} data")
        xline = np.linspace(invT.min(), invT.max(), 200)
        plt.plot(xline, m * xline + b, "-", label=f"{label} fit (Ea={Ea_eV:.3f} eV)")

    plt.xlabel("1/T (1/K)")
    plt.ylabel("ln(TOF)")
    plt.title(f"Arrhenius comparison (PCO = {pco})")
    plt.legend(fontsize=8)
    plt.tight_layout()

    safe_pco = str(pco).replace(".", "p")
    outpath = f"figures/lnTOF_vs_invT_compare_PCO{safe_pco}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()

    Ea0 = results["baseline"]
    return {
        "PCO": float(pco),
        "Ea_app_eV": float(Ea0),
        "dEa_E2f": float(results["E2f +Δ (O2 dissociation)"] - Ea0),
        "dEa_E3f": float(results["E3f +Δ (surface rxn)"] - Ea0),
        "dEa_E4f": float(results["E4f +Δ (CO2 desorp)"] - Ea0),
        "plot": outpath,
    }


def main():
    base_params = {
        "PCO": 1.0,
        "PO2": 0.2,
        "PCO2": 0.0,
        "A": 1e3,

        "E1f": 0.35, "E1r": 0.60,
        "E2f": 0.55, "E2r": 1.00,
        "E3f": 0.70, "E3r": 0.90,
        "E4f": 0.70, "E4r": 1.20,
    }

    T_values = np.linspace(400, 900, 26)
    delta_e = 0.05

    # Baseline TOF vs T at PCO=1.0
    tofs = sweep_tof_vs_T({**base_params, "T": 600.0}, T_values)

    plt.figure()
    plt.plot(T_values, tofs, marker="o")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Steady-state TOF (net r4)")
    plt.title("TOF vs Temperature")
    plt.tight_layout()
    plt.savefig("figures/tof_vs_T.png", dpi=300)
    plt.close()
    print("Saved figures/tof_vs_T.png")

    # Compare two regimes: low CO vs high CO
    regimes = [0.1, 2.0]
    summaries = []
    for pco in regimes:
        s = compare_Ea_app_at_pco(base_params, T_values, pco, delta_e=delta_e)
        summaries.append(s)
        print(f"Saved {s['plot']}")

    print("\nEa_app sensitivity summary (ΔEa_app in eV):")
    print("PCO    Ea_app    ΔE2f     ΔE3f     ΔE4f")
    for s in summaries:
        print(f"{s['PCO']:<4.1f}  {s['Ea_app_eV']:<7.4f}  {s['dEa_E2f']:+.4f}  {s['dEa_E3f']:+.4f}  {s['dEa_E4f']:+.4f}")


if __name__ == "__main__":
    main()
