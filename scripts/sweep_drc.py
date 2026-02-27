import copy
import numpy as np
import matplotlib.pyplot as plt

from src.simulate import simulate_to_steady_state
from src.model import rates


K_B_EV_PER_K = 8.617333262e-5  # eV/K


def compute_tof(params, t_final=80.0):
    """Return steady-state TOF = net r4."""
    _, theta_ss, _ = simulate_to_steady_state(params, t_final=t_final)
    _, _, _, r4 = rates(theta_ss, params)
    return float(r4)


def main():
    # Baseline parameters (keep aligned with run_baseline.py)
    base_params = {
        "PCO": 1.0,
        "PO2": 0.2,
        "PCO2": 0.0,
        "T": 600.0,
        "A": 1e3,

        "E1f": 0.35, "E1r": 0.60,
        "E2f": 0.55, "E2r": 1.00,
        "E3f": 0.70, "E3r": 0.90,
        "E4f": 0.70, "E4r": 1.20,
    }

    T = float(base_params["T"])
    delta_E = 0.01  # eV (small perturbation)

    steps = ["E1f", "E2f", "E3f", "E4f"]
    labels = {
        "E1f": "CO adsorption",
        "E2f": "O2 dissociation",
        "E3f": "Surface reaction",
        "E4f": "CO2 desorption",
    }

    tof0 = compute_tof(base_params)
    if tof0 <= 0:
        raise ValueError(f"Baseline TOF must be > 0 for log(). Got {tof0}")

    ln_tof0 = np.log(tof0)

    drc = {}

    for key in steps:
        p = copy.deepcopy(base_params)
        p[key] = p[key] - delta_E  # lower barrier slightly

        tof1 = compute_tof(p)
        if tof1 <= 0:
            raise ValueError(f"Perturbed TOF must be > 0 for log(). {key} gave {tof1}")

        ln_tof1 = np.log(tof1)

        # DRC_i ≈ (d ln TOF) / (d( -Ea/(kB*T) ))
        # Lowering Ea by delta_E increases (-Ea/(kB*T)) by +delta_E/(kB*T)
        denom = delta_E / (K_B_EV_PER_K * T)
        drc_val = (ln_tof1 - ln_tof0) / denom

        drc[labels[key]] = float(drc_val)

    # Print
    print(f"Baseline TOF: {tof0:.6e}")
    print("Degree of Rate Control (DRC):")
    for name, val in drc.items():
        print(f"  {name:15s}: {val:+.3f}")

    # Plot
    names = list(drc.keys())
    values = [drc[n] for n in names]

    plt.figure()
    plt.bar(names, values)
    plt.axhline(0.0)
    plt.ylabel("Degree of Rate Control (DRC)")
    plt.title("DRC at Baseline Conditions")
    plt.xticks(rotation=20
