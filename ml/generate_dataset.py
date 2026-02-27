import os
import json
import numpy as np
import pandas as pd

from src.simulate import simulate_to_steady_state
from src.model import rates


def get_tof(T, P_CO, base_params, t_final=80.0):
    params = dict(base_params)
    params["T"] = float(T)
    params["PCO"] = float(P_CO)

    sol, theta_ss, theta_star_ss = simulate_to_steady_state(params, t_final=t_final)
    r1, r2, r3, r4 = rates(theta_ss, params)
    return float(r4)


def main():
    os.makedirs("data", exist_ok=True)

    base_params = {
        "PO2": 0.2,
        "PCO2": 0.0,
        "A": 1e3,
        "E1f": 0.35, "E1r": 0.60,
        "E2f": 0.55, "E2r": 1.00,
        "E3f": 0.70, "E3r": 0.90,
        "E4f": 0.70, "E4r": 1.20,
        "PCO": 1.0,
        "T": 600.0,
    }

    T_values = np.linspace(400, 900, 15)
    P_values = np.linspace(0.05, 3.0, 25)

    rows = []
    failures = 0

    for T in T_values:
        for P in P_values:
            try:
                tof = get_tof(T, P, base_params)
                rows.append([float(T), float(P), float(tof)])
            except Exception:
                failures += 1
                rows.append([float(T), float(P), np.nan])

    df = pd.DataFrame(rows, columns=["T", "PCO", "TOF"])
    out_csv = "data/dataset.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "T_min": float(T_values.min()),
        "T_max": float(T_values.max()),
        "n_T": int(len(T_values)),
        "PCO_min": float(P_values.min()),
        "PCO_max": float(P_values.max()),
        "n_PCO": int(len(P_values)),
        "t_final": 80.0,
        "base_params": base_params,
        "rows": int(len(df)),
        "failures": int(failures),
    }
    out_meta = "data/dataset_meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out_csv} rows: {len(df)} (failures: {failures})")
    print(f"Saved: {out_meta}")


if __name__ == "__main__":
    main()