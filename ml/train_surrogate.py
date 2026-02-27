"""
ml/surrogate.py

Learn a surrogate model for microkinetic TOF = f(T, PCO)

Supports:
- Random split (interpolation)
- PCO hold-out (extrapolation)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ----------------------------
# SETTINGS
# ----------------------------
SPLIT_TYPE = "random"   # "random" or "holdout"
PCO_CUTOFF = 1.0
EPS = 1e-12
RANDOM_STATE = 42


# ----------------------------
# Feature engineering
# ----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # minimal sanity
    df = df[df["TOF"] > 0].copy()
    if (df["PCO"] <= 0).any():
        raise ValueError("PCO must be > 0 to use log(PCO).")

    df["invT"] = 1.0 / df["T"]
    df["logPCO"] = np.log(df["PCO"])
    df["logTOF"] = np.log(df["TOF"] + EPS)
    return df


# ----------------------------
# Plots
# ----------------------------
def parity_plot(true_tof: np.ndarray, pred_tof: np.ndarray, title: str, outpath: str):
    plt.figure()
    plt.scatter(true_tof, pred_tof)

    lo = min(true_tof.min(), pred_tof.min())
    hi = max(true_tof.max(), pred_tof.max())
    plt.plot([lo, hi], [lo, hi])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Microkinetic TOF")
    plt.ylabel("ML Predicted TOF")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def error_vs_pco_plot(pco: np.ndarray, true_log: np.ndarray, pred_log: np.ndarray, title: str, outpath: str):
    abs_err = np.abs(true_log - pred_log)

    plt.figure()
    plt.scatter(pco, abs_err)
    plt.xlabel("PCO")
    plt.ylabel("Absolute Error (logTOF)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ----------------------------
# Reporting
# ----------------------------
def save_metrics(path: str, info: dict):
    lines = []
    for k, v in info.items():
        if isinstance(v, float):
            # mix: some want fixed, some scientific
            if "MAE" in k:
                lines.append(f"{k}: {v:.3e}")
            else:
                lines.append(f"{k}: {v:.6f}")
        else:
            lines.append(f"{k}: {v}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs("figures/ml", exist_ok=True)

    df = pd.read_csv("data/dataset.csv")
    df = add_features(df)

    X = df[["invT", "logPCO"]]
    y = df["logTOF"]

    # ------------------------
    # Split
    # ------------------------
    if SPLIT_TYPE == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        split_name = "random"
        split_title = "Random split (interpolation)"

    elif SPLIT_TYPE == "holdout":
        train_mask = df["PCO"] <= PCO_CUTOFF
        test_mask = df["PCO"] > PCO_CUTOFF

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        split_name = f"holdout_pco_{str(PCO_CUTOFF).replace('.', 'p')}"
        split_title = f"PCO hold-out (extrapolation): PCO > {PCO_CUTOFF}"

    else:
        raise ValueError('Unknown SPLIT_TYPE. Use "random" or "holdout".')

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}", end="")
    if SPLIT_TYPE == "holdout":
        print(f" | PCO cutoff: {PCO_CUTOFF}")
    else:
        print()

    # keep test metadata aligned with X_test/y_test
    df_test = df.loc[X_test.index]

    # ------------------------
    # Model
    # ------------------------
    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)

    # ------------------------
    # Metrics (log space primary)
    # ------------------------
    r2_log = r2_score(y_test, pred_log)
    mae_log = mean_absolute_error(y_test, pred_log)

    true_tof = np.exp(y_test.values)
    pred_tof = np.exp(pred_log)

    r2_tof = r2_score(true_tof, pred_tof)
    mae_tof = mean_absolute_error(true_tof, pred_tof)

    print(f"R2  (logTOF): {r2_log:.3f}")
    print(f"MAE (logTOF): {mae_log:.3e}")
    print(f"R2  (TOF):    {r2_tof:.3f}")
    print(f"MAE (TOF):    {mae_tof:.3e}")

    # ------------------------
    # Save figures
    # ------------------------
    parity_path = f"figures/ml/parity_{split_name}.png"
    parity_plot(true_tof, pred_tof, title=f"Parity (log-log) | {split_title}", outpath=parity_path)
    print(f"Saved: {parity_path}")

    err_path = f"figures/ml/error_vs_pco_{split_name}.png"
    error_vs_pco_plot(
        pco=df_test["PCO"].values,
        true_log=y_test.values,
        pred_log=pred_log,
        title=f"Error vs PCO | {split_title}",
        outpath=err_path,
    )
    print(f"Saved: {err_path}")

    # ------------------------
    # Save metrics to file
    # ------------------------
    metrics_path = f"figures/ml/metrics_{split_name}.txt"
    metrics = {
        "split_type": SPLIT_TYPE,
        "pco_cutoff": PCO_CUTOFF if SPLIT_TYPE == "holdout" else "NA",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "R2_logTOF": float(r2_log),
        "MAE_logTOF": float(mae_log),
        "R2_TOF": float(r2_tof),
        "MAE_TOF": float(mae_tof),
        "parity_plot": parity_path,
        "error_plot": err_path,
        "dataset_path": "data/dataset.csv",
        "features": "invT=1/T, logPCO=log(PCO), target=log(TOF)",
        "model": "GradientBoostingRegressor(random_state=42)",
    }
    save_metrics(metrics_path, metrics)
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()