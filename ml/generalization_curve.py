import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/dataset.csv")
df = df[df["TOF"] > 0].copy()

# Physics features
df["invT"] = 1.0 / df["T"]
df["logPCO"] = np.log(df["PCO"])
df["invT_x_logPCO"] = df["invT"] * df["logPCO"]
df["logTOF"] = np.log(df["TOF"])

X_all = df[["invT", "logPCO", "invT_x_logPCO"]]
y_all = df["logTOF"]

# ----------------------------
# Cutoffs to test
# ----------------------------
cutoffs = [0.8, 1.0, 1.2, 1.4, 1.6]
r2_scores = []

for cutoff in cutoffs:
    train_mask = df["PCO"] <= cutoff
    test_mask  = df["PCO"] > cutoff

    if test_mask.sum() == 0:
        r2_scores.append(np.nan)
        continue

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]

    X_test  = X_all[test_mask]
    y_test  = y_all[test_mask]

    model = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.08,
        max_iter=400
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    r2_scores.append(r2)

    print(f"Cutoff {cutoff}: R2 (log) = {r2:.3f}")

# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(cutoffs, r2_scores, marker="o")
plt.xlabel("PCO Training Cutoff")
plt.ylabel("R2 (logTOF)")
plt.title("Surrogate Generalization vs PCO Cutoff")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/ml/generalization_curve.png", dpi=300)
plt.close()

print("Saved: figures/ml/generalization_curve.png")