import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FILE = BASE / "results" / "confidence_gated_hybrid_results.csv"

df = pd.read_csv(FILE)

best = None
rows = []

for esp_threshold in np.arange(0.005, 0.026, 0.005):

    for advantage_threshold in np.arange(0.01, 0.041, 0.005):

        selected_regrets = []
        correct = 0
        overrides = 0

        for _, r in df.iterrows():

            use_pairwise = (
                r["pairwise_candidate"] != r["esp_candidate"]
                and r["esp_margin"] <= esp_threshold
                and r["pairwise_advantage"] >= advantage_threshold
            )

            if use_pairwise:
                fidelity = r["pairwise_fidelity"]
                candidate = r["pairwise_candidate"]
                overrides += 1
            else:
                fidelity = r["esp_fidelity"]
                candidate = r["esp_candidate"]

            selected_regrets.append(
                r["oracle_fidelity"] - fidelity
            )

            if candidate == r["oracle_candidate"]:
                correct += 1

        accuracy = correct / len(df)
        regret = np.mean(selected_regrets)

        rows.append({
            "esp_threshold": esp_threshold,
            "advantage_threshold": advantage_threshold,
            "accuracy": accuracy,
            "average_regret": regret,
            "overrides": overrides
        })

        if best is None or regret < best["average_regret"]:
            best = rows[-1]


results = pd.DataFrame(rows)

print("=" * 80)
print("THRESHOLD SENSITIVITY")
print("=" * 80)

print("\nBest configuration (descriptive only: selected on the same")
print("evaluation decisions, so it is optimistic and must not be")
print("used to tune thresholds):")
print(best)

print("\nBest configurations:")
print(
    results.sort_values(
        ["average_regret", "accuracy"],
        ascending=[True, False]
    ).head(10).to_string(index=False)
)

print("\nCurrent configuration:")
print(
    results[
        np.isclose(results["esp_threshold"], 0.01) &
        np.isclose(results["advantage_threshold"], 0.02)
    ].to_string(index=False)
)

out = BASE / "results" / "threshold_sensitivity.csv"
results.to_csv(out, index=False)

print(f"\nSaved: {out}")
print("\nDONE")