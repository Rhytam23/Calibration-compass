import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import pandas as pd


# Load our dataset
df = pd.read_csv("results/calibration_dataset.csv")


print("=" * 70)
print("CALIBRATIONCOMPASS - DATASET INSPECTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Basic information
# ------------------------------------------------------------

print("\nDATASET SIZE")
print("-" * 70)

print(f"Rows:    {len(df)}")
print(f"Columns: {len(df.columns)}")


# ------------------------------------------------------------
# 2. Show all column names
# ------------------------------------------------------------

print("\nCOLUMNS")
print("-" * 70)

for column in df.columns:
    print(column)


# ------------------------------------------------------------
# 3. Check for missing values
# ------------------------------------------------------------

print("\nMISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()

print(missing)


# ------------------------------------------------------------
# 4. Check the fidelity range
# ------------------------------------------------------------

print("\nFIDELITY")
print("-" * 70)

print(f"Minimum: {df['fidelity'].min():.4f}")
print(f"Maximum: {df['fidelity'].max():.4f}")
print(f"Average: {df['fidelity'].mean():.4f}")


# ------------------------------------------------------------
# 5. Average fidelity by backend
# ------------------------------------------------------------

print("\nAVERAGE FIDELITY BY BACKEND")
print("-" * 70)

backend_fidelity = (
    df.groupby("backend")["fidelity"]
    .mean()
    .sort_values(ascending=False)
)

print(backend_fidelity)


# ------------------------------------------------------------
# 6. Find the best backend for each circuit
# ------------------------------------------------------------

print("\nBEST BACKEND FOR EACH CIRCUIT")
print("-" * 70)

best_rows = df.loc[
    df.groupby("circuit_id")["fidelity"].idxmax()
]

for _, row in best_rows.iterrows():

    print(
        f"Circuit {int(row['circuit_id']):2d}"
        f" → {row['backend']:12s}"
        f" → fidelity {row['fidelity']:.4f}"
    )


# ------------------------------------------------------------
# 7. How often does each backend win?
# ------------------------------------------------------------

print("\nBACKEND WIN COUNTS")
print("-" * 70)

winner_counts = best_rows["backend"].value_counts()

print(winner_counts)


# ------------------------------------------------------------
# 8. Compare ESP with actual fidelity
# ------------------------------------------------------------

print("\nESP VS ACTUAL FIDELITY")
print("-" * 70)

print(
    df[["esp", "fidelity"]]
    .corr()
)


# ------------------------------------------------------------
# 9. Show a few rows
# ------------------------------------------------------------

print("\nSAMPLE DATA")
print("-" * 70)

print(
    df.head(10).to_string(index=False)
)


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)