import pandas as pd

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

FILE = BASE / "results" / "multi_circuit_validation.csv"

df = pd.read_csv(FILE)

print("=" * 80)
print("CALIBRATIONCOMPASS - WITHIN-CIRCUIT MAPPING ANALYSIS")
print("=" * 80)

rows = []

for (circuit, backend), group in df.groupby(["circuit", "backend"]):

    fidelity_readout_corr = group["fidelity"].corr(
        group["avg_readout"]
    )

    fidelity_max_readout_corr = group["fidelity"].corr(
        group["max_readout"]
    )

    spread = (
        group["fidelity"].max()
        -
        group["fidelity"].min()
    )

    rows.append({
        "circuit": circuit,
        "backend": backend,
        "avg_readout_corr": fidelity_readout_corr,
        "max_readout_corr": fidelity_max_readout_corr,
        "fidelity_spread": spread,
        "best": group["fidelity"].max(),
        "worst": group["fidelity"].min(),
    })

results = pd.DataFrame(rows)

print("\n")
print(results.to_string(index=False))

print("\nAverage fidelity spread:")
print(
    f"{results['fidelity_spread'].mean():.6f}"
)

print("\nLargest mapping sensitivity:")
print(
    results.sort_values(
        "fidelity_spread",
        ascending=False
    ).head(3).to_string(index=False)
)

out = BASE / "results" / "within_circuit_analysis.csv"

results.to_csv(out, index=False)

print(f"\nSaved: {out}")
print("\nDONE")