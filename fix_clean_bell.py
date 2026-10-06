from pathlib import Path

import pandas as pd


print("=" * 90)
print("CALIBRATIONCOMPASS - FINAL BELL DATA REPAIR")
print("=" * 90)


BASE = Path(__file__).resolve().parent

CLEAN_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_clean.csv"
)

FIXED_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_fixed.csv"
)

OUTPUT_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_final2.csv"
)


if not CLEAN_FILE.exists():
    print("Clean dataset not found:")
    print(CLEAN_FILE)
    raise SystemExit

if not FIXED_FILE.exists():
    print("Previously fixed dataset not found:")
    print(FIXED_FILE)
    raise SystemExit


clean = pd.read_csv(CLEAN_FILE)
fixed = pd.read_csv(FIXED_FILE)


# Get the already-correct Bell fidelities
fixed_bell = fixed[
    fixed["circuit_type"] == "Bell"
][
    [
        "backend",
        "candidate",
        "fidelity"
    ]
].copy()


fixed_bell["candidate"] = (
    fixed_bell["candidate"]
    .astype(int)
)


clean["candidate"] = (
    clean["candidate"]
    .astype(int)
)


# Match each Bell row
for _, row in fixed_bell.iterrows():

    mask = (
        (clean["circuit_type"] == "Bell")
        &
        (clean["backend"] == row["backend"])
        &
        (clean["candidate"] == row["candidate"])
    )

    matches = clean.index[mask].tolist()

    if len(matches) != 1:
        print(
            "ERROR: expected exactly one matching row."
        )
        print(row)
        raise SystemExit

    clean.at[
        matches[0],
        "fidelity"
    ] = row["fidelity"]


clean.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 90)
print("FINAL DATASET SUMMARY")
print("=" * 90)

print(
    "Total rows:",
    len(clean)
)

print()

print(
    clean.groupby(
        "circuit_type"
    )["fidelity"].agg(
        ["count", "min", "max", "mean"]
    )
)

print()

print(
    clean.groupby(
        "backend"
    )["fidelity"].agg(
        ["count", "min", "max", "mean"]
    )
)

print()
print("Bell rows repaired:", len(fixed_bell))

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("DONE")
