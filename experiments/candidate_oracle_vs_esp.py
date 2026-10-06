import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import pandas as pd


# ============================================================
# 1. LOAD CANDIDATE DATA
# ============================================================

df = pd.read_csv(
    "results/candidate_benchmark.csv"
)

print("=" * 75)
print("CALIBRATIONCOMPASS - CANDIDATE ORACLE VS ESP")
print("=" * 75)

print(
    f"\nRows loaded: {len(df)}"
)


# ============================================================
# 2. ORACLE CANDIDATE
# ============================================================
#
# For each circuit + backend:
#
# choose the candidate with the highest actual fidelity.
# ============================================================

oracle = df.loc[
    df.groupby(
        [
            "circuit_id",
            "backend"
        ]
    )["fidelity"].idxmax()
].copy()


oracle = oracle.rename(
    columns={
        "candidate_seed":
            "oracle_seed",

        "fidelity":
            "oracle_fidelity"
    }
)


# ============================================================
# 3. ESP CANDIDATE
# ============================================================
#
# ESP is our quick physics-based estimate.
#
# Choose the candidate with the highest ESP.
# ============================================================

esp = df.loc[
    df.groupby(
        [
            "circuit_id",
            "backend"
        ]
    )["esp"].idxmax()
].copy()


esp = esp.rename(
    columns={
        "candidate_seed":
            "esp_seed",

        "fidelity":
            "esp_actual_fidelity",

        "esp":
            "esp_score"
    }
)


# ============================================================
# 4. COMBINE
# ============================================================

comparison = oracle[
    [
        "circuit_id",
        "backend",
        "oracle_seed",
        "oracle_fidelity"
    ]
].merge(
    esp[
        [
            "circuit_id",
            "backend",
            "esp_seed",
            "esp_score",
            "esp_actual_fidelity"
        ]
    ],
    on=[
        "circuit_id",
        "backend"
    ]
)


# ============================================================
# 5. CHECK WHETHER ESP FOUND THE ORACLE
# ============================================================

comparison["esp_correct"] = (
    comparison["oracle_seed"]
    ==
    comparison["esp_seed"]
)


# ============================================================
# 6. CALCULATE FIDELITY REGRET
# ============================================================
#
# How much fidelity did we lose by using ESP
# instead of the actual best candidate?
# ============================================================

comparison["fidelity_regret"] = (
    comparison["oracle_fidelity"]
    -
    comparison["esp_actual_fidelity"]
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("RESULTS")
print("=" * 75)

print(
    f"\nCandidate decisions: "
    f"{len(comparison)}"
)

print(
    f"\nESP selected the oracle candidate in: "
    f"{comparison['esp_correct'].sum()}/"
    f"{len(comparison)} cases"
)

print(
    f"ESP candidate-selection accuracy: "
    f"{comparison['esp_correct'].mean():.2%}"
)

print(
    f"\nAverage fidelity regret: "
    f"{comparison['fidelity_regret'].mean():.6f}"
)

print(
    f"Maximum fidelity regret: "
    f"{comparison['fidelity_regret'].max():.6f}"
)


# ============================================================
# 8. SHOW THE BIGGEST ESP FAILURES
# ============================================================

failures = comparison[
    comparison["esp_correct"] == False
].copy()


print("\n")
print("=" * 75)
print("BIGGEST ESP CANDIDATE FAILURES")
print("=" * 75)

print(
    f"\nTotal failures: "
    f"{len(failures)}"
)


if len(failures) > 0:

    failures = failures.sort_values(
        "fidelity_regret",
        ascending=False
    )

    print(
        failures[
            [
                "circuit_id",
                "backend",
                "oracle_seed",
                "oracle_fidelity",
                "esp_seed",
                "esp_score",
                "esp_actual_fidelity",
                "fidelity_regret"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

comparison.to_csv(
    "results/candidate_oracle_vs_esp.csv",
    index=False
)


print("\nSaved:")
print(
    "results/candidate_oracle_vs_esp.csv"
)


print("\n")
print("=" * 75)
print("EXPERIMENT COMPLETE")
print("=" * 75)