import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import pandas as pd


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

df = pd.read_csv(
    "results/calibration_dataset_v2.csv"
)


print("=" * 70)
print("CALIBRATIONCOMPASS - ORACLE VS ESP")
print("=" * 70)


# ============================================================
# 2. FIND THE ORACLE WINNER
# ============================================================
#
# The oracle chooses the backend with the highest
# actual simulated fidelity.
#
# This is our "perfect answer".
# ============================================================

oracle_rows = df.loc[
    df.groupby(
        ["circuit_id", "condition"]
    )["fidelity"].idxmax()
].copy()


oracle_rows = oracle_rows.rename(
    columns={
        "backend": "oracle_backend",
        "fidelity": "oracle_fidelity"
    }
)


# ============================================================
# 3. FIND THE ESP WINNER
# ============================================================
#
# ESP is a quick estimate.
# Higher ESP = better predicted execution quality.
# ============================================================

esp_rows = df.loc[
    df.groupby(
        ["circuit_id", "condition"]
    )["esp"].idxmax()
].copy()


esp_rows = esp_rows.rename(
    columns={
        "backend": "esp_backend",
        "fidelity": "esp_actual_fidelity",
        "esp": "esp_score"
    }
)


# ============================================================
# 4. COMBINE ORACLE AND ESP RESULTS
# ============================================================

comparison = pd.merge(
    oracle_rows[
        [
            "circuit_id",
            "condition",
            "oracle_backend",
            "oracle_fidelity"
        ]
    ],

    esp_rows[
        [
            "circuit_id",
            "condition",
            "esp_backend",
            "esp_score",
            "esp_actual_fidelity"
        ]
    ],

    on=[
        "circuit_id",
        "condition"
    ]
)


# ============================================================
# 5. CHECK WHETHER ESP CHOSE THE ORACLE WINNER
# ============================================================

comparison["esp_correct"] = (
    comparison["oracle_backend"]
    ==
    comparison["esp_backend"]
)


# ============================================================
# 6. CALCULATE FIDELITY REGRET
# ============================================================
#
# Regret = how much fidelity we lost by using ESP
# instead of the oracle.
#
# Example:
#
# Oracle = 0.95
# ESP     = 0.91
#
# Regret = 0.04
# ============================================================

comparison["fidelity_regret"] = (
    comparison["oracle_fidelity"]
    -
    comparison["esp_actual_fidelity"]
)


# ============================================================
# 7. PRINT MAIN RESULTS
# ============================================================

print("\nTOTAL DECISIONS")
print("-" * 70)

print(
    f"Total circuit-condition scenarios: "
    f"{len(comparison)}"
)


correct_count = comparison[
    "esp_correct"
].sum()

accuracy = correct_count / len(comparison)

print(
    f"ESP selected the oracle backend in: "
    f"{correct_count}/{len(comparison)} cases"
)

print(
    f"ESP backend-selection accuracy: "
    f"{accuracy:.2%}"
)


# ============================================================
# 8. AVERAGE FIDELITY REGRET
# ============================================================

average_regret = comparison[
    "fidelity_regret"
].mean()

maximum_regret = comparison[
    "fidelity_regret"
].max()


print("\nFIDELITY REGRET")
print("-" * 70)

print(
    f"Average fidelity lost by using ESP: "
    f"{average_regret:.6f}"
)

print(
    f"Maximum fidelity lost by using ESP: "
    f"{maximum_regret:.6f}"
)


# ============================================================
# 9. RESULTS BY CONDITION
# ============================================================

print("\nRESULTS BY HARDWARE CONDITION")
print("-" * 70)

condition_summary = (
    comparison
    .groupby("condition")
    .agg(
        esp_accuracy=("esp_correct", "mean"),
        average_regret=("fidelity_regret", "mean"),
        maximum_regret=("fidelity_regret", "max")
    )
)


condition_summary["esp_accuracy"] *= 100

print(
    condition_summary
)


# ============================================================
# 10. SHOW CASES WHERE ESP FAILED
# ============================================================

failed = comparison[
    comparison["esp_correct"] == False
].copy()


print("\nCASES WHERE ESP CHOSE THE WRONG BACKEND")
print("-" * 70)

print(
    f"Number of failures: {len(failed)}"
)


if len(failed) > 0:

    failed = failed.sort_values(
        "fidelity_regret",
        ascending=False
    )

    print(
        failed[
            [
                "circuit_id",
                "condition",
                "oracle_backend",
                "oracle_fidelity",
                "esp_backend",
                "esp_score",
                "esp_actual_fidelity",
                "fidelity_regret"
            ]
        ].head(20).to_string(index=False)
    )

else:

    print(
        "ESP matched the oracle in every case."
    )


# ============================================================
# 11. SAVE THE COMPARISON
# ============================================================

comparison.to_csv(
    "results/oracle_vs_esp.csv",
    index=False
)


print("\nSaved to:")
print("results/oracle_vs_esp.csv")


print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)