"""
CalibrationCompass - Pairwise Case Analysis

Diagnostic experiment.

We inspect the most important cases where:
    - Pairwise beats ESP
    - Pairwise hurts ESP

The goal is to understand which calibration/mapping
features distinguish successful corrections.

This script does NOT train a new model.
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "results"
    / "candidate_features_v2.csv"
)

RESULTS_FILE = (
    BASE_DIR
    / "results"
    / "cross_validation_pairwise_results.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 90)
print("CALIBRATIONCOMPASS - PAIRWISE CASE ANALYSIS")
print("=" * 90)


# ============================================================
# LOAD
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nMissing:\n{DATA_FILE}"
    )


if not RESULTS_FILE.exists():

    raise FileNotFoundError(
        f"\nMissing:\n{RESULTS_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
)

results = pd.read_csv(
    RESULTS_FILE
)


print(
    f"\nCandidate dataset rows: "
    f"{len(df)}"
)

print(
    f"Cross-validation decisions: "
    f"{len(results)}"
)


# ============================================================
# FIND CASES
# ============================================================

results["improvement"] = (
    results["esp_regret"]
    -
    results["pairwise_regret"]
)


successful = (
    results
    .sort_values(
        "improvement",
        ascending=False
    )
    .head(5)
)


failures = (
    results
    .sort_values(
        "pairwise_regret",
        ascending=False
    )
    .head(5)
)


# ============================================================
# FEATURES TO INSPECT
# ============================================================

preferred_features = [
    "esp",
    "actual_fidelity",

    "transpiled_depth",
    "one_qubit_gates",
    "two_qubit_gates",

    "edge_exposure",
    "max_edge_error",
    "avg_cz_error",
    "max_cz_error",

    "weighted_readout_total",
    "weighted_readout_2q",
    "avg_readout_error",
    "max_readout_error",
    "sum_readout_error",

    "top1_readout_risk",
    "top2_readout_risk",
    "activity_readout_corr",

    "avg_sx_error",
    "weighted_sx_total",

    "t2_exposure",
    "avg_t1",
    "min_t1",
    "avg_t2",
    "min_t2",
]


available_features = [
    feature
    for feature in preferred_features
    if feature in df.columns
]


# ============================================================
# DISPLAY FUNCTION
# ============================================================

def inspect_case(
    circuit_id,
    backend,
    title,
):

    print("\n")
    print("=" * 90)
    print(title)
    print(
        f"Circuit {int(circuit_id)} | "
        f"{backend}"
    )
    print("=" * 90)


    case = df[
        (df["circuit_id"] == circuit_id)
        &
        (df["backend"] == backend)
    ].copy()


    if len(case) == 0:

        print(
            "\nNo candidate rows found."
        )

        return


    # --------------------------------------------------------
    # Find candidate identities
    # --------------------------------------------------------

    result_row = results[
        (results["circuit_id"] == circuit_id)
        &
        (results["backend"] == backend)
    ]


    if len(result_row) == 0:

        print(
            "\nNo cross-validation result found."
        )

        return


    result_row = (
        result_row
        .iloc[0]
    )


    oracle_candidate = (
        result_row[
            "oracle_candidate"
        ]
    )

    esp_candidate = (
        result_row[
            "esp_candidate"
        ]
    )

    pairwise_candidate = (
        result_row[
            "pairwise_candidate"
        ]
    )


    print(
        f"\nOracle candidate: "
        f"{int(oracle_candidate)}"
    )

    print(
        f"ESP candidate: "
        f"{int(esp_candidate)}"
    )

    print(
        f"Pairwise candidate: "
        f"{int(pairwise_candidate)}"
    )


    # --------------------------------------------------------
    # Candidate ranking
    # --------------------------------------------------------

    case = case.sort_values(
        "actual_fidelity",
        ascending=False
    )


    display_columns = [
        "candidate_seed",
        "actual_fidelity",
        "esp",
    ]


    extra_columns = [
        feature
        for feature in available_features
        if feature not in display_columns
    ]


    print("\nCandidate ranking by actual fidelity:")

    print(
        case[
            display_columns
        ].to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Compare important candidates
    # --------------------------------------------------------

    candidate_ids = [
        oracle_candidate,
        esp_candidate,
        pairwise_candidate,
    ]


    candidate_ids = list(
        dict.fromkeys(
            candidate_ids
        )
    )


    print("\n")
    print(
        "Important calibration/mapping features:"
    )


    for feature in extra_columns:

        values = []


        for candidate in candidate_ids:

            row = case[
                case[
                    "candidate_seed"
                ] == candidate
            ]


            if len(row) == 0:

                values.append(
                    np.nan
                )

            else:

                value = row.iloc[0][
                    feature
                ]

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = np.nan

                values.append(
                    value
                )


        if all(
            pd.isna(value)
            for value in values
        ):

            continue


        print(
            f"\n{feature}"
        )


        labels = []


        for candidate, value in zip(
            candidate_ids,
            values
        ):

            labels.append(
                f"candidate {int(candidate)}: "
                f"{value:.6f}"
                if pd.notna(value)
                else
                f"candidate {int(candidate)}: NaN"
            )


        print(
            " | ".join(labels)
        )


# ============================================================
# SUCCESSFUL CASES
# ============================================================

print("\n")
print("=" * 90)
print("SUCCESSFUL PAIRWISE CORRECTIONS")
print("=" * 90)


for _, row in successful.iterrows():

    if (
        row["pairwise_regret"]
        <
        row["esp_regret"]
    ):

        inspect_case(
            row["circuit_id"],
            row["backend"],
            "PAIRWISE BEAT ESP"
        )


# ============================================================
# FAILURE CASES
# ============================================================

print("\n")
print("=" * 90)
print("PAIRWISE FAILURE CASES")
print("=" * 90)


for _, row in failures.iterrows():

    if (
        row["pairwise_regret"]
        >
        row["esp_regret"]
    ):

        inspect_case(
            row["circuit_id"],
            row["backend"],
            "PAIRWISE HURT ESP"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)


print(
    f"\nPairwise improvements: "
    f"{int((results['improvement'] > 0).sum())}"
)

print(
    f"Pairwise same: "
    f"{int((results['improvement'] == 0).sum())}"
)

print(
    f"Pairwise worse: "
    f"{int((results['improvement'] < 0).sum())}"
)


print("\n")
print("=" * 90)
print("DONE")
print("=" * 90)