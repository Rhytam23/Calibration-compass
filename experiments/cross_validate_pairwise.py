"""
CalibrationCompass - Leave-One-Circuit-Out Pairwise Evaluation

This experiment evaluates the pairwise candidate model across
ALL circuits instead of using only one fixed train/test split.

For every circuit:
    - Train on all other circuits
    - Test on the held-out circuit

Each circuit contains 3 backends and 6 candidates.

Therefore:
    10 circuits x 3 backends = 30 decisions

We compare:
    1. ESP
    2. Pairwise ML

This gives a much stronger estimate of generalization.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "results"
    / "candidate_features_v2.csv"
)

RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42


# ============================================================
# HELPER
# ============================================================

def find_column(df, names, description):

    for name in names:

        if name in df.columns:
            return name

    raise ValueError(
        f"\nCould not find {description}.\n"
        f"Tried: {names}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# START
# ============================================================

print("=" * 90)
print("CALIBRATIONCOMPASS - LEAVE-ONE-CIRCUIT-OUT PAIRWISE")
print("=" * 90)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
)

print(
    f"\nRows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# IDENTIFY IMPORTANT COLUMNS
# ============================================================

circuit_col = find_column(
    df,
    ["circuit_id", "circuit"],
    "circuit ID"
)

backend_col = find_column(
    df,
    ["backend"],
    "backend"
)

candidate_col = find_column(
    df,
    ["candidate_seed", "seed", "candidate"],
    "candidate seed"
)

fidelity_col = find_column(
    df,
    ["actual_fidelity", "fidelity"],
    "actual fidelity"
)

esp_col = find_column(
    df,
    ["esp", "ESP", "esp_score"],
    "ESP"
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

df = df.rename(
    columns={
        circuit_col: "circuit_id",
        backend_col: "backend",
        candidate_col: "candidate_seed",
        fidelity_col: "actual_fidelity",
        esp_col: "esp",
    }
)


# ============================================================
# REMOVE DUPLICATE COLUMNS
# ============================================================

if df.columns.duplicated().any():

    duplicates = df.columns[
        df.columns.duplicated()
    ].tolist()

    print(
        "\nRemoving duplicate columns:"
    )

    print(
        sorted(set(duplicates))
    )

    df = df.loc[
        :,
        ~df.columns.duplicated(
            keep="first"
        )
    ].copy()


# ============================================================
# CLEAN DATA
# ============================================================

df["circuit_id"] = pd.to_numeric(
    df["circuit_id"],
    errors="coerce"
)

df["candidate_seed"] = pd.to_numeric(
    df["candidate_seed"],
    errors="coerce"
)

df["actual_fidelity"] = pd.to_numeric(
    df["actual_fidelity"],
    errors="coerce"
)

df["esp"] = pd.to_numeric(
    df["esp"],
    errors="coerce"
)

df["backend"] = (
    df["backend"]
    .astype(str)
)


df = df.dropna(
    subset=[
        "circuit_id",
        "backend",
        "candidate_seed",
        "actual_fidelity",
        "esp",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    by=[
        "circuit_id",
        "backend",
        "candidate_seed",
    ]
).reset_index(
    drop=True
)


# ============================================================
# BUILD FEATURES
# ============================================================

DROP_COLUMNS = {
    "circuit_id",
    "candidate_seed",
    "actual_fidelity",
}


# Remove raw physical IDs.
for column in df.columns:

    lower = column.lower()

    if lower == "physical_qubits":

        DROP_COLUMNS.add(column)

    elif lower == "mapping":

        DROP_COLUMNS.add(column)

    elif lower.endswith("_physical"):

        DROP_COLUMNS.add(column)


feature_df = df.drop(
    columns=[
        column
        for column in DROP_COLUMNS
        if column in df.columns
    ],
    errors="ignore"
).copy()


# ============================================================
# DUPLICATE FEATURES
# ============================================================

if feature_df.columns.duplicated().any():

    feature_df = feature_df.loc[
        :,
        ~feature_df.columns.duplicated(
            keep="first"
        )
    ].copy()


# ============================================================
# MISSING VALUE INDICATORS
# ============================================================

missing_columns = []

for column in feature_df.columns:

    if feature_df[column].isna().any():

        missing_columns.append(
            column
        )

        feature_df[
            f"{column}__missing"
        ] = (
            feature_df[column]
            .isna()
            .astype(float)
        )


print(
    f"\nFeatures with missing values: "
    f"{len(missing_columns)}"
)


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

categorical_columns = (
    feature_df
    .select_dtypes(
        include=[
            "object",
            "string",
            "category",
        ]
    )
    .columns
    .tolist()
)


if categorical_columns:

    print(
        "\nCategorical columns encoded:"
    )

    for column in categorical_columns:

        print(
            f"  - {column}"
        )

    feature_df = pd.get_dummies(
        feature_df,
        columns=categorical_columns,
        dummy_na=True
    )


# ============================================================
# FINAL DUPLICATE CHECK
# ============================================================

if feature_df.columns.duplicated().any():

    feature_df = feature_df.loc[
        :,
        ~feature_df.columns.duplicated(
            keep="first"
        )
    ].copy()


# ============================================================
# NUMERIC
# ============================================================

non_numeric = (
    feature_df
    .select_dtypes(
        exclude=[
            np.number,
            "bool",
        ]
    )
    .columns
    .tolist()
)


if non_numeric:

    print(
        "\nDropping remaining non-numeric features:"
    )

    for column in non_numeric:

        print(
            f"  - {column}"
        )

    feature_df = feature_df.drop(
        columns=non_numeric
    )


feature_df = feature_df.astype(
    float
)


print(
    f"\nFinal feature count: "
    f"{feature_df.shape[1]}"
)


# ============================================================
# PAIR CREATION FUNCTION
# ============================================================

def create_pairwise_training_data(
    training_df,
    feature_df
):

    X_pairs = []

    y_pairs = []


    for (circuit, backend), group in training_df.groupby(
        ["circuit_id", "backend"],
        sort=True
    ):

        indices = (
            group.index
            .tolist()
        )


        for idx_a, idx_b in combinations(
            indices,
            2
        ):

            features_a = (
                feature_df
                .loc[idx_a]
                .values
            )

            features_b = (
                feature_df
                .loc[idx_b]
                .values
            )


            fidelity_a = (
                training_df.loc[
                    idx_a,
                    "actual_fidelity"
                ]
            )

            fidelity_b = (
                training_df.loc[
                    idx_b,
                    "actual_fidelity"
                ]
            )


            if (
                fidelity_a
                ==
                fidelity_b
            ):

                continue


            difference = (
                features_a
                -
                features_b
            )


            if fidelity_a > fidelity_b:

                label = 1

            else:

                label = 0


            # A - B
            X_pairs.append(
                difference
            )

            y_pairs.append(
                label
            )


            # B - A
            X_pairs.append(
                -difference
            )

            y_pairs.append(
                1 - label
            )


    return (
        np.asarray(
            X_pairs,
            dtype=float
        ),
        np.asarray(
            y_pairs,
            dtype=int
        ),
    )


# ============================================================
# ALL CIRCUITS
# ============================================================

circuits = sorted(
    df["circuit_id"]
    .unique()
    .tolist()
)


print(
    f"\nTotal circuits: "
    f"{len(circuits)}"
)

print(
    f"Total backend decisions: "
    f"{len(circuits) * 3}"
)


# ============================================================
# STORAGE
# ============================================================

all_results = []

all_pairwise_training_sizes = []


# ============================================================
# LEAVE-ONE-CIRCUIT-OUT LOOP
# ============================================================

for counter, test_circuit in enumerate(
    circuits,
    start=1
):

    print(
        f"\n[{counter}/{len(circuits)}] "
        f"Testing circuit {int(test_circuit)}..."
    )


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_mask = (
        df["circuit_id"]
        !=
        test_circuit
    )

    test_mask = (
        df["circuit_id"]
        ==
        test_circuit
    )


    train_df = df[
        train_mask
    ].copy()

    test_df = df[
        test_mask
    ].copy()


    # --------------------------------------------------------
    # Pairwise training data
    # --------------------------------------------------------

    X_train, y_train = (
        create_pairwise_training_data(
            train_df,
            feature_df
        )
    )


    all_pairwise_training_sizes.append(
        len(X_train)
    )


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = XGBClassifier(

        objective="binary:logistic",

        n_estimators=200,

        learning_rate=0.03,

        max_depth=2,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.7,

        reg_alpha=0.5,

        reg_lambda=5.0,

        random_state=RANDOM_STATE,

        eval_metric="logloss",

        tree_method="hist",
    )


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Evaluate each backend
    # --------------------------------------------------------

    for backend, group in test_df.groupby(
        "backend",
        sort=True
    ):

        indices = (
            group.index
            .tolist()
        )


        candidate_scores = []


        # ----------------------------------------------------
        # Score every candidate by predicted pairwise wins
        # ----------------------------------------------------

        for idx_a in indices:

            win_probabilities = []


            for idx_b in indices:

                if idx_a == idx_b:
                    continue


                features_a = (
                    feature_df
                    .loc[idx_a]
                    .values
                )

                features_b = (
                    feature_df
                    .loc[idx_b]
                    .values
                )


                difference = (
                    features_a
                    -
                    features_b
                ).reshape(
                    1,
                    -1
                )


                probability = (
                    model
                    .predict_proba(
                        difference
                    )[0, 1]
                )


                win_probabilities.append(
                    probability
                )


            pairwise_score = np.mean(
                win_probabilities
            )


            candidate_scores.append(
                {
                    "candidate_seed":
                        df.loc[
                            idx_a,
                            "candidate_seed"
                        ],

                    "pairwise_score":
                        pairwise_score,

                    "esp":
                        df.loc[
                            idx_a,
                            "esp"
                        ],

                    "actual_fidelity":
                        df.loc[
                            idx_a,
                            "actual_fidelity"
                        ],
                }
            )


        # ----------------------------------------------------
        # Oracle
        # ----------------------------------------------------

        oracle = max(
            candidate_scores,
            key=lambda x:
                x["actual_fidelity"]
        )


        # ----------------------------------------------------
        # ESP
        # ----------------------------------------------------

        esp = max(
            candidate_scores,
            key=lambda x:
                x["esp"]
        )


        # ----------------------------------------------------
        # Pairwise
        # ----------------------------------------------------

        pairwise = max(
            candidate_scores,
            key=lambda x:
                x["pairwise_score"]
        )


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_results.append(
            {
                "circuit_id":
                    test_circuit,

                "backend":
                    backend,

                "oracle_candidate":
                    oracle[
                        "candidate_seed"
                    ],

                "oracle_fidelity":
                    oracle[
                        "actual_fidelity"
                    ],

                "esp_candidate":
                    esp[
                        "candidate_seed"
                    ],

                "esp_fidelity":
                    esp[
                        "actual_fidelity"
                    ],

                "pairwise_candidate":
                    pairwise[
                        "candidate_seed"
                    ],

                "pairwise_fidelity":
                    pairwise[
                        "actual_fidelity"
                    ],
            }
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    all_results
)


# ============================================================
# ACCURACY
# ============================================================

results["esp_correct"] = (
    results["esp_candidate"]
    ==
    results["oracle_candidate"]
)

results["pairwise_correct"] = (
    results["pairwise_candidate"]
    ==
    results["oracle_candidate"]
)


esp_accuracy = (
    results["esp_correct"]
    .mean()
)

pairwise_accuracy = (
    results["pairwise_correct"]
    .mean()
)


# ============================================================
# REGRET
# ============================================================

results["esp_regret"] = (
    results["oracle_fidelity"]
    -
    results["esp_fidelity"]
)

results["pairwise_regret"] = (
    results["oracle_fidelity"]
    -
    results["pairwise_fidelity"]
)


# ============================================================
# IMPROVEMENT COUNTS
# ============================================================

improved = results[
    results["pairwise_regret"]
    <
    results["esp_regret"]
]

same = results[
    results["pairwise_regret"]
    ==
    results["esp_regret"]
]

worse = results[
    results["pairwise_regret"]
    >
    results["esp_regret"]
]


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("LEAVE-ONE-CIRCUIT-OUT RESULTS")
print("=" * 90)

print(
    f"\nTotal test decisions: "
    f"{len(results)}"
)

print(
    f"\nESP selection accuracy: "
    f"{esp_accuracy:.2%}"
)

print(
    f"Pairwise selection accuracy: "
    f"{pairwise_accuracy:.2%}"
)

print(
    f"\nESP average regret: "
    f"{results['esp_regret'].mean():.6f}"
)

print(
    f"Pairwise average regret: "
    f"{results['pairwise_regret'].mean():.6f}"
)

print(
    f"\nESP maximum regret: "
    f"{results['esp_regret'].max():.6f}"
)

print(
    f"Pairwise maximum regret: "
    f"{results['pairwise_regret'].max():.6f}"
)

print(
    f"\nPairwise improved over ESP: "
    f"{len(improved)}/{len(results)}"
)

print(
    f"Pairwise same as ESP: "
    f"{len(same)}/{len(results)}"
)

print(
    f"Pairwise worse than ESP: "
    f"{len(worse)}/{len(results)}"
)

print(
    f"\nAverage pairwise training samples per fold: "
    f"{np.mean(all_pairwise_training_sizes):.0f}"
)


# ============================================================
# BEST IMPROVEMENTS
# ============================================================

if len(improved) > 0:

    print("\n")
    print("=" * 90)
    print("BEST PAIRWISE IMPROVEMENTS OVER ESP")
    print("=" * 90)


    improvement_table = (
        improved
        .copy()
    )

    improvement_table[
        "improvement"
    ] = (
        improvement_table[
            "esp_regret"
        ]
        -
        improvement_table[
            "pairwise_regret"
        ]
    )


    improvement_table = (
        improvement_table
        .sort_values(
            "improvement",
            ascending=False
        )
        .head(10)
    )


    print(
        improvement_table[
            [
                "circuit_id",
                "backend",
                "oracle_candidate",
                "esp_candidate",
                "pairwise_candidate",
                "esp_regret",
                "pairwise_regret",
                "improvement",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# BIGGEST FAILURES
# ============================================================

print("\n")
print("=" * 90)
print("BIGGEST PAIRWISE FAILURES")
print("=" * 90)


failure_table = (
    results
    .sort_values(
        "pairwise_regret",
        ascending=False
    )
    .head(10)
)


print(
    failure_table[
        [
            "circuit_id",
            "backend",
            "oracle_candidate",
            "esp_candidate",
            "pairwise_candidate",
            "oracle_fidelity",
            "esp_fidelity",
            "pairwise_fidelity",
            "esp_regret",
            "pairwise_regret",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    RESULTS_DIR
    /
    "cross_validation_pairwise_results.csv"
)

results.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY FILE
# ============================================================

summary_file = (
    RESULTS_DIR
    /
    "cross_validation_pairwise_summary.csv"
)


summary = pd.DataFrame(
    {
        "metric": [
            "total_decisions",
            "esp_accuracy",
            "pairwise_accuracy",
            "esp_average_regret",
            "pairwise_average_regret",
            "esp_max_regret",
            "pairwise_max_regret",
            "pairwise_improved",
            "pairwise_same",
            "pairwise_worse",
        ],

        "value": [
            len(results),
            esp_accuracy,
            pairwise_accuracy,
            results[
                "esp_regret"
            ].mean(),
            results[
                "pairwise_regret"
            ].mean(),
            results[
                "esp_regret"
            ].max(),
            results[
                "pairwise_regret"
            ].max(),
            len(improved),
            len(same),
            len(worse),
        ],
    }
)


summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    f"\n{output_file}"
)

print(
    summary_file
)

print("\n")
print("=" * 90)
print("DONE")
print("=" * 90)