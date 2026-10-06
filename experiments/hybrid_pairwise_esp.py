"""
CalibrationCompass - Hybrid ESP + Pairwise Candidate Selection

Strategy:

    ESP is the default.

    Pairwise ML is used only when ESP is uncertain.

ESP uncertainty is measured by the gap between:
    top ESP candidate
    second-best ESP candidate

If the ESP margin is small:
    use Pairwise model

Otherwise:
    keep ESP

The threshold is learned ONLY from the training circuits
inside each leave-one-circuit-out fold.

Evaluation:
    30 circuits x 3 backends = 90 decisions

Compare:
    ESP
    Pairwise
    Hybrid
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
# LOAD DATA
# ============================================================

print("=" * 90)
print("CALIBRATIONCOMPASS - HYBRID ESP + PAIRWISE")
print("=" * 90)


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
# IMPORTANT COLUMNS
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
# NORMALIZE NAMES
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
# MISSINGNESS INDICATORS
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
# FINAL CLEANUP
# ============================================================

if feature_df.columns.duplicated().any():

    feature_df = feature_df.loc[
        :,
        ~feature_df.columns.duplicated(
            keep="first"
        )
    ].copy()


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
# CREATE PAIRWISE DATA
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


            if fidelity_a == fidelity_b:
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


            # A beats B
            X_pairs.append(
                difference
            )

            y_pairs.append(
                label
            )


            # B beats A
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
# SCORE ONE GROUP
# ============================================================

def get_pairwise_scores(
    group,
    feature_df,
    model
):

    indices = (
        group.index
        .tolist()
    )


    scores = []


    for idx_a in indices:

        probabilities = []


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


            probabilities.append(
                probability
            )


        scores.append(
            {
                "candidate_seed":
                    df.loc[
                        idx_a,
                        "candidate_seed"
                    ],

                "pairwise_score":
                    np.mean(
                        probabilities
                    ),

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

                "index":
                    idx_a,
            }
        )


    return scores


def make_model():

    return XGBClassifier(

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


def out_of_fold_scores(
    training_df,
    feature_df
):
    """Pairwise scores for each training group, produced by a model
    trained WITHOUT that group's circuit (leave-one-circuit-out).

    Learning the override threshold from in-sample scores would be
    overconfident, since the model has seen those groups' labels.
    """

    scores_by_group = {}

    for circuit in sorted(
        training_df["circuit_id"].unique()
    ):

        inner_train = training_df[
            training_df["circuit_id"] != circuit
        ]

        inner_test = training_df[
            training_df["circuit_id"] == circuit
        ]

        if inner_train.empty:
            continue

        X_inner, y_inner = (
            create_pairwise_training_data(
                inner_train,
                feature_df
            )
        )

        inner_model = make_model()

        inner_model.fit(
            X_inner,
            y_inner
        )

        for (c, backend), group in inner_test.groupby(
            ["circuit_id", "backend"],
            sort=True
        ):

            scores_by_group[(c, backend)] = (
                get_pairwise_scores(
                    group,
                    feature_df,
                    inner_model
                )
            )

    return scores_by_group


# ============================================================
# LEARN ESP UNCERTAINTY THRESHOLD
# ============================================================

def learn_threshold(
    training_df,
    feature_df,
    model
):

    # Out-of-fold pairwise scores, computed once and reused for
    # every candidate threshold.
    oof_scores = out_of_fold_scores(
        training_df,
        feature_df
    )

    training_margins = []


    # --------------------------------------------------------
    # Calculate ESP margins
    # --------------------------------------------------------

    for (_, _), group in training_df.groupby(
        ["circuit_id", "backend"],
        sort=True
    ):

        esp_values = sorted(
            group["esp"]
            .astype(float)
            .tolist(),
            reverse=True
        )


        if len(esp_values) < 2:
            continue


        margin = (
            esp_values[0]
            -
            esp_values[1]
        )


        training_margins.append(
            margin
        )


    if not training_margins:

        return 0.0


    # --------------------------------------------------------
    # Candidate thresholds
    # --------------------------------------------------------

    candidates = sorted(
        set(
            training_margins
        )
    )


    # Include a threshold below
    # the smallest observed margin.

    candidates = [
        0.0
    ] + candidates


    best_threshold = 0.0

    best_regret = np.inf

    best_accuracy = -1.0


    # --------------------------------------------------------
    # Evaluate each threshold
    # --------------------------------------------------------

    for threshold in candidates:

        regrets = []

        correct = 0

        total = 0


        for (circuit_key, backend_key), group in training_df.groupby(
            ["circuit_id", "backend"],
            sort=True
        ):

            group = group.copy()


            # ESP candidate
            esp_idx = group[
                "esp"
            ].idxmax()

            esp_candidate = (
                group.loc[
                    esp_idx,
                    "candidate_seed"
                ]
            )


            esp_fidelity = (
                group.loc[
                    esp_idx,
                    "actual_fidelity"
                ]
            )


            # ESP margin
            esp_sorted = sorted(
                group[
                    "esp"
                ].astype(float)
                .tolist(),
                reverse=True
            )


            margin = (
                esp_sorted[0]
                -
                esp_sorted[1]
            )


            # Pairwise candidate
            pair_scores = oof_scores[
                (circuit_key, backend_key)
            ]


            pairwise = max(
                pair_scores,
                key=lambda x:
                    x["pairwise_score"]
            )


            # ------------------------------------------------
            # Hybrid choice
            # ------------------------------------------------

            if margin <= threshold:

                selected_candidate = (
                    pairwise[
                        "candidate_seed"
                    ]
                )

                selected_fidelity = (
                    pairwise[
                        "actual_fidelity"
                    ]
                )

            else:

                selected_candidate = (
                    esp_candidate
                )

                selected_fidelity = (
                    esp_fidelity
                )


            oracle_idx = group[
                "actual_fidelity"
            ].idxmax()

            oracle_candidate = (
                group.loc[
                    oracle_idx,
                    "candidate_seed"
                ]
            )

            oracle_fidelity = (
                group.loc[
                    oracle_idx,
                    "actual_fidelity"
                ]
            )


            if (
                selected_candidate
                ==
                oracle_candidate
            ):

                correct += 1


            total += 1


            regrets.append(
                oracle_fidelity
                -
                selected_fidelity
            )


        accuracy = (
            correct / total
            if total
            else 0.0
        )

        average_regret = (
            np.mean(regrets)
            if regrets
            else np.inf
        )


        better = False


        if average_regret < best_regret:

            better = True

        elif (
            average_regret
            ==
            best_regret
            and
            accuracy > best_accuracy
        ):

            better = True


        if better:

            best_regret = (
                average_regret
            )

            best_accuracy = (
                accuracy
            )

            best_threshold = (
                threshold
            )


    return best_threshold


# ============================================================
# CIRCUITS
# ============================================================

circuits = sorted(
    df[
        "circuit_id"
    ].unique().tolist()
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

results = []


# ============================================================
# LEAVE-ONE-CIRCUIT-OUT
# ============================================================

for counter, test_circuit in enumerate(
    circuits,
    start=1
):

    print(
        f"\n[{counter}/{len(circuits)}] "
        f"Testing circuit "
        f"{int(test_circuit)}..."
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


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = make_model()


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Learn threshold from training circuits
    # --------------------------------------------------------

    threshold = learn_threshold(
        train_df,
        feature_df,
        model
    )


    # --------------------------------------------------------
    # Test each backend
    # --------------------------------------------------------

    for backend, group in test_df.groupby(
        "backend",
        sort=True
    ):

        group = group.copy()


        # ----------------------------------------------------
        # ESP
        # ----------------------------------------------------

        esp_idx = group[
            "esp"
        ].idxmax()

        esp_candidate = (
            group.loc[
                esp_idx,
                "candidate_seed"
            ]
        )

        esp_fidelity = (
            group.loc[
                esp_idx,
                "actual_fidelity"
            ]
        )


        # ----------------------------------------------------
        # ESP margin
        # ----------------------------------------------------

        esp_values = sorted(
            group[
                "esp"
            ]
            .astype(float)
            .tolist(),
            reverse=True
        )


        if len(esp_values) >= 2:

            esp_margin = (
                esp_values[0]
                -
                esp_values[1]
            )

        else:

            esp_margin = 0.0


        # ----------------------------------------------------
        # Pairwise
        # ----------------------------------------------------

        pair_scores = get_pairwise_scores(
            group,
            feature_df,
            model
        )


        pairwise = max(
            pair_scores,
            key=lambda x:
                x["pairwise_score"]
        )


        # ----------------------------------------------------
        # Hybrid
        # ----------------------------------------------------

        if esp_margin <= threshold:

            hybrid_candidate = (
                pairwise[
                    "candidate_seed"
                ]
            )

            hybrid_fidelity = (
                pairwise[
                    "actual_fidelity"
                ]
            )

            used_pairwise = True

        else:

            hybrid_candidate = (
                esp_candidate
            )

            hybrid_fidelity = (
                esp_fidelity
            )

            used_pairwise = False


        # ----------------------------------------------------
        # Oracle
        # ----------------------------------------------------

        oracle_idx = group[
            "actual_fidelity"
        ].idxmax()

        oracle_candidate = (
            group.loc[
                oracle_idx,
                "candidate_seed"
            ]
        )

        oracle_fidelity = (
            group.loc[
                oracle_idx,
                "actual_fidelity"
            ]
        )


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append(
            {
                "circuit_id":
                    test_circuit,

                "backend":
                    backend,

                "oracle_candidate":
                    oracle_candidate,

                "oracle_fidelity":
                    oracle_fidelity,

                "esp_candidate":
                    esp_candidate,

                "esp_fidelity":
                    esp_fidelity,

                "pairwise_candidate":
                    pairwise[
                        "candidate_seed"
                    ],

                "pairwise_fidelity":
                    pairwise[
                        "actual_fidelity"
                    ],

                "hybrid_candidate":
                    hybrid_candidate,

                "hybrid_fidelity":
                    hybrid_fidelity,

                "esp_margin":
                    esp_margin,

                "learned_threshold":
                    threshold,

                "used_pairwise":
                    used_pairwise,
            }
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    results
)


# ============================================================
# ACCURACY
# ============================================================

results["esp_correct"] = (
    results[
        "esp_candidate"
    ]
    ==
    results[
        "oracle_candidate"
    ]
)

results["pairwise_correct"] = (
    results[
        "pairwise_candidate"
    ]
    ==
    results[
        "oracle_candidate"
    ]
)

results["hybrid_correct"] = (
    results[
        "hybrid_candidate"
    ]
    ==
    results[
        "oracle_candidate"
    ]
)


# ============================================================
# REGRET
# ============================================================

results["esp_regret"] = (
    results[
        "oracle_fidelity"
    ]
    -
    results[
        "esp_fidelity"
    ]
)

results["pairwise_regret"] = (
    results[
        "oracle_fidelity"
    ]
    -
    results[
        "pairwise_fidelity"
    ]
)

results["hybrid_regret"] = (
    results[
        "oracle_fidelity"
    ]
    -
    results[
        "hybrid_fidelity"
    ]
)


# ============================================================
# METRICS
# ============================================================

esp_accuracy = (
    results[
        "esp_correct"
    ].mean()
)

pairwise_accuracy = (
    results[
        "pairwise_correct"
    ].mean()
)

hybrid_accuracy = (
    results[
        "hybrid_correct"
    ].mean()
)


esp_avg_regret = (
    results[
        "esp_regret"
    ].mean()
)

pairwise_avg_regret = (
    results[
        "pairwise_regret"
    ].mean()
)

hybrid_avg_regret = (
    results[
        "hybrid_regret"
    ].mean()
)


esp_max_regret = (
    results[
        "esp_regret"
    ].max()
)

pairwise_max_regret = (
    results[
        "pairwise_regret"
    ].max()
)

hybrid_max_regret = (
    results[
        "hybrid_regret"
    ].max()
)


# ============================================================
# COMPARISON
# ============================================================

hybrid_improved = results[
    results["hybrid_regret"]
    <
    results["esp_regret"]
]

hybrid_same = results[
    results["hybrid_regret"]
    ==
    results["esp_regret"]
]

hybrid_worse = results[
    results["hybrid_regret"]
    >
    results["esp_regret"]
]


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("HYBRID RESULTS")
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
    f"Hybrid selection accuracy: "
    f"{hybrid_accuracy:.2%}"
)


print(
    f"\nESP average regret: "
    f"{esp_avg_regret:.6f}"
)

print(
    f"Pairwise average regret: "
    f"{pairwise_avg_regret:.6f}"
)

print(
    f"Hybrid average regret: "
    f"{hybrid_avg_regret:.6f}"
)


print(
    f"\nESP maximum regret: "
    f"{esp_max_regret:.6f}"
)

print(
    f"Pairwise maximum regret: "
    f"{pairwise_max_regret:.6f}"
)

print(
    f"Hybrid maximum regret: "
    f"{hybrid_max_regret:.6f}"
)


print("\n")
print("=" * 90)
print("HYBRID VS ESP")
print("=" * 90)


print(
    f"\nHybrid improved over ESP: "
    f"{len(hybrid_improved)}/{len(results)}"
)

print(
    f"Hybrid same as ESP: "
    f"{len(hybrid_same)}/{len(results)}"
)

print(
    f"Hybrid worse than ESP: "
    f"{len(hybrid_worse)}/{len(results)}"
)


print(
    f"\nUsed pairwise override on: "
    f"{int(results['used_pairwise'].sum())}/"
    f"{len(results)} decisions"
)


# ============================================================
# DECISION TABLE
# ============================================================

print("\n")
print("=" * 90)
print("DECISION COMPARISON")
print("=" * 90)


print(
    results[
        [
            "circuit_id",
            "backend",
            "oracle_candidate",
            "esp_candidate",
            "pairwise_candidate",
            "hybrid_candidate",
            "oracle_fidelity",
            "esp_fidelity",
            "pairwise_fidelity",
            "hybrid_fidelity",
            "esp_margin",
            "learned_threshold",
            "used_pairwise",
            "esp_regret",
            "pairwise_regret",
            "hybrid_regret",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BIGGEST HYBRID IMPROVEMENTS
# ============================================================

improvement_table = results[
    results["hybrid_regret"]
    <
    results["esp_regret"]
].copy()


if len(improvement_table) > 0:

    improvement_table[
        "improvement"
    ] = (
        improvement_table[
            "esp_regret"
        ]
        -
        improvement_table[
            "hybrid_regret"
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


    print("\n")
    print("=" * 90)
    print("BIGGEST HYBRID IMPROVEMENTS")
    print("=" * 90)


    print(
        improvement_table[
            [
                "circuit_id",
                "backend",
                "esp_candidate",
                "pairwise_candidate",
                "hybrid_candidate",
                "esp_regret",
                "hybrid_regret",
                "improvement",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# BIGGEST HYBRID FAILURES
# ============================================================

failure_table = (
    results
    .sort_values(
        "hybrid_regret",
        ascending=False
    )
    .head(10)
)


print("\n")
print("=" * 90)
print("BIGGEST HYBRID FAILURES")
print("=" * 90)


print(
    failure_table[
        [
            "circuit_id",
            "backend",
            "oracle_candidate",
            "esp_candidate",
            "pairwise_candidate",
            "hybrid_candidate",
            "oracle_fidelity",
            "esp_fidelity",
            "hybrid_fidelity",
            "esp_regret",
            "hybrid_regret",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    RESULTS_DIR
    /
    "hybrid_pairwise_esp_results.csv"
)


results.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_file = (
    RESULTS_DIR
    /
    "hybrid_pairwise_esp_summary.csv"
)


summary = pd.DataFrame(
    {
        "metric": [
            "total_decisions",

            "esp_accuracy",
            "pairwise_accuracy",
            "hybrid_accuracy",

            "esp_average_regret",
            "pairwise_average_regret",
            "hybrid_average_regret",

            "esp_max_regret",
            "pairwise_max_regret",
            "hybrid_max_regret",

            "hybrid_improved",
            "hybrid_same",
            "hybrid_worse",

            "pairwise_overrides",
        ],

        "value": [
            len(results),

            esp_accuracy,
            pairwise_accuracy,
            hybrid_accuracy,

            esp_avg_regret,
            pairwise_avg_regret,
            hybrid_avg_regret,

            esp_max_regret,
            pairwise_max_regret,
            hybrid_max_regret,

            len(hybrid_improved),
            len(hybrid_same),
            len(hybrid_worse),

            int(
                results[
                    "used_pairwise"
                ].sum()
            ),
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