"""
CalibrationCompass - Blended ESP + Pairwise Selector

Final candidate score:

    blended_score =
        (1 - alpha) * normalized ESP
        + alpha * normalized pairwise score

alpha is learned on the training circuits.

Evaluation uses leave-one-circuit-out testing:

    30 circuits
    3 backends per circuit
    90 total decisions

Comparisons:
    ESP
    Pairwise
    Blended
"""


from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from xgboost import XGBClassifier


# ============================================================
# CONFIG
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
# HELPERS
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


def build_features(df):

    drop_columns = {
        "circuit_id",
        "candidate_seed",
        "actual_fidelity",
    }


    for column in df.columns:

        lower = column.lower()

        if lower == "physical_qubits":
            drop_columns.add(column)

        elif lower == "mapping":
            drop_columns.add(column)

        elif lower.endswith("_physical"):
            drop_columns.add(column)


    features = df.drop(
        columns=[
            c
            for c in drop_columns
            if c in df.columns
        ],
        errors="ignore"
    ).copy()


    if features.columns.duplicated().any():

        features = features.loc[
            :,
            ~features.columns.duplicated(
                keep="first"
            )
        ].copy()


    missing_columns = []


    original_columns = list(
        features.columns
    )


    for column in original_columns:

        if features[column].isna().any():

            missing_columns.append(
                column
            )

            features[
                f"{column}__missing"
            ] = (
                features[column]
                .isna()
                .astype(float)
            )


    categorical_columns = (
        features
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

        features = pd.get_dummies(
            features,
            columns=categorical_columns,
            dummy_na=True
        )


    if features.columns.duplicated().any():

        features = features.loc[
            :,
            ~features.columns.duplicated(
                keep="first"
            )
        ].copy()


    non_numeric = (
        features
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

        features = features.drop(
            columns=non_numeric
        )


    features = features.astype(
        float
    )


    return features, missing_columns


def create_pairwise_training_data(
    training_df,
    feature_df
):

    X = []
    y = []


    for (_, _), group in training_df.groupby(
        ["circuit_id", "backend"],
        sort=True
    ):

        indices = group.index.tolist()


        for idx_a, idx_b in combinations(
            indices,
            2
        ):

            a = feature_df.loc[
                idx_a
            ].values

            b = feature_df.loc[
                idx_b
            ].values


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
                a - b
            )


            if fidelity_a > fidelity_b:

                label = 1

            else:

                label = 0


            X.append(
                difference
            )

            y.append(
                label
            )


            X.append(
                -difference
            )

            y.append(
                1 - label
            )


    return (
        np.asarray(
            X,
            dtype=float
        ),
        np.asarray(
            y,
            dtype=int
        )
    )


def train_pairwise_model(
    training_df,
    feature_df
):

    X_train, y_train = (
        create_pairwise_training_data(
            training_df,
            feature_df
        )
    )


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


    return model


def minmax(series):

    series = pd.Series(
        series,
        dtype=float
    )


    if series.max() - series.min() < 1e-12:

        return pd.Series(
            0.5,
            index=series.index
        )


    return (
        (series - series.min())
        /
        (series.max() - series.min())
    )


def get_pairwise_scores(
    group,
    feature_df,
    model
):

    indices = group.index.tolist()

    scores = []


    for idx_a in indices:

        probabilities = []


        for idx_b in indices:

            if idx_a == idx_b:
                continue


            a = feature_df.loc[
                idx_a
            ].values

            b = feature_df.loc[
                idx_b
            ].values


            difference = (
                a - b
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
                "index": idx_a,

                "candidate_seed":
                    group.loc[
                        idx_a,
                        "candidate_seed"
                    ],

                "esp":
                    float(
                        group.loc[
                            idx_a,
                            "esp"
                        ]
                    ),

                "pairwise_score":
                    float(
                        np.mean(
                            probabilities
                        )
                    ),

                "actual_fidelity":
                    float(
                        group.loc[
                            idx_a,
                            "actual_fidelity"
                        ]
                    ),
            }
        )


    return pd.DataFrame(
        scores
    )


def choose_candidate(
    group_scores,
    alpha
):

    group_scores = (
        group_scores.copy()
    )


    group_scores[
        "esp_norm"
    ] = minmax(
        group_scores["esp"]
    )


    group_scores[
        "pairwise_norm"
    ] = minmax(
        group_scores["pairwise_score"]
    )


    group_scores[
        "blended_score"
    ] = (
        (1 - alpha)
        *
        group_scores["esp_norm"]
        +
        alpha
        *
        group_scores["pairwise_norm"]
    )


    return group_scores.loc[
        group_scores[
            "blended_score"
        ].idxmax()
    ]


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("CALIBRATIONCOMPASS - BLENDED ESP + PAIRWISE")
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
# CLEAN
# ============================================================

if df.columns.duplicated().any():

    df = df.loc[
        :,
        ~df.columns.duplicated(
            keep="first"
        )
    ].copy()


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
# FEATURES
# ============================================================

feature_df, missing_columns = (
    build_features(df)
)


print(
    f"\nFeatures with missing values: "
    f"{len(missing_columns)}"
)

print(
    f"Final feature count: "
    f"{feature_df.shape[1]}"
)


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
# TRAINING ALPHAS
# ============================================================

alpha_values = np.arange(
    0.0,
    1.01,
    0.05
)


# ============================================================
# RESULTS
# ============================================================

all_results = []


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


    train_df = df[
        df["circuit_id"]
        !=
        test_circuit
    ].copy()


    test_df = df[
        df["circuit_id"]
        ==
        test_circuit
    ].copy()


    # --------------------------------------------------------
    # Train pairwise model
    # --------------------------------------------------------

    model = train_pairwise_model(
        train_df,
        feature_df
    )


    # --------------------------------------------------------
    # Precompute pairwise scores for TRAIN groups
    # --------------------------------------------------------

    training_groups = []


    # Out-of-fold: score each training group with a model trained
    # without its circuit so alpha is not tuned on in-sample scores.
    for inner_circuit in sorted(
        train_df["circuit_id"].unique()
    ):

        inner_train = train_df[
            train_df["circuit_id"] != inner_circuit
        ]

        if inner_train.empty:
            continue

        inner_model = train_pairwise_model(
            inner_train,
            feature_df
        )

        inner_test = train_df[
            train_df["circuit_id"] == inner_circuit
        ]

        for (circuit, backend), group in inner_test.groupby(
            ["circuit_id", "backend"],
            sort=True
        ):

            pair_scores = get_pairwise_scores(
                group,
                feature_df,
                inner_model
            )


            training_groups.append(
                pair_scores
            )


    # --------------------------------------------------------
    # Learn alpha
    # --------------------------------------------------------

    best_alpha = 0.0

    best_train_regret = np.inf

    best_train_accuracy = -1.0


    for alpha in alpha_values:

        regrets = []

        correct = 0

        total = 0


        for group_scores in training_groups:

            selected = choose_candidate(
                group_scores,
                alpha
            )


            oracle_idx = (
                group_scores[
                    "actual_fidelity"
                ].idxmax()
            )


            oracle = group_scores.loc[
                oracle_idx
            ]


            if (
                selected[
                    "candidate_seed"
                ]
                ==
                oracle[
                    "candidate_seed"
                ]
            ):

                correct += 1


            total += 1


            regrets.append(
                oracle[
                    "actual_fidelity"
                ]
                -
                selected[
                    "actual_fidelity"
                ]
            )


        train_accuracy = (
            correct / total
        )

        train_regret = (
            np.mean(regrets)
        )


        better = False


        if train_regret < best_train_regret:

            better = True

        elif (
            train_regret
            ==
            best_train_regret
            and
            train_accuracy
            >
            best_train_accuracy
        ):

            better = True


        if better:

            best_alpha = alpha

            best_train_regret = (
                train_regret
            )

            best_train_accuracy = (
                train_accuracy
            )


    # --------------------------------------------------------
    # Test backends
    # --------------------------------------------------------

    for backend, group in test_df.groupby(
        "backend",
        sort=True
    ):

        group_scores = (
            get_pairwise_scores(
                group,
                feature_df,
                model
            )
        )


        # ----------------------------------------------------
        # ESP
        # ----------------------------------------------------

        esp = group_scores.loc[
            group_scores[
                "esp"
            ].idxmax()
        ]


        # ----------------------------------------------------
        # Pairwise
        # ----------------------------------------------------

        pairwise = group_scores.loc[
            group_scores[
                "pairwise_score"
            ].idxmax()
        ]


        # ----------------------------------------------------
        # Blended
        # ----------------------------------------------------

        blended = choose_candidate(
            group_scores,
            best_alpha
        )


        # ----------------------------------------------------
        # Oracle
        # ----------------------------------------------------

        oracle = group_scores.loc[
            group_scores[
                "actual_fidelity"
            ].idxmax()
        ]


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

                "blended_candidate":
                    blended[
                        "candidate_seed"
                    ],

                "blended_fidelity":
                    blended[
                        "actual_fidelity"
                    ],

                "alpha":
                    best_alpha,

                "training_accuracy":
                    best_train_accuracy,

                "training_regret":
                    best_train_regret,
            }
        )


# ============================================================
# FINAL RESULTS
# ============================================================

results = pd.DataFrame(
    all_results
)


# ============================================================
# CORRECTNESS
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

results["blended_correct"] = (
    results[
        "blended_candidate"
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

results["blended_regret"] = (
    results[
        "oracle_fidelity"
    ]
    -
    results[
        "blended_fidelity"
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

blended_accuracy = (
    results[
        "blended_correct"
    ].mean()
)


esp_average_regret = (
    results[
        "esp_regret"
    ].mean()
)

pairwise_average_regret = (
    results[
        "pairwise_regret"
    ].mean()
)

blended_average_regret = (
    results[
        "blended_regret"
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

blended_max_regret = (
    results[
        "blended_regret"
    ].max()
)


# ============================================================
# COMPARISON
# ============================================================

improved = results[
    results[
        "blended_regret"
    ]
    <
    results[
        "esp_regret"
    ]
]

same = results[
    results[
        "blended_regret"
    ]
    ==
    results[
        "esp_regret"
    ]
]

worse = results[
    results[
        "blended_regret"
    ]
    >
    results[
        "esp_regret"
    ]
]


# ============================================================
# PRINT
# ============================================================

print("\n")
print("=" * 90)
print("BLENDED MODEL RESULTS")
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
    f"Blended selection accuracy: "
    f"{blended_accuracy:.2%}"
)


print(
    f"\nESP average regret: "
    f"{esp_average_regret:.6f}"
)

print(
    f"Pairwise average regret: "
    f"{pairwise_average_regret:.6f}"
)

print(
    f"Blended average regret: "
    f"{blended_average_regret:.6f}"
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
    f"Blended maximum regret: "
    f"{blended_max_regret:.6f}"
)


# ============================================================
# ALPHA
# ============================================================

print("\n")
print("=" * 90)
print("LEARNED ALPHA")
print("=" * 90)


print(
    results[
        "alpha"
    ].value_counts()
    .sort_index()
    .to_string()
)


print(
    f"\nAverage alpha: "
    f"{results['alpha'].mean():.3f}"
)


# ============================================================
# ESP COMPARISON
# ============================================================

print("\n")
print("=" * 90)
print("BLENDED VS ESP")
print("=" * 90)


print(
    f"\nImproved over ESP: "
    f"{len(improved)}/{len(results)}"
)

print(
    f"Same as ESP: "
    f"{len(same)}/{len(results)}"
)

print(
    f"Worse than ESP: "
    f"{len(worse)}/{len(results)}"
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
            "blended_candidate",
            "oracle_fidelity",
            "esp_fidelity",
            "pairwise_fidelity",
            "blended_fidelity",
            "alpha",
            "esp_regret",
            "pairwise_regret",
            "blended_regret",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BIGGEST BLENDED IMPROVEMENTS
# ============================================================

if len(improved) > 0:

    improvement_table = (
        improved.copy()
    )


    improvement_table[
        "improvement"
    ] = (
        improvement_table[
            "esp_regret"
        ]
        -
        improvement_table[
            "blended_regret"
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
    print("BIGGEST BLENDED IMPROVEMENTS")
    print("=" * 90)


    print(
        improvement_table[
            [
                "circuit_id",
                "backend",
                "oracle_candidate",
                "esp_candidate",
                "pairwise_candidate",
                "blended_candidate",
                "esp_regret",
                "blended_regret",
                "improvement",
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
    "blended_pairwise_esp_results.csv"
)


results.to_csv(
    output_file,
    index=False
)


summary_file = (
    RESULTS_DIR
    /
    "blended_pairwise_esp_summary.csv"
)


summary = pd.DataFrame(
    {
        "metric": [
            "total_decisions",

            "esp_accuracy",
            "pairwise_accuracy",
            "blended_accuracy",

            "esp_average_regret",
            "pairwise_average_regret",
            "blended_average_regret",

            "esp_max_regret",
            "pairwise_max_regret",
            "blended_max_regret",

            "blended_improved",
            "blended_same",
            "blended_worse",

            "average_alpha",
        ],

        "value": [
            len(results),

            esp_accuracy,
            pairwise_accuracy,
            blended_accuracy,

            esp_average_regret,
            pairwise_average_regret,
            blended_average_regret,

            esp_max_regret,
            pairwise_max_regret,
            blended_max_regret,

            len(improved),
            len(same),
            len(worse),

            results["alpha"].mean(),
        ],
    }
)


summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# DONE
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