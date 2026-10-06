"""
CalibrationCompass - Confidence-Gated ESP + Pairwise Hybrid

ESP is the default candidate.

Pairwise ML can override ESP only when BOTH conditions hold:

1. ESP is uncertain
   -> small gap between best and second-best ESP

2. Pairwise model is confident
   -> pairwise winner has a sufficiently large advantage
      over the ESP-selected candidate

The two thresholds are learned ONLY from the training
circuits inside each leave-one-circuit-out fold.

Evaluation:
    30 circuits x 3 backends = 90 decisions
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


    # Remove raw physical IDs
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


    # Remove duplicate columns
    if features.columns.duplicated().any():

        features = features.loc[
            :,
            ~features.columns.duplicated(
                keep="first"
            )
        ].copy()


    # Missing indicators
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


    # Categorical columns
    categorical = (
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


    if categorical:

        features = pd.get_dummies(
            features,
            columns=categorical,
            dummy_na=True
        )


    # Remove duplicate columns again
    if features.columns.duplicated().any():

        features = features.loc[
            :,
            ~features.columns.duplicated(
                keep="first"
            )
        ].copy()


    # Remaining non-numeric
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


            fidelity_a = training_df.loc[
                idx_a,
                "actual_fidelity"
            ]

            fidelity_b = training_df.loc[
                idx_b,
                "actual_fidelity"
            ]


            if fidelity_a == fidelity_b:
                continue


            difference = a - b


            if fidelity_a > fidelity_b:

                label = 1

            else:

                label = 0


            # A versus B
            X.append(difference)
            y.append(label)


            # B versus A
            X.append(-difference)
            y.append(1 - label)


    return (
        np.asarray(
            X,
            dtype=float
        ),
        np.asarray(
            y,
            dtype=int
        ),
    )


def train_model(
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


def get_pairwise_candidate_scores(
    group,
    feature_df,
    model,
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

                "pairwise_score":
                    np.mean(
                        probabilities
                    ),

                "esp":
                    group.loc[
                        idx_a,
                        "esp"
                    ],

                "actual_fidelity":
                    group.loc[
                        idx_a,
                        "actual_fidelity"
                    ],
            }
        )


    return scores


def get_group_information(
    group,
    feature_df,
    model
):

    pair_scores = (
        get_pairwise_candidate_scores(
            group,
            feature_df,
            model
        )
    )


    # --------------------------------------------------------
    # ESP candidate
    # --------------------------------------------------------

    esp_sorted = group.sort_values(
        "esp",
        ascending=False
    )


    esp_row = (
        esp_sorted.iloc[0]
    )

    second_esp_row = (
        esp_sorted.iloc[1]
    )


    esp_candidate = (
        esp_row["candidate_seed"]
    )

    esp_fidelity = (
        esp_row["actual_fidelity"]
    )


    esp_margin = (
        esp_row["esp"]
        -
        second_esp_row["esp"]
    )


    # --------------------------------------------------------
    # Pairwise candidate
    # --------------------------------------------------------

    pairwise_row = max(
        pair_scores,
        key=lambda x:
            x["pairwise_score"]
    )


    pairwise_candidate = (
        pairwise_row[
            "candidate_seed"
        ]
    )

    pairwise_fidelity = (
        pairwise_row[
            "actual_fidelity"
        ]
    )


    pairwise_score = (
        pairwise_row[
            "pairwise_score"
        ]
    )


    # Pairwise score of ESP candidate
    esp_pairwise_score = next(
        item[
            "pairwise_score"
        ]
        for item in pair_scores
        if item["candidate_seed"]
        == esp_candidate
    )


    pairwise_advantage = (
        pairwise_score
        -
        esp_pairwise_score
    )


    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    oracle_row = group.loc[
        group[
            "actual_fidelity"
        ].idxmax()
    ]


    return {
        "oracle_candidate":
            oracle_row[
                "candidate_seed"
            ],

        "oracle_fidelity":
            oracle_row[
                "actual_fidelity"
            ],

        "esp_candidate":
            esp_candidate,

        "esp_fidelity":
            esp_fidelity,

        "esp_margin":
            esp_margin,

        "pairwise_candidate":
            pairwise_candidate,

        "pairwise_fidelity":
            pairwise_fidelity,

        "pairwise_score":
            pairwise_score,

        "pairwise_advantage":
            pairwise_advantage,
    }


# ============================================================
# LEARN TWO GATES
# ============================================================

def learn_gates(
    training_df,
    feature_df,
    model
):

    group_data = []

    # Out-of-fold scoring: each training group is scored by a model
    # trained WITHOUT its circuit, so the gates are not tuned on
    # predictions the model has already seen labels for.
    for circuit in sorted(
        training_df["circuit_id"].unique()
    ):

        inner_train = training_df[
            training_df["circuit_id"] != circuit
        ]

        if inner_train.empty:
            continue

        inner_model = train_model(
            inner_train,
            feature_df
        )

        inner_test = training_df[
            training_df["circuit_id"] == circuit
        ]

        for (_, _), group in inner_test.groupby(
            ["circuit_id", "backend"],
            sort=True
        ):

            info = get_group_information(
                group,
                feature_df,
                inner_model
            )

            group_data.append(
                info
            )


    # --------------------------------------------------------
    # Threshold ranges
    # --------------------------------------------------------

    esp_thresholds = np.arange(
        0.000,
        0.201,
        0.005
    )

    confidence_thresholds = np.arange(
        0.000,
        0.301,
        0.005
    )


    best_regret = np.inf

    best_accuracy = -1.0

    best_override_count = np.inf

    best_esp_threshold = 0.0

    best_confidence_threshold = 0.0


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    for esp_threshold in esp_thresholds:

        for confidence_threshold in (
            confidence_thresholds
        ):

            correct = 0

            regrets = []

            overrides = 0


            for info in group_data:

                use_pairwise = (
                    info[
                        "pairwise_candidate"
                    ]
                    !=
                    info[
                        "esp_candidate"
                    ]
                    and
                    info[
                        "esp_margin"
                    ]
                    <=
                    esp_threshold
                    and
                    info[
                        "pairwise_advantage"
                    ]
                    >=
                    confidence_threshold
                )


                if use_pairwise:

                    selected_candidate = (
                        info[
                            "pairwise_candidate"
                        ]
                    )

                    selected_fidelity = (
                        info[
                            "pairwise_fidelity"
                        ]
                    )

                    overrides += 1

                else:

                    selected_candidate = (
                        info[
                            "esp_candidate"
                        ]
                    )

                    selected_fidelity = (
                        info[
                            "esp_fidelity"
                        ]
                    )


                if (
                    selected_candidate
                    ==
                    info[
                        "oracle_candidate"
                    ]
                ):

                    correct += 1


                regrets.append(
                    info[
                        "oracle_fidelity"
                    ]
                    -
                    selected_fidelity
                )


            accuracy = (
                correct
                /
                len(group_data)
            )

            average_regret = (
                np.mean(regrets)
            )


            # Primary:
            # lower regret
            #
            # Secondary:
            # higher accuracy
            #
            # Tertiary:
            # fewer overrides

            better = False


            if (
                average_regret
                <
                best_regret
            ):

                better = True

            elif (
                average_regret
                ==
                best_regret
                and
                accuracy
                >
                best_accuracy
            ):

                better = True

            elif (
                average_regret
                ==
                best_regret
                and
                accuracy
                ==
                best_accuracy
                and
                overrides
                <
                best_override_count
            ):

                better = True


            if better:

                best_regret = (
                    average_regret
                )

                best_accuracy = (
                    accuracy
                )

                best_override_count = (
                    overrides
                )

                best_esp_threshold = (
                    esp_threshold
                )

                best_confidence_threshold = (
                    confidence_threshold
                )


    return (
        best_esp_threshold,
        best_confidence_threshold,
        best_accuracy,
        best_regret,
        best_override_count,
    )


# ============================================================
# APPLY HYBRID
# ============================================================

def evaluate_group(
    group,
    feature_df,
    model,
    esp_threshold,
    confidence_threshold,
):

    info = get_group_information(
        group,
        feature_df,
        model
    )


    use_pairwise = (
        info[
            "pairwise_candidate"
        ]
        !=
        info[
            "esp_candidate"
        ]
        and
        info[
            "esp_margin"
        ]
        <=
        esp_threshold
        and
        info[
            "pairwise_advantage"
        ]
        >=
        confidence_threshold
    )


    if use_pairwise:

        hybrid_candidate = (
            info[
                "pairwise_candidate"
            ]
        )

        hybrid_fidelity = (
            info[
                "pairwise_fidelity"
            ]
        )

    else:

        hybrid_candidate = (
            info[
                "esp_candidate"
            ]
        )

        hybrid_fidelity = (
            info[
                "esp_fidelity"
            ]
        )


    return {
        "oracle_candidate":
            info[
                "oracle_candidate"
            ],

        "oracle_fidelity":
            info[
                "oracle_fidelity"
            ],

        "esp_candidate":
            info[
                "esp_candidate"
            ],

        "esp_fidelity":
            info[
                "esp_fidelity"
            ],

        "pairwise_candidate":
            info[
                "pairwise_candidate"
            ],

        "pairwise_fidelity":
            info[
                "pairwise_fidelity"
            ],

        "hybrid_candidate":
            hybrid_candidate,

        "hybrid_fidelity":
            hybrid_fidelity,

        "esp_margin":
            info[
                "esp_margin"
            ],

        "pairwise_advantage":
            info[
                "pairwise_advantage"
            ],

        "used_pairwise":
            use_pairwise,

        "esp_threshold":
            esp_threshold,

        "confidence_threshold":
            confidence_threshold,
    }


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
# FIND COLUMNS
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
# NORMALIZE
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
    f"\nFinal feature count: "
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
# RESULTS STORAGE
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

    model = train_model(
        train_df,
        feature_df
    )


    # --------------------------------------------------------
    # Learn gates using training circuits
    # --------------------------------------------------------

    (
        esp_threshold,
        confidence_threshold,
        train_accuracy,
        train_regret,
        train_overrides,
    ) = learn_gates(
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

        info = evaluate_group(
            group,
            feature_df,
            model,
            esp_threshold,
            confidence_threshold,
        )


        all_results.append(
            {
                "circuit_id":
                    test_circuit,

                "backend":
                    backend,

                **info,

                "training_accuracy":
                    train_accuracy,

                "training_regret":
                    train_regret,

                "training_overrides":
                    train_overrides,
            }
        )


# ============================================================
# RESULTS
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

hybrid_average_regret = (
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
# IMPROVEMENT COUNTS
# ============================================================

improved = results[
    results[
        "hybrid_regret"
    ]
    <
    results[
        "esp_regret"
    ]
]

same = results[
    results[
        "hybrid_regret"
    ]
    ==
    results[
        "esp_regret"
    ]
]

worse = results[
    results[
        "hybrid_regret"
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
print("CONFIDENCE-GATED HYBRID RESULTS")
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
    f"Confidence-gated hybrid accuracy: "
    f"{hybrid_accuracy:.2%}"
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
    f"Confidence-gated hybrid average regret: "
    f"{hybrid_average_regret:.6f}"
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
    f"Confidence-gated hybrid maximum regret: "
    f"{hybrid_max_regret:.6f}"
)


print("\n")
print("=" * 90)
print("HYBRID VS ESP")
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


print(
    f"\nPairwise override used on: "
    f"{int(results['used_pairwise'].sum())}/"
    f"{len(results)} decisions"
)


# ============================================================
# THRESHOLDS
# ============================================================

print("\n")
print("=" * 90)
print("LEARNED GATES")
print("=" * 90)


print(
    f"\nESP uncertainty threshold "
    f"(average): "
    f"{results['esp_threshold'].mean():.6f}"
)

print(
    f"Pairwise confidence threshold "
    f"(average): "
    f"{results['confidence_threshold'].mean():.6f}"
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
            "pairwise_advantage",
            "esp_threshold",
            "confidence_threshold",
            "used_pairwise",
            "esp_regret",
            "hybrid_regret",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# IMPROVEMENTS
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
    print("BIGGEST IMPROVEMENTS")
    print("=" * 90)


    print(
        improvement_table[
            [
                "circuit_id",
                "backend",
                "oracle_candidate",
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
# FAILURES
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
# SAVE
# ============================================================

output_file = (
    RESULTS_DIR
    /
    "confidence_gated_hybrid_results.csv"
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
    "confidence_gated_hybrid_summary.csv"
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

            "improved_over_esp",
            "same_as_esp",
            "worse_than_esp",

            "pairwise_overrides",
        ],

        "value": [
            len(results),

            esp_accuracy,
            pairwise_accuracy,
            hybrid_accuracy,

            esp_average_regret,
            pairwise_average_regret,
            hybrid_average_regret,

            esp_max_regret,
            pairwise_max_regret,
            hybrid_max_regret,

            len(improved),
            len(same),
            len(worse),

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