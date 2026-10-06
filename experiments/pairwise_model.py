import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "results/hard_benchmark.csv"

TEST_CIRCUIT_COUNT = 10

HARD_CASE_THRESHOLD = 0.01


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 75)
print("CALIBRATIONCOMPASS - PAIRWISE ML MODEL")
print("=" * 75)

print(f"\nRows loaded: {len(df)}")


# ============================================================
# 2. FEATURES USED TO COMPARE TWO BACKENDS
# ============================================================

features = [
    "original_qubits",
    "original_depth",

    "transpiled_depth",
    "one_qubit_gates",
    "two_qubit_gates",

    "esp",

    "active_physical_qubits",

    "avg_gate_error",
    "max_gate_error",

    "avg_readout_error",
    "max_readout_error",

    "avg_t1",
    "min_t1",

    "avg_t2",
    "min_t2"
]


# ============================================================
# 3. TRAIN / TEST CIRCUITS
# ============================================================

unique_circuits = sorted(
    df["circuit_id"].unique()
)

test_circuits = unique_circuits[
    -TEST_CIRCUIT_COUNT:
]

train_circuits = unique_circuits[
    :-TEST_CIRCUIT_COUNT
]


train_df = df[
    df["circuit_id"].isin(
        train_circuits
    )
].copy()


test_df = df[
    df["circuit_id"].isin(
        test_circuits
    )
].copy()


print("\nTRAIN / TEST")
print("-" * 75)

print(
    f"Training circuits: {len(train_circuits)}"
)

print(
    f"Testing circuits:  {len(test_circuits)}"
)


# ============================================================
# 4. CREATE PAIRS
# ============================================================

backend_names = [
    "Fez",
    "Sherbrooke",
    "Torino"
]


def create_pair_dataset(data):

    rows = []

    # Each circuit + day is one decision
    for (
        circuit_id,
        day
    ), group in data.groupby(
        [
            "circuit_id",
            "day"
        ]
    ):

        group = group.set_index(
            "backend"
        )

        for i in range(
            len(backend_names)
        ):

            for j in range(
                i + 1,
                len(backend_names)
            ):

                backend_a = (
                    backend_names[i]
                )

                backend_b = (
                    backend_names[j]
                )

                if (
                    backend_a not in group.index
                    or backend_b not in group.index
                ):
                    continue

                a = group.loc[
                    backend_a
                ]

                b = group.loc[
                    backend_b
                ]

                # ------------------------------------------------
                # Create DIFFERENCE features
                # ------------------------------------------------

                row = {

                    "circuit_id":
                        circuit_id,

                    "day":
                        day,

                    "backend_a":
                        backend_a,

                    "backend_b":
                        backend_b,

                    # Target:
                    # 1 means A is actually better
                    # 0 means B is actually better

                    "target":
                        int(
                            a["fidelity"]
                            >
                            b["fidelity"]
                        )
                }

                # ------------------------------------------------
                # Hardware differences
                # ------------------------------------------------

                for feature in features:

                    row[
                        f"diff_{feature}"
                    ] = (
                        a[feature]
                        -
                        b[feature]
                    )

                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 5. BUILD TRAIN / TEST PAIRS
# ============================================================

pair_train = create_pair_dataset(
    train_df
)

pair_test = create_pair_dataset(
    test_df
)


# Augment training pairs with the swapped (B, A) copy so the model
# cannot learn a bias from the fixed backend ordering.
_diff_columns = [
    column
    for column in pair_train.columns
    if column.startswith("diff_")
]

_flipped = pair_train.copy()
_flipped[_diff_columns] = -_flipped[_diff_columns]
_flipped["target"] = 1 - _flipped["target"]
_flipped[["backend_a", "backend_b"]] = (
    pair_train[["backend_b", "backend_a"]].values
)

pair_train = pd.concat(
    [pair_train, _flipped],
    ignore_index=True
)


print("\nPAIR DATASET")
print("-" * 75)

print(
    f"Training pairs: {len(pair_train)}"
)

print(
    f"Testing pairs:  {len(pair_test)}"
)


# ============================================================
# 6. FEATURES FOR MODEL
# ============================================================

pair_features = [
    column
    for column in pair_train.columns
    if column.startswith("diff_")
]


X_train = pair_train[
    pair_features
]

y_train = pair_train[
    "target"
]

X_test = pair_test[
    pair_features
]

y_test = pair_test[
    "target"
]


# ============================================================
# 7. CREATE CLASSIFIER
# ============================================================

model = XGBClassifier(

    n_estimators=400,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 8. TRAIN
# ============================================================

print("\nTraining pairwise model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 9. PAIRWISE ACCURACY
# ============================================================

pair_predictions = model.predict(
    X_test
)

pair_accuracy = accuracy_score(
    y_test,
    pair_predictions
)


print("\nPAIRWISE PREDICTION")
print("-" * 75)

print(
    f"Pairwise accuracy: "
    f"{pair_accuracy:.2%}"
)


# ============================================================
# 10. USE PAIRWISE PREDICTIONS TO SELECT BACKEND
# ============================================================

pair_test = pair_test.copy()

pair_test[
    "probability_a_wins"
] = model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# Voting system
# ------------------------------------------------------------
#
# Every backend receives a score.
#
# If A is predicted to beat B:
#     A gets A's probability
#     B gets 1 - probability
#
# Highest total score wins.
# ------------------------------------------------------------

decision_rows = []

for (
    circuit_id,
    day
), group in pair_test.groupby(
    [
        "circuit_id",
        "day"
    ]
):

    scores = {
        "Fez": 0.0,
        "Sherbrooke": 0.0,
        "Torino": 0.0
    }

    for _, row in group.iterrows():

        a = row[
            "backend_a"
        ]

        b = row[
            "backend_b"
        ]

        probability = row[
            "probability_a_wins"
        ]

        scores[a] += probability

        scores[b] += (
            1 - probability
        )

    predicted_backend = max(
        scores,
        key=scores.get
    )

    decision_rows.append({

        "circuit_id":
            circuit_id,

        "day":
            day,

        "predicted_backend":
            predicted_backend,

        "Fez_score":
            scores["Fez"],

        "Sherbrooke_score":
            scores["Sherbrooke"],

        "Torino_score":
            scores["Torino"]
    })


decisions = pd.DataFrame(
    decision_rows
)


# ============================================================
# 11. FIND ORACLE WINNERS
# ============================================================

oracle_rows = test_df.loc[
    test_df.groupby(
        [
            "circuit_id",
            "day"
        ]
    )["fidelity"].idxmax()
].copy()


oracle_rows = oracle_rows.rename(
    columns={
        "backend":
            "oracle_backend",

        "fidelity":
            "oracle_fidelity"
    }
)


# ============================================================
# 12. FIND ESP WINNERS
# ============================================================

esp_rows = test_df.loc[
    test_df.groupby(
        [
            "circuit_id",
            "day"
        ]
    )["esp"].idxmax()
].copy()


esp_rows = esp_rows.rename(
    columns={
        "backend":
            "esp_backend",

        "fidelity":
            "esp_selected_fidelity"
    }
)


# ============================================================
# 13. COMBINE RESULTS
# ============================================================

comparison = decisions.merge(

    oracle_rows[
        [
            "circuit_id",
            "day",
            "oracle_backend",
            "oracle_fidelity"
        ]
    ],

    on=[
        "circuit_id",
        "day"
    ]
).merge(

    esp_rows[
        [
            "circuit_id",
            "day",
            "esp_backend",
            "esp_selected_fidelity"
        ]
    ],

    on=[
        "circuit_id",
        "day"
    ]
)


# ============================================================
# 14. ACCURACY
# ============================================================

comparison[
    "pairwise_correct"
] = (
    comparison[
        "predicted_backend"
    ]
    ==
    comparison[
        "oracle_backend"
    ]
)


comparison[
    "esp_correct"
] = (
    comparison[
        "esp_backend"
    ]
    ==
    comparison[
        "oracle_backend"
    ]
)


print("\n")
print("=" * 75)
print("BACKEND SELECTION")
print("=" * 75)

print(
    f"\nESP accuracy: "
    f"{comparison['esp_correct'].mean():.2%}"
)

print(
    f"Pairwise ML accuracy: "
    f"{comparison['pairwise_correct'].mean():.2%}"
)


# ============================================================
# 15. FIDELITY REGRET
# ============================================================

# Get the actual fidelity achieved by the ML-selected backend.

selected_fidelity = []

for _, row in comparison.iterrows():

    selected = row[
        "predicted_backend"
    ]

    value = test_df[
        (
            test_df["circuit_id"]
            == row["circuit_id"]
        )
        &
        (
            test_df["day"]
            == row["day"]
        )
        &
        (
            test_df["backend"]
            == selected
        )
    ]["fidelity"].iloc[0]

    selected_fidelity.append(
        value
    )


comparison[
    "ml_selected_fidelity"
] = selected_fidelity


comparison[
    "ml_regret"
] = (
    comparison[
        "oracle_fidelity"
    ]
    -
    comparison[
        "ml_selected_fidelity"
    ]
)


comparison[
    "esp_regret"
] = (
    comparison[
        "oracle_fidelity"
    ]
    -
    comparison[
        "esp_selected_fidelity"
    ]
)


print("\n")
print("=" * 75)
print("FIDELITY REGRET")
print("=" * 75)

print(
    f"\nESP average regret: "
    f"{comparison['esp_regret'].mean():.6f}"
)

print(
    f"Pairwise ML average regret: "
    f"{comparison['ml_regret'].mean():.6f}"
)

print(
    f"\nESP maximum regret: "
    f"{comparison['esp_regret'].max():.6f}"
)

print(
    f"Pairwise ML maximum regret: "
    f"{comparison['ml_regret'].max():.6f}"
)


# ============================================================
# 16. HARD CASES
# ============================================================

sorted_test = test_df.sort_values(
    [
        "circuit_id",
        "day",
        "fidelity"
    ],
    ascending=[
        True,
        True,
        False
    ]
)


top_two = (
    sorted_test
    .groupby(
        [
            "circuit_id",
            "day"
        ]
    )["fidelity"]
    .apply(
        lambda x:
        list(x.head(2))
    )
)


hard_records = []


for key, values in top_two.items():

    if len(values) < 2:
        continue

    gap = values[0] - values[1]

    hard_records.append({

        "circuit_id":
            key[0],

        "day":
            key[1],

        "gap":
            gap
    })


hard_df = pd.DataFrame(
    hard_records
)


comparison = comparison.merge(
    hard_df,
    on=[
        "circuit_id",
        "day"
    ]
)


hard_cases = comparison[
    comparison["gap"]
    <= HARD_CASE_THRESHOLD
]


print("\n")
print("=" * 75)
print("HARD CASES")
print("=" * 75)

print(
    f"\nHard cases: "
    f"{len(hard_cases)}/"
    f"{len(comparison)}"
)


if len(hard_cases) > 0:

    print(
        f"ESP accuracy: "
        f"{hard_cases['esp_correct'].mean():.2%}"
    )

    print(
        f"Pairwise ML accuracy: "
        f"{hard_cases['pairwise_correct'].mean():.2%}"
    )


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        pair_features,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\n")
print("=" * 75)
print("TOP PAIRWISE FEATURES")
print("=" * 75)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

comparison.to_csv(
    "results/pairwise_comparison.csv",
    index=False
)

importance.to_csv(
    "results/pairwise_feature_importance.csv",
    index=False
)


print("\nSaved:")

print(
    "results/pairwise_comparison.csv"
)

print(
    "results/pairwise_feature_importance.csv"
)


print("\n" + "=" * 75)
print("PAIRWISE EXPERIMENT COMPLETE")
print("=" * 75)