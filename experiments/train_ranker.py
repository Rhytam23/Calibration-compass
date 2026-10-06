import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import pandas as pd

from xgboost import XGBRanker


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
print("CALIBRATIONCOMPASS - XGBOOST RANKING MODEL")
print("=" * 75)

print(f"\nRows loaded: {len(df)}")


# ============================================================
# 2. FEATURES
# ============================================================

base_features = [
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
# 3. SORT DATA
# ============================================================
#
# Every circuit/day becomes one ranking group.
#
# Example:
#
# Circuit 10 + Day 4
#     Fez
#     Torino
#     Sherbrooke
#
# The model must rank those three.
# ============================================================

df = df.sort_values(
    [
        "circuit_id",
        "day",
        "backend"
    ]
).reset_index(drop=True)


# ============================================================
# 4. TRAIN / TEST SPLIT
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

print(
    f"Training rows:     {len(train_df)}"
)

print(
    f"Testing rows:      {len(test_df)}"
)


# ============================================================
# 5. CREATE FEATURES WITH BACKEND NAME
# ============================================================

def make_features(
    data,
    include_backend=True
):

    X = data[
        base_features
    ].copy()

    if include_backend:

        backend_encoded = pd.get_dummies(
            data["backend"],
            prefix="backend",
            dtype=float
        )

        X = pd.concat(
            [
                X,
                backend_encoded
            ],
            axis=1
        )

    return X


# ============================================================
# 6. MAKE TRAIN / TEST FEATURES
# ============================================================

X_train = make_features(
    train_df,
    include_backend=True
)

X_test = make_features(
    test_df,
    include_backend=True
)


# Make sure both have exactly the same columns.
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# ============================================================
# 7. TARGET
# ============================================================
#
# The ranking model tries to place higher-fidelity candidates
# above lower-fidelity candidates.
#
# Actual fidelity is the relevance score.
# ============================================================

y_train = train_df[
    "fidelity"
]


# ============================================================
# 8. CREATE RANKING GROUPS
# ============================================================
#
# Every circuit + day has exactly 3 backends.
#
# So:
#
# Circuit 0, Day 0 → group size 3
# Circuit 0, Day 1 → group size 3
# ...
# ============================================================

train_groups = (
    train_df
    .groupby(
        [
            "circuit_id",
            "day"
        ]
    )
    .size()
    .tolist()
)


# ============================================================
# 9. CREATE THE RANKING MODEL
# ============================================================

ranker = XGBRanker(

    objective="rank:pairwise",

    n_estimators=500,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    eval_metric="ndcg",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\nTraining ranking model...")

ranker.fit(
    X_train,
    y_train,
    group=train_groups
)

print("Training complete.")


# ============================================================
# 11. PREDICT RANKING SCORES
# ============================================================

test_df[
    "ranking_score"
] = ranker.predict(
    X_test
)


# ============================================================
# 12. ORACLE WINNER
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
# 13. RANKER WINNER
# ============================================================

ranker_rows = test_df.loc[
    test_df.groupby(
        [
            "circuit_id",
            "day"
        ]
    )["ranking_score"].idxmax()
].copy()


ranker_rows = ranker_rows.rename(
    columns={
        "backend":
            "ranker_backend",

        "fidelity":
            "ranker_selected_fidelity"
    }
)


# ============================================================
# 14. ESP WINNER
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
# 15. ALWAYS-FEZ BASELINE
# ============================================================

fez_rows = test_df[
    test_df["backend"] == "Fez"
].copy()


fez_rows = fez_rows.rename(
    columns={
        "fidelity":
            "fez_selected_fidelity"
    }
)


# ============================================================
# 16. COMBINE RESULTS
# ============================================================

comparison = oracle_rows[
    [
        "circuit_id",
        "day",
        "oracle_backend",
        "oracle_fidelity"
    ]
].merge(
    ranker_rows[
        [
            "circuit_id",
            "day",
            "ranker_backend",
            "ranker_selected_fidelity"
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
).merge(
    fez_rows[
        [
            "circuit_id",
            "day",
            "fez_selected_fidelity"
        ]
    ],
    on=[
        "circuit_id",
        "day"
    ]
)


# ============================================================
# 17. CORRECTNESS
# ============================================================

comparison[
    "ranker_correct"
] = (
    comparison[
        "ranker_backend"
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


comparison[
    "fez_correct"
] = (
    comparison[
        "oracle_backend"
    ]
    ==
    "Fez"
)


# ============================================================
# 18. REGRET
# ============================================================

comparison[
    "ranker_regret"
] = (
    comparison[
        "oracle_fidelity"
    ]
    -
    comparison[
        "ranker_selected_fidelity"
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


comparison[
    "fez_regret"
] = (
    comparison[
        "oracle_fidelity"
    ]
    -
    comparison[
        "fez_selected_fidelity"
    ]
)


# ============================================================
# 19. GENERAL SELECTION RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("BACKEND SELECTION")
print("=" * 75)

print(
    f"\nAlways Fez: "
    f"{comparison['fez_correct'].mean():.2%}"
)

print(
    f"ESP: "
    f"{comparison['esp_correct'].mean():.2%}"
)

print(
    f"XGBoost ranking: "
    f"{comparison['ranker_correct'].mean():.2%}"
)


# ============================================================
# 20. REGRET RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FIDELITY REGRET")
print("=" * 75)

print(
    f"\nAlways Fez average regret: "
    f"{comparison['fez_regret'].mean():.6f}"
)

print(
    f"ESP average regret: "
    f"{comparison['esp_regret'].mean():.6f}"
)

print(
    f"Ranking model average regret: "
    f"{comparison['ranker_regret'].mean():.6f}"
)


print(
    f"\nAlways Fez maximum regret: "
    f"{comparison['fez_regret'].max():.6f}"
)

print(
    f"ESP maximum regret: "
    f"{comparison['esp_regret'].max():.6f}"
)

print(
    f"Ranking model maximum regret: "
    f"{comparison['ranker_regret'].max():.6f}"
)


# ============================================================
# 21. HARD CASES
# ============================================================

# Determine the gap between the best and second-best backend.

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
].copy()


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
        f"Always Fez: "
        f"{hard_cases['fez_correct'].mean():.2%}"
    )

    print(
        f"ESP: "
        f"{hard_cases['esp_correct'].mean():.2%}"
    )

    print(
        f"XGBoost ranking: "
        f"{hard_cases['ranker_correct'].mean():.2%}"
    )


# ============================================================
# 22. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        X_train.columns,

    "importance":
        ranker.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\n")
print("=" * 75)
print("TOP RANKING FEATURES")
print("=" * 75)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# 23. SAVE RESULTS
# ============================================================

test_df.to_csv(
    "results/ranker_predictions.csv",
    index=False
)


comparison.to_csv(
    "results/ranker_comparison.csv",
    index=False
)


importance.to_csv(
    "results/ranker_feature_importance.csv",
    index=False
)


print("\n")
print("=" * 75)
print("RANKING EXPERIMENT COMPLETE")
print("=" * 75)

print("\nSaved:")

print(
    "results/ranker_predictions.csv"
)

print(
    "results/ranker_comparison.csv"
)

print(
    "results/ranker_feature_importance.csv"
)