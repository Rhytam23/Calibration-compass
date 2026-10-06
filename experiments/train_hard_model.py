import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
print("CALIBRATIONCOMPASS - HARD BENCHMARK ML")
print("=" * 75)

print(f"\nRows loaded: {len(df)}")


# ============================================================
# 2. CREATE ML TARGET
# ============================================================
#
# We want ML to learn:
#
#       actual fidelity - ESP
#
# This tells the model how much ESP is overestimating
# or underestimating the real execution quality.
# ============================================================

df["residual"] = (
    df["fidelity"] - df["esp"]
)


# ============================================================
# 3. FEATURES
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
# 4. TRAIN / TEST SPLIT
# ============================================================
#
# We keep entire circuits together.
#
# First 30 circuits:
#     training
#
# Last 10 circuits:
#     testing
#
# Therefore the model never sees those 10 circuits during training.
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
    df["circuit_id"].isin(train_circuits)
].copy()


test_df = df[
    df["circuit_id"].isin(test_circuits)
].copy()


print("\nTRAIN / TEST SPLIT")
print("-" * 75)

print(
    f"Training circuits: "
    f"{len(train_circuits)}"
)

print(
    f"Testing circuits:  "
    f"{len(test_circuits)}"
)

print(
    f"Training rows:     "
    f"{len(train_df)}"
)

print(
    f"Testing rows:      "
    f"{len(test_df)}"
)


# ============================================================
# 5. FUNCTION TO TRAIN MODEL
# ============================================================

def train_model(
    train_data,
    test_data,
    include_backend=True
):

    # --------------------------------------------
    # Base features
    # --------------------------------------------

    X_train = train_data[
        features
    ].copy()

    X_test = test_data[
        features
    ].copy()

    # --------------------------------------------
    # Add backend identity if requested
    # --------------------------------------------

    if include_backend:

        train_backend = pd.get_dummies(
            train_data["backend"],
            prefix="backend",
            dtype=float
        )

        test_backend = pd.get_dummies(
            test_data["backend"],
            prefix="backend",
            dtype=float
        )

        # Make columns identical
        test_backend = test_backend.reindex(
            columns=train_backend.columns,
            fill_value=0
        )

        X_train = pd.concat(
            [
                X_train,
                train_backend
            ],
            axis=1
        )

        X_test = pd.concat(
            [
                X_test,
                test_backend
            ],
            axis=1
        )

    # --------------------------------------------
    # Target
    # --------------------------------------------

    y_train = train_data[
        "residual"
    ]

    # --------------------------------------------
    # Model
    # --------------------------------------------

    model = XGBRegressor(

        n_estimators=500,

        max_depth=4,

        learning_rate=0.03,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42,

        n_jobs=-1

    )

    # --------------------------------------------
    # Train
    # --------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------
    # Predict residual
    # --------------------------------------------

    predicted_residual = model.predict(
        X_test
    )

    # --------------------------------------------
    # Convert residual to fidelity
    # --------------------------------------------

    predicted_fidelity = (
        test_data["esp"].values
        +
        predicted_residual
    )

    # Fidelity must be between 0 and 1
    predicted_fidelity = np.clip(
        predicted_fidelity,
        0,
        1
    )

    return (
        model,
        predicted_fidelity,
        X_train.columns
    )


# ============================================================
# 6. TRAIN MODEL WITH BACKEND IDENTITY
# ============================================================

print("\n")
print("=" * 75)
print("MODEL A - WITH BACKEND IDENTITY")
print("=" * 75)

model_with_backend, pred_with_backend, columns = (
    train_model(
        train_df,
        test_df,
        include_backend=True
    )
)


# ============================================================
# 7. TRAIN MODEL WITHOUT BACKEND IDENTITY
# ============================================================

print("\n")
print("=" * 75)
print("MODEL B - WITHOUT BACKEND IDENTITY")
print("=" * 75)

model_without_backend, pred_without_backend, columns2 = (
    train_model(
        train_df,
        test_df,
        include_backend=False
    )
)


# ============================================================
# 8. SAVE PREDICTIONS
# ============================================================

results = test_df.copy()

results[
    "ml_fidelity"
] = pred_with_backend

results[
    "ml_no_backend_fidelity"
] = pred_without_backend


# ============================================================
# 9. FIDELITY PREDICTION METRICS
# ============================================================

actual = results[
    "fidelity"
].values

esp = results[
    "esp"
].values


# ESP
esp_mae = mean_absolute_error(
    actual,
    esp
)

esp_rmse = np.sqrt(
    mean_squared_error(
        actual,
        esp
    )
)


# ML with backend
ml_mae = mean_absolute_error(
    actual,
    results["ml_fidelity"]
)

ml_rmse = np.sqrt(
    mean_squared_error(
        actual,
        results["ml_fidelity"]
    )
)


# ML without backend
ml_nb_mae = mean_absolute_error(
    actual,
    results["ml_no_backend_fidelity"]
)

ml_nb_rmse = np.sqrt(
    mean_squared_error(
        actual,
        results["ml_no_backend_fidelity"]
    )
)


print("\n")
print("=" * 75)
print("FIDELITY PREDICTION")
print("=" * 75)

print(
    f"\nESP MAE: "
    f"{esp_mae:.6f}"
)

print(
    f"ML with backend MAE: "
    f"{ml_mae:.6f}"
)

print(
    f"ML without backend MAE: "
    f"{ml_nb_mae:.6f}"
)

print()

print(
    f"ESP RMSE: "
    f"{esp_rmse:.6f}"
)

print(
    f"ML with backend RMSE: "
    f"{ml_rmse:.6f}"
)

print(
    f"ML without backend RMSE: "
    f"{ml_nb_rmse:.6f}"
)


# ============================================================
# 10. IMPROVEMENT
# ============================================================

if esp_mae > 0:

    improvement = (
        (esp_mae - ml_mae)
        / esp_mae
        * 100
    )

else:

    improvement = 0


print(
    f"\nML improvement over ESP: "
    f"{improvement:.2f}%"
)


# ============================================================
# 11. FIND WINNERS
# ============================================================

# Oracle = highest actual fidelity
oracle_rows = results.loc[
    results.groupby(
        [
            "circuit_id",
            "day"
        ]
    )["fidelity"].idxmax()
].copy()


oracle_rows = oracle_rows.rename(
    columns={
        "backend": "oracle_backend",
        "fidelity": "oracle_fidelity"
    }
)


# ESP winner
esp_rows = results.loc[
    results.groupby(
        [
            "circuit_id",
            "day"
        ]
    )["esp"].idxmax()
].copy()


esp_rows = esp_rows.rename(
    columns={
        "backend": "esp_backend",
        "fidelity": "esp_selected_fidelity"
    }
)


# ML winner
ml_rows = results.loc[
    results.groupby(
        [
            "circuit_id",
            "day"
        ]
    )["ml_fidelity"].idxmax()
].copy()


ml_rows = ml_rows.rename(
    columns={
        "backend": "ml_backend",
        "fidelity": "ml_selected_fidelity"
    }
)


# ML without backend identity
ml_nb_rows = results.loc[
    results.groupby(
        [
            "circuit_id",
            "day"
        ]
    )["ml_no_backend_fidelity"].idxmax()
].copy()


ml_nb_rows = ml_nb_rows.rename(
    columns={
        "backend": "ml_no_backend",
        "fidelity": "ml_nb_selected_fidelity"
    }
)


# ============================================================
# 12. COMBINE WINNERS
# ============================================================

comparison = oracle_rows[
    [
        "circuit_id",
        "day",
        "oracle_backend",
        "oracle_fidelity"
    ]
].merge(
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
    ml_rows[
        [
            "circuit_id",
            "day",
            "ml_backend",
            "ml_selected_fidelity"
        ]
    ],
    on=[
        "circuit_id",
        "day"
    ]
).merge(
    ml_nb_rows[
        [
            "circuit_id",
            "day",
            "ml_no_backend",
            "ml_nb_selected_fidelity"
        ]
    ],
    on=[
        "circuit_id",
        "day"
    ]
)


# ============================================================
# 13. SELECTION ACCURACY
# ============================================================

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
    "ml_correct"
] = (
    comparison[
        "ml_backend"
    ]
    ==
    comparison[
        "oracle_backend"
    ]
)

comparison[
    "ml_no_backend_correct"
] = (
    comparison[
        "ml_no_backend"
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
    f"ML accuracy: "
    f"{comparison['ml_correct'].mean():.2%}"
)

print(
    f"ML without backend identity: "
    f"{comparison['ml_no_backend_correct'].mean():.2%}"
)


# ============================================================
# 14. FIDELITY REGRET
# ============================================================

comparison[
    "esp_regret"
] = (
    comparison["oracle_fidelity"]
    -
    comparison["esp_selected_fidelity"]
)

comparison[
    "ml_regret"
] = (
    comparison["oracle_fidelity"]
    -
    comparison["ml_selected_fidelity"]
)

comparison[
    "ml_no_backend_regret"
] = (
    comparison["oracle_fidelity"]
    -
    comparison["ml_nb_selected_fidelity"]
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
    f"ML average regret: "
    f"{comparison['ml_regret'].mean():.6f}"
)

print(
    f"ML without backend average regret: "
    f"{comparison['ml_no_backend_regret'].mean():.6f}"
)


print(
    f"\nESP maximum regret: "
    f"{comparison['esp_regret'].max():.6f}"
)

print(
    f"ML maximum regret: "
    f"{comparison['ml_regret'].max():.6f}"
)


# ============================================================
# 15. IDENTIFY HARD DECISIONS
# ============================================================

sorted_results = results.sort_values(
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
    sorted_results
    .groupby(
        [
            "circuit_id",
            "day"
        ]
    )["fidelity"]
    .apply(
        lambda x: list(x.head(2))
    )
)


hard_records = []


for key, values in top_two.items():

    if len(values) < 2:
        continue

    best = values[0]
    second = values[1]

    gap = best - second

    hard_records.append({

        "circuit_id": key[0],

        "day": key[1],

        "best_fidelity": best,

        "second_best_fidelity": second,

        "gap": gap

    })


hard_df = pd.DataFrame(
    hard_records
)


comparison = comparison.merge(
    hard_df[
        [
            "circuit_id",
            "day",
            "gap"
        ]
    ],
    on=[
        "circuit_id",
        "day"
    ]
)


# ============================================================
# 16. HARD-CASE ACCURACY
# ============================================================

hard_cases = comparison[
    comparison["gap"]
    <= HARD_CASE_THRESHOLD
].copy()


print("\n")
print("=" * 75)
print("HARD-CASE PERFORMANCE")
print("=" * 75)

print(
    f"\nHard cases: "
    f"{len(hard_cases)}/"
    f"{len(comparison)}"
)

if len(hard_cases) > 0:

    print(
        f"ESP hard-case accuracy: "
        f"{hard_cases['esp_correct'].mean():.2%}"
    )

    print(
        f"ML hard-case accuracy: "
        f"{hard_cases['ml_correct'].mean():.2%}"
    )

    print(
        f"ML without backend hard-case accuracy: "
        f"{hard_cases['ml_no_backend_correct'].mean():.2%}"
    )


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": columns,

    "importance":
        model_with_backend.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\n")
print("=" * 75)
print("TOP ML FEATURES")
print("=" * 75)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results.to_csv(
    "results/hard_ml_predictions.csv",
    index=False
)

comparison.to_csv(
    "results/hard_ml_selection.csv",
    index=False
)

importance.to_csv(
    "results/hard_ml_feature_importance.csv",
    index=False
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 75)
print("HARD ML EXPERIMENT COMPLETE")
print("=" * 75)

print("\nSaved:")

print(
    "results/hard_ml_predictions.csv"
)

print(
    "results/hard_ml_selection.csv"
)

print(
    "results/hard_ml_feature_importance.csv"
)