import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "results/candidate_features.csv"
)

print("=" * 90)
print("CALIBRATIONCOMPASS - CANDIDATE ML MODEL")
print("=" * 90)

print()
print("Total rows:", len(df))


# ============================================================
# SPLIT BY WHOLE CIRCUITS
# ============================================================

all_circuits = sorted(
    df["circuit_id"].unique()
)

# Fixed split so the experiment is reproducible.
# Train on 7 circuits, test on 3 completely unseen circuits.
test_circuits = [2, 5, 8]

train_circuits = [
    c for c in all_circuits
    if c not in test_circuits
]

train_df = df[
    df["circuit_id"].isin(train_circuits)
].copy()

test_df = df[
    df["circuit_id"].isin(test_circuits)
].copy()

print()
print("Training circuits:", train_circuits)
print("Testing circuits:", test_circuits)

print()
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    "original_qubits",
    "original_depth",
    "transpiled_depth",
    "one_qubit_gates",
    "two_qubit_gates",
    "esp",
    "active_physical_qubits",
    "avg_readout_error",
    "max_readout_error",
    "sum_readout_error",
    "readout_product",
    "avg_cz_error",
    "max_cz_error",
    "edge_exposure",
    "max_edge_error",
    "cz_product",
    "avg_t1",
    "min_t1",
    "avg_t2",
    "min_t2"
]


# Add backend as categorical information.
combined = pd.concat(
    [
        train_df[["backend"]],
        test_df[["backend"]]
    ]
)

backend_encoded = pd.get_dummies(
    combined,
    columns=["backend"],
    dtype=int
)

backend_train = backend_encoded.iloc[
    :len(train_df)
].reset_index(drop=True)

backend_test = backend_encoded.iloc[
    len(train_df):
].reset_index(drop=True)


X_train = pd.concat(
    [
        train_df[feature_columns]
        .reset_index(drop=True),
        backend_train
    ],
    axis=1
)

X_test = pd.concat(
    [
        test_df[feature_columns]
        .reset_index(drop=True),
        backend_test
    ],
    axis=1
)

y_train = train_df[
    "actual_fidelity"
]

y_test = test_df[
    "actual_fidelity"
]


# ============================================================
# TRAIN XGBOOST
# ============================================================

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

test_df["ml_prediction"] = (
    model.predict(X_test)
)


# ============================================================
# REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    test_df["ml_prediction"]
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_df["ml_prediction"]
    )
)

print()
print("=" * 90)
print("FIDELITY PREDICTION")
print("=" * 90)

print()
print("ML MAE :", f"{mae:.6f}")
print("ML RMSE:", f"{rmse:.6f}")


# ============================================================
# ESP BASELINE
# ============================================================

esp_predictions = test_df[
    "esp"
]

esp_mae = mean_absolute_error(
    y_test,
    esp_predictions
)

esp_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        esp_predictions
    )
)

print()
print("ESP MAE :", f"{esp_mae:.6f}")
print("ESP RMSE:", f"{esp_rmse:.6f}")


# ============================================================
# CANDIDATE SELECTION
# ============================================================

selection_results = []

for (circuit_id, backend), group in (
    test_df.groupby(
        ["circuit_id", "backend"]
    )
):

    # Actual best candidate
    actual_best = group.loc[
        group["actual_fidelity"].idxmax()
    ]

    # ESP choice
    esp_choice = group.loc[
        group["esp"].idxmax()
    ]

    # ML choice
    ml_choice = group.loc[
        group["ml_prediction"].idxmax()
    ]

    actual_fidelity = (
        actual_best["actual_fidelity"]
    )

    esp_fidelity = (
        esp_choice["actual_fidelity"]
    )

    ml_fidelity = (
        ml_choice["actual_fidelity"]
    )

    selection_results.append({

        "circuit_id":
            circuit_id,

        "backend":
            backend,

        "actual_best_seed":
            int(
                actual_best[
                    "candidate_seed"
                ]
            ),

        "esp_selected_seed":
            int(
                esp_choice[
                    "candidate_seed"
                ]
            ),

        "ml_selected_seed":
            int(
                ml_choice[
                    "candidate_seed"
                ]
            ),

        "actual_best_fidelity":
            actual_fidelity,

        "esp_selected_fidelity":
            esp_fidelity,

        "ml_selected_fidelity":
            ml_fidelity,

        "esp_regret":
            actual_fidelity - esp_fidelity,

        "ml_regret":
            actual_fidelity - ml_fidelity,

        "esp_correct":
            int(
                esp_choice[
                    "candidate_seed"
                ]
                ==
                actual_best[
                    "candidate_seed"
                ]
            ),

        "ml_correct":
            int(
                ml_choice[
                    "candidate_seed"
                ]
                ==
                actual_best[
                    "candidate_seed"
                ]
            )
    })


selection_df = pd.DataFrame(
    selection_results
)


# ============================================================
# SELECTION RESULTS
# ============================================================

print()
print("=" * 90)
print("CANDIDATE SELECTION")
print("=" * 90)

print()

print(
    selection_df.to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

total_decisions = len(
    selection_df
)

esp_accuracy = (
    selection_df["esp_correct"]
    .mean()
)

ml_accuracy = (
    selection_df["ml_correct"]
    .mean()
)

esp_avg_regret = (
    selection_df["esp_regret"]
    .mean()
)

ml_avg_regret = (
    selection_df["ml_regret"]
    .mean()
)

esp_max_regret = (
    selection_df["esp_regret"]
    .max()
)

ml_max_regret = (
    selection_df["ml_regret"]
    .max()
)

print()
print("=" * 90)
print("SELECTION SUMMARY")
print("=" * 90)

print()
print(
    "Total test decisions:",
    total_decisions
)

print(
    "ESP selection accuracy:",
    f"{esp_accuracy:.2%}"
)

print(
    "ML selection accuracy:",
    f"{ml_accuracy:.2%}"
)

print(
    "ESP average regret:",
    f"{esp_avg_regret:.6f}"
)

print(
    "ML average regret:",
    f"{ml_avg_regret:.6f}"
)

print(
    "ESP maximum regret:",
    f"{esp_max_regret:.6f}"
)

print(
    "ML maximum regret:",
    f"{ml_max_regret:.6f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("=" * 90)
print("TOP FEATURES")
print("=" * 90)

print()

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_df.to_csv(
    "results/candidate_ml_predictions.csv",
    index=False
)

selection_df.to_csv(
    "results/candidate_ml_selection.csv",
    index=False
)

importance.to_csv(
    "results/candidate_ml_feature_importance.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 90)
print("MODEL TRAINING COMPLETE")
print("=" * 90)

print()
print("Saved:")
print("results/candidate_ml_predictions.csv")
print("results/candidate_ml_selection.csv")
print("results/candidate_ml_feature_importance.csv")