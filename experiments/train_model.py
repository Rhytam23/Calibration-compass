import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "results/calibration_dataset_v2.csv"
)

print("=" * 70)
print("CALIBRATIONCOMPASS - FIRST ML MODEL")
print("=" * 70)


# ============================================================
# 2. CREATE THE TARGET
# ============================================================
#
# Instead of asking ML to predict fidelity directly,
# we ask it to learn where ESP is wrong.
#
# residual = actual fidelity - ESP
#
# Example:
#
# actual = 0.920
# ESP    = 0.915
# residual = +0.005
# ============================================================

df["residual"] = (
    df["fidelity"] - df["esp"]
)


# ============================================================
# 3. REMOVE COLUMNS WE DON'T WANT THE MODEL TO LEARN
# ============================================================

columns_to_remove = [
    "circuit_id",
    "seed",
    "fidelity",
    "residual",
    "drift",
    "condition"
]


feature_df = df.drop(
    columns=columns_to_remove
)


# ============================================================
# 4. CONVERT BACKEND NAME INTO NUMBERS
# ============================================================

feature_df = pd.get_dummies(
    feature_df,
    columns=["backend"],
    dtype=float
)


# ============================================================
# 5. SPLIT BY CIRCUIT
# ============================================================
#
# We deliberately keep complete circuits together.
# The test circuits are completely unseen during training.
# ============================================================

unique_circuits = sorted(
    df["circuit_id"].unique()
)

split_point = int(
    len(unique_circuits) * 0.8
)

train_circuits = unique_circuits[
    :split_point
]

test_circuits = unique_circuits[
    split_point:
]


train_mask = df["circuit_id"].isin(
    train_circuits
)

test_mask = df["circuit_id"].isin(
    test_circuits
)


X = feature_df

y = df["residual"]


X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]


print("\nTRAIN / TEST SPLIT")
print("-" * 70)

print(
    f"Training circuits: {len(train_circuits)}"
)

print(
    f"Testing circuits:  {len(test_circuits)}"
)

print(
    f"Training rows:     {len(X_train)}"
)

print(
    f"Testing rows:      {len(X_test)}"
)


# ============================================================
# 6. CREATE THE XGBOOST MODEL
# ============================================================

model = XGBRegressor(

    n_estimators=300,

    max_depth=3,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42

)


# ============================================================
# 7. TRAIN
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 8. PREDICT RESIDUALS
# ============================================================

predicted_residual = model.predict(
    X_test
)


# ============================================================
# 9. CREATE ML FIDELITY PREDICTION
# ============================================================

test_results = df[
    test_mask
].copy()

test_results["predicted_residual"] = (
    predicted_residual
)

test_results["predicted_fidelity"] = (
    test_results["esp"]
    +
    test_results["predicted_residual"]
)


# ============================================================
# 10. CLIP PREDICTIONS TO VALID RANGE
# ============================================================

test_results["predicted_fidelity"] = (
    test_results["predicted_fidelity"]
    .clip(0, 1)
)


# ============================================================
# 11. MEASURE ESP ERROR
# ============================================================

esp_mae = mean_absolute_error(
    test_results["fidelity"],
    test_results["esp"]
)

esp_rmse = np.sqrt(
    mean_squared_error(
        test_results["fidelity"],
        test_results["esp"]
    )
)


# ============================================================
# 12. MEASURE ML ERROR
# ============================================================

ml_mae = mean_absolute_error(
    test_results["fidelity"],
    test_results["predicted_fidelity"]
)

ml_rmse = np.sqrt(
    mean_squared_error(
        test_results["fidelity"],
        test_results["predicted_fidelity"]
    )
)


# ============================================================
# 13. PRINT PREDICTION RESULTS
# ============================================================

print("\nFIDELITY PREDICTION")
print("-" * 70)

print(
    f"ESP MAE: "
    f"{esp_mae:.6f}"
)

print(
    f"ML  MAE: "
    f"{ml_mae:.6f}"
)

print()

print(
    f"ESP RMSE: "
    f"{esp_rmse:.6f}"
)

print(
    f"ML  RMSE: "
    f"{ml_rmse:.6f}"
)


# ============================================================
# 14. CALCULATE IMPROVEMENT
# ============================================================

if esp_mae != 0:

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
# 15. TEST BACKEND SELECTION
# ============================================================

print("\nBACKEND SELECTION ON UNSEEN CIRCUITS")
print("-" * 70)


# For every circuit-condition pair,
# choose the backend with the highest predicted fidelity.

predicted_winners = test_results.loc[
    test_results.groupby(
        [
            "circuit_id",
            "condition"
        ]
    )["predicted_fidelity"].idxmax()
].copy()


# Find actual oracle winner

oracle_winners = test_results.loc[
    test_results.groupby(
        [
            "circuit_id",
            "condition"
        ]
    )["fidelity"].idxmax()
].copy()


# Merge the two

winner_comparison = pd.merge(

    predicted_winners[
        [
            "circuit_id",
            "condition",
            "backend"
        ]
    ].rename(
        columns={
            "backend": "ml_backend"
        }
    ),

    oracle_winners[
        [
            "circuit_id",
            "condition",
            "backend"
        ]
    ].rename(
        columns={
            "backend": "oracle_backend"
        }
    ),

    on=[
        "circuit_id",
        "condition"
    ]
)


winner_comparison["correct"] = (
    winner_comparison["ml_backend"]
    ==
    winner_comparison["oracle_backend"]
)


selection_accuracy = (
    winner_comparison["correct"]
    .mean()
)


print(
    f"ML backend selection accuracy: "
    f"{selection_accuracy:.2%}"
)


# ============================================================
# 16. CALCULATE ML FIDELITY REGRET
# ============================================================

selected_rows = test_results.merge(

    predicted_winners[
        [
            "circuit_id",
            "condition",
            "backend"
        ]
    ],

    on=[
        "circuit_id",
        "condition",
        "backend"
    ]
)


# Actual fidelity of the backend ML selected

ml_regret_table = predicted_winners[
    [
        "circuit_id",
        "condition",
        "backend",
        "predicted_fidelity"
    ]
].merge(

    oracle_winners[
        [
            "circuit_id",
            "condition",
            "fidelity"
        ]
    ],

    on=[
        "circuit_id",
        "condition"
    ]
)


ml_regret_table["regret"] = (
    ml_regret_table["fidelity"]
    -
    test_results.merge(
        predicted_winners[
            [
                "circuit_id",
                "condition",
                "backend"
            ]
        ],
        on=[
            "circuit_id",
            "condition",
            "backend"
        ]
    )["fidelity"].values
)


print(
    f"Average ML fidelity regret: "
    f"{ml_regret_table['regret'].mean():.6f}"
)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": X_train.columns,

    "importance": model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\nTOP FEATURES")
print("-" * 70)

print(
    importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

test_results.to_csv(
    "results/ml_predictions.csv",
    index=False
)


importance.to_csv(
    "results/ml_feature_importance.csv",
    index=False
)


print("\nResults saved to:")

print("results/ml_predictions.csv")

print("results/ml_feature_importance.csv")


print("\n" + "=" * 70)
print("ML EXPERIMENT COMPLETE")
print("=" * 70)