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

DATA_FILE = "results/calibration_dataset_v2.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 75)
print("CALIBRATIONCOMPASS - MODEL VALIDATION")
print("=" * 75)

print(f"\nLoaded {len(df)} rows")


# ============================================================
# 2. CREATE TARGET
# ============================================================
#
# The model learns:
#
#     residual = actual fidelity - ESP
#
# Then:
#
#     predicted fidelity = ESP + predicted residual
# ============================================================

df["residual"] = df["fidelity"] - df["esp"]


# ============================================================
# 3. CHOOSE FEATURES
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
# 4. CREATE FUNCTION FOR MODEL TRAINING
# ============================================================

def train_and_predict(train_df, test_df, include_backend=True):
    """
    Train XGBoost and predict fidelity on test data.

    include_backend=True:
        Backend name is given to the model.

    include_backend=False:
        Backend name is removed.
        This tests whether the model can learn from
        actual hardware properties instead of memorizing
        backend identity.
    """

    # --------------------------------------------
    # Select features
    # --------------------------------------------

    train_features = train_df[base_features].copy()
    test_features = test_df[base_features].copy()

    # --------------------------------------------
    # Optionally include backend
    # --------------------------------------------

    if include_backend:

        train_features = pd.concat(
            [
                train_features,
                pd.get_dummies(
                    train_df["backend"],
                    prefix="backend",
                    dtype=float
                )
            ],
            axis=1
        )

        test_features = pd.concat(
            [
                test_features,
                pd.get_dummies(
                    test_df["backend"],
                    prefix="backend",
                    dtype=float
                )
            ],
            axis=1
        )

        # Make sure test and train have exactly
        # the same columns.
        test_features = test_features.reindex(
            columns=train_features.columns,
            fill_value=0
        )

    # --------------------------------------------
    # Create target
    # --------------------------------------------

    y_train = train_df["residual"]

    # --------------------------------------------
    # Create model
    # --------------------------------------------

    model = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    # --------------------------------------------
    # Train
    # --------------------------------------------

    model.fit(
        train_features,
        y_train
    )

    # --------------------------------------------
    # Predict residual
    # --------------------------------------------

    predicted_residual = model.predict(
        test_features
    )

    # --------------------------------------------
    # Convert residual to fidelity prediction
    # --------------------------------------------

    predicted_fidelity = (
        test_df["esp"].values
        +
        predicted_residual
    )

    predicted_fidelity = np.clip(
        predicted_fidelity,
        0,
        1
    )

    return model, predicted_fidelity, train_features.columns


# ============================================================
# TEST 1
# ============================================================
#
# HARD CONDITION GENERALIZATION
#
# Train on:
#   normal
#   readout_4x
#   readout_8x
#   gate_4x
#
# Test on:
#   combined_4x
#
# AND:
#   use the last 6 circuits as unseen circuits.
#
# This means the model sees neither those circuits
# nor the combined condition during training.
# ============================================================

print("\n")
print("=" * 75)
print("TEST 1 - UNSEEN CONDITION + UNSEEN CIRCUITS")
print("=" * 75)


train_conditions = [
    "normal",
    "readout_4x",
    "readout_8x",
    "gate_4x"
]

test_condition = "combined_4x"


# Last 6 circuits are our completely unseen circuits
unique_circuits = sorted(
    df["circuit_id"].unique()
)

test_circuits = unique_circuits[-6:]


train_df = df[
    df["condition"].isin(train_conditions)
    &
    ~df["circuit_id"].isin(test_circuits)
].copy()


test_df = df[
    (df["condition"] == test_condition)
    &
    df["circuit_id"].isin(test_circuits)
].copy()


print("\nTraining conditions:")
print(train_conditions)

print("\nTesting condition:")
print(test_condition)

print("\nTraining circuits:")
print(
    sorted(train_df["circuit_id"].unique())
)

print("\nTesting circuits:")
print(
    sorted(test_df["circuit_id"].unique())
)

print(
    f"\nTraining rows: {len(train_df)}"
)

print(
    f"Testing rows: {len(test_df)}"
)


# --------------------------------------------
# Train model
# --------------------------------------------

model, predicted_fidelity, columns = train_and_predict(
    train_df,
    test_df,
    include_backend=True
)


# --------------------------------------------
# Evaluate ML
# --------------------------------------------

actual_fidelity = test_df["fidelity"].values

ml_mae = mean_absolute_error(
    actual_fidelity,
    predicted_fidelity
)

ml_rmse = np.sqrt(
    mean_squared_error(
        actual_fidelity,
        predicted_fidelity
    )
)


# --------------------------------------------
# Evaluate ESP
# --------------------------------------------

esp_values = test_df["esp"].values

esp_mae = mean_absolute_error(
    actual_fidelity,
    esp_values
)

esp_rmse = np.sqrt(
    mean_squared_error(
        actual_fidelity,
        esp_values
    )
)


print("\nFIDELITY PREDICTION")
print("-" * 75)

print(
    f"ESP MAE: {esp_mae:.6f}"
)

print(
    f"ML  MAE: {ml_mae:.6f}"
)

print(
    f"ESP RMSE: {esp_rmse:.6f}"
)

print(
    f"ML  RMSE: {ml_rmse:.6f}"
)


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
# TEST 2
# ============================================================
#
# REMOVE BACKEND IDENTITY
#
# The model is NOT allowed to know:
#
#   backend = Fez
#   backend = Torino
#   backend = Sherbrooke
#
# It can only use actual hardware measurements.
# ============================================================

print("\n")
print("=" * 75)
print("TEST 2 - REMOVE BACKEND IDENTITY")
print("=" * 75)


# Use the same train/test split
# so this is a fair comparison.

model_no_backend, predicted_no_backend, columns_no_backend = (
    train_and_predict(
        train_df,
        test_df,
        include_backend=False
    )
)


# --------------------------------------------
# Measure performance
# --------------------------------------------

no_backend_mae = mean_absolute_error(
    actual_fidelity,
    predicted_no_backend
)

no_backend_rmse = np.sqrt(
    mean_squared_error(
        actual_fidelity,
        predicted_no_backend
    )
)


print("\nFIDELITY PREDICTION")
print("-" * 75)

print(
    f"ML WITH backend name    MAE: "
    f"{ml_mae:.6f}"
)

print(
    f"ML WITHOUT backend name MAE: "
    f"{no_backend_mae:.6f}"
)

print(
    f"\nML WITH backend name    RMSE: "
    f"{ml_rmse:.6f}"
)

print(
    f"ML WITHOUT backend name RMSE: "
    f"{no_backend_rmse:.6f}"
)


# ============================================================
# TEST 3
# ============================================================
#
# BACKEND SELECTION
#
# We compare:
#
#   ESP choice
#   ML choice
#   Oracle choice
# ============================================================

print("\n")
print("=" * 75)
print("TEST 3 - BACKEND SELECTION")
print("=" * 75)


selection_df = test_df.copy()

selection_df["predicted_fidelity"] = predicted_fidelity

selection_df["predicted_no_backend"] = (
    predicted_no_backend
)


# --------------------------------------------
# Oracle winner
# --------------------------------------------

oracle_rows = selection_df.loc[
    selection_df.groupby(
        ["circuit_id", "condition"]
    )["fidelity"].idxmax()
].copy()

oracle_rows = oracle_rows.rename(
    columns={
        "backend": "oracle_backend"
    }
)


# --------------------------------------------
# ESP winner
# --------------------------------------------

esp_rows = selection_df.loc[
    selection_df.groupby(
        ["circuit_id", "condition"]
    )["esp"].idxmax()
].copy()

esp_rows = esp_rows.rename(
    columns={
        "backend": "esp_backend"
    }
)


# --------------------------------------------
# ML winner
# --------------------------------------------

ml_rows = selection_df.loc[
    selection_df.groupby(
        ["circuit_id", "condition"]
    )["predicted_fidelity"].idxmax()
].copy()

ml_rows = ml_rows.rename(
    columns={
        "backend": "ml_backend"
    }
)


# --------------------------------------------
# ML without backend identity
# --------------------------------------------

ml_no_backend_rows = selection_df.loc[
    selection_df.groupby(
        ["circuit_id", "condition"]
    )["predicted_no_backend"].idxmax()
].copy()

ml_no_backend_rows = ml_no_backend_rows.rename(
    columns={
        "backend": "ml_no_backend"
    }
)


# --------------------------------------------
# Combine results
# --------------------------------------------

selection = oracle_rows[
    [
        "circuit_id",
        "condition",
        "oracle_backend"
    ]
].merge(
    esp_rows[
        [
            "circuit_id",
            "condition",
            "esp_backend"
        ]
    ],
    on=[
        "circuit_id",
        "condition"
    ]
).merge(
    ml_rows[
        [
            "circuit_id",
            "condition",
            "ml_backend"
        ]
    ],
    on=[
        "circuit_id",
        "condition"
    ]
).merge(
    ml_no_backend_rows[
        [
            "circuit_id",
            "condition",
            "ml_no_backend"
        ]
    ],
    on=[
        "circuit_id",
        "condition"
    ]
)


# --------------------------------------------
# Check correctness
# --------------------------------------------

selection["esp_correct"] = (
    selection["esp_backend"]
    ==
    selection["oracle_backend"]
)

selection["ml_correct"] = (
    selection["ml_backend"]
    ==
    selection["oracle_backend"]
)

selection["ml_no_backend_correct"] = (
    selection["ml_no_backend"]
    ==
    selection["oracle_backend"]
)


print("\nBACKEND SELECTION ACCURACY")
print("-" * 75)

print(
    f"ESP: "
    f"{selection['esp_correct'].mean():.2%}"
)

print(
    f"ML with backend identity: "
    f"{selection['ml_correct'].mean():.2%}"
)

print(
    f"ML without backend identity: "
    f"{selection['ml_no_backend_correct'].mean():.2%}"
)


# ============================================================
# 4. SAVE RESULTS
# ============================================================

selection.to_csv(
    "results/validation_selection.csv",
    index=False
)


test_output = test_df.copy()

test_output["ml_predicted_fidelity"] = predicted_fidelity

test_output["ml_no_backend_fidelity"] = (
    predicted_no_backend
)

test_output.to_csv(
    "results/validation_predictions.csv",
    index=False
)


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("VALIDATION COMPLETE")
print("=" * 75)

print("\nSaved:")
print("results/validation_selection.csv")
print("results/validation_predictions.csv")