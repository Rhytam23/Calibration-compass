import math
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


print("=" * 90)
print("CALIBRATIONCOMPASS - FINAL REAL HARDWARE MODEL TEST")
print("=" * 90)


BASE = Path(__file__).resolve().parent

DATA_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_final.csv"
)


if not DATA_FILE.exists():
    print("Dataset not found:")
    print(DATA_FILE)
    raise SystemExit


df = pd.read_csv(DATA_FILE)

df["candidate"] = pd.to_numeric(
    df["candidate"],
    errors="coerce"
)

df["fidelity"] = pd.to_numeric(
    df["fidelity"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "candidate",
        "fidelity"
    ]
).reset_index(drop=True)


print()
print("Rows:", len(df))


# ------------------------------------------------------------
# Numeric features
# ------------------------------------------------------------

excluded = {
    "fidelity",
    "candidate"
}

numeric_features = []

for column in df.columns:

    if column in excluded:
        continue

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):
        numeric_features.append(column)


print()
print("Numeric features:")

for feature in numeric_features:
    print(" ", feature)


# ------------------------------------------------------------
# Backend encoding
# ------------------------------------------------------------

backend_encoded = pd.get_dummies(
    df["backend"],
    prefix="backend",
    dtype=float
)

X = pd.concat(
    [
        df[numeric_features].reset_index(
            drop=True
        ),
        backend_encoded.reset_index(
            drop=True
        )
    ],
    axis=1
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

y = df["fidelity"]


# ------------------------------------------------------------
# Leave-one-circuit-out validation
# ------------------------------------------------------------

circuits = sorted(
    df["circuit_type"].unique()
)

results = []


for test_circuit in circuits:

    print()
    print("-" * 90)
    print("TEST CIRCUIT:", test_circuit)
    print("-" * 90)

    train_mask = (
        df["circuit_type"]
        != test_circuit
    )

    test_mask = (
        df["circuit_type"]
        == test_circuit
    )

    X_train = X.loc[
        train_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    test_df = df.loc[
        test_mask
    ].copy()

    model = XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    test_df["predicted_fidelity"] = (
        model.predict(X_test)
    )


    # --------------------------------------------------------
    # Calibration baseline
    # --------------------------------------------------------

    test_df["calibration_score"] = (
        test_df["readout_sum"].fillna(0)
        +
        test_df["2Q_gate_error_sum"].fillna(0)
    )


    # --------------------------------------------------------
    # Actual best
    # --------------------------------------------------------

    actual = test_df.loc[
        test_df["fidelity"].idxmax()
    ]


    # --------------------------------------------------------
    # Model best
    # --------------------------------------------------------

    predicted = test_df.loc[
        test_df["predicted_fidelity"].idxmax()
    ]


    # --------------------------------------------------------
    # Calibration best
    # --------------------------------------------------------

    calibration = test_df.loc[
        test_df["calibration_score"].idxmin()
    ]


    model_regret = (
        actual["fidelity"]
        -
        predicted["fidelity"]
    )

    calibration_regret = (
        actual["fidelity"]
        -
        calibration["fidelity"]
    )


    model_match = (
        predicted["backend"]
        == actual["backend"]
        and
        int(predicted["candidate"])
        ==
        int(actual["candidate"])
    )


    calibration_match = (
        calibration["backend"]
        == actual["backend"]
        and
        int(calibration["candidate"])
        ==
        int(actual["candidate"])
    )


    print(
        "Actual best:",
        actual["backend"],
        "candidate",
        int(actual["candidate"]),
        "|",
        f"{actual['fidelity']:.6f}"
    )

    print(
        "Model best:",
        predicted["backend"],
        "candidate",
        int(predicted["candidate"]),
        "| predicted",
        f"{predicted['predicted_fidelity']:.6f}"
    )

    print(
        "Calibration best:",
        calibration["backend"],
        "candidate",
        int(calibration["candidate"]),
        "| score",
        f"{calibration['calibration_score']:.6f}"
    )

    print(
        "Model regret:",
        f"{model_regret:.6f}"
    )

    print(
        "Calibration regret:",
        f"{calibration_regret:.6f}"
    )

    print(
        "Model exact match:",
        model_match
    )

    print(
        "Calibration exact match:",
        calibration_match
    )


    results.append({
        "circuit": test_circuit,
        "model_match": model_match,
        "calibration_match": calibration_match,
        "model_regret": model_regret,
        "calibration_regret": calibration_regret
    })


# ------------------------------------------------------------
# Final metrics
# ------------------------------------------------------------

results_df = pd.DataFrame(results)


all_predictions = []

for test_circuit in circuits:

    train_mask = (
        df["circuit_type"]
        != test_circuit
    )

    test_mask = (
        df["circuit_type"]
        == test_circuit
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(
        X.loc[train_mask],
        y.loc[train_mask]
    )

    predictions = model.predict(
        X.loc[test_mask]
    )

    actual_values = y.loc[
        test_mask
    ].values

    for actual_value, predicted_value in zip(
        actual_values,
        predictions
    ):

        all_predictions.append({
            "actual": actual_value,
            "predicted": predicted_value
        })


pred_df = pd.DataFrame(
    all_predictions
)


mae = np.mean(
    np.abs(
        pred_df["actual"]
        -
        pred_df["predicted"]
    )
)

rmse = math.sqrt(
    np.mean(
        (
            pred_df["actual"]
            -
            pred_df["predicted"]
        ) ** 2
    )
)


model_accuracy = (
    results_df["model_match"].mean()
)

calibration_accuracy = (
    results_df["calibration_match"].mean()
)

model_avg_regret = (
    results_df["model_regret"].mean()
)

calibration_avg_regret = (
    results_df["calibration_regret"].mean()
)


print()
print("=" * 90)
print("FINAL RESULTS")
print("=" * 90)

print()
print(
    "Model MAE:",
    f"{mae:.6f}"
)

print(
    "Model RMSE:",
    f"{rmse:.6f}"
)

print()
print(
    "Model selection accuracy:",
    f"{model_accuracy * 100:.2f}%"
)

print(
    "Calibration selection accuracy:",
    f"{calibration_accuracy * 100:.2f}%"
)

print()
print(
    "Model average regret:",
    f"{model_avg_regret:.6f}"
)

print(
    "Calibration average regret:",
    f"{calibration_avg_regret:.6f}"
)


print()
print("=" * 90)
print("PER-CIRCUIT SUMMARY")
print("=" * 90)

for _, row in results_df.iterrows():

    print(
        f"{row['circuit']:<10}"
        f"Model match: {str(row['model_match']):<6}"
        f" | Calibration match: "
        f"{str(row['calibration_match']):<6}"
        f" | Model regret: "
        f"{row['model_regret']:.6f}"
        f" | Calibration regret: "
        f"{row['calibration_regret']:.6f}"
    )


print()
print("DONE")