import math
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


print("=" * 90)
print("CALIBRATIONCOMPASS - REAL HARDWARE MODEL VALIDATION")
print("=" * 90)


BASE = Path(__file__).resolve().parent

DATA_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_final.csv"
)

OUTPUT_FILE = (
    BASE
    / "results"
    / "real_model_predictions.csv"
)


if not DATA_FILE.exists():

    print("Dataset not found:")
    print(DATA_FILE)
    raise SystemExit


df = pd.read_csv(DATA_FILE)

print()
print("Dataset rows:", len(df))


# ------------------------------------------------------------
# Clean data
# ------------------------------------------------------------

df["fidelity"] = pd.to_numeric(
    df["fidelity"],
    errors="coerce"
)

df = df.dropna(
    subset=["fidelity"]
).reset_index(drop=True)


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

    if pd.api.types.is_numeric_dtype(df[column]):

        if df[column].notna().any():
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

X_numeric = df[
    numeric_features
].copy()

X = pd.concat(
    [
        X_numeric.reset_index(drop=True),
        backend_encoded.reset_index(drop=True)
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
    df["circuit_type"].dropna().unique()
)

print()
print("Circuits:", circuits)

all_predictions = []


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
        n_estimators=150,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    test_df[
        "predicted_fidelity"
    ] = predictions

    # Calibration baseline:
    # lower readout + 2Q gate error = better
    test_df[
        "calibration_score"
    ] = (
        test_df["readout_sum"].fillna(0)
        +
        test_df["2Q_gate_error_sum"].fillna(0)
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    model_best_index = (
        test_df[
            "predicted_fidelity"
        ].idxmax()
    )

    model_best = test_df.loc[
        model_best_index
    ]

    # --------------------------------------------------------
    # Calibration baseline
    # --------------------------------------------------------

    calibration_best_index = (
        test_df[
            "calibration_score"
        ].idxmin()
    )

    calibration_best = test_df.loc[
        calibration_best_index
    ]

    # --------------------------------------------------------
    # Actual best
    # --------------------------------------------------------

    actual_best_index = (
        test_df[
            "fidelity"
        ].idxmax()
    )

    actual_best = test_df.loc[
        actual_best_index
    ]

    model_regret = (
        actual_best["fidelity"]
        -
        model_best["fidelity"]
    )

    calibration_regret = (
        actual_best["fidelity"]
        -
        calibration_best["fidelity"]
    )

    model_match = (
        model_best["backend"]
        == actual_best["backend"]
        and
        int(model_best["candidate"])
        ==
        int(actual_best["candidate"])
    )

    calibration_match = (
        calibration_best["backend"]
        == actual_best["backend"]
        and
        int(calibration_best["candidate"])
        ==
        int(actual_best["candidate"])
    )

    print(
        "Actual best:",
        actual_best["backend"],
        "candidate",
        int(actual_best["candidate"]),
        "| fidelity",
        f"{actual_best['fidelity']:.6f}"
    )

    print(
        "Model best:",
        model_best["backend"],
        "candidate",
        int(model_best["candidate"]),
        "| predicted",
        f"{model_best['predicted_fidelity']:.6f}"
    )

    print(
        "Calibration best:",
        calibration_best["backend"],
        "candidate",
        int(calibration_best["candidate"]),
        "| score",
        f"{calibration_best['calibration_score']:.6f}"
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

    all_predictions.append(
        test_df
    )


# ------------------------------------------------------------
# Combine predictions
# ------------------------------------------------------------

prediction_df = pd.concat(
    all_predictions,
    ignore_index=True
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

model_mae = np.mean(
    np.abs(
        prediction_df["predicted_fidelity"]
        -
        prediction_df["fidelity"]
    )
)

model_rmse = math.sqrt(
    np.mean(
        (
            prediction_df["predicted_fidelity"]
            -
            prediction_df["fidelity"]
        ) ** 2
    )
)


# Selection metrics
selection_results = []

for circuit_name in circuits:

    group = prediction_df[
        prediction_df["circuit_type"]
        == circuit_name
    ]

    actual_best = group.loc[
        group["fidelity"].idxmax()
    ]

    model_best = group.loc[
        group["predicted_fidelity"].idxmax()
    ]

    calibration_best = group.loc[
        group["calibration_score"].idxmin()
    ]

    selection_results.append({

        "circuit": circuit_name,

        "model_match":
            model_best["backend"]
            == actual_best["backend"]
            and
            int(model_best["candidate"])
            ==
            int(actual_best["candidate"]),

        "calibration_match":
            calibration_best["backend"]
            == actual_best["backend"]
            and
            int(calibration_best["candidate"])
            ==
            int(actual_best["candidate"]),

        "model_regret":
            actual_best["fidelity"]
            -
            model_best["fidelity"],

        "calibration_regret":
            actual_best["fidelity"]
            -
            calibration_best["fidelity"]
    })


selection_df = pd.DataFrame(
    selection_results
)


model_accuracy = (
    selection_df["model_match"].mean()
)

calibration_accuracy = (
    selection_df["calibration_match"].mean()
)

model_avg_regret = (
    selection_df["model_regret"].mean()
)

calibration_avg_regret = (
    selection_df[
        "calibration_regret"
    ].mean()
)


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print()
print("=" * 90)
print("FINAL REAL-HARDWARE VALIDATION")
print("=" * 90)

print()
print(
    "Model MAE:",
    f"{model_mae:.6f}"
)

print(
    "Model RMSE:",
    f"{model_rmse:.6f}"
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


# ------------------------------------------------------------
# Per-circuit results
# ------------------------------------------------------------

print()
print("=" * 90)
print("PER-CIRCUIT RESULTS")
print("=" * 90)

for _, row in selection_df.iterrows():

    print(
        f"{row['circuit']:<10}"
        f"Model match: "
        f"{str(row['model_match']):<6}"
        f"| Calibration match: "
        f"{str(row['calibration_match']):<6}"
        f"| Model regret: "
        f"{row['model_regret']:.6f}"
        f"| Calibration regret: "
        f"{row['calibration_regret']:.6f}"
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("DONE")