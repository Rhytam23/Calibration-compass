import os
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


print("=" * 90)
print("CALIBRATIONCOMPASS - LEARNED MODEL VS LIVE HARDWARE")
print("=" * 90)

BASE = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(
    BASE,
    "results",
    "candidate_features_v2.csv"
)

LIVE_FILE = os.path.join(
    BASE,
    "results",
    "live_calibration_analysis.csv"
)

if not os.path.exists(TRAIN_FILE):
    print("Training file not found:")
    print(TRAIN_FILE)
    raise SystemExit

if not os.path.exists(LIVE_FILE):
    print("Live analysis file not found:")
    print(LIVE_FILE)
    raise SystemExit


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
live_df = pd.read_csv(LIVE_FILE)

print()
print("Training rows:", len(train_df))
print("Live rows:", len(live_df))


# ------------------------------------------------------------
# Find common numeric features
# ------------------------------------------------------------

excluded = {
    "actual_fidelity",
    "fidelity",
    "candidate",
    "job_id"
}

common = []

for column in train_df.columns:

    if column in excluded:
        continue

    if column not in live_df.columns:
        continue

    if (
        pd.api.types.is_numeric_dtype(train_df[column])
        and
        pd.api.types.is_numeric_dtype(live_df[column])
    ):
        common.append(column)


print()
print("Common numeric features:")

for column in common:
    print(" ", column)


# ------------------------------------------------------------
# Add backend as categorical feature
# ------------------------------------------------------------

use_backend = (
    "backend" in train_df.columns
    and
    "backend" in live_df.columns
)

if use_backend:

    train_backends = set(train_df["backend"].astype(str))
    live_backends = set(live_df["backend"].astype(str))

    # Training uses simulated fake backends (Fez/Sherbrooke/Torino);
    # live data uses real devices (ibm_fez/...). With no shared names
    # the backend one-hot columns would be all-zero on live rows, so
    # they carry no information and are skipped.
    if not (train_backends & live_backends):
        print()
        print(
            "Backend names do not overlap between training "
            f"{sorted(train_backends)} and live "
            f"{sorted(live_backends)}; not using backend as a feature."
        )
        use_backend = False

if use_backend:
    print()
    print("Using backend as categorical feature.")


if len(common) == 0:
    print()
    print("No common numeric features were found.")
    raise SystemExit


# ------------------------------------------------------------
# Build feature matrices
# ------------------------------------------------------------

X_train = train_df[common].copy()
X_live = live_df[common].copy()

if use_backend:

    combined_backend = pd.concat(
        [
            train_df[["backend"]],
            live_df[["backend"]]
        ],
        ignore_index=True
    )

    backend_encoded = pd.get_dummies(
        combined_backend,
        columns=["backend"],
        dtype=float
    )

    train_backend = backend_encoded.iloc[
        :len(train_df)
    ].reset_index(drop=True)

    live_backend = backend_encoded.iloc[
        len(train_df):
    ].reset_index(drop=True)

    X_train = pd.concat(
        [
            X_train.reset_index(drop=True),
            train_backend
        ],
        axis=1
    )

    X_live = pd.concat(
        [
            X_live.reset_index(drop=True),
            live_backend
        ],
        axis=1
    )


# Replace invalid values

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_live = X_live.replace(
    [np.inf, -np.inf],
    np.nan
)

# Make absolutely sure columns are identical

X_live = X_live.reindex(
    columns=X_train.columns
)

y_train = train_df["actual_fidelity"]


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

model = XGBRegressor(
    n_estimators=250,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Predict live hardware
# ------------------------------------------------------------

predictions = model.predict(X_live)

live_df["predicted_fidelity"] = predictions

live_df["prediction_error"] = (
    live_df["predicted_fidelity"]
    - live_df["fidelity"]
)


# ------------------------------------------------------------
# Performance
# ------------------------------------------------------------

mae = mean_absolute_error(
    live_df["fidelity"],
    live_df["predicted_fidelity"]
)

rmse = np.sqrt(
    mean_squared_error(
        live_df["fidelity"],
        live_df["predicted_fidelity"]
    )
)

print()
print("=" * 90)
print("LIVE HARDWARE PREDICTION")
print("=" * 90)

print(
    "MAE:",
    f"{mae:.6f}"
)

print(
    "RMSE:",
    f"{rmse:.6f}"
)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

print()
print(
    f"{'backend':<16}"
    f"{'candidate':>10}"
    f"{'actual':>12}"
    f"{'predicted':>12}"
    f"{'error':>12}"
)

for _, row in live_df.iterrows():

    print(
        f"{row['backend']:<16}"
        f"{int(row['candidate']):>10}"
        f"{row['fidelity']:>12.6f}"
        f"{row['predicted_fidelity']:>12.6f}"
        f"{row['prediction_error']:>12.6f}"
    )


# ------------------------------------------------------------
# Model decision
# ------------------------------------------------------------

predicted_best = live_df.loc[
    live_df["predicted_fidelity"].idxmax()
]

actual_best = live_df.loc[
    live_df["fidelity"].idxmax()
]

print()
print("=" * 90)
print("MODEL DECISION VS ACTUAL HARDWARE")
print("=" * 90)

print()
print("Model prediction:")
print("Backend:", predicted_best["backend"])
print("Candidate:", int(predicted_best["candidate"]))
print(
    "Predicted fidelity:",
    f"{predicted_best['predicted_fidelity']:.6f}"
)

print()
print("Actual hardware:")
print("Backend:", actual_best["backend"])
print("Candidate:", int(actual_best["candidate"]))
print(
    "Actual fidelity:",
    f"{actual_best['fidelity']:.6f}"
)

match = (
    predicted_best["backend"]
    == actual_best["backend"]
    and
    int(predicted_best["candidate"])
    == int(actual_best["candidate"])
)

print()
print(
    "Prediction matches actual best:",
    match
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_file = os.path.join(
    BASE,
    "results",
    "live_model_predictions.csv"
)

live_df.to_csv(
    output_file,
    index=False
)

print()
print("Saved:")
print(output_file)

print()
print("DONE")