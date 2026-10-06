from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent
DATA = BASE / "results" / "candidate_features_v2.csv"

ESP_MARGIN = 0.01
PAIRWISE_ADVANTAGE = 0.02


def build_features(df):
    drop = {"circuit_id", "candidate_seed", "actual_fidelity"}

    for c in df.columns:
        if c.lower() == "physical_qubits" or c.lower() == "mapping" or c.lower().endswith("_physical"):
            drop.add(c)

    X = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    X = X.loc[:, ~X.columns.duplicated()]

    for c in list(X.columns):
        if X[c].isna().any():
            X[f"{c}__missing"] = X[c].isna().astype(float)

    cats = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    if cats:
        X = pd.get_dummies(X, columns=cats, dummy_na=True)

    X = X.loc[:, ~X.columns.duplicated()]

    non_numeric = X.select_dtypes(
        exclude=[np.number, "bool"]
    ).columns.tolist()

    if non_numeric:
        X = X.drop(columns=non_numeric)

    return X.astype(float)


def make_pairs(df, X):
    XP, y = [], []

    for _, group in df.groupby(["circuit_id", "backend"]):
        ids = group.index.tolist()

        for a, b in combinations(ids, 2):
            fa = X.loc[a].values
            fb = X.loc[b].values

            ya = df.loc[a, "actual_fidelity"]
            yb = df.loc[b, "actual_fidelity"]

            if ya == yb:
                continue

            d = fa - fb
            label = 1 if ya > yb else 0

            XP.append(d)
            y.append(label)

            XP.append(-d)
            y.append(1 - label)

    return np.asarray(XP), np.asarray(y)


def train_model(df, X):
    XP, y = make_pairs(df, X)

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
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
    )

    model.fit(XP, y)
    return model


def get_scores(group, X, model):
    rows = []

    for a in group.index:
        probs = []

        for b in group.index:
            if a == b:
                continue

            d = (
                X.loc[a].values -
                X.loc[b].values
            ).reshape(1, -1)

            probs.append(
                model.predict_proba(d)[0, 1]
            )

        rows.append({
            "candidate": group.loc[a, "candidate_seed"],
            "esp": group.loc[a, "esp"],
            "pairwise": np.mean(probs),
            "fidelity": group.loc[a, "actual_fidelity"]
        })

    return pd.DataFrame(rows)


print("=" * 80)
print("CALIBRATIONCOMPASS")
print("=" * 80)

df = pd.read_csv(DATA)

df = df.rename(columns={
    "circuit": "circuit_id",
    "seed": "candidate_seed",
    "fidelity": "actual_fidelity",
    "ESP": "esp"
})

df["circuit_id"] = pd.to_numeric(df["circuit_id"], errors="coerce")
df["candidate_seed"] = pd.to_numeric(df["candidate_seed"], errors="coerce")
df["actual_fidelity"] = pd.to_numeric(df["actual_fidelity"], errors="coerce")
df["esp"] = pd.to_numeric(df["esp"], errors="coerce")

df["backend"] = df["backend"].astype(str)

df = df.dropna(
    subset=[
        "circuit_id",
        "candidate_seed",
        "actual_fidelity",
        "esp"
    ]
).reset_index(drop=True)

df = df.sort_values(
    ["circuit_id", "backend", "candidate_seed"]
).reset_index(drop=True)

X = build_features(df)

print(f"\nCircuits: {df['circuit_id'].nunique()}")
print(f"Backends: {df['backend'].nunique()}")
print(
    "Candidates per group: "
    f"{int(df.groupby(['circuit_id', 'backend']).size().median())}"
)

print("\nAvailable circuits:")
print(
    sorted(
        df["circuit_id"].unique().astype(int).tolist()
    )
)

backends = sorted(df["backend"].unique())

try:
    circuit_id = int(
        input("\nEnter circuit ID: ")
    )
except ValueError:
    raise SystemExit("Circuit ID must be an integer.")

backend = input(
    f"Enter backend ({' / '.join(backends)}): "
).strip()

group = df[
    (df["circuit_id"] == circuit_id) &
    (df["backend"].str.lower() == backend.lower())
].copy()

if group.empty:
    raise SystemExit(
        f"No candidates for circuit {circuit_id} on backend "
        f"'{backend}'. Backends: {', '.join(backends)}"
    )

# Hold the queried circuit out of training so the recommendation is
# out-of-sample rather than scored by a model that saw its labels.
train_mask = df["circuit_id"] != circuit_id

print("\nTraining pairwise model (queried circuit held out)...")

model = train_model(df[train_mask], X[train_mask])

print("Model trained.")

scores = get_scores(
    group,
    X,
    model
)

esp = scores.loc[
    scores["esp"].idxmax()
]

pairwise = scores.loc[
    scores["pairwise"].idxmax()
]

esp_sorted = scores.sort_values(
    "esp",
    ascending=False
)

esp_margin = (
    esp_sorted.iloc[0]["esp"] -
    esp_sorted.iloc[1]["esp"]
)

esp_pairwise = scores.loc[
    scores["candidate"] == esp["candidate"],
    "pairwise"
].iloc[0]

pairwise_advantage = (
    pairwise["pairwise"] -
    esp_pairwise
)

override = (
    pairwise["candidate"] != esp["candidate"]
    and
    esp_margin <= ESP_MARGIN
    and
    pairwise_advantage >= PAIRWISE_ADVANTAGE
)

selected = pairwise if override else esp

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print(
    f"\nBackend: {backend}"
)

print(
    f"Recommended candidate: "
    f"{int(selected['candidate'])}"
)

if override:
    print("\nCalibrationCompass OVERRIDES ESP.")
    print(
        f"ESP margin: {esp_margin:.6f}"
    )
    print(
        f"Pairwise advantage: "
        f"{pairwise_advantage:.6f}"
    )
    print(
        "Reason: ESP is uncertain and the "
        "pairwise model strongly prefers another mapping."
    )
else:
    print("\nCalibrationCompass keeps ESP.")
    print(
        f"ESP margin: {esp_margin:.6f}"
    )
    print(
        "Reason: ESP is sufficiently decisive, "
        "so no override was triggered."
    )

print("\nCandidate ranking:")
print(
    scores.sort_values(
        "pairwise",
        ascending=False
    ).to_string(index=False)
)

print("\nDONE")