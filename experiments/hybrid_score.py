import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import pandas as pd


# ============================================================
# LOAD NOISE PROFILE
# ============================================================

df = pd.read_csv(
    "results/candidate_noise_profile.csv"
)


# ============================================================
# CREATE CALIBRATION LOSSES
# ============================================================

df["gate_loss"] = 1.0 - df["gate_only"]

df["readout_loss"] = (
    1.0 - df["readout_only"]
)

df["thermal_loss"] = (
    1.0 - df["thermal_only"]
)


# ============================================================
# HYBRID SCORES
# ============================================================

# Equal weighting
df["score_equal"] = (
    df["gate_loss"]
    +
    df["readout_loss"]
)


# Gate-weighted
df["score_gate_heavy"] = (
    1.5 * df["gate_loss"]
    +
    1.0 * df["readout_loss"]
)


# Readout-weighted
df["score_readout_heavy"] = (
    1.0 * df["gate_loss"]
    +
    1.5 * df["readout_loss"]
)


# Include thermal slightly
df["score_with_thermal"] = (
    df["gate_loss"]
    +
    df["readout_loss"]
    +
    0.25 * df["thermal_loss"]
)


# Multiplicative survival-style score
df["score_multiplicative"] = (
    (1.0 - df["gate_loss"])
    *
    (1.0 - df["readout_loss"])
)


# ============================================================
# ACTUAL RESULT
# ============================================================

df["actual_rank"] = (
    df["full_backend"]
    .rank(
        ascending=False,
        method="min"
    )
)


# ============================================================
# PRINT
# ============================================================

print("=" * 100)
print("CALIBRATIONCOMPASS - HYBRID CALIBRATION SCORE")
print("=" * 100)

print()

print(
    df[
        [
            "candidate_seed",
            "full_backend",
            "gate_loss",
            "readout_loss",
            "thermal_loss",
            "score_equal",
            "score_gate_heavy",
            "score_readout_heavy",
            "score_with_thermal",
            "score_multiplicative"
        ]
    ]
    .sort_values(
        "full_backend",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# WINNERS
# ============================================================

print()
print("=" * 100)
print("PREDICTED WINNERS")
print("=" * 100)

score_columns = [
    "score_equal",
    "score_gate_heavy",
    "score_readout_heavy",
    "score_with_thermal"
]


for column in score_columns:

    winner = df.loc[
        df[column].idxmin()
    ]

    print(
        f"{column:<25}"
        f"Seed {int(winner['candidate_seed'])}"
    )


winner = df.loc[
    df["score_multiplicative"].idxmax()
]

print(
    f"{'score_multiplicative':<25}"
    f"Seed {int(winner['candidate_seed'])}"
)


actual_winner = df.loc[
    df["full_backend"].idxmax()
]

print()
print(
    "ACTUAL WINNER:",
    int(actual_winner["candidate_seed"])
)


# ============================================================
# RANK CORRELATION
# ============================================================

print()
print("=" * 100)
print("RANK CORRELATION WITH ACTUAL FIDELITY")
print("=" * 100)

for column in score_columns:

    # Lower score is better
    predicted = (
        -df[column]
    )

    correlation = (
        predicted.corr(
            df["full_backend"],
            method="spearman"
        )
    )

    print(
        f"{column:<25}"
        f"{correlation:.4f}"
    )


correlation = (
    df["score_multiplicative"]
    .corr(
        df["full_backend"],
        method="spearman"
    )
)

print(
    f"{'score_multiplicative':<25}"
    f"{correlation:.4f}"
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "results/hybrid_score.csv",
    index=False
)

print()
print("Saved:")
print("results/hybrid_score.csv")

print()
print("=" * 100)
print("HYBRID SCORE COMPLETE")
print("=" * 100)