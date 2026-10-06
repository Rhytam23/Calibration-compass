"""
CalibrationCompass - Readout-Aware Candidate Selector

This experiment tests whether mapping-aware readout calibration
features can improve candidate selection over ESP.

Selectors tested:

1. ESP
2. Lowest weighted_readout_total
3. Lowest weighted_readout_2q
4. Lowest top2_readout_risk
5. Readout composite

The test uses all 30 circuit/backend decisions.

No ML is trained here.
"""


from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "results"
    / "candidate_features_v2.csv"
)

RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# HELPERS
# ============================================================

def find_column(df, names, description):

    for name in names:

        if name in df.columns:
            return name

    raise ValueError(
        f"\nCould not find {description}.\n"
        f"Tried: {names}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


def numeric(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# START
# ============================================================

print("=" * 90)
print("CALIBRATIONCOMPASS - READOUT-AWARE SELECTOR")
print("=" * 90)


# ============================================================
# LOAD
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
)


print(
    f"\nRows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

circuit_col = find_column(
    df,
    ["circuit_id", "circuit"],
    "circuit ID"
)

backend_col = find_column(
    df,
    ["backend"],
    "backend"
)

candidate_col = find_column(
    df,
    ["candidate_seed", "seed", "candidate"],
    "candidate seed"
)

fidelity_col = find_column(
    df,
    ["actual_fidelity", "fidelity"],
    "actual fidelity"
)

esp_col = find_column(
    df,
    ["esp", "ESP", "esp_score"],
    "ESP"
)


# ============================================================
# NORMALIZE
# ============================================================

df = df.rename(
    columns={
        circuit_col: "circuit_id",
        backend_col: "backend",
        candidate_col: "candidate_seed",
        fidelity_col: "actual_fidelity",
        esp_col: "esp",
    }
)


# ============================================================
# CLEAN
# ============================================================

df["circuit_id"] = pd.to_numeric(
    df["circuit_id"],
    errors="coerce"
)

df["candidate_seed"] = pd.to_numeric(
    df["candidate_seed"],
    errors="coerce"
)

df["actual_fidelity"] = pd.to_numeric(
    df["actual_fidelity"],
    errors="coerce"
)

df["esp"] = pd.to_numeric(
    df["esp"],
    errors="coerce"
)

df["backend"] = (
    df["backend"]
    .astype(str)
)


df = df.dropna(
    subset=[
        "circuit_id",
        "backend",
        "candidate_seed",
        "actual_fidelity",
        "esp",
    ]
).reset_index(
    drop=True
)


# ============================================================
# CHECK READOUT FEATURES
# ============================================================

required_readout_features = [
    "weighted_readout_total",
    "weighted_readout_2q",
    "top2_readout_risk",
]


missing = [
    column
    for column in required_readout_features
    if column not in df.columns
]


if missing:

    raise ValueError(
        "\nMissing required readout features:\n"
        +
        "\n".join(missing)
    )


for column in required_readout_features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# NORMALIZE READOUT FEATURES WITHIN EACH GROUP
# ============================================================

df["readout_total_norm"] = 0.0

df["readout_2q_norm"] = 0.0

df["top2_readout_norm"] = 0.0


for (circuit, backend), group in df.groupby(
    ["circuit_id", "backend"],
    sort=True
):

    idx = group.index


    # --------------------------------------------------------
    # Function for within-group normalization
    # --------------------------------------------------------

    def normalize(series):

        series = series.copy()

        if series.isna().all():

            return pd.Series(
                0.0,
                index=series.index
            )

        series = series.fillna(
            series.median()
        )

        minimum = series.min()

        maximum = series.max()

        if (
            maximum - minimum
            <
            1e-12
        ):

            return pd.Series(
                0.0,
                index=series.index
            )

        return (
            (series - minimum)
            /
            (maximum - minimum)
        )


    df.loc[
        idx,
        "readout_total_norm"
    ] = normalize(
        group[
            "weighted_readout_total"
        ]
    )

    df.loc[
        idx,
        "readout_2q_norm"
    ] = normalize(
        group[
            "weighted_readout_2q"
        ]
    )

    df.loc[
        idx,
        "top2_readout_norm"
    ] = normalize(
        group[
            "top2_readout_risk"
        ]
    )


# ============================================================
# READOUT COMPOSITE
# ============================================================

df["readout_composite"] = (
    0.50
    *
    df["readout_total_norm"]
    +
    0.30
    *
    df["readout_2q_norm"]
    +
    0.20
    *
    df["top2_readout_norm"]
)


# ============================================================
# BUILD DECISIONS
# ============================================================

records = []


for (circuit, backend), group in df.groupby(
    ["circuit_id", "backend"],
    sort=True
):

    group = group.copy()


    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    oracle_idx = group[
        "actual_fidelity"
    ].idxmax()

    oracle = group.loc[
        oracle_idx
    ]


    # --------------------------------------------------------
    # ESP
    # --------------------------------------------------------

    esp_idx = group[
        "esp"
    ].idxmax()

    esp = group.loc[
        esp_idx
    ]


    # --------------------------------------------------------
    # Lowest total readout
    # --------------------------------------------------------

    total_idx = group[
        "weighted_readout_total"
    ].idxmin()

    total_readout = group.loc[
        total_idx
    ]


    # --------------------------------------------------------
    # Lowest 2Q readout
    # --------------------------------------------------------

    readout_2q_idx = group[
        "weighted_readout_2q"
    ].idxmin()

    readout_2q = group.loc[
        readout_2q_idx
    ]


    # --------------------------------------------------------
    # Lowest top-2 readout risk
    # --------------------------------------------------------

    top2_idx = group[
        "top2_readout_risk"
    ].idxmin()

    top2 = group.loc[
        top2_idx
    ]


    # --------------------------------------------------------
    # Readout composite
    # --------------------------------------------------------

    composite_idx = group[
        "readout_composite"
    ].idxmin()

    composite = group.loc[
        composite_idx
    ]


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    records.append(
        {
            "circuit_id":
                circuit,

            "backend":
                backend,

            "oracle_candidate":
                oracle[
                    "candidate_seed"
                ],

            "oracle_fidelity":
                oracle[
                    "actual_fidelity"
                ],

            "esp_candidate":
                esp[
                    "candidate_seed"
                ],

            "esp_fidelity":
                esp[
                    "actual_fidelity"
                ],

            "readout_total_candidate":
                total_readout[
                    "candidate_seed"
                ],

            "readout_total_fidelity":
                total_readout[
                    "actual_fidelity"
                ],

            "readout_2q_candidate":
                readout_2q[
                    "candidate_seed"
                ],

            "readout_2q_fidelity":
                readout_2q[
                    "actual_fidelity"
                ],

            "top2_candidate":
                top2[
                    "candidate_seed"
                ],

            "top2_fidelity":
                top2[
                    "actual_fidelity"
                ],

            "readout_composite_candidate":
                composite[
                    "candidate_seed"
                ],

            "readout_composite_fidelity":
                composite[
                    "actual_fidelity"
                ],
        }
    )


results = pd.DataFrame(
    records
)


# ============================================================
# REGRETS
# ============================================================

results["esp_regret"] = (
    results[
        "oracle_fidelity"
    ]
    -
    results[
        "esp_fidelity"
    ]
)

results["readout_total_regret"] = (
    results[
        "oracle_fidelity"
    ]
    -
    results[
        "readout_total_fidelity"
    ]
)

results["readout_2q_regret"] = (
    results[
        "oracle_fidelity"
    ]
    -
    results[
        "readout_2q_fidelity"
    ]
)

results["top2_regret"] = (
    results[
        "oracle_fidelity"
    ]
    -
    results[
        "top2_fidelity"
    ]
)

results["readout_composite_regret"] = (
    results[
        "oracle_fidelity"
    ]
    -
    results[
        "readout_composite_fidelity"
    ]
)


# ============================================================
# CORRECTNESS
# ============================================================

results["esp_correct"] = (
    results[
        "esp_candidate"
    ]
    ==
    results[
        "oracle_candidate"
    ]
)

results["readout_total_correct"] = (
    results[
        "readout_total_candidate"
    ]
    ==
    results[
        "oracle_candidate"
    ]
)

results["readout_2q_correct"] = (
    results[
        "readout_2q_candidate"
    ]
    ==
    results[
        "oracle_candidate"
    ]
)

results["top2_correct"] = (
    results[
        "top2_candidate"
    ]
    ==
    results[
        "oracle_candidate"
    ]
)

results["readout_composite_correct"] = (
    results[
        "readout_composite_candidate"
    ]
    ==
    results[
        "oracle_candidate"
    ]
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("READOUT-AWARE RESULTS")
print("=" * 90)


print(
    f"\nTotal decisions: "
    f"{len(results)}"
)


print(
    "\nSelection accuracy:"
)


print(
    f"ESP: "
    f"{results['esp_correct'].mean():.2%}"
)

print(
    f"Lowest weighted readout: "
    f"{results['readout_total_correct'].mean():.2%}"
)

print(
    f"Lowest weighted 2Q readout: "
    f"{results['readout_2q_correct'].mean():.2%}"
)

print(
    f"Lowest top-2 readout risk: "
    f"{results['top2_correct'].mean():.2%}"
)

print(
    f"Readout composite: "
    f"{results['readout_composite_correct'].mean():.2%}"
)


print(
    "\nAverage regret:"
)


print(
    f"ESP: "
    f"{results['esp_regret'].mean():.6f}"
)

print(
    f"Lowest weighted readout: "
    f"{results['readout_total_regret'].mean():.6f}"
)

print(
    f"Lowest weighted 2Q readout: "
    f"{results['readout_2q_regret'].mean():.6f}"
)

print(
    f"Lowest top-2 readout risk: "
    f"{results['top2_regret'].mean():.6f}"
)

print(
    f"Readout composite: "
    f"{results['readout_composite_regret'].mean():.6f}"
)


print(
    "\nMaximum regret:"
)


print(
    f"ESP: "
    f"{results['esp_regret'].max():.6f}"
)

print(
    f"Lowest weighted readout: "
    f"{results['readout_total_regret'].max():.6f}"
)

print(
    f"Lowest weighted 2Q readout: "
    f"{results['readout_2q_regret'].max():.6f}"
)

print(
    f"Lowest top-2 readout risk: "
    f"{results['top2_regret'].max():.6f}"
)

print(
    f"Readout composite: "
    f"{results['readout_composite_regret'].max():.6f}"
)


# ============================================================
# DECISION TABLE
# ============================================================

print("\n")
print("=" * 90)
print("DECISION COMPARISON")
print("=" * 90)


print(
    results[
        [
            "circuit_id",
            "backend",
            "oracle_candidate",
            "esp_candidate",
            "readout_total_candidate",
            "readout_2q_candidate",
            "top2_candidate",
            "readout_composite_candidate",
            "esp_regret",
            "readout_total_regret",
            "readout_2q_regret",
            "top2_regret",
            "readout_composite_regret",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    RESULTS_DIR
    /
    "readout_aware_selector_results.csv"
)


results.to_csv(
    output_file,
    index=False
)


print("\n")
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    f"\n{output_file}"
)

print("\n")
print("=" * 90)
print("DONE")
print("=" * 90)