import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider
from qiskit.quantum_info import hellinger_fidelity


# ============================================================
# SAME CIRCUIT GENERATOR
# ============================================================

def make_random_circuit(seed):

    rng = np.random.default_rng(seed)

    num_qubits = int(rng.integers(4, 9))
    circuit_depth = int(rng.integers(3, 9))

    qc = QuantumCircuit(num_qubits, num_qubits)

    for _ in range(circuit_depth):

        for q in range(num_qubits):

            choice = rng.integers(0, 3)

            if choice == 0:
                qc.h(q)

            elif choice == 1:
                qc.ry(
                    float(rng.uniform(0, 2 * np.pi)),
                    q
                )

            else:
                qc.rz(
                    float(rng.uniform(0, 2 * np.pi)),
                    q
                )

        number_of_entangling_gates = max(
            1,
            num_qubits // 2
        )

        for _ in range(number_of_entangling_gates):

            q1, q2 = rng.choice(
                num_qubits,
                size=2,
                replace=False
            )

            qc.cx(
                int(q1),
                int(q2)
            )

    qc.measure(
        range(num_qubits),
        range(num_qubits)
    )

    return qc


# ============================================================
# READOUT ERROR
# ============================================================

def get_readout_error(backend, physical):

    try:

        props = backend.target["measure"][(physical,)]

        if props is not None:
            return float(props.error)

    except (KeyError, TypeError):

        pass

    return None


# ============================================================
# FINAL LOGICAL -> PHYSICAL MAPPING
# ============================================================

def get_mapping(transpiled):

    final = transpiled.layout.final_index_layout(
        filter_ancillas=True
    )

    mapping = {}

    for logical, physical in enumerate(final):

        if logical >= transpiled.num_clbits:
            break

        mapping[logical] = physical

    return mapping


# ============================================================
# APPLY INDEPENDENT READOUT BIT FLIPS
# ============================================================

def apply_readout_noise(
    probabilities,
    mapping,
    backend
):

    result = {}

    n = len(next(iter(probabilities.keys())))

    # Start with original distribution
    result = dict(probabilities)

    # Apply each logical qubit's readout channel
    # independently.
    for logical, physical in mapping.items():

        error = get_readout_error(
            backend,
            physical
        )

        if error is None:
            continue

        new_distribution = {}

        bit_position = n - 1 - logical

        for state, probability in result.items():

            flipped = list(state)

            if flipped[bit_position] == "0":
                flipped[bit_position] = "1"
            else:
                flipped[bit_position] = "0"

            flipped_state = "".join(flipped)

            # Probability staying unchanged
            new_distribution[state] = (
                new_distribution.get(state, 0.0)
                + probability * (1.0 - error)
            )

            # Probability of flipping
            new_distribution[flipped_state] = (
                new_distribution.get(flipped_state, 0.0)
                + probability * error
            )

        result = new_distribution

    return result


# ============================================================
# PREDICT ONE CANDIDATE
# ============================================================

def predict_candidate(
    circuit,
    backend,
    seed
):

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed
    )

    mapping = get_mapping(transpiled)

    # --------------------------------------------------------
    # Ideal probability distribution
    # --------------------------------------------------------

    simulator = AerSimulator()

    ideal_result = simulator.run(
        transpiled,
        shots=50000,
        seed_simulator=12345
    ).result()

    ideal_counts = ideal_result.get_counts()

    total = sum(ideal_counts.values())

    ideal_probabilities = {
        state: count / total
        for state, count in ideal_counts.items()
    }

    # --------------------------------------------------------
    # Predict readout-only output
    # --------------------------------------------------------

    predicted_noisy = apply_readout_noise(
        ideal_probabilities,
        mapping,
        backend
    )

    predicted_fidelity = hellinger_fidelity(
        ideal_probabilities,
        predicted_noisy
    )

    # --------------------------------------------------------
    # Simple calibration score
    # --------------------------------------------------------

    readout_sum = 0.0

    for logical, physical in mapping.items():

        error = get_readout_error(
            backend,
            physical
        )

        if error is not None:
            readout_sum += error

    return {
        "candidate_seed": seed,
        "predicted_readout_fidelity": predicted_fidelity,
        "readout_error_sum": readout_sum,
        "mapping": mapping
    }


# ============================================================
# RUN
# ============================================================

CIRCUIT_SEED = 20000

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(
    CIRCUIT_SEED
)

candidate_seeds = [
    11,
    22,
    33,
    44,
    55,
    66
]


# ============================================================
# LOAD ACTUAL BENCHMARK
# ============================================================

actual = pd.read_csv(
    "results/candidate_benchmark.csv"
)

actual = actual[
    (actual["circuit_id"] == 0)
    &
    (actual["backend"] == "Torino")
][
    [
        "candidate_seed",
        "fidelity"
    ]
]


# ============================================================
# PREDICT
# ============================================================

rows = []

for seed in candidate_seeds:

    prediction = predict_candidate(
        circuit,
        backend,
        seed
    )

    row = {
        "candidate_seed":
            seed,

        "predicted_readout_fidelity":
            prediction[
                "predicted_readout_fidelity"
            ],

        "readout_error_sum":
            prediction[
                "readout_error_sum"
            ],

        "actual_fidelity":
            float(
                actual.loc[
                    actual["candidate_seed"] == seed,
                    "fidelity"
                ].iloc[0]
            )
    }

    rows.append(row)


df = pd.DataFrame(rows)


# ============================================================
# RANKINGS
# ============================================================

df["predicted_rank"] = (
    df["predicted_readout_fidelity"]
    .rank(
        ascending=False,
        method="min"
    )
)

df["actual_rank"] = (
    df["actual_fidelity"]
    .rank(
        ascending=False,
        method="min"
    )
)

df = df.sort_values(
    "predicted_readout_fidelity",
    ascending=False
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 100)
print("CALIBRATIONCOMPASS - READOUT PREDICTOR")
print("=" * 100)

print()

print(
    df[
        [
            "candidate_seed",
            "predicted_readout_fidelity",
            "actual_fidelity",
            "readout_error_sum",
            "predicted_rank",
            "actual_rank"
        ]
    ].to_string(index=False)
)


# ============================================================
# WINNERS
# ============================================================

predicted_winner = df.iloc[0]

actual_winner = df.loc[
    df["actual_fidelity"].idxmax()
]

print()
print("=" * 100)
print("WINNER COMPARISON")
print("=" * 100)

print()
print(
    "Predicted winner:",
    int(predicted_winner["candidate_seed"])
)

print(
    "Actual winner:",
    int(actual_winner["candidate_seed"])
)

print()
print(
    "Predicted winner fidelity:",
    f"{predicted_winner['predicted_readout_fidelity']:.6f}"
)

print(
    "Actual winner fidelity:",
    f"{actual_winner['actual_fidelity']:.6f}"
)


# ============================================================
# TOP-1 AGREEMENT
# ============================================================

if (
    int(predicted_winner["candidate_seed"])
    ==
    int(actual_winner["candidate_seed"])
):

    print()
    print("TOP-1 PREDICTION: CORRECT")

else:

    print()
    print("TOP-1 PREDICTION: WRONG")


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "results/readout_predictor.csv",
    index=False
)

print()
print("Saved:")
print("results/readout_predictor.csv")

print()
print("=" * 100)
print("PREDICTOR COMPLETE")
print("=" * 100)