import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import fake_provider


# ============================================================
# SAME CIRCUIT GENERATOR AS candidate_benchmark.py
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

            qc.cx(int(q1), int(q2))

    qc.measure(
        range(num_qubits),
        range(num_qubits)
    )

    return qc


# ============================================================
# GET CALIBRATION ERROR
# ============================================================

def get_error(backend, instruction_name, qubits):

    qubits = tuple(qubits)

    try:
        props = backend.target[instruction_name][qubits]

        if props is not None:
            return props.error

    except (KeyError, TypeError):
        pass

    # Try reversed direction for 2-qubit gates
    if len(qubits) == 2:

        try:
            props = backend.target[instruction_name][
                (qubits[1], qubits[0])
            ]

            if props is not None:
                return props.error

        except (KeyError, TypeError):
            pass

    return None


# ============================================================
# ANALYZE ONE CANDIDATE
# ============================================================

def analyze_candidate(circuit, backend, seed):

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed
    )

    physical_qubits = set()
    cz_errors = []
    readout_errors = []

    # --------------------------------------------------------
    # Inspect operations
    # --------------------------------------------------------

    for instruction in transpiled.data:

        operation = instruction.operation

        if operation.name == "cz":

            q0 = transpiled.find_bit(
                instruction.qubits[0]
            ).index

            q1 = transpiled.find_bit(
                instruction.qubits[1]
            ).index

            physical_qubits.add(q0)
            physical_qubits.add(q1)

            error = get_error(
                backend,
                "cz",
                (q0, q1)
            )

            if error is not None:
                cz_errors.append(error)

        elif operation.name == "measure":

            q = transpiled.find_bit(
                instruction.qubits[0]
            ).index

            physical_qubits.add(q)

            error = get_error(
                backend,
                "measure",
                (q,)
            )

            if error is not None:
                readout_errors.append(error)

    # --------------------------------------------------------
    # Calibration metrics
    # --------------------------------------------------------

    if cz_errors:

        avg_cz_error = float(
            np.mean(cz_errors)
        )

        max_cz_error = float(
            np.max(cz_errors)
        )

        cz_success = float(
            np.prod(
                1.0 - np.array(cz_errors)
            )
        )

    else:

        avg_cz_error = np.nan
        max_cz_error = np.nan
        cz_success = np.nan

    if readout_errors:

        avg_readout_error = float(
            np.mean(readout_errors)
        )

        max_readout_error = float(
            np.max(readout_errors)
        )

        readout_success = float(
            np.prod(
                1.0 - np.array(readout_errors)
            )
        )

    else:

        avg_readout_error = np.nan
        max_readout_error = np.nan
        readout_success = np.nan

    # Simple calibration-only score
    if np.isnan(cz_success):
        calibration_score = readout_success
    elif np.isnan(readout_success):
        calibration_score = cz_success
    else:
        calibration_score = (
            cz_success * readout_success
        )

    return {
        "candidate_seed": seed,
        "depth": transpiled.depth(),
        "two_qubit_gates": transpiled.count_ops().get("cz", 0),
        "physical_qubits_used": len(physical_qubits),
        "physical_qubit_list": sorted(physical_qubits),
        "avg_cz_error": avg_cz_error,
        "max_cz_error": max_cz_error,
        "avg_readout_error": avg_readout_error,
        "max_readout_error": max_readout_error,
        "cz_success": cz_success,
        "readout_success": readout_success,
        "calibration_score": calibration_score
    }


# ============================================================
# SETTINGS
# ============================================================

CIRCUIT_ID = 0
CIRCUIT_SEED = 20000

CANDIDATE_SEEDS = [
    11, 22, 33, 44, 55, 66
]

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(
    CIRCUIT_SEED
)


# ============================================================
# LOAD EXISTING FIDELITY RESULTS
# ============================================================

benchmark_file = (
    "results/candidate_benchmark.csv"
)

benchmark = pd.read_csv(
    benchmark_file
)

benchmark = benchmark[
    (benchmark["circuit_id"] == CIRCUIT_ID)
    &
    (benchmark["backend"] == "Torino")
].copy()


# ============================================================
# ANALYZE CANDIDATES
# ============================================================

results = []

for seed in CANDIDATE_SEEDS:

    result = analyze_candidate(
        circuit,
        backend,
        seed
    )

    results.append(result)


df = pd.DataFrame(results)


# ============================================================
# ADD ACTUAL FIDELITY
# ============================================================

df = df.merge(
    benchmark[
        [
            "candidate_seed",
            "fidelity",
            "esp"
        ]
    ],
    on="candidate_seed",
    how="left"
)


# ============================================================
# SORT BY CALIBRATION SCORE
# ============================================================

df = df.sort_values(
    "calibration_score",
    ascending=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 100)
print("CALIBRATIONCOMPASS - CALIBRATION ANALYSIS")
print("=" * 100)

print()
print("Circuit:", CIRCUIT_ID)
print("Backend: Torino")
print("Logical qubits:", circuit.num_qubits)

print()
print(
    df[
        [
            "candidate_seed",
            "fidelity",
            "esp",
            "depth",
            "avg_cz_error",
            "max_cz_error",
            "avg_readout_error",
            "calibration_score"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SHOW PHYSICAL MAPPINGS
# ============================================================

print()
print("=" * 100)
print("PHYSICAL QUBITS USED")
print("=" * 100)

for _, row in df.iterrows():

    print()
    print(
        f"Seed {int(row['candidate_seed'])}: "
        f"{row['physical_qubit_list']}"
    )


# ============================================================
# BEST ACCORDING TO EACH METHOD
# ============================================================

best_calibration = df.iloc[0]

best_fidelity = df.loc[
    df["fidelity"].idxmax()
]

print()
print("=" * 100)
print("WINNERS")
print("=" * 100)

print()
print(
    "Calibration score winner:",
    int(best_calibration["candidate_seed"])
)

print(
    "Actual fidelity winner:",
    int(best_fidelity["candidate_seed"])
)

print(
    "Calibration winner fidelity:",
    f"{best_calibration['fidelity']:.6f}"
)

print(
    "Actual best fidelity:",
    f"{best_fidelity['fidelity']:.6f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    "results/calibration_analysis.csv"
)

df.to_csv(
    output_file,
    index=False
)

print()
print("Saved:")
print(output_file)

print()
print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)