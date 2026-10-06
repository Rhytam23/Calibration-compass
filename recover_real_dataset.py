import ast
import math
from pathlib import Path

import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService


print("=" * 90)
print("CALIBRATIONCOMPASS - RECOVER REAL HARDWARE DATASET")
print("=" * 90)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset.csv"
)

OUTPUT_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_fixed.csv"
)

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)


JOBS = {
    "ibm_fez": "db2cmf42ljfc73d496a0",
    "ibm_kingston": "db2cmjk2ljfc73d496e0",
    "ibm_marrakesh": "db2cmnnr11fs7396hmq0"
}


SEEDS = [11, 22, 33, 44, 55, 66]


# Exact mappings obtained from the completed jobs

MAPPINGS = {

    "ibm_fez": {

        "Bell": {
            11: [110, 109, 101],
            22: [43, 42, 13],
            33: [65, 77, 2],
            44: [63, 56, 91],
            55: [89, 78, 98],
            66: [69, 78, 26]
        },

        "Ring": {
            11: [107, 105, 106],
            22: [16, 23, 3],
            33: [22, 36, 21],
            44: [66, 68, 67],
            55: [98, 111, 91],
            66: [51, 71, 58]
        }
    },

    "ibm_kingston": {

        "Bell": {
            11: [110, 109, 101],
            22: [43, 42, 13],
            33: [65, 77, 2],
            44: [63, 56, 91],
            55: [89, 78, 98],
            66: [69, 78, 26]
        },

        "Ring": {
            11: [107, 105, 106],
            22: [16, 23, 3],
            33: [22, 36, 21],
            44: [66, 68, 67],
            55: [98, 111, 91],
            66: [51, 71, 58]
        }
    },

    "ibm_marrakesh": {

        "Bell": {
            11: [110, 109, 101],
            22: [43, 42, 13],
            33: [65, 77, 2],
            44: [63, 56, 91],
            55: [89, 78, 98],
            66: [69, 78, 26]
        },

        "Ring": {
            11: [107, 105, 106],
            22: [16, 23, 3],
            33: [22, 36, 21],
            44: [66, 68, 67],
            55: [98, 111, 91],
            66: [51, 71, 58]
        }
    }
}


# ------------------------------------------------------------
# Circuits
# ------------------------------------------------------------

def make_circuit(name):

    if name == "Bell":

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)

        return qc

    if name == "Ring":

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

        return qc

    raise ValueError(
        f"Unknown circuit: {name}"
    )


# ------------------------------------------------------------
# Ideal probability distribution
# ------------------------------------------------------------

def ideal_distribution(circuit):

    state = Statevector.from_instruction(
        circuit
    )

    probabilities = (
        state.probabilities_dict()
    )

    return {
        str(key).replace(" ", ""):
        float(value)
        for key, value in probabilities.items()
        if float(value) > 1e-12
    }


# ------------------------------------------------------------
# Convert 156-bit hardware result to 3 logical bits
# ------------------------------------------------------------

def project_counts(counts, mapping):

    projected = {}

    for raw_key, count in counts.items():

        key = str(raw_key).replace(
            " ",
            ""
        )

        if len(key) < 156:
            raise ValueError(
                "Unexpected bitstring length: "
                f"{len(key)}"
            )

        logical_bits = []

        # Qiskit displays q_(n-1) on the left.
        # mapping[logical_qubit] = physical_qubit.

        for logical_qubit in [2, 1, 0]:

            physical_qubit = mapping[
                logical_qubit
            ]

            position = (
                len(key)
                - 1
                - physical_qubit
            )

            logical_bits.append(
                key[position]
            )

        logical_key = "".join(
            logical_bits
        )

        projected[logical_key] = (
            projected.get(
                logical_key,
                0
            )
            + int(count)
        )

    return projected


# ------------------------------------------------------------
# Hellinger fidelity
# ------------------------------------------------------------

def hellinger_fidelity(
    ideal_probs,
    measured_counts
):

    total = sum(
        measured_counts.values()
    )

    if total == 0:
        return 0.0

    measured_probs = {
        key: value / total
        for key, value
        in measured_counts.items()
    }

    keys = set(
        ideal_probs
    ) | set(
        measured_probs
    )

    coefficient = 0.0

    for key in keys:

        coefficient += math.sqrt(
            ideal_probs.get(
                key,
                0.0
            )
            *
            measured_probs.get(
                key,
                0.0
            )
        )

    return coefficient ** 2


# ------------------------------------------------------------
# Check input
# ------------------------------------------------------------

if not INPUT_FILE.exists():

    print(
        "Input dataset not found:"
    )

    print(INPUT_FILE)

    raise SystemExit


df = pd.read_csv(
    INPUT_FILE
)


# ------------------------------------------------------------
# Recover Bell/Ring results
# ------------------------------------------------------------

recovered = 0


for backend_name, job_id in JOBS.items():

    print()
    print("-" * 90)
    print(
        "RECOVERING:",
        backend_name
    )
    print(
        "Job:",
        job_id
    )
    print("-" * 90)

    job = service.job(
        job_id
    )

    print(
        "Job status:",
        job.status()
    )

    result = job.result()

    index = 0

    for circuit_name in [
        "Bell",
        "Ring"
    ]:

        circuit = make_circuit(
            circuit_name
        )

        ideal_probs = (
            ideal_distribution(
                circuit
            )
        )

        for seed in SEEDS:

            mapping = MAPPINGS[
                backend_name
            ][circuit_name][seed]

            counts = (
                result[index]
                .data.meas
                .get_counts()
            )

            projected_counts = (
                project_counts(
                    counts,
                    mapping
                )
            )

            fidelity = (
                hellinger_fidelity(
                    ideal_probs,
                    projected_counts
                )
            )

            mask = (
                (df["backend"] == backend_name)
                &
                (
                    df["circuit_type"]
                    == circuit_name
                )
                &
                (
                    df["candidate"].astype(int)
                    == seed
                )
            )

            matching_rows = df.index[
                mask
            ].tolist()

            if len(matching_rows) != 1:

                print(
                    "ERROR: expected one matching row, "
                    f"found {len(matching_rows)}"
                )

                raise SystemExit

            row_index = matching_rows[0]

            df.at[
                row_index,
                "mapping"
            ] = str(mapping)

            df.at[
                row_index,
                "active_physical_qubits"
            ] = str(sorted(mapping))

            df.at[
                row_index,
                "fidelity"
            ] = fidelity

            df.at[
                row_index,
                "logical_counts"
            ] = str(
                projected_counts
            )

            df.at[
                row_index,
                "job_id"
            ] = job_id

            recovered += 1

            print(
                f"{circuit_name:<8} "
                f"candidate {seed:>2} | "
                f"mapping {str(mapping):<20} | "
                f"fidelity {fidelity:.6f}"
            )

            index += 1


# ------------------------------------------------------------
# Save corrected dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 90)
print("RECOVERY SUMMARY")
print("=" * 90)

print(
    "Recovered Bell/Ring rows:",
    recovered
)

print(
    "Total dataset rows:",
    len(df)
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("DONE")