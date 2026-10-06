import ast
import math
from pathlib import Path

import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService


print("=" * 90)
print("CALIBRATIONCOMPASS - FIX BELL DATASET")
print("=" * 90)


BASE = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_v2.csv"
)

OUTPUT_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_final.csv"
)

JOBS = {
    "ibm_fez": "db2cmf42ljfc73d496a0",
    "ibm_kingston": "db2cmjk2ljfc73d496e0",
    "ibm_marrakesh": "db2cmnnr11fs7396hmq0"
}

SEEDS = [11, 22, 33, 44, 55, 66]

BELL_MAPPINGS = {
    11: [110, 109, 101],
    22: [43, 42, 13],
    33: [65, 77, 2],
    44: [63, 56, 91],
    55: [89, 78, 98],
    66: [69, 78, 26]
}


service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)


def bell_circuit():

    qc = QuantumCircuit(3)

    qc.h(0)
    qc.cx(0, 1)

    return qc


def bell_ideal_distribution():

    qc = bell_circuit()

    state = Statevector.from_instruction(qc)

    full = state.probabilities_dict()

    ideal = {}

    for key, probability in full.items():

        key = str(key).replace(" ", "")

        # Keep q1 and q0 only.
        # Qiskit key ordering is q2 q1 q0.

        logical_key = key[-2:]

        ideal[logical_key] = (
            ideal.get(logical_key, 0.0)
            + float(probability)
        )

    return ideal


def project_counts(counts, mapping):

    projected = {}

    for raw_key, count in counts.items():

        key = str(raw_key).replace(" ", "")

        if len(key) != 156:

            raise ValueError(
                f"Expected 156 bits, got {len(key)}"
            )

        # Bell uses logical q0 and q1.
        # Qiskit displays q1 before q0.

        q1_position = (
            len(key) - 1 - mapping[1]
        )

        q0_position = (
            len(key) - 1 - mapping[0]
        )

        logical_key = (
            key[q1_position]
            + key[q0_position]
        )

        projected[logical_key] = (
            projected.get(
                logical_key,
                0
            )
            + int(count)
        )

    return projected


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

    keys = (
        set(ideal_probs)
        |
        set(measured_probs)
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


if not INPUT_FILE.exists():

    print("Input file not found:")
    print(INPUT_FILE)
    raise SystemExit


df = pd.read_csv(INPUT_FILE)

ideal = bell_ideal_distribution()

print()
print("Bell ideal distribution:")
print(ideal)

fixed = 0


for backend_name, job_id in JOBS.items():

    print()
    print("-" * 90)
    print("BACKEND:", backend_name)
    print("-" * 90)

    job = service.job(job_id)

    print(
        "Job status:",
        job.status()
    )

    result = job.result()

    # Bell occupies the first six circuits
    # in each 12-circuit saved job.

    for index, seed in enumerate(SEEDS):

        counts = (
            result[index]
            .data.meas
            .get_counts()
        )

        mapping = BELL_MAPPINGS[seed]

        projected = project_counts(
            counts,
            mapping
        )

        fidelity = hellinger_fidelity(
            ideal,
            projected
        )

        mask = (
            (df["backend"] == backend_name)
            &
            (df["circuit_type"] == "Bell")
            &
            (
                df["candidate"].astype(int)
                == seed
            )
        )

        indices = df.index[mask].tolist()

        if len(indices) != 1:

            print(
                "ERROR: expected one Bell row."
            )

            raise SystemExit

        row_index = indices[0]

        df.at[
            row_index,
            "fidelity"
        ] = fidelity

        fixed += 1

        print(
            f"Candidate {seed:>2} | "
            f"fidelity {fidelity:.6f}"
        )


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    "Bell rows fixed:",
    fixed
)

print(
    "Total rows:",
    len(df)
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("DONE")