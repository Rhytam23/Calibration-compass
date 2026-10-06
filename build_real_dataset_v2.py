import ast
import csv
import math
from pathlib import Path

import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService


print("=" * 90)
print("CALIBRATIONCOMPASS - BUILD REAL DATASET V2")
print("=" * 90)

BASE = Path(__file__).resolve().parent

GHZ_FILE = (
    BASE
    / "results"
    / "live_hardware_validation.csv"
)

OUTPUT_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_v2.csv"
)

BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh"
]

SEEDS = [11, 22, 33, 44, 55, 66]

JOBS = {
    "ibm_fez": "db2cmf42ljfc73d496a0",
    "ibm_kingston": "db2cmjk2ljfc73d496e0",
    "ibm_marrakesh": "db2cmnnr11fs7396hmq0"
}

# Exact Bell/Ring mappings recovered from the completed jobs.
MAPPINGS = {
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


# ------------------------------------------------------------
# IBM connection
# ------------------------------------------------------------

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)


# ------------------------------------------------------------
# Circuits
# ------------------------------------------------------------

def make_circuit(name):

    if name == "GHZ":
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        return qc

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

    raise ValueError(name)


# ------------------------------------------------------------
# Ideal distribution
# ------------------------------------------------------------

def ideal_distribution(circuit, active_logical=None):
    """Ideal outcome distribution, marginalised onto active_logical qubits.

    Keys are ordered like project_counts(): highest active logical qubit
    on the left.
    """

    state = Statevector.from_instruction(circuit)

    probabilities = state.probabilities_dict()

    n = circuit.num_qubits

    if active_logical is None:
        active_logical = list(range(n))

    marginal = {}

    for key, value in probabilities.items():
        key = str(key).replace(" ", "")
        short = "".join(
            key[n - 1 - q] for q in reversed(active_logical)
        )
        marginal[short] = marginal.get(short, 0.0) + float(value)

    return {k: v for k, v in marginal.items() if v > 1e-12}


# ------------------------------------------------------------
# Project 156-bit hardware counts onto logical qubits
# ------------------------------------------------------------

def project_counts(counts, mapping, active_logical_qubits):

    projected = {}

    for raw_key, count in counts.items():

        key = str(raw_key).replace(" ", "")

        if len(key) != 156:
            raise ValueError(
                f"Expected 156-bit result, got {len(key)} bits."
            )

        bits = []

        # Qiskit displays the highest-index bit on the left.
        # Therefore logical bits are reconstructed in q2,q1,q0 order.

        for logical_qubit in reversed(active_logical_qubits):

            physical_qubit = mapping[logical_qubit]

            position = len(key) - 1 - physical_qubit

            bits.append(key[position])

        logical_key = "".join(bits)

        projected[logical_key] = (
            projected.get(logical_key, 0)
            + int(count)
        )

    return projected


# ------------------------------------------------------------
# Hellinger fidelity
# ------------------------------------------------------------

def hellinger_fidelity(ideal_probs, counts):

    total = sum(counts.values())

    if total == 0:
        return 0.0

    measured = {
        key: value / total
        for key, value in counts.items()
    }

    keys = set(ideal_probs) | set(measured)

    coefficient = 0.0

    for key in keys:
        coefficient += math.sqrt(
            ideal_probs.get(key, 0.0)
            *
            measured.get(key, 0.0)
        )

    return coefficient ** 2


# ------------------------------------------------------------
# Feature extraction from the exact job properties
# ------------------------------------------------------------

def extract_features(compiled, props, mapping, active_logical):

    active_physical = [
        mapping[q]
        for q in active_logical
    ]

    gate_error_sum = 0.0
    max_gate_error = 0.0

    two_q_error_sum = 0.0
    max_two_q_error = 0.0

    gate_length_sum = 0.0
    two_q_gate_length_sum = 0.0

    gate_count = 0
    one_q_count = 0
    two_q_count = 0

    exact_2q_edges = []

    for item in compiled.data:

        operation = item.operation
        qargs = item.qubits
        name = operation.name.lower()

        if name in [
            "measure",
            "barrier",
            "delay",
            "reset"
        ]:
            continue

        physical = [
            compiled.find_bit(q).index
            for q in qargs
        ]

        gate_count += 1

        if len(physical) == 2:
            two_q_count += 1
        else:
            one_q_count += 1

        try:
            error = props.gate_error(
                name,
                physical
            )
        except Exception:
            error = None

        if error is not None:

            gate_error_sum += error

            max_gate_error = max(
                max_gate_error,
                error
            )

            if len(physical) == 2:

                two_q_error_sum += error

                max_two_q_error = max(
                    max_two_q_error,
                    error
                )

                exact_2q_edges.append(
                    str(tuple(physical))
                )

        try:
            length = props.gate_length(
                name,
                physical
            )
        except Exception:
            length = None

        if length is not None:

            gate_length_sum += length

            if len(physical) == 2:
                two_q_gate_length_sum += length

    readout_errors = []

    t1_values = []
    t2_values = []

    for q in active_physical:

        try:
            value = props.readout_error(q)
        except Exception:
            value = None

        if value is not None:
            readout_errors.append(value)

        try:
            value = props.t1(q)
        except Exception:
            value = None

        if value is not None:
            t1_values.append(value)

        try:
            value = props.t2(q)
        except Exception:
            value = None

        if value is not None:
            t2_values.append(value)

    readout_sum = sum(readout_errors)

    avg_readout = (
        readout_sum / len(readout_errors)
        if readout_errors
        else math.nan
    )

    max_readout = (
        max(readout_errors)
        if readout_errors
        else math.nan
    )

    if readout_errors:

        success = 1.0

        for error in readout_errors:
            success *= 1.0 - error

        readout_product_loss = 1.0 - success

    else:
        readout_product_loss = math.nan

    return {
        "mapping": str(mapping),
        "active_physical_qubits": str(active_physical),
        "gate_count": gate_count,
        "1Q_gates": one_q_count,
        "2Q_gates": two_q_count,
        "readout_sum": readout_sum,
        "avg_readout": avg_readout,
        "max_readout": max_readout,
        "readout_product_loss": readout_product_loss,
        "gate_error_sum": gate_error_sum,
        "max_gate_error": max_gate_error,
        "2Q_gate_error_sum": two_q_error_sum,
        "max_2Q_gate_error": max_two_q_error,
        "gate_length_sum": gate_length_sum,
        "2Q_gate_length_sum": two_q_gate_length_sum,
        "avg_t1": (
            sum(t1_values) / len(t1_values)
            if t1_values
            else math.nan
        ),
        "min_t1": (
            min(t1_values)
            if t1_values
            else math.nan
        ),
        "avg_t2": (
            sum(t2_values) / len(t2_values)
            if t2_values
            else math.nan
        ),
        "min_t2": (
            min(t2_values)
            if t2_values
            else math.nan
        ),
        "2Q_edges": ";".join(exact_2q_edges)
    }


# ------------------------------------------------------------
# Load previous GHZ results
# ------------------------------------------------------------

if not GHZ_FILE.exists():

    print("Missing:")
    print(GHZ_FILE)
    raise SystemExit

ghz_df = pd.read_csv(GHZ_FILE)

rows = []


# ------------------------------------------------------------
# Process all three saved hardware jobs
# ------------------------------------------------------------

for backend_name in BACKENDS:

    print()
    print("-" * 90)
    print("PROCESSING:", backend_name)
    print("-" * 90)

    job_id = JOBS[backend_name]

    backend = service.backend(backend_name)

    job = service.job(job_id)

    print(
        "Job status:",
        job.status()
    )

    # IMPORTANT:
    # These are the properties associated with this job,
    # rather than the current live calibration snapshot.

    props = job.properties()

    if props is None:

        print(
            "ERROR: job properties unavailable."
        )

        raise SystemExit

    calibration_time = props.last_update_date

    job_time = job.creation_date

    print(
        "Job creation:",
        job_time
    )

    print(
        "Job calibration:",
        calibration_time
    )

    # --------------------------------------------------------
    # GHZ
    # --------------------------------------------------------

    backend_ghz = ghz_df[
        ghz_df["backend"] == backend_name
    ]

    ghz_circuit = make_circuit("GHZ")

    for _, old_row in backend_ghz.iterrows():

        seed = int(old_row["candidate"])

        compiled = transpile(
            ghz_circuit,
            backend=backend,
            optimization_level=3,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=seed
        )

        mapping = ast.literal_eval(
            str(old_row["mapping"])
        )

        features = extract_features(
            compiled,
            props,
            mapping,
            [0, 1, 2]
        )

        if job_time is not None and calibration_time is not None:

            calibration_age = (
                job_time - calibration_time
            ).total_seconds() / 60.0

        else:
            calibration_age = math.nan

        row = {
            "circuit_type": "GHZ",
            "backend": backend_name,
            "candidate": seed,
            "depth": compiled.depth(),
            "calibration_time": str(calibration_time),
            "job_time": str(job_time),
            "calibration_age_minutes": calibration_age,
            "fidelity": float(old_row["fidelity"])
        }

        row.update(features)

        rows.append(row)

    # --------------------------------------------------------
    # Bell + Ring
    # --------------------------------------------------------

    job_result = job.result()

    result_index = 0

    for circuit_name in ["Bell", "Ring"]:

        circuit = make_circuit(
            circuit_name
        )

        if circuit_name == "Bell":
            active_logical = [0, 1]
        else:
            active_logical = [0, 1, 2]

        ideal_probs = ideal_distribution(
            circuit,
            active_logical
        )

        for seed in SEEDS:

            mapping = MAPPINGS[
                circuit_name
            ][seed]

            counts = (
                job_result[result_index]
                .data.meas
                .get_counts()
            )

            projected = project_counts(
                counts,
                mapping,
                active_logical
            )

            fidelity = hellinger_fidelity(
                ideal_probs,
                projected
            )

            compiled = transpile(
                circuit,
                backend=backend,
                optimization_level=3,
                layout_method="sabre",
                routing_method="sabre",
                seed_transpiler=seed
            )

            features = extract_features(
                compiled,
                props,
                mapping,
                active_logical
            )

            if job_time is not None and calibration_time is not None:

                calibration_age = (
                    job_time - calibration_time
                ).total_seconds() / 60.0

            else:
                calibration_age = math.nan

            row = {
                "circuit_type": circuit_name,
                "backend": backend_name,
                "candidate": seed,
                "depth": compiled.depth(),
                "calibration_time": str(calibration_time),
                "job_time": str(job_time),
                "calibration_age_minutes": calibration_age,
                "fidelity": fidelity
            }

            row.update(features)

            rows.append(row)

            print(
                f"{circuit_name:<8} "
                f"candidate {seed:>2} | "
                f"fidelity {fidelity:.6f}"
            )

            result_index += 1


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_columns = [
    "circuit_type",
    "backend",
    "candidate",
    "mapping",
    "active_physical_qubits",
    "depth",
    "gate_count",
    "1Q_gates",
    "2Q_gates",
    "readout_sum",
    "avg_readout",
    "max_readout",
    "readout_product_loss",
    "gate_error_sum",
    "max_gate_error",
    "2Q_gate_error_sum",
    "max_2Q_gate_error",
    "gate_length_sum",
    "2Q_gate_length_sum",
    "avg_t1",
    "min_t1",
    "avg_t2",
    "min_t2",
    "2Q_edges",
    "calibration_time",
    "job_time",
    "calibration_age_minutes",
    "fidelity"
]

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=output_columns
    )

    writer.writeheader()

    for row in rows:

        writer.writerow({
            column: row.get(
                column,
                math.nan
            )
            for column in output_columns
        })


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 90)
print("REAL DATASET V2 SUMMARY")
print("=" * 90)

print(
    "Total rows:",
    len(rows)
)

for circuit_name in ["GHZ", "Bell", "Ring"]:

    count = sum(
        1
        for row in rows
        if row["circuit_type"] == circuit_name
    )

    print(
        circuit_name,
        "rows:",
        count
    )

for backend_name in BACKENDS:

    count = sum(
        1
        for row in rows
        if row["backend"] == backend_name
    )

    print(
        backend_name,
        "rows:",
        count
    )

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("DONE")