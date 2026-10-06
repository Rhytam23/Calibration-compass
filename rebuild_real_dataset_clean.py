import ast
import math
from pathlib import Path

import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService


print("=" * 90)
print("CALIBRATIONCOMPASS - CLEAN REAL HARDWARE DATASET")
print("=" * 90)


BASE = Path(__file__).resolve().parent

GHZ_RESULTS = (
    BASE
    / "results"
    / "live_hardware_validation.csv"
)

OUTPUT_FILE = (
    BASE
    / "results"
    / "real_hardware_dataset_clean.csv"
)


BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh"
]

SEEDS = [11, 22, 33, 44, 55, 66]


# Existing completed GHZ jobs
GHZ_JOBS = {
    "ibm_fez": "db2cdcfr11fs7396hcc0",
    "ibm_kingston": "db2cdfs2ljfc73d48s7g",
    "ibm_marrakesh": "db2cdl7r11fs7396hcmg"
}


# Existing completed Bell + Ring jobs
BR_JOBS = {
    "ibm_fez": "db2cmf42ljfc73d496a0",
    "ibm_kingston": "db2cmjk2ljfc73d496e0",
    "ibm_marrakesh": "db2cmnnr11fs7396hmq0"
}


# Exact Bell mappings recovered from the completed jobs
BELL_MAPPINGS = {
    "ibm_fez": {
        11: [110, 109, 101],
        22: [43, 42, 13],
        33: [65, 77, 2],
        44: [63, 56, 91],
        55: [89, 78, 98],
        66: [69, 78, 26]
    },
    "ibm_kingston": {
        11: [110, 109, 101],
        22: [43, 42, 13],
        33: [65, 77, 2],
        44: [63, 56, 91],
        55: [89, 78, 98],
        66: [69, 78, 26]
    },
    "ibm_marrakesh": {
        11: [110, 109, 101],
        22: [43, 42, 13],
        33: [65, 77, 2],
        44: [63, 56, 91],
        55: [89, 78, 98],
        66: [69, 78, 26]
    }
}


# Exact Ring mappings recovered from the completed jobs
RING_MAPPINGS = {
    "ibm_fez": {
        11: [107, 105, 106],
        22: [16, 23, 3],
        33: [22, 36, 21],
        44: [66, 68, 67],
        55: [98, 111, 91],
        66: [51, 71, 58]
    },
    "ibm_kingston": {
        11: [107, 105, 106],
        22: [16, 23, 3],
        33: [22, 36, 21],
        44: [66, 68, 67],
        55: [98, 111, 91],
        66: [51, 71, 58]
    },
    "ibm_marrakesh": {
        11: [107, 105, 106],
        22: [16, 23, 3],
        33: [22, 36, 21],
        44: [66, 68, 67],
        55: [98, 111, 91],
        66: [51, 71, 58]
    }
}


service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)


def make_circuit(name):

    qc = QuantumCircuit(3)

    if name == "GHZ":
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

    elif name == "Bell":
        qc.h(0)
        qc.cx(0, 1)

    elif name == "Ring":
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

    else:
        raise ValueError(
            f"Unknown circuit: {name}"
        )

    return qc


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


def project_counts(
    counts,
    mapping,
    active_logical
):

    projected = {}

    for raw_key, count in counts.items():

        key = str(raw_key).replace(
            " ",
            ""
        )

        if len(key) != 156:

            raise ValueError(
                "Expected 156-bit result, "
                f"got {len(key)} bits."
            )

        bits = []

        # Qiskit displays the highest-index bit on the left.
        # Reconstruct logical q2,q1,q0 order.

        for logical_qubit in reversed(
            active_logical
        ):

            physical_qubit = mapping[
                logical_qubit
            ]

            position = (
                len(key)
                - 1
                - physical_qubit
            )

            bits.append(
                key[position]
            )

        logical_key = "".join(bits)

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
        for key, value in measured_counts.items()
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


def extract_features(
    compiled,
    props,
    mapping,
    active_logical
):

    active_physical = [
        mapping[q]
        for q in active_logical
    ]

    gate_error_sum = 0.0
    max_gate_error = 0.0

    two_q_gate_error_sum = 0.0
    max_two_q_gate_error = 0.0

    gate_length_sum = 0.0
    two_q_gate_length_sum = 0.0

    gate_count = 0
    one_q_count = 0
    two_q_count = 0

    edges = []

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

                two_q_gate_error_sum += error

                max_two_q_gate_error = max(
                    max_two_q_gate_error,
                    error
                )

                edges.append(
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

        readout_product_loss = (
            1.0 - success
        )

    else:

        readout_product_loss = math.nan

    return {
        "mapping": str(mapping),

        "active_physical_qubits":
            str(active_physical),

        "gate_count":
            gate_count,

        "1Q_gates":
            one_q_count,

        "2Q_gates":
            two_q_count,

        "readout_sum":
            readout_sum,

        "avg_readout":
            avg_readout,

        "max_readout":
            max_readout,

        "readout_product_loss":
            readout_product_loss,

        "gate_error_sum":
            gate_error_sum,

        "max_gate_error":
            max_gate_error,

        "2Q_gate_error_sum":
            two_q_gate_error_sum,

        "max_2Q_gate_error":
            max_two_q_gate_error,

        "gate_length_sum":
            gate_length_sum,

        "2Q_gate_length_sum":
            two_q_gate_length_sum,

        "avg_t1":
            (
                sum(t1_values)
                / len(t1_values)
                if t1_values
                else math.nan
            ),

        "min_t1":
            (
                min(t1_values)
                if t1_values
                else math.nan
            ),

        "avg_t2":
            (
                sum(t2_values)
                / len(t2_values)
                if t2_values
                else math.nan
            ),

        "min_t2":
            (
                min(t2_values)
                if t2_values
                else math.nan
            ),

        "2Q_edges":
            ";".join(edges)
    }


if not GHZ_RESULTS.exists():

    print(
        "Missing GHZ result file:"
    )

    print(GHZ_RESULTS)

    raise SystemExit


ghz_df = pd.read_csv(
    GHZ_RESULTS
)

rows = []


for backend_name in BACKENDS:

    print()
    print("-" * 90)
    print(
        "PROCESSING:",
        backend_name
    )
    print("-" * 90)

    backend = service.backend(
        backend_name
    )

    # ========================================================
    # GHZ
    # ========================================================

    ghz_job = service.job(
        GHZ_JOBS[backend_name]
    )

    ghz_result = ghz_job.result()
    ghz_props = ghz_job.properties()

    ghz_job_time = ghz_job.creation_date
    ghz_calibration_time = (
        ghz_props.last_update_date
    )

    print(
        "GHZ job:",
        GHZ_JOBS[backend_name]
    )

    print(
        "GHZ status:",
        ghz_job.status()
    )

    print(
        "GHZ job time:",
        ghz_job_time
    )

    print(
        "GHZ calibration:",
        ghz_calibration_time
    )

    ghz_backend_rows = ghz_df[
        ghz_df["backend"]
        == backend_name
    ]

    for index, seed in enumerate(SEEDS):

        saved = ghz_backend_rows[
            ghz_backend_rows[
                "candidate"
            ].astype(int)
            == seed
        ]

        if len(saved) != 1:

            print(
                "ERROR: missing GHZ mapping "
                f"for candidate {seed}"
            )

            raise SystemExit

        mapping = ast.literal_eval(
            str(
                saved.iloc[0][
                    "mapping"
                ]
            )
        )

        circuit = make_circuit(
            "GHZ"
        )

        compiled = transpile(
            circuit,
            backend=backend,
            optimization_level=3,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=seed
        )

        counts = (
            ghz_result[index]
            .data.meas
            .get_counts()
        )

        example_key = (
            str(next(iter(counts)))
            .replace(" ", "")
            if counts
            else ""
        )

        if len(example_key) == 3:

            projected = {
                str(key).replace(
                    " ",
                    ""
                ): int(value)
                for key, value
                in counts.items()
            }

        else:

            projected = project_counts(
                counts,
                mapping,
                [0, 1, 2]
            )

        ideal = ideal_distribution(
            circuit
        )

        fidelity = hellinger_fidelity(
            ideal,
            projected
        )

        features = extract_features(
            compiled,
            ghz_props,
            mapping,
            [0, 1, 2]
        )

        calibration_age = (
            ghz_job_time
            - ghz_calibration_time
        ).total_seconds() / 60.0

        row = {
            "circuit_type":
                "GHZ",

            "backend":
                backend_name,

            "candidate":
                seed,

            "depth":
                compiled.depth(),

            "calibration_time":
                str(ghz_calibration_time),

            "job_time":
                str(ghz_job_time),

            "calibration_age_minutes":
                calibration_age,

            "fidelity":
                fidelity
        }

        row.update(features)

        rows.append(row)

        print(
            f"GHZ  candidate {seed:>2} | "
            f"fidelity {fidelity:.6f}"
        )


    # ========================================================
    # BELL + RING
    # ========================================================

    br_job = service.job(
        BR_JOBS[backend_name]
    )

    br_result = br_job.result()
    br_props = br_job.properties()

    br_job_time = br_job.creation_date
    br_calibration_time = (
        br_props.last_update_date
    )

    print()
    print(
        "Bell/Ring job:",
        BR_JOBS[backend_name]
    )

    print(
        "Bell/Ring status:",
        br_job.status()
    )

    print(
        "Bell/Ring job time:",
        br_job_time
    )

    print(
        "Bell/Ring calibration:",
        br_calibration_time
    )

    result_index = 0

    for circuit_name in [
        "Bell",
        "Ring"
    ]:

        if circuit_name == "Bell":

            active_logical = [
                0,
                1
            ]

            mapping_table = (
                BELL_MAPPINGS
            )

        else:

            active_logical = [
                0,
                1,
                2
            ]

            mapping_table = (
                RING_MAPPINGS
            )

        circuit = make_circuit(
            circuit_name
        )

        ideal = ideal_distribution(
            circuit,
            active_logical
        )

        for seed in SEEDS:

            mapping = mapping_table[
                backend_name
            ][seed]

            counts = (
                br_result[result_index]
                .data.meas
                .get_counts()
            )

            projected = project_counts(
                counts,
                mapping,
                active_logical
            )

            fidelity = hellinger_fidelity(
                ideal,
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
                br_props,
                mapping,
                active_logical
            )

            calibration_age = (
                br_job_time
                - br_calibration_time
            ).total_seconds() / 60.0

            row = {
                "circuit_type":
                    circuit_name,

                "backend":
                    backend_name,

                "candidate":
                    seed,

                "depth":
                    compiled.depth(),

                "calibration_time":
                    str(br_calibration_time),

                "job_time":
                    str(br_job_time),

                "calibration_age_minutes":
                    calibration_age,

                "fidelity":
                    fidelity
            }

            row.update(features)

            rows.append(row)

            print(
                f"{circuit_name:<5} "
                f"candidate {seed:>2} | "
                f"fidelity {fidelity:.6f}"
            )

            result_index += 1


# ============================================================
# Save dataset
# ============================================================

columns = [
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


df = pd.DataFrame(rows)

for column in columns:

    if column not in df.columns:
        df[column] = math.nan


df = df[columns]

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 90)
print("CLEAN DATASET SUMMARY")
print("=" * 90)

print(
    "Total rows:",
    len(df)
)

print()

print(
    df.groupby(
        "circuit_type"
    )["fidelity"].agg(
        ["count", "min", "max", "mean"]
    )
)

print()

print(
    df.groupby(
        "backend"
    )["fidelity"].agg(
        ["count", "min", "max", "mean"]
    )
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("DONE")