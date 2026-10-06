import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import fake_provider


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_CIRCUITS = 10

CANDIDATE_SEEDS = [
    11, 22, 33, 44, 55, 66
]

backend_classes = {
    "Sherbrooke": fake_provider.FakeSherbrooke,
    "Torino": fake_provider.FakeTorino,
    "Fez": fake_provider.FakeFez
}


# ============================================================
# CIRCUIT GENERATOR
# ============================================================

def make_random_circuit(seed):

    rng = np.random.default_rng(seed)

    num_qubits = int(rng.integers(4, 9))
    circuit_depth = int(rng.integers(3, 9))

    qc = QuantumCircuit(
        num_qubits,
        num_qubits
    )

    for _ in range(circuit_depth):

        for q in range(num_qubits):

            choice = rng.integers(0, 3)

            if choice == 0:
                qc.h(q)

            elif choice == 1:
                qc.ry(
                    float(
                        rng.uniform(
                            0,
                            2 * np.pi
                        )
                    ),
                    q
                )

            else:
                qc.rz(
                    float(
                        rng.uniform(
                            0,
                            2 * np.pi
                        )
                    ),
                    q
                )

        number_of_entangling_gates = max(
            1,
            num_qubits // 2
        )

        for _ in range(
            number_of_entangling_gates
        ):

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
# GET INSTRUCTION ERROR
# ============================================================

def get_error(
    backend,
    instruction,
    qubits
):

    try:

        props = backend.target[
            instruction
        ][tuple(qubits)]

        if props is not None:
            return props.error

    except (
        KeyError,
        TypeError
    ):

        pass

    return None


# ============================================================
# BUILD FEATURES FOR ONE CANDIDATE
# ============================================================

def extract_features(
    circuit,
    backend,
    backend_name,
    circuit_id,
    circuit_seed,
    candidate_seed,
    actual_fidelity
):

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=candidate_seed
    )

    operations = transpiled.count_ops()

    # --------------------------------------------------------
    # BASIC CIRCUIT FEATURES
    # --------------------------------------------------------

    one_q = 0
    two_q = 0

    for name, count in operations.items():

        if name in [
            "measure",
            "barrier"
        ]:
            continue

        if name in [
            "cx",
            "cz",
            "ecr",
            "swap",
            "rzz"
        ]:
            two_q += count

        else:
            one_q += count

    # --------------------------------------------------------
    # MAPPING
    # --------------------------------------------------------

    final_layout = (
        transpiled.layout
        .final_index_layout(
            filter_ancillas=True
        )
    )

    logical_to_physical = {}

    for logical, physical in enumerate(
        final_layout
    ):

        if logical >= circuit.num_qubits:
            break

        logical_to_physical[
            logical
        ] = physical

    physical_qubits = sorted(
        logical_to_physical.values()
    )

    # --------------------------------------------------------
    # READOUT FEATURES
    # --------------------------------------------------------

    readout_errors = []

    for physical in physical_qubits:

        error = get_error(
            backend,
            "measure",
            (physical,)
        )

        if error is not None:
            readout_errors.append(
                error
            )

    if readout_errors:

        avg_readout = float(
            np.mean(readout_errors)
        )

        max_readout = float(
            np.max(readout_errors)
        )

        sum_readout = float(
            np.sum(readout_errors)
        )

        readout_product = float(
            np.prod(
                1.0 -
                np.array(
                    readout_errors
                )
            )
        )

    else:

        avg_readout = np.nan
        max_readout = np.nan
        sum_readout = np.nan
        readout_product = np.nan

    # --------------------------------------------------------
    # CZ FEATURES
    # --------------------------------------------------------

    cz_errors = []
    edge_exposure = 0.0
    max_edge_error = 0.0

    edge_counts = {}

    for instruction in transpiled.data:

        if instruction.operation.name not in [
            "cz",
            "cx",
            "ecr"
        ]:
            continue

        q0 = transpiled.find_bit(
            instruction.qubits[0]
        ).index

        q1 = transpiled.find_bit(
            instruction.qubits[1]
        ).index

        edge = tuple(
            sorted(
                (q0, q1)
            )
        )

        edge_counts[edge] = (
            edge_counts.get(edge, 0)
            + 1
        )

        error = get_error(
            backend,
            instruction.operation.name,
            (q0, q1)
        )

        if error is None:

            error = get_error(
                backend,
                instruction.operation.name,
                (q1, q0)
            )

        if error is not None:

            cz_errors.append(error)

            edge_exposure += error

            max_edge_error = max(
                max_edge_error,
                error
            )

    if cz_errors:

        avg_cz_error = float(
            np.mean(cz_errors)
        )

        max_cz_error_observed = float(
            np.max(cz_errors)
        )

        cz_product = float(
            np.prod(
                1.0 -
                np.array(
                    cz_errors
                )
            )
        )

    else:

        avg_cz_error = np.nan
        max_cz_error_observed = np.nan
        cz_product = np.nan

    # --------------------------------------------------------
    # QUBIT T1 / T2
    # --------------------------------------------------------

    t1_values = []
    t2_values = []

    for physical in physical_qubits:

        try:

            props = backend.qubit_properties(
                physical
            )

            if props.t1 is not None:
                t1_values.append(
                    props.t1
                )

            if props.t2 is not None:
                t2_values.append(
                    props.t2
                )

        except Exception:
            pass

    if t1_values:

        avg_t1 = float(
            np.mean(t1_values)
        )

        min_t1 = float(
            np.min(t1_values)
        )

    else:

        avg_t1 = np.nan
        min_t1 = np.nan

    if t2_values:

        avg_t2 = float(
            np.mean(t2_values)
        )

        min_t2 = float(
            np.min(t2_values)
        )

    else:

        avg_t2 = np.nan
        min_t2 = np.nan

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "circuit_id":
            circuit_id,

        "circuit_seed":
            circuit_seed,

        "backend":
            backend_name,

        "candidate_seed":
            candidate_seed,

        "original_qubits":
            circuit.num_qubits,

        "original_depth":
            circuit.depth(),

        "transpiled_depth":
            transpiled.depth(),

        "one_qubit_gates":
            one_q,

        "two_qubit_gates":
            two_q,

        "esp":
            calculate_esp(
                backend,
                transpiled
            ),

        "active_physical_qubits":
            len(physical_qubits),

        "avg_readout_error":
            avg_readout,

        "max_readout_error":
            max_readout,

        "sum_readout_error":
            sum_readout,

        "readout_product":
            readout_product,

        "avg_cz_error":
            avg_cz_error,

        "max_cz_error":
            max_cz_error_observed,

        "edge_exposure":
            edge_exposure,

        "max_edge_error":
            max_edge_error,

        "cz_product":
            cz_product,

        "avg_t1":
            avg_t1,

        "min_t1":
            min_t1,

        "avg_t2":
            avg_t2,

        "min_t2":
            min_t2,

        "physical_qubits":
            ",".join(
                map(
                    str,
                    physical_qubits
                )
            ),

        "actual_fidelity":
            actual_fidelity
    }


# ============================================================
# ESP
# ============================================================

def calculate_esp(
    backend,
    transpiled
):

    esp = 1.0

    for instruction in transpiled.data:

        error = get_error(
            backend,
            instruction.operation.name,
            tuple(
                transpiled.find_bit(
                    q
                ).index
                for q in instruction.qubits
            )
        )

        if error is not None:

            esp *= max(
                0.0,
                1.0 - error
            )

    return esp


# ============================================================
# LOAD ACTUAL RESULTS
# ============================================================

benchmark = pd.read_csv(
    "results/candidate_benchmark.csv"
)


# ============================================================
# BUILD DATASET
# ============================================================

rows = []

print("=" * 90)
print(
    "CALIBRATIONCOMPASS - "
    "CANDIDATE FEATURE DATASET"
)
print("=" * 90)

for circuit_id in range(
    NUMBER_OF_CIRCUITS
):

    circuit_seed = (
        20000 + circuit_id
    )

    circuit = make_random_circuit(
        circuit_seed
    )

    for backend_name, BackendClass in (
        backend_classes.items()
    ):

        backend = BackendClass()

        for candidate_seed in CANDIDATE_SEEDS:

            match = benchmark[
                (benchmark["circuit_id"] == circuit_id)
                &
                (benchmark["backend"] == backend_name)
                &
                (
                    benchmark["candidate_seed"]
                    == candidate_seed
                )
            ]

            actual_fidelity = float(
                match["fidelity"].iloc[0]
            )

            row = extract_features(
                circuit,
                backend,
                backend_name,
                circuit_id,
                circuit_seed,
                candidate_seed,
                actual_fidelity
            )

            rows.append(row)

    print(
        f"Circuit {circuit_id + 1}/"
        f"{NUMBER_OF_CIRCUITS} complete"
    )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(rows)

output_file = (
    "results/candidate_features.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("DATASET COMPLETE")
print("=" * 90)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print()
print(
    "Missing values:"
)

print(
    df.isna().sum()
    .loc[
        lambda x: x > 0
    ]
    .to_string()
)

print()
print("Saved:")
print(output_file)

print()
print("=" * 90)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 90)