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
# CALIBRATION HELPERS
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


def get_qubit_property(
    backend,
    physical,
    property_name
):

    try:

        props = backend.qubit_properties(
            physical
        )

        return getattr(
            props,
            property_name
        )

    except Exception:

        return None


# ============================================================
# LOGICAL ACTIVITY
# ============================================================

def get_logical_activity(circuit):

    activity = {
        q: {
            "one_q": 0,
            "one_q_physical": 0,
            "two_q": 0,
            "total": 0
        }
        for q in range(
            circuit.num_qubits
        )
    }

    for instruction in circuit.data:

        if instruction.operation.name in [
            "measure",
            "barrier"
        ]:
            continue

        for qubit in instruction.qubits:

            logical = circuit.find_bit(
                qubit
            ).index

            activity[logical]["total"] += 1

            if len(instruction.qubits) == 1:

                activity[logical]["one_q"] += 1

                # rz is a virtual (error-free) gate on IBM
                # hardware, so it carries no sx error.
                if instruction.operation.name != "rz":

                    activity[logical]["one_q_physical"] += 1

            elif len(instruction.qubits) == 2:

                activity[logical]["two_q"] += 1

    return activity


# ============================================================
# GET FINAL LOGICAL -> PHYSICAL MAPPING
# ============================================================

def get_mapping(transpiled):

    final = (
        transpiled.layout
        .final_index_layout(
            filter_ancillas=True
        )
    )

    mapping = {}

    for logical, physical in enumerate(
        final
    ):

        if logical >= transpiled.num_qubits:
            break

        mapping[logical] = physical

    return mapping


# ============================================================
# CANDIDATE FEATURES
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

    mapping = get_mapping(
        transpiled
    )

    physical_qubits = sorted(
        mapping.values()
    )

    # --------------------------------------------------------
    # LOGICAL ACTIVITY
    # --------------------------------------------------------

    activity = get_logical_activity(
        circuit
    )

    # --------------------------------------------------------
    # MAPPING-AWARE QUBIT FEATURES
    # --------------------------------------------------------

    mapped_readout = {}
    mapped_sx = {}
    mapped_t1 = {}
    mapped_t2 = {}

    for logical in range(
        circuit.num_qubits
    ):

        physical = mapping[logical]

        readout = get_error(
            backend,
            "measure",
            (physical,)
        )

        sx = get_error(
            backend,
            "sx",
            (physical,)
        )

        t1 = get_qubit_property(
            backend,
            physical,
            "t1"
        )

        t2 = get_qubit_property(
            backend,
            physical,
            "t2"
        )

        mapped_readout[logical] = (
            readout
            if readout is not None
            else 0.0
        )

        mapped_sx[logical] = (
            sx
            if sx is not None
            else 0.0
        )

        mapped_t1[logical] = (
            t1
            if t1 is not None
            else np.nan
        )

        mapped_t2[logical] = (
            t2
            if t2 is not None
            else np.nan
        )

    # --------------------------------------------------------
    # READOUT × LOGICAL ACTIVITY
    # --------------------------------------------------------

    weighted_readout_total = 0.0
    weighted_readout_2q = 0.0

    weighted_sx_total = 0.0

    activity_readout_values = []

    for logical in range(
        circuit.num_qubits
    ):

        ro = mapped_readout[logical]

        sx = mapped_sx[logical]

        total_activity = (
            activity[logical]["total"]
        )

        twoq_activity = (
            activity[logical]["two_q"]
        )

        weighted_readout_total += (
            total_activity * ro
        )

        weighted_readout_2q += (
            twoq_activity * ro
        )

        weighted_sx_total += (
            activity[logical]["one_q_physical"] * sx
        )

        activity_readout_values.append(
            (
                twoq_activity,
                ro
            )
        )

    # --------------------------------------------------------
    # CORRELATION:
    #
    # Positive value means highly active logical qubits
    # tend to be sitting on noisier readout qubits.
    # --------------------------------------------------------

    activities = np.array([
        x[0]
        for x in activity_readout_values
    ], dtype=float)

    readouts = np.array([
        x[1]
        for x in activity_readout_values
    ], dtype=float)

    if (
        len(activities) >= 2
        and np.std(activities) > 0
        and np.std(readouts) > 0
    ):

        activity_readout_corr = float(
            np.corrcoef(
                activities,
                readouts
            )[0, 1]
        )

    else:

        activity_readout_corr = 0.0

    # --------------------------------------------------------
    # WORST LOGICAL ASSIGNMENTS
    # --------------------------------------------------------

    weighted_values = [
        activity[q]["two_q"]
        * mapped_readout[q]
        for q in range(
            circuit.num_qubits
        )
    ]

    weighted_values.sort(
        reverse=True
    )

    top1_readout_risk = (
        weighted_values[0]
        if weighted_values
        else 0.0
    )

    top2_readout_risk = (
        sum(weighted_values[:2])
        if weighted_values
        else 0.0
    )

    # --------------------------------------------------------
    # MAPPED QUBIT AGGREGATES
    # --------------------------------------------------------

    ro_values = list(
        mapped_readout.values()
    )

    sx_values = list(
        mapped_sx.values()
    )

    t1_values = [
        x for x in mapped_t1.values()
        if not np.isnan(x)
    ]

    t2_values = [
        x for x in mapped_t2.values()
        if not np.isnan(x)
    ]

    # Weighted T2 exposure
    t2_exposure = 0.0

    for logical in range(
        circuit.num_qubits
    ):

        t2 = mapped_t2[logical]

        if (
            t2 is not None
            and not np.isnan(t2)
            and t2 > 0
        ):

            t2_exposure += (
                activity[logical]["total"]
                / t2
            )

    # --------------------------------------------------------
    # ACTUAL TRANSPILED EDGE EXPOSURE
    # --------------------------------------------------------

    edge_errors = []

    edge_exposure = 0.0

    max_edge_error = 0.0

    edge_counts = {}

    for instruction in transpiled.data:

        if instruction.operation.name not in [
            "cx",
            "cz",
            "ecr",
            "swap",
            "rzz"
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
            edge_counts.get(
                edge,
                0
            ) + 1
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

            edge_errors.append(
                error
            )

            edge_exposure += (
                error
            )

            max_edge_error = max(
                max_edge_error,
                error
            )

    # --------------------------------------------------------
    # T1 / T2
    # --------------------------------------------------------

    avg_t1 = (
        float(np.mean(t1_values))
        if t1_values
        else np.nan
    )

    min_t1 = (
        float(np.min(t1_values))
        if t1_values
        else np.nan
    )

    avg_t2 = (
        float(np.mean(t2_values))
        if t2_values
        else np.nan
    )

    min_t2 = (
        float(np.min(t2_values))
        if t2_values
        else np.nan
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    row = {

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
            float(np.mean(ro_values)),

        "max_readout_error":
            float(np.max(ro_values)),

        "sum_readout_error":
            float(np.sum(ro_values)),

        "weighted_readout_total":
            weighted_readout_total,

        "weighted_readout_2q":
            weighted_readout_2q,

        "top1_readout_risk":
            top1_readout_risk,

        "top2_readout_risk":
            top2_readout_risk,

        "activity_readout_corr":
            activity_readout_corr,

        "avg_sx_error":
            float(np.mean(sx_values)),

        "weighted_sx_total":
            weighted_sx_total,

        "avg_cz_error":
            (
                float(np.mean(edge_errors))
                if edge_errors
                else np.nan
            ),

        "max_cz_error":
            (
                float(np.max(edge_errors))
                if edge_errors
                else np.nan
            ),

        "edge_exposure":
            edge_exposure,

        "max_edge_error":
            max_edge_error,

        "t2_exposure":
            t2_exposure,

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

    # --------------------------------------------------------
    # ADD PER-LOGICAL-QUBIT FEATURES
    # --------------------------------------------------------

    for logical in range(
        circuit.num_qubits
    ):

        row[
            f"logical_{logical}_physical"
        ] = mapping[logical]

        row[
            f"logical_{logical}_readout"
        ] = mapped_readout[logical]

        row[
            f"logical_{logical}_activity_2q"
        ] = activity[logical]["two_q"]

        row[
            f"logical_{logical}_activity_total"
        ] = activity[logical]["total"]

    return row


# ============================================================
# ESP
# ============================================================

def calculate_esp(
    backend,
    transpiled
):

    esp = 1.0

    for instruction in transpiled.data:

        qubits = tuple(
            transpiled.find_bit(
                q
            ).index
            for q in instruction.qubits
        )

        error = get_error(
            backend,
            instruction.operation.name,
            qubits
        )

        if error is not None:

            esp *= max(
                0.0,
                1.0 - error
            )

    return esp


# ============================================================
# LOAD ACTUAL BENCHMARK
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
    "MAPPING-AWARE FEATURE DATASET"
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

        for candidate_seed in (
            CANDIDATE_SEEDS
        ):

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
    "results/candidate_features_v2.csv"
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
print("Missing values:")

missing = (
    df.isna()
    .sum()
)

missing = missing[
    missing > 0
]

if missing.empty:

    print("None")

else:

    print(
        missing.to_string()
    )

print()
print("Saved:")
print(output_file)

print()
print("=" * 90)
print("MAPPING-AWARE FEATURE EXTRACTION COMPLETE")
print("=" * 90)