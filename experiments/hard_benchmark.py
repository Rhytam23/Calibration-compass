import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider
from qiskit.quantum_info import hellinger_fidelity
from qiskit.transpiler import InstructionProperties


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_CIRCUITS = int(_os.environ.get("CC_NUM_CIRCUITS", 40))
NUMBER_OF_DAYS = 8
SHOTS = 2000


# ============================================================
# 1. CREATE RANDOM CIRCUITS
# ============================================================

def make_random_circuit(seed):

    rng = np.random.default_rng(seed)

    # Number of qubits
    num_qubits = int(
        rng.integers(4, 9)
    )

    # Circuit depth
    circuit_depth = int(
        rng.integers(3, 9)
    )

    qc = QuantumCircuit(
        num_qubits,
        num_qubits
    )

    for _ in range(circuit_depth):

        # -------------------------------
        # Single-qubit gates
        # -------------------------------

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

        # -------------------------------
        # Two-qubit gates
        # -------------------------------

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
# 2. COUNTS → PROBABILITIES
# ============================================================

def counts_to_probabilities(
    counts,
    shots
):

    return {
        bitstring: count / shots
        for bitstring, count in counts.items()
    }


# ============================================================
# 3. GET INSTRUCTION ERROR
# ============================================================

def get_instruction_error(
    backend,
    instruction_name,
    qargs
):

    try:

        properties = (
            backend.target[
                instruction_name
            ][qargs]
        )

        if properties is None:
            return None

        return properties.error

    except (
        KeyError,
        TypeError
    ):

        return None


# ============================================================
# 4. GET HARDWARE FEATURES
# ============================================================

def get_hardware_features(
    backend,
    transpiled
):

    active_qubits = set()

    gate_errors = []
    readout_errors = []

    for instruction in transpiled.data:

        operation = instruction.operation

        operation_name = operation.name

        physical_qubits = tuple(
            transpiled.find_bit(q).index
            for q in instruction.qubits
        )

        active_qubits.update(
            physical_qubits
        )

        # -------------------------------
        # Measurement error
        # -------------------------------

        if operation_name == "measure":

            error = get_instruction_error(
                backend,
                "measure",
                physical_qubits
            )

            if error is not None:
                readout_errors.append(
                    error
                )

        # -------------------------------
        # Gate error
        # -------------------------------

        else:

            error = get_instruction_error(
                backend,
                operation_name,
                physical_qubits
            )

            if error is not None:
                gate_errors.append(
                    error
                )

    # -------------------------------
    # T1 / T2
    # -------------------------------

    t1_values = []
    t2_values = []

    for qubit in active_qubits:

        try:

            properties = (
                backend.qubit_properties(
                    qubit
                )
            )

            if properties is not None:

                if properties.t1 is not None:
                    t1_values.append(
                        properties.t1
                    )

                if properties.t2 is not None:
                    t2_values.append(
                        properties.t2
                    )

        except (
            NotImplementedError,
            IndexError
        ):

            pass

    def average(values):

        if not values:
            return np.nan

        return float(
            np.mean(values)
        )

    def minimum(values):

        if not values:
            return np.nan

        return float(
            np.min(values)
        )

    return {

        "active_physical_qubits":
            len(active_qubits),

        "avg_gate_error":
            average(gate_errors),

        "max_gate_error":
            (
                float(
                    max(gate_errors)
                )
                if gate_errors
                else np.nan
            ),

        "avg_readout_error":
            average(readout_errors),

        "max_readout_error":
            (
                float(
                    max(readout_errors)
                )
                if readout_errors
                else np.nan
            ),

        "avg_t1":
            average(t1_values),

        "min_t1":
            minimum(t1_values),

        "avg_t2":
            average(t2_values),

        "min_t2":
            minimum(t2_values),
    }


# ============================================================
# 5. CALCULATE ESP
# ============================================================

def calculate_esp(
    backend,
    transpiled
):

    esp = 1.0

    for instruction in transpiled.data:

        operation = instruction.operation

        operation_name = operation.name

        physical_qubits = tuple(
            transpiled.find_bit(q).index
            for q in instruction.qubits
        )

        error = get_instruction_error(
            backend,
            operation_name,
            physical_qubits
        )

        if error is not None:

            success_probability = max(
                0.0,
                1.0 - error
            )

            esp *= success_probability

    return esp


# ============================================================
# 6. FIND MEASUREMENT QUBITS
# ============================================================

def get_measurement_qubits(
    transpiled
):

    qubits = []

    for instruction in transpiled.data:

        if instruction.operation.name == "measure":

            physical_qubit = (
                transpiled
                .find_bit(
                    instruction.qubits[0]
                )
                .index
            )

            qubits.append(
                physical_qubit
            )

    return list(
        dict.fromkeys(qubits)
    )


# ============================================================
# 7. FIND TWO-QUBIT OPERATIONS
# ============================================================

def get_two_qubit_operations(
    transpiled
):

    operations = []

    for instruction in transpiled.data:

        if (
            instruction.operation.num_qubits
            == 2
        ):

            qargs = tuple(
                transpiled.find_bit(q).index
                for q in instruction.qubits
            )

            operations.append(
                (
                    instruction.operation.name,
                    qargs
                )
            )

    return operations


# ============================================================
# 8. CREATE A RANDOM "HARDWARE DAY"
# ============================================================

def apply_random_drift(
    backend,
    transpiled,
    rng,
    is_normal_day=False
):

    changes = []

    # --------------------------------------------------------
    # NORMAL DAY
    # --------------------------------------------------------

    if is_normal_day:

        return "normal"

    # --------------------------------------------------------
    # FIND CANDIDATE QUBITS / GATES
    # --------------------------------------------------------

    measurement_qubits = (
        get_measurement_qubits(
            transpiled
        )
    )

    two_qubit_operations = (
        get_two_qubit_operations(
            transpiled
        )
    )

    # --------------------------------------------------------
    # RANDOM READOUT DEGRADATION
    # --------------------------------------------------------

    if measurement_qubits:

        # 60% chance
        if rng.random() < 0.60:

            qubit = int(
                rng.choice(
                    measurement_qubits
                )
            )

            properties = (
                backend.target[
                    "measure"
                ][(qubit,)]
            )

            if (
                properties is not None
                and properties.error is not None
            ):

                old_error = (
                    properties.error
                )

                factor = float(
                    rng.uniform(1.5, 6.0)
                )

                new_error = min(
                    old_error * factor,
                    0.40
                )

                backend.target.update_instruction_properties(
                    instruction="measure",
                    qargs=(qubit,),
                    properties=InstructionProperties(
                        duration=properties.duration,
                        error=new_error
                    )
                )

                changes.append(
                    f"readout q{qubit}: "
                    f"{old_error:.4f}"
                    f"->{new_error:.4f}"
                )

    # --------------------------------------------------------
    # SECOND READOUT DEGRADATION
    # --------------------------------------------------------

    if measurement_qubits:

        # 35% chance
        if rng.random() < 0.35:

            qubit = int(
                rng.choice(
                    measurement_qubits
                )
            )

            properties = (
                backend.target[
                    "measure"
                ][(qubit,)]
            )

            if (
                properties is not None
                and properties.error is not None
            ):

                old_error = (
                    properties.error
                )

                factor = float(
                    rng.uniform(1.5, 4.0)
                )

                new_error = min(
                    old_error * factor,
                    0.40
                )

                backend.target.update_instruction_properties(
                    instruction="measure",
                    qargs=(qubit,),
                    properties=InstructionProperties(
                        duration=properties.duration,
                        error=new_error
                    )
                )

                changes.append(
                    f"readout q{qubit}: "
                    f"{old_error:.4f}"
                    f"->{new_error:.4f}"
                )

    # --------------------------------------------------------
    # TWO-QUBIT GATE DEGRADATION
    # --------------------------------------------------------

    if two_qubit_operations:

        # 65% chance
        if rng.random() < 0.65:

            gate_name, qargs = (
                two_qubit_operations[
                    rng.integers(
                        0,
                        len(
                            two_qubit_operations
                        )
                    )
                ]
            )

            properties = (
                backend.target[
                    gate_name
                ][qargs]
            )

            if (
                properties is not None
                and properties.error is not None
            ):

                old_error = (
                    properties.error
                )

                factor = float(
                    rng.uniform(1.5, 5.0)
                )

                new_error = min(
                    old_error * factor,
                    0.30
                )

                backend.target.update_instruction_properties(
                    instruction=gate_name,
                    qargs=qargs,
                    properties=InstructionProperties(
                        duration=properties.duration,
                        error=new_error
                    )
                )

                changes.append(
                    f"{gate_name}{qargs}: "
                    f"{old_error:.4f}"
                    f"->{new_error:.4f}"
                )

    # --------------------------------------------------------
    # IF NOTHING CHANGED, FORCE ONE CHANGE
    # --------------------------------------------------------

    if not changes:

        if measurement_qubits:

            qubit = int(
                measurement_qubits[0]
            )

            properties = (
                backend.target[
                    "measure"
                ][(qubit,)]
            )

            if (
                properties is not None
                and properties.error is not None
            ):

                old_error = (
                    properties.error
                )

                new_error = min(
                    old_error * 2.0,
                    0.40
                )

                backend.target.update_instruction_properties(
                    instruction="measure",
                    qargs=(qubit,),
                    properties=InstructionProperties(
                        duration=properties.duration,
                        error=new_error
                    )
                )

                changes.append(
                    f"readout q{qubit}: "
                    f"{old_error:.4f}"
                    f"->{new_error:.4f}"
                )

    return " | ".join(changes)


# ============================================================
# 9. EVALUATE ONE CANDIDATE
# ============================================================

def evaluate(
    transpiled,
    backend
):

    # --------------------------------------------------------
    # IDEAL SIMULATION
    # --------------------------------------------------------

    ideal_simulator = AerSimulator()

    ideal_job = ideal_simulator.run(
        transpiled,
        shots=SHOTS,
        seed_simulator=1234
    )

    ideal_counts = (
        ideal_job
        .result()
        .get_counts()
    )

    # --------------------------------------------------------
    # NOISY SIMULATION
    # --------------------------------------------------------

    noisy_simulator = (
        AerSimulator.from_backend(
            backend
        )
    )

    noisy_job = noisy_simulator.run(
        transpiled,
        shots=SHOTS,
        seed_simulator=5678
    )

    noisy_counts = (
        noisy_job
        .result()
        .get_counts()
    )

    # --------------------------------------------------------
    # CONVERT TO PROBABILITIES
    # --------------------------------------------------------

    ideal_probabilities = (
        counts_to_probabilities(
            ideal_counts,
            SHOTS
        )
    )

    noisy_probabilities = (
        counts_to_probabilities(
            noisy_counts,
            SHOTS
        )
    )

    # --------------------------------------------------------
    # HELLINGER FIDELITY
    # --------------------------------------------------------

    fidelity = hellinger_fidelity(
        ideal_probabilities,
        noisy_probabilities
    )

    # --------------------------------------------------------
    # CIRCUIT STATISTICS
    # --------------------------------------------------------

    operations = (
        transpiled.count_ops()
    )

    one_qubit_gates = 0
    two_qubit_gates = 0

    for gate_name, count in operations.items():

        if gate_name in [
            "measure",
            "barrier"
        ]:

            continue

        if (
            gate_name in [
                "cx",
                "ecr",
                "cz",
                "swap",
                "rzz"
            ]
        ):

            two_qubit_gates += count

        else:

            one_qubit_gates += count

    # --------------------------------------------------------
    # HARDWARE FEATURES
    # --------------------------------------------------------

    hardware_features = (
        get_hardware_features(
            backend,
            transpiled
        )
    )

    # --------------------------------------------------------
    # ESP
    # --------------------------------------------------------

    esp = calculate_esp(
        backend,
        transpiled
    )

    return {

        "transpiled_depth":
            transpiled.depth(),

        "one_qubit_gates":
            one_qubit_gates,

        "two_qubit_gates":
            two_qubit_gates,

        "esp":
            esp,

        "fidelity":
            fidelity,

        **hardware_features
    }


# ============================================================
# 10. BACKENDS
# ============================================================

backend_classes = {

    "Sherbrooke":
        fake_provider.FakeSherbrooke,

    "Torino":
        fake_provider.FakeTorino,

    "Fez":
        fake_provider.FakeFez
}


# ============================================================
# 11. GENERATE THE DATA
# ============================================================

results = []

print("=" * 75)
print("CALIBRATIONCOMPASS - HARD DYNAMIC BENCHMARK")
print("=" * 75)

print(
    f"\nCircuits: {NUMBER_OF_CIRCUITS}"
)

print(
    f"Backends: {len(backend_classes)}"
)

print(
    f"Hardware days: {NUMBER_OF_DAYS}"
)

print(
    f"Expected rows: "
    f"{NUMBER_OF_CIRCUITS * len(backend_classes) * NUMBER_OF_DAYS}"
)

print(
    f"Shots: {SHOTS}"
)


for circuit_id in range(
    NUMBER_OF_CIRCUITS
):

    circuit_seed = (
        10000 + circuit_id
    )

    circuit = make_random_circuit(
        circuit_seed
    )

    print(
        f"\nCircuit "
        f"{circuit_id + 1}/"
        f"{NUMBER_OF_CIRCUITS} "
        f"({circuit.num_qubits} qubits)"
    )

    # --------------------------------------------------------
    # Each circuit gets several simulated days
    # --------------------------------------------------------

    for day in range(
        NUMBER_OF_DAYS
    ):

        scenario_id = (
            f"c{circuit_id}_day{day}"
        )

        # Different random seed for each day
        scenario_seed = (
            50000
            + circuit_id * 100
            + day
        )

        # ----------------------------------------------------
        # Each backend gets its OWN hardware state
        # ----------------------------------------------------

        for backend_name, BackendClass in (
            backend_classes.items()
        ):

            # Clean backend for transpilation
            base_backend = (
                BackendClass()
            )

            transpiled = transpile(
                circuit,
                backend=base_backend,
                optimization_level=1,
                seed_transpiler=scenario_seed
            )

            # Fresh backend for this simulated day
            scenario_backend = (
                BackendClass()
            )

            rng = np.random.default_rng(
                scenario_seed
                + {"Sherbrooke": 0, "Torino": 1, "Fez": 2}.get(backend_name, 3)
            )

            # Day 0 = normal
            # Other days = random drift
            is_normal_day = (
                day == 0
            )

            drift = apply_random_drift(
                scenario_backend,
                transpiled,
                rng,
                is_normal_day
            )

            result = evaluate(
                transpiled,
                scenario_backend
            )

            results.append({

                "circuit_id":
                    circuit_id,

                "circuit_seed":
                    circuit_seed,

                "day":
                    day,

                "scenario_id":
                    scenario_id,

                "backend":
                    backend_name,

                "original_qubits":
                    circuit.num_qubits,

                "original_depth":
                    circuit.depth(),

                "drift":
                    drift,

                **result
            })

        print(
            f"  Day {day + 1}/"
            f"{NUMBER_OF_DAYS} complete"
        )


# ============================================================
# 12. SAVE DATASET
# ============================================================

df = pd.DataFrame(
    results
)

output_file = (
    "results/hard_benchmark.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 13. FIND ORACLE WINNERS
# ============================================================

oracle_rows = df.loc[
    df.groupby(
        [
            "circuit_id",
            "day"
        ]
    )["fidelity"].idxmax()
].copy()

oracle_rows = oracle_rows.rename(
    columns={
        "backend":
            "oracle_backend",

        "fidelity":
            "oracle_fidelity"
    }
)


# ============================================================
# 14. FIND ESP WINNERS
# ============================================================

esp_rows = df.loc[
    df.groupby(
        [
            "circuit_id",
            "day"
        ]
    )["esp"].idxmax()
].copy()

esp_rows = esp_rows.rename(
    columns={
        "backend":
            "esp_backend",

        "fidelity":
            "esp_actual_fidelity"
    }
)


# ============================================================
# 15. COMPARE ESP WITH ORACLE
# ============================================================

comparison = oracle_rows[
    [
        "circuit_id",
        "day",
        "oracle_backend",
        "oracle_fidelity"
    ]
].merge(
    esp_rows[
        [
            "circuit_id",
            "day",
            "esp_backend",
            "esp_actual_fidelity"
        ]
    ],
    on=[
        "circuit_id",
        "day"
    ]
)

comparison["esp_correct"] = (
    comparison["oracle_backend"]
    ==
    comparison["esp_backend"]
)

comparison["esp_regret"] = (
    comparison["oracle_fidelity"]
    -
    comparison[
        "esp_actual_fidelity"
    ]
)


# ============================================================
# 16. FIND HOW HARD EACH DECISION WAS
# ============================================================

sorted_fidelities = (
    df.sort_values(
        [
            "circuit_id",
            "day",
            "fidelity"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )
)


top_two = (
    sorted_fidelities
    .groupby(
        [
            "circuit_id",
            "day"
        ]
    )["fidelity"]
    .apply(
        lambda x:
        list(x.head(2))
    )
)


hard_cases = []

for key, values in top_two.items():

    if len(values) < 2:
        continue

    best = values[0]
    second = values[1]

    gap = best - second

    hard_cases.append({

        "circuit_id":
            key[0],

        "day":
            key[1],

        "best_fidelity":
            best,

        "second_best_fidelity":
            second,

        "gap":
            gap
    })


hard_cases_df = pd.DataFrame(
    hard_cases
)


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("HARD BENCHMARK SUMMARY")
print("=" * 75)

print(
    f"\nTotal rows: {len(df)}"
)

print(
    f"Total decisions: {len(comparison)}"
)


# ------------------------------------------------------------
# ESP accuracy
# ------------------------------------------------------------

esp_accuracy = (
    comparison[
        "esp_correct"
    ].mean()
)

print(
    f"\nESP backend-selection accuracy: "
    f"{esp_accuracy:.2%}"
)


# ------------------------------------------------------------
# ESP regret
# ------------------------------------------------------------

print(
    f"Average ESP fidelity regret: "
    f"{comparison['esp_regret'].mean():.6f}"
)

print(
    f"Maximum ESP fidelity regret: "
    f"{comparison['esp_regret'].max():.6f}"
)


# ------------------------------------------------------------
# Backend wins
# ------------------------------------------------------------

print(
    "\nORACLE BACKEND WIN COUNTS"
)

print(
    comparison[
        "oracle_backend"
    ].value_counts()
)


# ------------------------------------------------------------
# HARD CASES
# ------------------------------------------------------------

if len(hard_cases_df) > 0:

    print(
        "\nHARD CASES"
    )

    print(
        "A case is considered hard when "
        "the best and second-best backend "
        "are within 1 percentage point."
    )

    hard_threshold = 0.01

    number_of_hard_cases = (
        hard_cases_df[
            "gap"
        ]
        <= hard_threshold
    ).sum()

    print(
        f"\nHard cases: "
        f"{number_of_hard_cases}/"
        f"{len(hard_cases_df)}"
    )

    print(
        f"Percentage hard: "
        f"{number_of_hard_cases / len(hard_cases_df):.2%}"
    )

    print(
        "\nSmallest backend gaps:"
    )

    print(
        hard_cases_df.sort_values(
            "gap"
        ).head(15).to_string(
            index=False
        )
    )


# ============================================================
# 18. SAVE ANALYSIS
# ============================================================

comparison.to_csv(
    "results/hard_oracle_vs_esp.csv",
    index=False
)

hard_cases_df.to_csv(
    "results/hard_cases.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "results/hard_benchmark.csv"
)

print(
    "results/hard_oracle_vs_esp.csv"
)

print(
    "results/hard_cases.csv"
)


print(
    "\n" + "=" * 75
)

print(
    "HARD BENCHMARK COMPLETE"
)

print(
    "=" * 75
)