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

NUMBER_OF_CIRCUITS = int(_os.environ.get("CC_NUM_CIRCUITS", 30))
SHOTS = 2000


# ============================================================
# 1. CREATE RANDOM CIRCUITS
# ============================================================

def make_random_circuit(seed):

    rng = np.random.default_rng(seed)

    num_qubits = int(rng.integers(4, 9))
    circuit_depth = int(rng.integers(3, 9))

    qc = QuantumCircuit(num_qubits, num_qubits)

    for _ in range(circuit_depth):

        # Single-qubit gates
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

        # Two-qubit gates
        number_of_entangling_gates = max(1, num_qubits // 2)

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
# 2. COUNTS → PROBABILITIES
# ============================================================

def counts_to_probabilities(counts, shots):

    return {
        bitstring: count / shots
        for bitstring, count in counts.items()
    }


# ============================================================
# 3. FIND A HARDWARE ERROR
# ============================================================

def get_instruction_error(backend, instruction_name, qargs):

    try:

        properties = backend.target[instruction_name][qargs]

        if properties is None:
            return None

        return properties.error

    except (KeyError, TypeError):

        return None


# ============================================================
# 4. GET HARDWARE FEATURES
# ============================================================

def get_hardware_features(backend, transpiled):

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

        active_qubits.update(physical_qubits)

        if operation_name == "measure":

            error = get_instruction_error(
                backend,
                "measure",
                physical_qubits
            )

            if error is not None:
                readout_errors.append(error)

        else:

            error = get_instruction_error(
                backend,
                operation_name,
                physical_qubits
            )

            if error is not None:
                gate_errors.append(error)

    t1_values = []
    t2_values = []

    for qubit in active_qubits:

        try:

            properties = backend.qubit_properties(qubit)

            if properties is not None:

                if properties.t1 is not None:
                    t1_values.append(properties.t1)

                if properties.t2 is not None:
                    t2_values.append(properties.t2)

        except (NotImplementedError, IndexError):

            pass

    def average(values):

        if not values:
            return np.nan

        return float(np.mean(values))

    def minimum(values):

        if not values:
            return np.nan

        return float(np.min(values))

    return {
        "active_physical_qubits": len(active_qubits),

        "avg_gate_error": average(gate_errors),

        "max_gate_error": (
            float(max(gate_errors))
            if gate_errors else np.nan
        ),

        "avg_readout_error": average(readout_errors),

        "max_readout_error": (
            float(max(readout_errors))
            if readout_errors else np.nan
        ),

        "avg_t1": average(t1_values),
        "min_t1": minimum(t1_values),

        "avg_t2": average(t2_values),
        "min_t2": minimum(t2_values),
    }


# ============================================================
# 5. CALCULATE ESP
# ============================================================

def calculate_esp(backend, transpiled):

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

            esp *= max(0.0, 1.0 - error)

    return esp


# ============================================================
# 6. FIND ONE MEASUREMENT QUBIT
# ============================================================

def find_measurement_qubit(transpiled):

    for instruction in transpiled.data:

        if instruction.operation.name == "measure":

            return transpiled.find_bit(
                instruction.qubits[0]
            ).index

    return None


# ============================================================
# 7. FIND ONE TWO-QUBIT GATE
# ============================================================

def find_two_qubit_gate(transpiled):

    for instruction in transpiled.data:

        if instruction.operation.num_qubits == 2:

            qargs = tuple(
                transpiled.find_bit(q).index
                for q in instruction.qubits
            )

            return instruction.operation.name, qargs

    return None, None


# ============================================================
# 8. APPLY SYNTHETIC HARDWARE DRIFT
# ============================================================

def apply_drift(
    backend,
    transpiled,
    condition
):

    changed_description = "none"

    # --------------------------------------------------------
    # READOUT DRIFT
    # --------------------------------------------------------

    if condition in [
        "readout_4x",
        "readout_8x",
        "combined_4x"
    ]:

        qubit = find_measurement_qubit(transpiled)

        if qubit is not None:

            properties = backend.target["measure"][(qubit,)]

            if properties is not None:

                old_error = properties.error

                if old_error is not None:

                    if condition == "readout_8x":
                        factor = 8
                    else:
                        factor = 4

                    new_error = min(
                        old_error * factor,
                        0.50
                    )

                    backend.target.update_instruction_properties(
                        instruction="measure",
                        qargs=(qubit,),
                        properties=InstructionProperties(
                            duration=properties.duration,
                            error=new_error
                        )
                    )

                    changed_description = (
                        f"measure q{qubit}: "
                        f"{old_error:.6f} -> {new_error:.6f}"
                    )

    # --------------------------------------------------------
    # TWO-QUBIT GATE DRIFT
    # --------------------------------------------------------

    if condition in [
        "gate_4x",
        "combined_4x"
    ]:

        gate_name, qargs = find_two_qubit_gate(
            transpiled
        )

        if gate_name is not None:

            properties = backend.target[
                gate_name
            ][qargs]

            if properties is not None:

                old_error = properties.error

                if old_error is not None:

                    new_error = min(
                        old_error * 4,
                        0.50
                    )

                    backend.target.update_instruction_properties(
                        instruction=gate_name,
                        qargs=qargs,
                        properties=InstructionProperties(
                            duration=properties.duration,
                            error=new_error
                        )
                    )

                    if changed_description == "none":

                        changed_description = (
                            f"{gate_name} {qargs}: "
                            f"{old_error:.6f} -> {new_error:.6f}"
                        )

                    else:

                        changed_description += (
                            f" | {gate_name} {qargs}: "
                            f"{old_error:.6f} -> {new_error:.6f}"
                        )

    return changed_description


# ============================================================
# 9. EVALUATE ONE SCENARIO
# ============================================================

def evaluate(
    transpiled,
    backend
):

    # Ideal simulation
    ideal_simulator = AerSimulator()

    ideal_job = ideal_simulator.run(
        transpiled,
        shots=SHOTS,
        seed_simulator=1234
    )

    ideal_counts = ideal_job.result().get_counts()

    # Noisy simulation
    noisy_simulator = AerSimulator.from_backend(
        backend
    )

    noisy_job = noisy_simulator.run(
        transpiled,
        shots=SHOTS,
        seed_simulator=5678
    )

    noisy_counts = noisy_job.result().get_counts()

    # Convert to probabilities
    ideal_probabilities = counts_to_probabilities(
        ideal_counts,
        SHOTS
    )

    noisy_probabilities = counts_to_probabilities(
        noisy_counts,
        SHOTS
    )

    # Fidelity
    fidelity = hellinger_fidelity(
        ideal_probabilities,
        noisy_probabilities
    )

    # Circuit statistics
    operations = transpiled.count_ops()

    one_qubit_gates = 0
    two_qubit_gates = 0

    for gate_name, count in operations.items():

        if gate_name in ["measure", "barrier"]:
            continue

        if gate_name in [
            "cx",
            "ecr",
            "cz",
            "swap",
            "rzz"
        ]:

            two_qubit_gates += count

        else:

            one_qubit_gates += count

    hardware_features = get_hardware_features(
        backend,
        transpiled
    )

    esp = calculate_esp(
        backend,
        transpiled
    )

    return {
        "transpiled_depth": transpiled.depth(),
        "one_qubit_gates": one_qubit_gates,
        "two_qubit_gates": two_qubit_gates,
        "esp": esp,
        "fidelity": fidelity,
        **hardware_features
    }


# ============================================================
# 10. BACKENDS
# ============================================================

backend_classes = {
    "Sherbrooke": fake_provider.FakeSherbrooke,
    "Torino": fake_provider.FakeTorino,
    "Fez": fake_provider.FakeFez
}


# ============================================================
# 11. HARDWARE CONDITIONS
# ============================================================

conditions = [
    "normal",
    "readout_4x",
    "readout_8x",
    "gate_4x",
    "combined_4x"
]


# ============================================================
# 12. GENERATE DATASET
# ============================================================

results = []

total_rows = (
    NUMBER_OF_CIRCUITS
    * len(backend_classes)
    * len(conditions)
)

print("=" * 70)
print("CALIBRATIONCOMPASS - DYNAMIC DATASET V2")
print("=" * 70)

print(f"\nCircuits: {NUMBER_OF_CIRCUITS}")
print(f"Backends: {len(backend_classes)}")
print(f"Conditions: {len(conditions)}")
print(f"Expected rows: {total_rows}")
print(f"Shots: {SHOTS}")


for circuit_id in range(NUMBER_OF_CIRCUITS):

    seed = 5000 + circuit_id

    circuit = make_random_circuit(seed)

    print(
        f"\nCircuit "
        f"{circuit_id + 1}/{NUMBER_OF_CIRCUITS} "
        f"({circuit.num_qubits} qubits)"
    )

    for backend_name, BackendClass in backend_classes.items():

        # Create a clean backend for transpilation
        base_backend = BackendClass()

        transpiled = transpile(
            circuit,
            backend=base_backend,
            optimization_level=1
        )

        for condition in conditions:

            # Create a NEW backend for this scenario
            scenario_backend = BackendClass()

            # Apply synthetic drift
            drift_description = apply_drift(
                scenario_backend,
                transpiled,
                condition
            )

            result = evaluate(
                transpiled,
                scenario_backend
            )

            results.append({

                "circuit_id": circuit_id,

                "seed": seed,

                "original_qubits": circuit.num_qubits,

                "original_depth": circuit.depth(),

                "backend": backend_name,

                "condition": condition,

                "drift": drift_description,

                **result

            })

        print(
            f"  {backend_name}: completed"
        )


# ============================================================
# 13. SAVE DATASET
# ============================================================

df = pd.DataFrame(results)

output_file = (
    "results/calibration_dataset_v2.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 14. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET V2 COMPLETE")
print("=" * 70)

print(f"\nRows generated: {len(df)}")

print("\nAverage fidelity by condition:")

print(
    df.groupby("condition")["fidelity"]
    .mean()
)


print("\nAverage fidelity by backend:")

print(
    df.groupby("backend")["fidelity"]
    .mean()
)


print("\nDataset saved to:")

print(output_file)


print("\nEXPERIMENT COMPLETE")