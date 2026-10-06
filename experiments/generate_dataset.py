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
# SETTINGS
# ============================================================

NUMBER_OF_CIRCUITS = int(_os.environ.get("CC_NUM_CIRCUITS", 30))
SHOTS = 2000


# ============================================================
# 1. CREATE A RANDOM QUANTUM CIRCUIT
# ============================================================

def make_random_circuit(seed):

    rng = np.random.default_rng(seed)

    # Randomly choose circuit size
    num_qubits = int(rng.integers(4, 9))

    # Randomly choose circuit depth
    circuit_depth = int(rng.integers(3, 9))

    qc = QuantumCircuit(num_qubits, num_qubits)

    # Create several layers
    for _ in range(circuit_depth):

        # --------------------------------------------
        # Single-qubit gates
        # --------------------------------------------

        for q in range(num_qubits):

            gate_choice = rng.integers(0, 3)

            if gate_choice == 0:
                qc.h(q)

            elif gate_choice == 1:
                qc.ry(
                    float(rng.uniform(0, 2 * np.pi)),
                    q
                )

            else:
                qc.rz(
                    float(rng.uniform(0, 2 * np.pi)),
                    q
                )

        # --------------------------------------------
        # Random two-qubit gates
        # --------------------------------------------

        number_of_entangling_gates = max(1, num_qubits // 2)

        for _ in range(number_of_entangling_gates):

            q1, q2 = rng.choice(
                num_qubits,
                size=2,
                replace=False
            )

            qc.cx(int(q1), int(q2))

    # Measure everything
    qc.measure(
        range(num_qubits),
        range(num_qubits)
    )

    return qc


# ============================================================
# 2. CONVERT COUNTS TO PROBABILITIES
# ============================================================

def counts_to_probabilities(counts, shots):

    return {
        bitstring: count / shots
        for bitstring, count in counts.items()
    }


# ============================================================
# 3. GET ERROR OF A HARDWARE INSTRUCTION
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
# 4. EXTRACT HARDWARE FEATURES
# ============================================================

def get_hardware_features(backend, transpiled):

    active_qubits = set()

    gate_errors = []
    readout_errors = []

    # --------------------------------------------
    # Look through the transpiled circuit
    # --------------------------------------------

    for instruction in transpiled.data:

        operation = instruction.operation
        operation_name = operation.name

        # Get physical qubit numbers
        physical_qubits = tuple(
            transpiled.find_bit(q).index
            for q in instruction.qubits
        )

        # Save active physical qubits
        active_qubits.update(physical_qubits)

        # Measurement error
        if operation_name == "measure":

            error = get_instruction_error(
                backend,
                "measure",
                physical_qubits
            )

            if error is not None:
                readout_errors.append(error)

        # Other gate errors
        else:

            error = get_instruction_error(
                backend,
                operation_name,
                physical_qubits
            )

            if error is not None:
                gate_errors.append(error)

    # --------------------------------------------
    # Get T1 and T2
    # --------------------------------------------

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

    # --------------------------------------------
    # Helper for averages
    # --------------------------------------------

    def average(values):

        if len(values) == 0:
            return np.nan

        return float(np.mean(values))

    def minimum(values):

        if len(values) == 0:
            return np.nan

        return float(np.min(values))

    # --------------------------------------------
    # Hardware feature dictionary
    # --------------------------------------------

    return {
        "active_physical_qubits": len(active_qubits),

        "avg_gate_error": average(gate_errors),
        "max_gate_error": max(gate_errors)
        if gate_errors else np.nan,

        "avg_readout_error": average(readout_errors),
        "max_readout_error": max(readout_errors)
        if readout_errors else np.nan,

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
            operation_name if operation_name != "measure"
            else "measure",
            physical_qubits
        )

        if error is not None:

            # Convert error probability into success probability
            esp *= max(0.0, 1.0 - error)

    return esp


# ============================================================
# 6. EVALUATE ONE CIRCUIT ON ONE BACKEND
# ============================================================

def evaluate(circuit, backend, ideal_simulator, noisy_simulator):

    # --------------------------------------------
    # Transpile for this backend
    # --------------------------------------------

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1
    )

    # --------------------------------------------
    # Ideal simulation
    # --------------------------------------------

    ideal_job = ideal_simulator.run(
        transpiled,
        shots=SHOTS,
        seed_simulator=1234
    )

    ideal_counts = ideal_job.result().get_counts()

    # --------------------------------------------
    # Noisy simulation
    # --------------------------------------------

    noisy_job = noisy_simulator.run(
        transpiled,
        shots=SHOTS,
        seed_simulator=5678
    )

    noisy_counts = noisy_job.result().get_counts()

    # --------------------------------------------
    # Convert to probabilities
    # --------------------------------------------

    ideal_probabilities = counts_to_probabilities(
        ideal_counts,
        SHOTS
    )

    noisy_probabilities = counts_to_probabilities(
        noisy_counts,
        SHOTS
    )

    # --------------------------------------------
    # Calculate Hellinger fidelity
    # --------------------------------------------

    fidelity = hellinger_fidelity(
        ideal_probabilities,
        noisy_probabilities
    )

    # --------------------------------------------
    # Circuit statistics
    # --------------------------------------------

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

    # --------------------------------------------
    # Hardware information
    # --------------------------------------------

    hardware_features = get_hardware_features(
        backend,
        transpiled
    )

    # --------------------------------------------
    # ESP
    # --------------------------------------------

    esp = calculate_esp(
        backend,
        transpiled
    )

    # --------------------------------------------
    # Return everything
    # --------------------------------------------

    return {
        "transpiled_depth": transpiled.depth(),
        "one_qubit_gates": one_qubit_gates,
        "two_qubit_gates": two_qubit_gates,

        "esp": esp,

        "fidelity": fidelity,

        **hardware_features
    }


# ============================================================
# 7. LOAD OUR BACKENDS
# ============================================================

backends = {
    "Sherbrooke": fake_provider.FakeSherbrooke(),
    "Torino": fake_provider.FakeTorino(),
    "Fez": fake_provider.FakeFez()
}


# ============================================================
# 8. CREATE SIMULATORS
# ============================================================

ideal_simulator = AerSimulator()

noisy_simulators = {
    name: AerSimulator.from_backend(backend)
    for name, backend in backends.items()
}


# ============================================================
# 9. GENERATE DATA
# ============================================================

results = []

print("=" * 70)
print("CALIBRATIONCOMPASS - DATASET GENERATOR")
print("=" * 70)

print(f"\nGenerating {NUMBER_OF_CIRCUITS} circuits...")
print(f"Testing {len(backends)} backends...")
print(f"Shots per simulation: {SHOTS}")

print(
    f"\nTotal training examples: "
    f"{NUMBER_OF_CIRCUITS * len(backends)}"
)


for circuit_number in range(NUMBER_OF_CIRCUITS):

    seed = 1000 + circuit_number

    circuit = make_random_circuit(seed)

    print(
        f"\nCircuit {circuit_number + 1}"
        f"/{NUMBER_OF_CIRCUITS} "
        f"({circuit.num_qubits} qubits)"
    )

    for backend_name, backend in backends.items():

        print(f"  Running on {backend_name}...")

        result = evaluate(
            circuit,
            backend,
            ideal_simulator,
            noisy_simulators[backend_name]
        )

        results.append({
            "circuit_id": circuit_number,
            "seed": seed,
            "original_qubits": circuit.num_qubits,
            "original_depth": circuit.depth(),

            "backend": backend_name,

            **result
        })

        print(
            f"    Fidelity: {result['fidelity']:.4f} | "
            f"ESP: {result['esp']:.4f}"
        )


# ============================================================
# 10. SAVE DATASET
# ============================================================

df = pd.DataFrame(results)

output_file = "results/calibration_dataset.csv"

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 11. SHOW SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET COMPLETE")
print("=" * 70)

print(f"\nRows generated: {len(df)}")

print("\nAverage fidelity by backend:")

print(
    df.groupby("backend")["fidelity"]
    .mean()
    .sort_values(ascending=False)
)


print("\nDataset saved to:")
print(output_file)

print("\nFirst 5 rows:")

print(
    df.head().to_string(index=False)
)