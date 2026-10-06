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
# 1. CREATE DIFFERENT TYPES OF QUANTUM CIRCUITS
# ============================================================

def make_ghz():
    """4-qubit GHZ circuit."""
    qc = QuantumCircuit(4, 4)

    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    qc.measure(range(4), range(4))

    return qc


def make_qft_like():
    """Small QFT-style circuit."""
    qc = QuantumCircuit(4, 4)

    qc.h(0)
    qc.cp(np.pi / 2, 1, 0)
    qc.cp(np.pi / 4, 2, 0)
    qc.cp(np.pi / 8, 3, 0)

    qc.h(1)
    qc.cp(np.pi / 2, 2, 1)
    qc.cp(np.pi / 4, 3, 1)

    qc.h(2)
    qc.cp(np.pi / 2, 3, 2)

    qc.h(3)

    qc.measure(range(4), range(4))

    return qc


def make_qaoa_like():
    """Small QAOA-style circuit."""
    qc = QuantumCircuit(3, 3)

    # Put all qubits into superposition
    for q in range(3):
        qc.h(q)

    # Problem layer
    qc.rzz(0.8, 0, 1)
    qc.rzz(0.8, 1, 2)
    qc.rzz(0.8, 0, 2)

    # Mixer layer
    for q in range(3):
        qc.rx(1.2, q)

    qc.measure(range(3), range(3))

    return qc


def make_hardware_efficient():
    """Small hardware-efficient style circuit."""
    qc = QuantumCircuit(4, 4)

    # Single-qubit layer
    for q in range(4):
        qc.ry(0.7 + 0.1 * q, q)
        qc.rz(0.4 + 0.1 * q, q)

    # Entangling layer
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # Second single-qubit layer
    for q in range(4):
        qc.ry(0.3, q)

    qc.measure(range(4), range(4))

    return qc


# ============================================================
# 2. CONVERT COUNTS INTO PROBABILITIES
# ============================================================

def counts_to_probabilities(counts, shots):
    """Convert measurement counts into probabilities."""

    probabilities = {}

    for bitstring, count in counts.items():
        probabilities[bitstring] = count / shots

    return probabilities


# ============================================================
# 3. RUN A CIRCUIT AND CALCULATE FIDELITY
# ============================================================

def evaluate_circuit(circuit, backend, shots=5000):

    # --------------------------------------------------------
    # Transpile for the selected hardware
    # --------------------------------------------------------

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1
    )

    # --------------------------------------------------------
    # Ideal simulation
    # --------------------------------------------------------

    ideal_simulator = AerSimulator()

    ideal_job = ideal_simulator.run(
        transpiled,
        shots=shots
    )

    ideal_result = ideal_job.result()

    ideal_counts = ideal_result.get_counts()

    # --------------------------------------------------------
    # Noisy simulation using the backend's noise model
    # --------------------------------------------------------

    noisy_simulator = AerSimulator.from_backend(backend)

    noisy_job = noisy_simulator.run(
        transpiled,
        shots=shots
    )

    noisy_result = noisy_job.result()

    noisy_counts = noisy_result.get_counts()

    # --------------------------------------------------------
    # Convert to probabilities
    # --------------------------------------------------------

    ideal_probabilities = counts_to_probabilities(
        ideal_counts,
        shots
    )

    noisy_probabilities = counts_to_probabilities(
        noisy_counts,
        shots
    )

    # --------------------------------------------------------
    # Calculate Hellinger fidelity
    # --------------------------------------------------------

    fidelity = hellinger_fidelity(
        ideal_probabilities,
        noisy_probabilities
    )

    # --------------------------------------------------------
    # Circuit statistics
    # --------------------------------------------------------

    operations = transpiled.count_ops()

    two_qubit_gates = 0

    for gate in ["cx", "ecr", "cz", "swap", "rzz"]:
        two_qubit_gates += operations.get(gate, 0)

    return {
        "fidelity": fidelity,
        "depth": transpiled.depth(),
        "two_qubit_gates": two_qubit_gates
    }


# ============================================================
# 4. CREATE OUR CIRCUIT COLLECTION
# ============================================================

circuits = {
    "GHZ": make_ghz(),
    "QFT_like": make_qft_like(),
    "QAOA_like": make_qaoa_like(),
    "Hardware_efficient": make_hardware_efficient()
}


# ============================================================
# 5. LOAD OUR IBM FAKE BACKENDS
# ============================================================

backends = {
    "Sherbrooke": fake_provider.FakeSherbrooke(),
    "Torino": fake_provider.FakeTorino(),
    "Fez": fake_provider.FakeFez()
}


# ============================================================
# 6. RUN THE FULL BENCHMARK
# ============================================================

results = []

print("=" * 70)
print("CALIBRATIONCOMPASS - FIDELITY BENCHMARK")
print("=" * 70)

for circuit_name, circuit in circuits.items():

    print("\n" + "-" * 70)
    print(f"CIRCUIT: {circuit_name}")
    print("-" * 70)

    for backend_name, backend in backends.items():

        print(f"\nRunning on {backend_name}...")

        result = evaluate_circuit(
            circuit,
            backend,
            shots=5000
        )

        results.append({
            "circuit": circuit_name,
            "backend": backend_name,
            "fidelity": result["fidelity"],
            "depth": result["depth"],
            "two_qubit_gates": result["two_qubit_gates"]
        })

        print(f"Fidelity:       {result['fidelity']:.4f}")
        print(f"Depth:          {result['depth']}")
        print(f"Two-qubit gates:{result['two_qubit_gates']}")


# ============================================================
# 7. SAVE THE DATASET
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    "results/fidelity_benchmark.csv",
    index=False
)


# ============================================================
# 8. SHOW FINAL TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(df.to_string(index=False))

print("\nSaved to:")
print("results/fidelity_benchmark.csv")

print("\nEXPERIMENT COMPLETE")