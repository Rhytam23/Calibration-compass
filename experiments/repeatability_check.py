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
# SAME CIRCUIT
# ============================================================

def make_random_circuit(seed):
    rng = np.random.default_rng(seed)

    num_qubits = int(rng.integers(4, 9))
    circuit_depth = int(rng.integers(3, 9))

    qc = QuantumCircuit(num_qubits, num_qubits)

    for _ in range(circuit_depth):

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

        number_of_entangling_gates = max(
            1,
            num_qubits // 2
        )

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
# SETTINGS
# ============================================================

CIRCUIT_SEED = 20000
CANDIDATES = [22, 33]

SHOTS = 2000
REPEATS = 10

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(CIRCUIT_SEED)


# ============================================================
# RUN
# ============================================================

results = []

for candidate_seed in CANDIDATES:

    print()
    print("=" * 80)
    print(f"CANDIDATE SEED {candidate_seed}")
    print("=" * 80)

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=candidate_seed
    )

    # Ideal simulator
    ideal_simulator = AerSimulator()

    ideal_result = ideal_simulator.run(
        transpiled,
        shots=SHOTS,
        seed_simulator=12345
    ).result()

    ideal_counts = ideal_result.get_counts()

    ideal_probabilities = {
        key: value / SHOTS
        for key, value in ideal_counts.items()
    }

    # Noisy simulator
    noisy_simulator = AerSimulator.from_backend(
        backend
    )

    fidelities = []

    for repeat in range(REPEATS):

        noisy_result = noisy_simulator.run(
            transpiled,
            shots=SHOTS,
            seed_simulator=1000 + repeat
        ).result()

        noisy_counts = noisy_result.get_counts()

        noisy_probabilities = {
            key: value / SHOTS
            for key, value in noisy_counts.items()
        }

        fidelity = hellinger_fidelity(
            ideal_probabilities,
            noisy_probabilities
        )

        fidelities.append(fidelity)

        results.append({
            "candidate_seed": candidate_seed,
            "repeat": repeat + 1,
            "fidelity": fidelity
        })

        print(
            f"Repeat {repeat + 1:2d}: "
            f"{fidelity:.6f}"
        )

    print()
    print(
        f"Mean fidelity: "
        f"{np.mean(fidelities):.6f}"
    )

    print(
        f"Std deviation: "
        f"{np.std(fidelities):.6f}"
    )


# ============================================================
# SUMMARY
# ============================================================

df = pd.DataFrame(results)

print()
print("=" * 80)
print("REPEATABILITY SUMMARY")
print("=" * 80)

summary = (
    df.groupby("candidate_seed")["fidelity"]
    .agg(["mean", "std", "min", "max"])
)

print(summary.to_string())


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "results/repeatability_check.csv",
    index=False
)

print()
print("Saved:")
print("results/repeatability_check.csv")

print()
print("=" * 80)
print("CHECK COMPLETE")
print("=" * 80)