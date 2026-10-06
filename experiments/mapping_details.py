import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import fake_provider


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


def inspect_candidate(circuit, backend, seed):

    print()
    print("=" * 70)
    print(f"CANDIDATE SEED: {seed}")
    print("=" * 70)

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed
    )

    print("\nTRANSPILED STATISTICS")
    print("-" * 70)
    print("Depth:", transpiled.depth())
    print("Gate counts:", dict(transpiled.count_ops()))

    layout = transpiled.layout

    if layout is None:
        print("\nNO LAYOUT INFORMATION")
        return

    # ---------------------------------------------------------
    # INITIAL MAPPING
    # ---------------------------------------------------------

    print("\nINITIAL LOGICAL -> PHYSICAL MAPPING")
    print("-" * 70)

    try:
        initial = layout.initial_index_layout(filter_ancillas=True)

        for logical, physical in enumerate(initial):
            print(
                f"logical q[{logical}] -> physical {physical}"
            )

    except Exception as e:
        print("ERROR:", repr(e))

    # ---------------------------------------------------------
    # FINAL MAPPING
    # ---------------------------------------------------------

    print("\nFINAL LOGICAL -> PHYSICAL MAPPING")
    print("-" * 70)

    try:
        final = layout.final_index_layout(filter_ancillas=True)

        for logical, physical in enumerate(final):
            print(
                f"logical q[{logical}] -> physical {physical}"
            )

    except Exception as e:
        print("ERROR:", repr(e))

    # ---------------------------------------------------------
    # ROUTING PERMUTATION
    # ---------------------------------------------------------

    print("\nROUTING PERMUTATION")
    print("-" * 70)

    try:
        print(layout.routing_permutation())

    except Exception as e:
        print("ERROR:", repr(e))

    # ---------------------------------------------------------
    # PHYSICAL TWO-QUBIT INTERACTIONS
    # ---------------------------------------------------------

    print("\nTWO-QUBIT OPERATIONS")
    print("-" * 70)

    count = 0

    for instruction in transpiled.data:

        if len(instruction.qubits) != 2:
            continue

        q0 = transpiled.find_bit(
            instruction.qubits[0]
        ).index

        q1 = transpiled.find_bit(
            instruction.qubits[1]
        ).index

        print(
            f"{count:2d}: "
            f"{instruction.operation.name} "
            f"({q0}, {q1})"
        )

        count += 1

    print("\nTotal 2Q operations:", count)


# ============================================================
# REAL CIRCUIT 0
# ============================================================

CIRCUIT_SEED = 20000
CANDIDATE_SEEDS = [11, 22, 33, 44, 55, 66]

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(CIRCUIT_SEED)

print("=" * 70)
print("CALIBRATIONCOMPASS MAPPING INSPECTOR")
print("=" * 70)

print("Circuit seed:", CIRCUIT_SEED)
print("Logical qubits:", circuit.num_qubits)
print("Original depth:", circuit.depth())

for seed in CANDIDATE_SEEDS:
    inspect_candidate(
        circuit,
        backend,
        seed
    )

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)