import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import fake_provider


# ============================================================
# SAME CIRCUIT GENERATOR
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
# PHYSICAL SX ERROR
# ============================================================

def get_sx_error(backend, q):

    try:
        props = backend.target["sx"][(q,)]

        if props is not None:
            return props.error

    except (KeyError, TypeError):
        pass

    return None


# ============================================================
# ANALYZE LOGICAL CIRCUIT
# ============================================================

def logical_activity(circuit):

    activity = {}

    for q in range(circuit.num_qubits):

        activity[q] = {
            "one_qubit": 0,
            "two_qubit": 0,
            "total": 0
        }

    for instruction in circuit.data:

        for q in instruction.qubits:

            logical = circuit.find_bit(q).index

            if instruction.operation.name in [
                "measure",
                "barrier"
            ]:
                continue

            activity[logical]["total"] += 1

            if len(instruction.qubits) == 1:
                activity[logical]["one_qubit"] += 1

            elif len(instruction.qubits) == 2:
                activity[logical]["two_qubit"] += 1

    return activity


# ============================================================
# GET FINAL MAPPING
# ============================================================

def get_mapping(transpiled):

    layout = transpiled.layout

    final = layout.final_index_layout(
        filter_ancillas=True
    )

    mapping = {}

    for logical, physical in enumerate(final):

        mapping[logical] = physical

    return mapping


# ============================================================
# RUN
# ============================================================

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(20000)

activity = logical_activity(circuit)


print("=" * 90)
print("CALIBRATIONCOMPASS - LOGICAL MAPPING ANALYSIS")
print("=" * 90)

print("\nLOGICAL CIRCUIT ACTIVITY")
print("-" * 90)

print(
    f"{'LOGICAL':<10}"
    f"{'1Q GATES':>12}"
    f"{'2Q PARTICIPATION':>20}"
    f"{'TOTAL':>12}"
)

for logical in range(circuit.num_qubits):

    a = activity[logical]

    print(
        f"q[{logical}]"
        f"{a['one_qubit']:>12}"
        f"{a['two_qubit']:>20}"
        f"{a['total']:>12}"
    )


# ============================================================
# COMPARE SEED 22 / 33
# ============================================================

for seed in [22, 33]:

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed
    )

    mapping = get_mapping(transpiled)

    print()
    print("=" * 90)
    print(f"SEED {seed}")
    print("=" * 90)

    print(
        f"{'LOGICAL':<10}"
        f"{'PHYSICAL':>12}"
        f"{'2Q ACTIVITY':>15}"
        f"{'SX ERROR':>15}"
        f"{'WEIGHTED SX':>18}"
    )

    print("-" * 90)

    total_weighted = 0.0

    for logical in range(circuit.num_qubits):

        physical = mapping[logical]

        twoq = activity[logical]["two_qubit"]

        sx_error = get_sx_error(
            backend,
            physical
        )

        if sx_error is None:
            weighted = np.nan
        else:
            weighted = twoq * sx_error
            total_weighted += weighted

        print(
            f"q[{logical}]"
            f"{physical:>12}"
            f"{twoq:>15}"
            f"{(np.nan if sx_error is None else sx_error):>15.6f}"
            f"{weighted:>18.6f}"
        )

    print("-" * 90)

    print(
        "TOTAL ACTIVITY-WEIGHTED SX ERROR:",
        f"{total_weighted:.6f}"
    )


print()
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)