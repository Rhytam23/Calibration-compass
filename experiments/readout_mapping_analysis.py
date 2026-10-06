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
# GET READOUT ERROR
# ============================================================

def get_readout_error(backend, physical):

    try:
        props = backend.target["measure"][(physical,)]

        if props is not None:
            return props.error

    except (KeyError, TypeError):
        pass

    return None


# ============================================================
# GET FINAL MAPPING
# ============================================================

def get_mapping(transpiled):

    final = transpiled.layout.final_index_layout(
        filter_ancillas=True
    )

    mapping = {}

    for logical, physical in enumerate(final):

        mapping[logical] = physical

    return mapping


# ============================================================
# LOGICAL CIRCUIT ACTIVITY
# ============================================================

def get_activity(circuit):

    activity = {
        q: {
            "one_qubit": 0,
            "two_qubit": 0,
            "total": 0
        }
        for q in range(circuit.num_qubits)
    }

    for instruction in circuit.data:

        if instruction.operation.name in [
            "measure",
            "barrier"
        ]:
            continue

        for q in instruction.qubits:

            logical = circuit.find_bit(q).index

            activity[logical]["total"] += 1

            if len(instruction.qubits) == 1:
                activity[logical]["one_qubit"] += 1

            elif len(instruction.qubits) == 2:
                activity[logical]["two_qubit"] += 1

    return activity


# ============================================================
# RUN
# ============================================================

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(20000)

activity = get_activity(circuit)

print("=" * 100)
print("CALIBRATIONCOMPASS - READOUT-AWARE MAPPING")
print("=" * 100)

print()
print("LOGICAL CIRCUIT ACTIVITY")
print("-" * 100)

print(
    f"{'LOGICAL':<10}"
    f"{'1Q':>8}"
    f"{'2Q':>8}"
    f"{'TOTAL':>10}"
)

for q in range(circuit.num_qubits):

    a = activity[q]

    print(
        f"q[{q}]"
        f"{a['one_qubit']:>8}"
        f"{a['two_qubit']:>8}"
        f"{a['total']:>10}"
    )


# ============================================================
# CANDIDATES
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
    print("=" * 100)
    print(f"CANDIDATE SEED {seed}")
    print("=" * 100)

    print(
        f"{'LOGICAL':<10}"
        f"{'PHYSICAL':>10}"
        f"{'2Q ACTIVITY':>15}"
        f"{'READOUT ERR':>15}"
        f"{'ACTIVITY × ERR':>20}"
    )

    print("-" * 100)

    total_risk = 0.0

    for logical in range(circuit.num_qubits):

        physical = mapping[logical]

        readout_error = get_readout_error(
            backend,
            physical
        )

        two_qubit_activity = (
            activity[logical]["two_qubit"]
        )

        if readout_error is None:

            weighted_risk = np.nan

        else:

            weighted_risk = (
                two_qubit_activity
                * readout_error
            )

            total_risk += weighted_risk

        print(
            f"q[{logical}]"
            f"{physical:>10}"
            f"{two_qubit_activity:>15}"
            f"{(np.nan if readout_error is None else readout_error):>15.6f}"
            f"{weighted_risk:>20.6f}"
        )

    print("-" * 100)

    print(
        "TOTAL ACTIVITY-WEIGHTED READOUT RISK:",
        f"{total_risk:.6f}"
    )


print()
print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)