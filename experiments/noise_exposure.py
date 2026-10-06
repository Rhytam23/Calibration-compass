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


def get_prop(backend, instruction, qubits):
    try:
        return backend.target[instruction][tuple(qubits)]
    except (KeyError, TypeError):
        return None


def inspect(seed, circuit, backend):

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed
    )

    one_qubit_counts = {}
    two_qubit_counts = {}

    for instruction in transpiled.data:

        name = instruction.operation.name

        if len(instruction.qubits) == 1:

            q = transpiled.find_bit(
                instruction.qubits[0]
            ).index

            key = (q, name)

            one_qubit_counts[key] = (
                one_qubit_counts.get(key, 0) + 1
            )

        elif len(instruction.qubits) == 2:

            q0 = transpiled.find_bit(
                instruction.qubits[0]
            ).index

            q1 = transpiled.find_bit(
                instruction.qubits[1]
            ).index

            edge = tuple(sorted((q0, q1)))

            two_qubit_counts[edge] = (
                two_qubit_counts.get(edge, 0) + 1
            )

    print()
    print("=" * 90)
    print(f"SEED {seed}")
    print("=" * 90)

    print()
    print("DEPTH:", transpiled.depth())
    print("GATES:", dict(transpiled.count_ops()))

    print()
    print("PHYSICAL QUBIT NOISE EXPOSURE")
    print("-" * 90)

    qubits = set()

    for (q, name) in one_qubit_counts:
        qubits.add(q)

    for q0, q1 in two_qubit_counts:
        qubits.add(q0)
        qubits.add(q1)

    print(
        f"{'QUBIT':<8}"
        f"{'SX':>8}"
        f"{'RZ':>8}"
        f"{'T1(us)':>14}"
        f"{'T2(us)':>14}"
        f"{'SX ERR':>14}"
    )

    for q in sorted(qubits):

        sx_count = one_qubit_counts.get(
            (q, "sx"),
            0
        )

        rz_count = one_qubit_counts.get(
            (q, "rz"),
            0
        )

        try:
            props = backend.qubit_properties(q)

            t1 = (
                props.t1 * 1e6
                if props.t1 is not None
                else np.nan
            )

            t2 = (
                props.t2 * 1e6
                if props.t2 is not None
                else np.nan
            )

        except Exception:
            t1 = np.nan
            t2 = np.nan

        sx_props = get_prop(
            backend,
            "sx",
            (q,)
        )

        sx_error = (
            sx_props.error
            if sx_props is not None
            else np.nan
        )

        print(
            f"{q:<8}"
            f"{sx_count:>8}"
            f"{rz_count:>8}"
            f"{t1:>14.2f}"
            f"{t2:>14.2f}"
            f"{sx_error:>14.6f}"
        )

    print()
    print("PHYSICAL CZ LINKS")
    print("-" * 90)

    print(
        f"{'EDGE':<12}"
        f"{'USES':>8}"
        f"{'CZ ERR':>14}"
    )

    for edge, count in sorted(
        two_qubit_counts.items()
    ):

        props = get_prop(
            backend,
            "cz",
            edge
        )

        if props is None:
            props = get_prop(
                backend,
                "cz",
                (edge[1], edge[0])
            )

        error = (
            props.error
            if props is not None
            else np.nan
        )

        print(
            f"{edge[0]}-{edge[1]:<7}"
            f"{count:>8}"
            f"{error:>14.6f}"
        )


CIRCUIT_SEED = 20000

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(
    CIRCUIT_SEED
)

print("=" * 90)
print("CALIBRATIONCOMPASS - FULL NOISE EXPOSURE")
print("=" * 90)

print("Circuit:", CIRCUIT_SEED)
print("Logical qubits:", circuit.num_qubits)

for seed in [22, 33]:

    inspect(
        seed,
        circuit,
        backend
    )

print()
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)