import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_ibm_runtime import fake_provider
from qiskit.quantum_info import hellinger_fidelity


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
# READOUT ERROR
# ============================================================

def get_readout_error(backend, physical):

    try:
        props = backend.target["measure"][(physical,)]

        if props is not None and props.error is not None:
            return float(props.error)

    except (KeyError, TypeError):
        pass

    return None


# ============================================================
# FINAL LOGICAL -> PHYSICAL MAPPING
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
# RUN ONE CANDIDATE
# ============================================================

def analyze_candidate(
    candidate_seed,
    circuit,
    backend,
    shots
):

    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=candidate_seed
    )

    mapping = get_mapping(transpiled)

    # Physical qubits actually used for logical data qubits
    used_physical = list(mapping.values())

    # --------------------------------------------------------
    # IDEAL REFERENCE
    # --------------------------------------------------------

    ideal_simulator = AerSimulator()

    ideal_result = ideal_simulator.run(
        transpiled,
        shots=shots,
        seed_simulator=12345
    ).result()

    ideal_counts = ideal_result.get_counts()

    ideal_probabilities = {
        state: count / shots
        for state, count in ideal_counts.items()
    }

    # --------------------------------------------------------
    # TEST EACH PHYSICAL QUBIT'S READOUT ERROR ALONE
    # --------------------------------------------------------

    rows = []

    for physical in used_physical:

        error = get_readout_error(
            backend,
            physical
        )

        if error is None:
            continue

        # Symmetric readout error model:
        #
        # 0 -> 1 with probability p
        # 1 -> 0 with probability p
        #
        readout_error = ReadoutError([
            [1.0 - error, error],
            [error, 1.0 - error]
        ])

        noise_model = NoiseModel()

        noise_model.add_readout_error(
            readout_error,
            [physical]
        )

        simulator = AerSimulator(
            noise_model=noise_model
        )

        noisy_result = simulator.run(
            transpiled,
            shots=shots,
            seed_simulator=54321
        ).result()

        noisy_counts = noisy_result.get_counts()

        noisy_probabilities = {
            state: count / shots
            for state, count in noisy_counts.items()
        }

        fidelity = hellinger_fidelity(
            ideal_probabilities,
            noisy_probabilities
        )

        fidelity_loss = 1.0 - fidelity

        # Find which logical qubit occupies this physical qubit
        logical = None

        for logical_q, physical_q in mapping.items():

            if physical_q == physical:
                logical = logical_q
                break

        rows.append({
            "candidate": candidate_seed,
            "logical_qubit": logical,
            "physical_qubit": physical,
            "readout_error": error,
            "fidelity_with_this_qubit_only": fidelity,
            "fidelity_loss": fidelity_loss
        })

    return rows


# ============================================================
# SETTINGS
# ============================================================

CIRCUIT_SEED = 20000
SHOTS = 10000

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(
    CIRCUIT_SEED
)


# ============================================================
# RUN
# ============================================================

all_results = []

for candidate_seed in [22, 33]:

    print()
    print("=" * 90)
    print(f"CANDIDATE SEED {candidate_seed}")
    print("=" * 90)

    rows = analyze_candidate(
        candidate_seed,
        circuit,
        backend,
        SHOTS
    )

    rows = sorted(
        rows,
        key=lambda x: x["fidelity_loss"],
        reverse=True
    )

    print()
    print(
        f"{'LOGICAL':<10}"
        f"{'PHYSICAL':<12}"
        f"{'READOUT ERR':<15}"
        f"{'FIDELITY':<15}"
        f"{'FIDELITY LOSS':<15}"
    )

    print("-" * 90)

    for row in rows:

        print(
            f"q[{row['logical_qubit']}]"
            f"{row['physical_qubit']:>10}"
            f"{row['readout_error']:>15.6f}"
            f"{row['fidelity_with_this_qubit_only']:>15.6f}"
            f"{row['fidelity_loss']:>15.6f}"
        )

        all_results.append(row)

    total_loss = sum(
        row["fidelity_loss"]
        for row in rows
    )

    print("-" * 90)

    print(
        "Sum of individual qubit fidelity losses:",
        f"{total_loss:.6f}"
    )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(all_results)

df.to_csv(
    "results/readout_sensitivity.csv",
    index=False
)

print()
print("=" * 90)
print("READOUT SENSITIVITY COMPLETE")
print("=" * 90)

print()
print("Saved:")
print("results/readout_sensitivity.csv")