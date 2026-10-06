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

NUMBER_OF_CIRCUITS = 10
SHOTS = 2000

# Different transpiler seeds give us different candidates.
CANDIDATE_SEEDS = [11, 22, 33, 44, 55, 66]


# ============================================================
# 1. CREATE RANDOM QUANTUM CIRCUITS
# ============================================================

def make_random_circuit(seed):

    rng = np.random.default_rng(seed)

    num_qubits = int(
        rng.integers(4, 9)
    )

    circuit_depth = int(
        rng.integers(3, 9)
    )

    qc = QuantumCircuit(
        num_qubits,
        num_qubits
    )

    for _ in range(circuit_depth):

        # -------------------------------
        # Single-qubit operations
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
        # Two-qubit operations
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
# 2. COUNTS -> PROBABILITIES
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
# 3. COUNT TWO-QUBIT GATES
# ============================================================

def count_two_qubit_gates(
    transpiled
):

    operations = transpiled.count_ops()

    two_qubit_gates = 0

    for gate_name in [
        "cx",
        "ecr",
        "cz",
        "swap",
        "rzz"
    ]:

        two_qubit_gates += (
            operations.get(
                gate_name,
                0
            )
        )

    return two_qubit_gates


# ============================================================
# 4. CALCULATE ESP
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

            esp *= max(
                0.0,
                1.0 - error
            )

    return esp


# ============================================================
# 5. EVALUATE ONE CANDIDATE
# ============================================================

def evaluate_candidate(
    circuit,
    backend,
    seed
):

    # --------------------------------------------------------
    # Create the candidate compilation
    # --------------------------------------------------------
    #
    # SABRE uses the seed to explore different choices.
    # Different seeds can therefore produce different
    # layouts/routing decisions.
    # --------------------------------------------------------

    transpiled = transpile(

        circuit,

        backend=backend,

        optimization_level=1,

        layout_method="sabre",

        routing_method="sabre",

        seed_transpiler=seed
    )

    # --------------------------------------------------------
    # Ideal simulation
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
    # Noisy simulation
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
    # Convert counts to probabilities
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
    # Calculate Hellinger fidelity
    # --------------------------------------------------------

    fidelity = hellinger_fidelity(
        ideal_probabilities,
        noisy_probabilities
    )

    # --------------------------------------------------------
    # Circuit statistics
    # --------------------------------------------------------

    operations = (
        transpiled.count_ops()
    )

    one_qubit_gates = 0

    for gate_name, count in operations.items():

        if gate_name in [
            "measure",
            "barrier"
        ]:

            continue

        if gate_name not in [
            "cx",
            "ecr",
            "cz",
            "swap",
            "rzz"
        ]:

            one_qubit_gates += count

    two_qubit_gates = (
        count_two_qubit_gates(
            transpiled
        )
    )

    esp = calculate_esp(
        backend,
        transpiled
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "fidelity":
            fidelity,

        "depth":
            transpiled.depth(),

        "one_qubit_gates":
            one_qubit_gates,

        "two_qubit_gates":
            two_qubit_gates,

        "esp":
            esp

    }


# ============================================================
# 6. BACKENDS
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
# 7. RUN EXPERIMENT
# ============================================================

results = []

print("=" * 75)
print("CALIBRATIONCOMPASS - CANDIDATE BENCHMARK")
print("=" * 75)

print(
    f"\nCircuits: "
    f"{NUMBER_OF_CIRCUITS}"
)

print(
    f"Backends: "
    f"{len(backend_classes)}"
)

print(
    f"Candidates per backend: "
    f"{len(CANDIDATE_SEEDS)}"
)

print(
    f"Shots: "
    f"{SHOTS}"
)

print(
    f"\nExpected candidate executions: "
    f"{NUMBER_OF_CIRCUITS * len(backend_classes) * len(CANDIDATE_SEEDS)}"
)


for circuit_id in range(
    NUMBER_OF_CIRCUITS
):

    circuit_seed = (
        20000 + circuit_id
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

    for backend_name, BackendClass in (
        backend_classes.items()
    ):

        print(
            f"  Backend: "
            f"{backend_name}"
        )

        backend = BackendClass()

        for seed in CANDIDATE_SEEDS:

            result = evaluate_candidate(
                circuit,
                backend,
                seed
            )

            results.append({

                "circuit_id":
                    circuit_id,

                "circuit_seed":
                    circuit_seed,

                "backend":
                    backend_name,

                "candidate_seed":
                    seed,

                "original_qubits":
                    circuit.num_qubits,

                "original_depth":
                    circuit.depth(),

                **result

            })

        print(
            "    candidates completed"
        )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

df = pd.DataFrame(
    results
)

output_file = (
    "results/candidate_benchmark.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 9. ANALYZE CANDIDATE QUALITY
# ============================================================

print("\n")
print("=" * 75)
print("CANDIDATE ANALYSIS")
print("=" * 75)


# ------------------------------------------------------------
# Best candidate for each circuit/backend
# ------------------------------------------------------------

best_candidates = df.loc[
    df.groupby(
        [
            "circuit_id",
            "backend"
        ]
    )["fidelity"].idxmax()
].copy()


print(
    "\nBEST CANDIDATE BY CIRCUIT AND BACKEND"
)

print(
    best_candidates[
        [
            "circuit_id",
            "backend",
            "candidate_seed",
            "fidelity",
            "depth",
            "two_qubit_gates",
            "esp"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Fidelity spread
# ------------------------------------------------------------

spread = (
    df.groupby(
        [
            "circuit_id",
            "backend"
        ]
    )["fidelity"]
    .agg(
        [
            "min",
            "max"
        ]
    )
)

spread["difference"] = (
    spread["max"]
    -
    spread["min"]
)


print(
    "\nCANDIDATE FIDELITY SPREAD"
)

print(
    f"Average difference between "
    f"best and worst candidate: "
    f"{spread['difference'].mean():.6f}"
)

print(
    f"Maximum difference: "
    f"{spread['difference'].max():.6f}"
)


# ------------------------------------------------------------
# Biggest candidate differences
# ------------------------------------------------------------

print(
    "\nLARGEST CANDIDATE DIFFERENCES"
)

largest_spreads = (
    spread
    .sort_values(
        "difference",
        ascending=False
    )
    .head(15)
)

print(
    largest_spreads.to_string()
)


# ============================================================
# 10. FIND SURPRISING CASES
# ============================================================
#
# Cases where the candidate with MORE two-qubit gates
# nevertheless has BETTER fidelity than another candidate.
#
# This shows why simple gate-count optimization is not
# always enough.
# ============================================================

surprising_cases = []

for (
    circuit_id,
    backend
), group in df.groupby(
    [
        "circuit_id",
        "backend"
    ]
):

    rows = group.to_dict(
        "records"
    )

    for a in rows:

        for b in rows:

            if (
                a["candidate_seed"]
                == b["candidate_seed"]
            ):
                continue

            if (
                a["two_qubit_gates"]
                >
                b["two_qubit_gates"]
                and
                a["fidelity"]
                >
                b["fidelity"]
            ):

                surprising_cases.append({

                    "circuit_id":
                        circuit_id,

                    "backend":
                        backend,

                    "better_fidelity_seed":
                        a["candidate_seed"],

                    "better_fidelity":
                        a["fidelity"],

                    "better_fidelity_2q":
                        a["two_qubit_gates"],

                    "worse_fidelity_seed":
                        b["candidate_seed"],

                    "worse_fidelity":
                        b["fidelity"],

                    "worse_fidelity_2q":
                        b["two_qubit_gates"]

                })


if surprising_cases:

    surprising_df = (
        pd.DataFrame(
            surprising_cases
        )
        .sort_values(
            "better_fidelity",
            ascending=False
        )
        .drop_duplicates(
            subset=[
                "circuit_id",
                "backend"
            ]
        )
    )

    print(
        "\nCASES WHERE MORE 2Q GATES "
        "STILL PRODUCED BETTER FIDELITY"
    )

    print(
        surprising_df
        .head(15)
        .to_string(index=False)
    )

else:

    surprising_df = pd.DataFrame()

    print(
        "\nNo surprising cases found."
    )


# ============================================================
# 11. SAVE ANALYSIS FILES
# ============================================================

best_candidates.to_csv(
    "results/best_candidates.csv",
    index=False
)

spread.reset_index().to_csv(
    "results/candidate_spread.csv",
    index=False
)

if not surprising_df.empty:

    surprising_df.to_csv(
        "results/surprising_candidates.csv",
        index=False
    )


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 75)
print("CANDIDATE BENCHMARK COMPLETE")
print("=" * 75)

print(
    "\nSaved:"
)

print(
    "results/candidate_benchmark.csv"
)

print(
    "results/best_candidates.csv"
)

print(
    "results/candidate_spread.csv"
)

if not surprising_df.empty:

    print(
        "results/surprising_candidates.csv"
    )