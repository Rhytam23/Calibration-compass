import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
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
# SETTINGS
# ============================================================

CIRCUIT_SEED = 20000
SHOTS = 10000

CANDIDATES = [11, 22, 33, 44, 55, 66]

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(
    CIRCUIT_SEED
)


# ============================================================
# NOISE MODELS
# ============================================================

noise_models = {

    "gate_only":
        NoiseModel.from_backend(
            backend,
            gate_error=True,
            readout_error=False,
            thermal_relaxation=False
        ),

    "thermal_only":
        NoiseModel.from_backend(
            backend,
            gate_error=False,
            readout_error=False,
            thermal_relaxation=True
        ),

    "readout_only":
        NoiseModel.from_backend(
            backend,
            gate_error=False,
            readout_error=True,
            thermal_relaxation=False
        ),

    "gate_plus_thermal":
        NoiseModel.from_backend(
            backend,
            gate_error=True,
            readout_error=False,
            thermal_relaxation=True
        ),

    "full":
        NoiseModel.from_backend(
            backend,
            gate_error=True,
            readout_error=True,
            thermal_relaxation=True
        )
}


# ============================================================
# RUN
# ============================================================

results = []


for seed in CANDIDATES:

    print()
    print("=" * 90)
    print(f"CANDIDATE SEED {seed}")
    print("=" * 90)

    candidate = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed
    )

    # --------------------------------------------------------
    # IDEAL REFERENCE
    # --------------------------------------------------------

    ideal_simulator = AerSimulator()

    ideal_result = ideal_simulator.run(
        candidate,
        shots=SHOTS,
        seed_simulator=12345
    ).result()

    ideal_counts = ideal_result.get_counts()

    ideal_probabilities = {
        state: count / SHOTS
        for state, count in ideal_counts.items()
    }

    # --------------------------------------------------------
    # FULL BACKEND BASELINE
    # --------------------------------------------------------

    baseline_simulator = AerSimulator.from_backend(
        backend
    )

    baseline_result = baseline_simulator.run(
        candidate,
        shots=SHOTS,
        seed_simulator=5000
    ).result()

    baseline_counts = (
        baseline_result.get_counts()
    )

    baseline_probabilities = {
        state: count / SHOTS
        for state, count in baseline_counts.items()
    }

    baseline_fidelity = hellinger_fidelity(
        ideal_probabilities,
        baseline_probabilities
    )

    print(
        f"Full backend: {baseline_fidelity:.6f}"
    )

    row = {
        "candidate_seed": seed,
        "full_backend": baseline_fidelity
    }

    # --------------------------------------------------------
    # INDIVIDUAL NOISE COMPONENTS
    # --------------------------------------------------------

    for name, noise_model in noise_models.items():

        simulator = AerSimulator(
            noise_model=noise_model
        )

        result = simulator.run(
            candidate,
            shots=SHOTS,
            seed_simulator=5000
        ).result()

        counts = result.get_counts()

        probabilities = {
            state: count / SHOTS
            for state, count in counts.items()
        }

        fidelity = hellinger_fidelity(
            ideal_probabilities,
            probabilities
        )

        row[name] = fidelity

        print(
            f"{name:<20}: {fidelity:.6f}"
        )

    results.append(row)


# ============================================================
# SUMMARY
# ============================================================

df = pd.DataFrame(results)


print()
print("=" * 100)
print("COMPLETE NOISE PROFILE")
print("=" * 100)

print()

print(
    df[
        [
            "candidate_seed",
            "full_backend",
            "gate_only",
            "thermal_only",
            "readout_only",
            "gate_plus_thermal",
            "full"
        ]
    ].to_string(index=False)
)


# ============================================================
# CALCULATE INDIVIDUAL LOSSES
# ============================================================

df["gate_loss"] = (
    1.0 - df["gate_only"]
)

df["thermal_loss"] = (
    1.0 - df["thermal_only"]
)

df["readout_loss"] = (
    1.0 - df["readout_only"]
)

df["gate_thermal_loss"] = (
    1.0 - df["gate_plus_thermal"]
)

df["total_loss"] = (
    1.0 - df["full"]
)


print()
print("=" * 100)
print("NOISE LOSSES")
print("=" * 100)

print()

print(
    df[
        [
            "candidate_seed",
            "gate_loss",
            "thermal_loss",
            "readout_loss",
            "gate_thermal_loss",
            "total_loss"
        ]
    ].to_string(index=False)
)


# ============================================================
# RANK EACH COMPONENT
# ============================================================

for column in [
    "full",
    "gate_only",
    "thermal_only",
    "readout_only",
    "gate_plus_thermal"
]:

    rank_column = (
        column + "_rank"
    )

    df[rank_column] = (
        df[column]
        .rank(
            ascending=False,
            method="min"
        )
    )


print()
print("=" * 100)
print("RANKING COMPARISON")
print("=" * 100)

print()

print(
    df[
        [
            "candidate_seed",
            "full_rank",
            "gate_only_rank",
            "thermal_only_rank",
            "readout_only_rank",
            "gate_plus_thermal_rank"
        ]
    ]
    .sort_values("full_rank")
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "results/candidate_noise_profile.csv",
    index=False
)

print()
print("Saved:")
print("results/candidate_noise_profile.csv")

print()
print("=" * 100)
print("NOISE PROFILE COMPLETE")
print("=" * 100)