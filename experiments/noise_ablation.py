import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import numpy as np

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
# SETTINGS
# ============================================================

CIRCUIT_SEED = 20000
CANDIDATES = [22, 33]

SHOTS = 10000

backend = fake_provider.FakeTorino()

circuit = make_random_circuit(
    CIRCUIT_SEED
)


# ============================================================
# CREATE CANDIDATE CIRCUITS
# ============================================================

candidates = {}

for seed in CANDIDATES:

    candidates[seed] = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed
    )


# ============================================================
# NOISE MODELS
# ============================================================

noise_models = {

    "full":
        NoiseModel.from_backend(
            backend,
            gate_error=True,
            readout_error=True,
            thermal_relaxation=True
        ),

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
        )
}


# ============================================================
# RUN IDEAL + NOISY
# ============================================================

results = []


for seed in CANDIDATES:

    candidate = candidates[seed]

    print()
    print("=" * 90)
    print(f"CANDIDATE SEED {seed}")
    print("=" * 90)

    # --------------------------------------------------------
    # Ideal reference
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
    # Full AerSimulator.from_backend baseline
    # --------------------------------------------------------

    print()
    print("BASELINE: AerSimulator.from_backend")

    full_backend_simulator = (
        AerSimulator.from_backend(backend)
    )

    result = full_backend_simulator.run(
        candidate,
        shots=SHOTS,
        seed_simulator=5000
    ).result()

    counts = result.get_counts()

    probabilities = {
        state: count / SHOTS
        for state, count in counts.items()
    }

    baseline_fidelity = hellinger_fidelity(
        ideal_probabilities,
        probabilities
    )

    print(
        f"Fidelity: {baseline_fidelity:.6f}"
    )

    results.append({
        "candidate": seed,
        "noise_model": "backend_baseline",
        "fidelity": baseline_fidelity
    })

    # --------------------------------------------------------
    # Noise ablation
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

        results.append({
            "candidate": seed,
            "noise_model": name,
            "fidelity": fidelity
        })

        print(
            f"{name:<20} "
            f"{fidelity:.6f}"
        )


# ============================================================
# SUMMARY TABLE
# ============================================================

print()
print("=" * 90)
print("NOISE ABLATION SUMMARY")
print("=" * 90)

for seed in CANDIDATES:

    print()
    print(f"CANDIDATE {seed}")
    print("-" * 60)

    for row in results:

        if row["candidate"] != seed:
            continue

        print(
            f"{row['noise_model']:<25}"
            f"{row['fidelity']:.6f}"
        )


# ============================================================
# GAP BETWEEN CANDIDATES
# ============================================================

print()
print("=" * 90)
print("SEED 33 - SEED 22 FIDELITY GAP")
print("=" * 90)

for noise_name in [
    "backend_baseline",
    "full",
    "gate_only",
    "thermal_only",
    "readout_only",
    "gate_plus_thermal"
]:

    values = {
        row["candidate"]: row["fidelity"]
        for row in results
        if row["noise_model"] == noise_name
    }

    gap = (
        values[CANDIDATES[1]]
        - values[CANDIDATES[0]]
    )

    print(
        f"{noise_name:<25}"
        f"{gap:+.6f}"
    )


# ============================================================
# SAVE
# ============================================================

import pandas as pd

df = pd.DataFrame(results)

df.to_csv(
    "results/noise_ablation.csv",
    index=False
)

print()
print("Saved:")
print("results/noise_ablation.csv")

print()
print("=" * 90)
print("NOISE ABLATION COMPLETE")
print("=" * 90)