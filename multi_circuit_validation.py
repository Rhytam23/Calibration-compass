import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "."))

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import (
    FakeFez,
    FakeSherbrooke,
    FakeTorino,
)

BACKENDS = {
    "Fez": FakeFez(),
    "Sherbrooke": FakeSherbrooke(),
    "Torino": FakeTorino(),
}

SEEDS = [11, 22, 33, 44, 55, 66]


def hellinger(p, q):
    keys = set(p) | set(q)
    overlap = sum(
        np.sqrt(p.get(k, 0) * q.get(k, 0))
        for k in keys
    )
    return overlap ** 2


def probs(counts, shots):
    return {
        k: v / shots
        for k, v in counts.items()
    }


circuits = []

# GHZ
qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure([0, 1, 2], [0, 1, 2])
circuits.append(("GHZ", qc))

# Bell pair
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
circuits.append(("Bell", qc))

# Ring
qc = QuantumCircuit(4, 4)
qc.h(range(4))
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.cx(3, 0)
qc.measure(range(4), range(4))
circuits.append(("Ring", qc))

# Hardware efficient
qc = QuantumCircuit(4, 4)
for q in range(4):
    qc.h(q)
    qc.rx(0.4, q)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.measure(range(4), range(4))
circuits.append(("HardwareEfficient", qc))

# QFT-like
qc = QuantumCircuit(4, 4)
for q in range(4):
    qc.h(q)
qc.cp(np.pi / 2, 0, 1)
qc.cp(np.pi / 4, 0, 2)
qc.cp(np.pi / 8, 0, 3)
qc.cp(np.pi / 2, 1, 2)
qc.cp(np.pi / 4, 1, 3)
qc.cp(np.pi / 2, 2, 3)
qc.measure(range(4), range(4))
circuits.append(("QFT_like", qc))


rows = []

print("=" * 80)
print("CALIBRATIONCOMPASS - MULTI-CIRCUIT VALIDATION")
print("=" * 80)

for circuit_name, circuit in circuits:

    print(f"\nTesting {circuit_name}...")

    ideal = AerSimulator().run(
        circuit,
        shots=3000,
        seed_simulator=123
    ).result()

    ideal_probs = probs(
        ideal.get_counts(),
        3000
    )

    for backend_name, backend in BACKENDS.items():

        for seed in SEEDS:

            compiled = transpile(
                circuit,
                backend=backend,
                optimization_level=3,
                layout_method="sabre",
                routing_method="sabre",
                seed_transpiler=seed,
            )

            noisy = AerSimulator.from_backend(
                backend
            )

            result = noisy.run(
                compiled,
                shots=3000,
                seed_simulator=seed
            ).result()

            fidelity = hellinger(
                ideal_probs,
                probs(
                    result.get_counts(),
                    3000
                )
            )

            physical = (
                compiled.layout
                .final_index_layout()
            )

            readout = [
                backend.properties().readout_error(q)
                for q in physical
                if q is not None and q < backend.num_qubits
            ]

            rows.append({
                "circuit": circuit_name,
                "backend": backend_name,
                "candidate": seed,
                "fidelity": fidelity,
                "avg_readout": np.mean(readout),
                "max_readout": np.max(readout),
                "depth": compiled.depth(),
            })


df = pd.DataFrame(rows)

print("\nRESULTS")
print("=" * 80)

summary = df.groupby("circuit").agg(
    best_fidelity=("fidelity", "max"),
    worst_fidelity=("fidelity", "min"),
    avg_fidelity=("fidelity", "mean"),
)

print(summary)

print("\nCORRELATION")
print("=" * 80)

within = [
    g["fidelity"].corr(g["avg_readout"])
    for _, g in df.groupby(["circuit", "backend"])
]

print(
    "Mean within-(circuit, backend) fidelity vs avg readout: "
    f"{pd.Series(within).mean():.4f}  "
    "(pooled value below is confounded by circuit/backend)"
)

print(
    f"Overall fidelity vs avg readout: "
    f"{df['fidelity'].corr(df['avg_readout']):.4f}"
)

print(
    f"Overall fidelity vs max readout: "
    f"{df['fidelity'].corr(df['max_readout']):.4f}"
)

out = (
    "results/multi_circuit_validation.csv"
)

df.to_csv(out, index=False)

print(f"\nSaved: {out}")
print("\nDONE")