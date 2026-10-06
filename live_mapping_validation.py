import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "."))

from itertools import combinations

import numpy as np
import pandas as pd
from qiskit import qasm2, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import (
    FakeFez,
    FakeSherbrooke,
    FakeTorino,
)

SEEDS = [11, 22, 33, 44, 55, 66]

BACKENDS = {
    "Fez": FakeFez(),
    "Sherbrooke": FakeSherbrooke(),
    "Torino": FakeTorino(),
}

QASM = """OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

h q[0];
cx q[0],q[1];
cx q[1],q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
"""


def probs(counts, shots):
    return {
        k: v / shots
        for k, v in counts.items()
    }


def hellinger(p, q):
    keys = set(p) | set(q)

    overlap = sum(
        np.sqrt(
            p.get(k, 0) *
            q.get(k, 0)
        )
        for k in keys
    )

    return overlap ** 2


print("=" * 80)
print("CALIBRATIONCOMPASS - LIVE MAPPING VALIDATION")
print("=" * 80)

circuit = qasm2.loads(QASM)

ideal = AerSimulator().run(
    circuit,
    shots=5000,
    seed_simulator=123
).result()

ideal_probs = probs(
    ideal.get_counts(),
    5000
)

rows = []

for backend_name, backend in BACKENDS.items():

    print(f"\nTesting {backend_name}...")

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
            shots=5000,
            seed_simulator=seed
        ).result()

        fidelity = hellinger(
            ideal_probs,
            probs(
                result.get_counts(),
                5000
            )
        )

        layout = compiled.layout

        physical = []

        for logical in range(
            circuit.num_qubits
        ):
            physical.append(
                layout.final_index_layout()[
                    logical
                ]
            )

        readout = [
            backend.properties().readout_error(q)
            for q in physical
        ]

        rows.append({
            "backend": backend_name,
            "candidate": seed,
            "fidelity": fidelity,
            "avg_readout": np.mean(readout),
            "max_readout": np.max(readout),
            "mapping": str(physical),
        })

df = pd.DataFrame(rows)

print("\nRESULTS")
print("=" * 80)

print(
    df.sort_values(
        ["backend", "fidelity"],
        ascending=[True, False]
    ).to_string(index=False)
)

print("\nCORRELATION")
print("=" * 80)

print(
    f"Fidelity vs average readout: "
    f"{df['fidelity'].corr(df['avg_readout']):.4f}"
)

print(
    f"Fidelity vs maximum readout: "
    f"{df['fidelity'].corr(df['max_readout']):.4f}"
)

out = (
    "results/live_mapping_validation.csv"
)

df.to_csv(
    out,
    index=False
)

print(f"\nSaved: {out}")
print("\nDONE")