import pandas as pd

from qiskit import qasm2, transpile
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


print("=" * 80)
print("CALIBRATIONCOMPASS - LIVE MAPPING ANALYSIS")
print("=" * 80)

backend_name = input(
    "\nBackend (Fez / Sherbrooke / Torino): "
).strip()

if backend_name not in BACKENDS:
    raise SystemExit(
        f"Unknown backend '{backend_name}'. "
        f"Choose one of: {', '.join(BACKENDS)}"
    )

backend = BACKENDS[backend_name]

circuit = qasm2.loads(QASM)

properties = backend.properties()

rows = []

for seed in SEEDS:

    compiled = transpile(
        circuit,
        backend=backend,
        optimization_level=3,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed,
    )

    layout = compiled.layout

    mapping = []
    physical_qubits = []

    if layout is not None:

        try:
            final_layout = layout.final_index_layout()
        except Exception:
            final_layout = []

        for logical in range(circuit.num_qubits):

            if logical < len(final_layout):
                physical = final_layout[logical]
                physical_qubits.append(physical)
            else:
                physical = "?"

            mapping.append(
                f"q{logical}->{physical}"
            )

    readout_errors = []

    for physical in physical_qubits:

        try:
            error = properties.readout_error(
                physical
            )

            readout_errors.append(error)

        except Exception:
            pass

    rows.append({
        "candidate": seed,
        "depth": compiled.depth(),
        "2Q_gates": sum(
            1
            for inst in compiled.data
            if inst.operation.name in [
                "cx",
                "cz",
                "ecr"
            ]
        ),
        "avg_readout_error": (
            sum(readout_errors)
            / len(readout_errors)
            if readout_errors
            else 0
        ),
        "mapping": ", ".join(mapping),
    })


df = pd.DataFrame(rows)

print("\n")

print(
    df.to_string(index=False)
)

print("\nDONE")