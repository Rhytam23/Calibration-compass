import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "."))

import csv
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService


print("=" * 90)
print("CALIBRATIONCOMPASS - LIVE RESULT ANALYSIS")
print("=" * 90)

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)

BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh"
]

SEEDS = [11, 22, 33, 44, 55, 66]

RESULT_FILE = Path(
    "results/live_hardware_validation.csv"
)

if not RESULT_FILE.exists():
    print("Result file not found:")
    print(RESULT_FILE)
    raise SystemExit


# ------------------------------------------------------------
# Load previous hardware results
# ------------------------------------------------------------

rows = []

with open(
    RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        row["candidate"] = int(row["candidate"])
        row["fidelity"] = float(row["fidelity"])
        row["total_risk"] = float(row["total_risk"])
        rows.append(row)


# ------------------------------------------------------------
# Recreate the same GHZ circuit
# ------------------------------------------------------------

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()


# ------------------------------------------------------------
# Analyze exact physical interactions
# ------------------------------------------------------------

analysis = []

for backend_name in BACKENDS:

    print()
    print("-" * 90)
    print("ANALYZING:", backend_name)
    print("-" * 90)

    backend = service.backend(backend_name)
    # NOTE: this is the CURRENT calibration, not the calibration at the
    # time the validation jobs ran, so risk features can differ from
    # what the hardware actually experienced.
    props = backend.properties(refresh=True)

    backend_rows = [
        r for r in rows
        if r["backend"] == backend_name
    ]

    for result_row in backend_rows:

        seed = result_row["candidate"]

        compiled = transpile(
            qc,
            backend=backend,
            optimization_level=3,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=seed
        )

        two_qubit_details = []

        total_gate_error = 0.0
        total_gate_length = 0.0

        for item in compiled.data:

            operation = item.operation
            qargs = item.qubits
            name = operation.name.lower()

            if name in [
                "measure",
                "barrier",
                "delay",
                "reset"
            ]:
                continue

            physical_indices = [
                compiled.find_bit(q).index
                for q in qargs
            ]

            try:
                error = props.gate_error(
                    name,
                    physical_indices
                )

                if error is not None:
                    total_gate_error += error
            except Exception:
                error = None

            try:
                length = props.gate_length(
                    name,
                    physical_indices
                )

                if length is not None:
                    total_gate_length += length
            except Exception:
                length = None

            if len(physical_indices) == 2:

                two_qubit_details.append({
                    "gate": name,
                    "qubits": physical_indices,
                    "error": error,
                    "length": length
                })

        physical_qubits = [
            int(x)
            for x in result_row["mapping"]
            .replace("[", "")
            .replace("]", "")
            .split(",")
            if x.strip()
        ]

        readout_errors = []

        for q in physical_qubits:

            try:
                readout = props.readout_error(q)
            except Exception:
                readout = None

            if readout is not None:
                readout_errors.append(readout)

        avg_readout = (
            sum(readout_errors) / len(readout_errors)
            if readout_errors
            else 0.0
        )

        max_readout = (
            max(readout_errors)
            if readout_errors
            else 0.0
        )

        t1_values = []
        t2_values = []

        for q in physical_qubits:

            try:
                t1 = props.t1(q)
                if t1 is not None:
                    t1_values.append(t1)
            except Exception:
                pass

            try:
                t2 = props.t2(q)
                if t2 is not None:
                    t2_values.append(t2)
            except Exception:
                pass

        avg_t1 = (
            sum(t1_values) / len(t1_values)
            if t1_values
            else 0.0
        )

        avg_t2 = (
            sum(t2_values) / len(t2_values)
            if t2_values
            else 0.0
        )

        analysis.append({
            "backend": backend_name,
            "candidate": seed,
            "fidelity": result_row["fidelity"],
            "risk": result_row["total_risk"],
            "avg_readout": avg_readout,
            "max_readout": max_readout,
            "gate_error": total_gate_error,
            "gate_length": total_gate_length,
            "avg_t1": avg_t1,
            "avg_t2": avg_t2,
            "2q_details": two_qubit_details
        })


# ------------------------------------------------------------
# Print exact 2Q calibration details
# ------------------------------------------------------------

print()
print("=" * 90)
print("EXACT 2Q GATE CALIBRATION")
print("=" * 90)

for backend_name in BACKENDS:

    print()
    print("BACKEND:", backend_name)

    backend_rows = [
        r for r in analysis
        if r["backend"] == backend_name
    ]

    for r in sorted(
        backend_rows,
        key=lambda x: x["candidate"]
    ):

        print()
        print(
            "Candidate:",
            r["candidate"],
            "| Fidelity:",
            f"{r['fidelity']:.6f}"
        )

        if not r["2q_details"]:
            print("  No 2Q gates found.")
            continue

        for gate in r["2q_details"]:

            print(
                " ",
                gate["gate"],
                gate["qubits"],
                "| error:",
                f"{gate['error']:.8f}"
                if gate["error"] is not None
                else "N/A",
                "| length:",
                f"{gate['length'] * 1e9:.2f} ns"
                if gate["length"] is not None
                else "N/A"
            )


# ------------------------------------------------------------
# Candidate comparison for Marrakesh
# ------------------------------------------------------------

print()
print("=" * 90)
print("MARRAKESH: CANDIDATE 22 VS 55")
print("=" * 90)

for candidate in [22, 55]:

    matches = [
        r for r in analysis
        if r["backend"] == "ibm_marrakesh"
        and r["candidate"] == candidate
    ]

    if not matches:
        continue

    r = matches[0]

    print()
    print("Candidate:", candidate)
    print("Fidelity:", f"{r['fidelity']:.6f}")
    print("Risk:", f"{r['risk']:.6f}")
    print("Average readout:", f"{r['avg_readout']:.6f}")
    print("Maximum readout:", f"{r['max_readout']:.6f}")
    print("Total gate error:", f"{r['gate_error']:.8f}")
    print(
        "Total gate length:",
        f"{r['gate_length'] * 1e9:.2f} ns"
    )
    print("Average T1:", f"{r['avg_t1']:.8e}")
    print("Average T2:", f"{r['avg_t2']:.8e}")


# ------------------------------------------------------------
# Overall correlation analysis
# ------------------------------------------------------------

def correlation(values_x, values_y):

    n = len(values_x)

    if n < 2:
        return 0.0

    mean_x = sum(values_x) / n
    mean_y = sum(values_y) / n

    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(values_x, values_y)
    )

    denominator_x = sum(
        (x - mean_x) ** 2
        for x in values_x
    )

    denominator_y = sum(
        (y - mean_y) ** 2
        for y in values_y
    )

    denominator = (
        denominator_x * denominator_y
    ) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


fidelity_values = [
    r["fidelity"]
    for r in analysis
]

risk_values = [
    r["risk"]
    for r in analysis
]

readout_values = [
    r["avg_readout"]
    for r in analysis
]

gate_values = [
    r["gate_error"]
    for r in analysis
]

length_values = [
    r["gate_length"]
    for r in analysis
]

t1_values = [
    r["avg_t1"]
    for r in analysis
]

t2_values = [
    r["avg_t2"]
    for r in analysis
]


print()
print("=" * 90)
print("FEATURE VS HARDWARE FIDELITY")
print("=" * 90)

print(
    "Total risk correlation:",
    f"{correlation(risk_values, fidelity_values):.4f}"
)

print(
    "Readout correlation:",
    f"{correlation(readout_values, fidelity_values):.4f}"
)

print(
    "Gate error correlation:",
    f"{correlation(gate_values, fidelity_values):.4f}"
)

print(
    "Gate length correlation:",
    f"{correlation(length_values, fidelity_values):.4f}"
)

print(
    "T1 correlation:",
    f"{correlation(t1_values, fidelity_values):.4f}"
)

print(
    "T2 correlation:",
    f"{correlation(t2_values, fidelity_values):.4f}"
)


# ------------------------------------------------------------
# Save analysis
# ------------------------------------------------------------

output_file = Path(
    "results/live_calibration_analysis.csv"
)

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "backend",
            "candidate",
            "fidelity",
            "risk",
            "avg_readout",
            "max_readout",
            "gate_error",
            "gate_length",
            "avg_t1",
            "avg_t2"
        ]
    )

    writer.writeheader()

    for r in analysis:

        writer.writerow({
            "backend": r["backend"],
            "candidate": r["candidate"],
            "fidelity": r["fidelity"],
            "risk": r["risk"],
            "avg_readout": r["avg_readout"],
            "max_readout": r["max_readout"],
            "gate_error": r["gate_error"],
            "gate_length": r["gate_length"],
            "avg_t1": r["avg_t1"],
            "avg_t2": r["avg_t2"]
        })


print()
print("Analysis saved to:")
print(output_file)

print()
print("DONE")