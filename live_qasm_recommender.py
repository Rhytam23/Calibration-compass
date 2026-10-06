import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "."))

from pathlib import Path

from qiskit import QuantumCircuit, transpile, qasm2
from qiskit_ibm_runtime import QiskitRuntimeService


print("=" * 90)
print("CALIBRATIONCOMPASS - LIVE QASM RECOMMENDER")
print("=" * 90)


# ============================================================
# CONFIGURATION
# ============================================================

BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh"
]

SEEDS = [11, 22, 33, 44, 55, 66]

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)


# ============================================================
# QASM INPUT
# ============================================================

print()
print("Paste your OpenQASM 2 circuit.")
print("Finish by entering a line containing only: END")
print()

qasm_lines = []

while True:

    line = input()

    if line.strip() == "END":
        break

    qasm_lines.append(line)


qasm_text = "\n".join(qasm_lines)


if not qasm_text.strip():

    print("No QASM was provided.")
    raise SystemExit


# ============================================================
# PARSE CIRCUIT
# ============================================================

try:

    circuit = qasm2.loads(
        qasm_text
    )

except Exception as exc:

    print()
    print("QASM parsing failed.")
    print("Reason:", exc)
    raise SystemExit


print()
print("=" * 90)
print("CIRCUIT")
print("=" * 90)

print(
    "Logical qubits:",
    circuit.num_qubits
)

print(
    "Depth:",
    circuit.depth()
)

print(
    "Operations:",
    circuit.count_ops()
)


# ============================================================
# BACKEND + CANDIDATE ANALYSIS
# ============================================================

all_results = []


for backend_name in BACKENDS:

    print()
    print("-" * 90)
    print("CHECKING:", backend_name)
    print("-" * 90)

    try:

        backend = service.backend(
            backend_name
        )

        status = backend.status()

        print(
            "Status:",
            status.status_msg
        )

        print(
            "Pending jobs:",
            status.pending_jobs
        )

        if not status.operational:

            print(
                "SKIPPED: backend is not operational."
            )

            continue

        if circuit.num_qubits > backend.num_qubits:

            print(
                "SKIPPED: circuit needs",
                circuit.num_qubits,
                "qubits but backend has",
                backend.num_qubits
            )

            continue

        props = backend.properties(
            refresh=True
        )

        calibration_time = (
            props.last_update_date
        )

        print(
            "Calibration:",
            calibration_time
        )

    except Exception as exc:

        print(
            "SKIPPED: could not access backend."
        )

        print(
            "Reason:",
            exc
        )

        continue


    # ========================================================
    # Generate candidates
    # ========================================================

    for seed in SEEDS:

        try:

            compiled = transpile(
                circuit,
                backend=backend,
                optimization_level=3,
                layout_method="sabre",
                routing_method="sabre",
                seed_transpiler=seed
            )

        except Exception as exc:

            print(
                "Candidate",
                seed,
                "failed to transpile:",
                exc
            )

            continue


        used_qubits = set()
        measured_qubits = set()

        total_gate_error = 0.0
        two_q_gate_error = 0.0

        gate_count = 0
        two_q_count = 0

        two_q_edges = []


        # ====================================================
        # Inspect compiled circuit
        # ====================================================

        for item in compiled.data:

            operation = item.operation
            qargs = item.qubits

            name = operation.name.lower()

            physical = [
                compiled.find_bit(q).index
                for q in qargs
            ]


            if name == "measure":

                for q in physical:
                    measured_qubits.add(q)

                continue


            if name in [
                "barrier",
                "delay",
                "reset"
            ]:

                continue


            for q in physical:
                used_qubits.add(q)


            gate_count += 1


            if len(physical) == 2:

                two_q_count += 1

                edge = tuple(
                    sorted(physical)
                )

                if edge not in two_q_edges:
                    two_q_edges.append(edge)


            try:

                error = props.gate_error(
                    name,
                    physical
                )

            except Exception:

                error = None


            if error is not None:

                total_gate_error += error

                if len(physical) == 2:
                    two_q_gate_error += error


        # ====================================================
        # Readout calibration
        # ====================================================

        readout_errors = []


        # Use measured qubits when measurements exist.
        # Otherwise use active circuit qubits.

        readout_targets = (
            measured_qubits
            if measured_qubits
            else used_qubits
        )


        for q in sorted(readout_targets):

            try:

                error = props.readout_error(q)

            except Exception:

                error = None


            if error is not None:
                readout_errors.append(error)


        if readout_errors:

            readout_sum = sum(
                readout_errors
            )

            max_readout = max(
                readout_errors
            )

            readout_success = 1.0

            for error in readout_errors:

                readout_success *= (
                    1.0 - error
                )

            readout_loss = (
                1.0
                -
                readout_success
            )

        else:

            readout_sum = 0.0
            max_readout = 0.0
            readout_loss = 0.0


        # ====================================================
        # Combined risk
        # ====================================================

        gate_success = 1.0

        # Convert cumulative gate error into
        # a bounded failure probability.

        gate_operations = []

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

            physical = [
                compiled.find_bit(q).index
                for q in qargs
            ]

            try:

                error = props.gate_error(
                    name,
                    physical
                )

            except Exception:

                error = None


            if error is not None:
                gate_operations.append(
                    error
                )


        for error in gate_operations:

            gate_success *= (
                1.0 - error
            )


        gate_loss = (
            1.0
            -
            gate_success
        )


        combined_risk = (
            1.0
            -
            (
                (1.0 - readout_loss)
                *
                (1.0 - gate_loss)
            )
        )


        all_results.append({

            "backend":
                backend_name,

            "candidate":
                seed,

            "physical_qubits":
                sorted(used_qubits),

            "measured_qubits":
                sorted(measured_qubits),

            "readout_sum":
                readout_sum,

            "max_readout":
                max_readout,

            "readout_loss":
                readout_loss,

            "gate_loss":
                gate_loss,

            "gate_error_sum":
                total_gate_error,

            "2Q_gate_error_sum":
                two_q_gate_error,

            "combined_risk":
                combined_risk,

            "gate_count":
                gate_count,

            "2Q_gates":
                two_q_count,

            "depth":
                compiled.depth(),

            "2Q_edges":
                two_q_edges,

            "calibration_time":
                calibration_time
        })


# ============================================================
# Check results
# ============================================================

if not all_results:

    print()
    print("No usable backend candidates were generated.")
    raise SystemExit


# ============================================================
# GLOBAL RANKING
# ============================================================

all_results.sort(
    key=lambda x: x["combined_risk"]
)


print()
print("=" * 90)
print("GLOBAL CALIBRATION RANKING")
print("=" * 90)

print(
    f"{'backend':<16}"
    f"{'candidate':>10}"
    f"{'physical qubits':>24}"
    f"{'readout':>12}"
    f"{'gate':>12}"
    f"{'risk':>12}"
)


for result in all_results:

    print(
        f"{result['backend']:<16}"
        f"{result['candidate']:>10}"
        f"{str(result['physical_qubits']):>24}"
        f"{result['readout_loss']:>12.6f}"
        f"{result['gate_loss']:>12.6f}"
        f"{result['combined_risk']:>12.6f}"
    )


# ============================================================
# BEST PER BACKEND
# ============================================================

print()
print("=" * 90)
print("BEST CANDIDATE PER BACKEND")
print("=" * 90)


available_backends = sorted(
    set(
        result["backend"]
        for result in all_results
    )
)


for backend_name in available_backends:

    candidates = [
        result
        for result in all_results
        if result["backend"] == backend_name
    ]


    best = min(
        candidates,
        key=lambda x: x["combined_risk"]
    )


    print()
    print(
        "Backend:",
        backend_name
    )

    print(
        "Candidate:",
        best["candidate"]
    )

    print(
        "Physical qubits:",
        best["physical_qubits"]
    )

    print(
        "2Q edges:",
        best["2Q_edges"]
    )

    print(
        "Readout loss:",
        f"{best['readout_loss']:.6f}"
    )

    print(
        "Gate loss:",
        f"{best['gate_loss']:.6f}"
    )

    print(
        "Combined calibration risk:",
        f"{best['combined_risk']:.6f}"
    )


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

best = all_results[0]


print()
print("=" * 90)
print("CALIBRATIONCOMPASS RECOMMENDATION")
print("=" * 90)

print()
print(
    "Backend:",
    best["backend"]
)

print(
    "Candidate:",
    best["candidate"]
)

print(
    "Physical qubits:",
    best["physical_qubits"]
)

print(
    "2Q edges:",
    best["2Q_edges"]
)

print(
    "Circuit depth:",
    best["depth"]
)

print(
    "2Q gates:",
    best["2Q_gates"]
)

print(
    "Readout loss:",
    f"{best['readout_loss']:.6f}"
)

print(
    "Gate loss:",
    f"{best['gate_loss']:.6f}"
)

print(
    "Combined calibration risk:",
    f"{best['combined_risk']:.6f}"
)

print()
print(
    "Note:"
)

print(
    "This is a live calibration-risk recommendation."
)

print(
    "It is not a guaranteed prediction of hardware fidelity."
)


# ============================================================
# SAVE CSV
# ============================================================

output_file = Path(
    "results/live_qasm_recommendations.csv"
)

output_file.parent.mkdir(
    exist_ok=True
)


import csv


fieldnames = [
    "backend",
    "candidate",
    "physical_qubits",
    "measured_qubits",
    "readout_sum",
    "max_readout",
    "readout_loss",
    "gate_loss",
    "gate_error_sum",
    "2Q_gate_error_sum",
    "combined_risk",
    "gate_count",
    "2Q_gates",
    "depth",
    "2Q_edges",
    "calibration_time"
]


with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for result in all_results:

        writer.writerow(
            {
                key: result[key]
                for key in fieldnames
            }
        )


print()
print(
    "Results saved to:",
    output_file
)

print()
print("DONE")