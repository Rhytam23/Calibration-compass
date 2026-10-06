import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "."))

import math
import csv
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


print("=" * 90)
print("CALIBRATIONCOMPASS - LIVE HARDWARE VALIDATION")
print("=" * 90)

SHOTS = 1024

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

# ------------------------------------------------------------
# Test circuit: 3-qubit GHZ
# ------------------------------------------------------------

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()


def get_calibration_score(compiled, props):
    used_qubits = set()
    readout_risk = 0.0
    gate_risk = 0.0
    two_qubit_count = 0

    for item in compiled.data:

        operation = item.operation
        qargs = item.qubits
        name = operation.name.lower()

        physical_indices = [
            compiled.find_bit(q).index
            for q in qargs
        ]

        for q in physical_indices:
            used_qubits.add(q)

        if name == "measure":
            continue

        if name in ["barrier", "delay", "reset"]:
            continue

        if len(physical_indices) == 2:
            two_qubit_count += 1

        try:
            error = props.gate_error(
                name,
                physical_indices
            )

            if error is not None:
                gate_risk += error

        except Exception:
            pass

    for q in sorted(used_qubits):
        try:
            error = props.readout_error(q)

            if error is not None:
                readout_risk += error

        except Exception:
            pass

    return {
        "mapping": [
            int(p) for p in compiled.layout.final_index_layout()
        ] if compiled.layout is not None else sorted(used_qubits),
        "readout_risk": readout_risk,
        "gate_risk": gate_risk,
        "total_risk": readout_risk + gate_risk,
        "two_qubit_count": two_qubit_count,
        "depth": compiled.depth()
    }


def ghz_fidelity(counts):
    total = sum(counts.values())

    if total == 0:
        return 0.0

    p000 = counts.get("000", 0) / total
    p111 = counts.get("111", 0) / total

    return (
        math.sqrt(0.5 * p000)
        + math.sqrt(0.5 * p111)
    ) ** 2


all_results = []

for backend_name in BACKENDS:

    print()
    print("-" * 90)
    print("CHECKING:", backend_name)
    print("-" * 90)

    try:
        backend = service.backend(backend_name)
        status = backend.status()

        print("Status:", status.status_msg)
        print("Pending jobs:", status.pending_jobs)

        # Skip paused / maintenance / offline backends
        if not status.operational:
            print("SKIPPED: backend is not operational.")
            continue

        props = backend.properties(refresh=True)

    except Exception as exc:
        print("SKIPPED: could not access backend.")
        print("Reason:", exc)
        continue

    circuits = []
    metadata = []

    for seed in SEEDS:

        compiled = transpile(
            qc,
            backend=backend,
            optimization_level=3,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=seed
        )

        score = get_calibration_score(
            compiled,
            props
        )

        circuits.append(compiled)

        metadata.append({
            "backend": backend_name,
            "candidate": seed,
            **score
        })

    print("Submitting", len(circuits), "candidates...")

    try:
        sampler = SamplerV2(mode=backend)

        job = sampler.run(
            circuits,
            shots=SHOTS
        )

        print("Job ID:", job.job_id())
        print("Waiting for results...")

        result = job.result()

    except Exception as exc:
        print("Backend execution failed.")
        print("Reason:", exc)
        continue

    for i, meta in enumerate(metadata):

        try:
            counts = result[i].data.meas.get_counts()
            fidelity = ghz_fidelity(counts)

            record = {
                **meta,
                "fidelity": fidelity,
                "counts_000": counts.get("000", 0),
                "counts_111": counts.get("111", 0),
                "job_id": job.job_id()
            }

            all_results.append(record)

            print(
                f"Candidate {meta['candidate']:>2} | "
                f"mapping {str(meta['mapping']):<18} | "
                f"risk {meta['total_risk']:.6f} | "
                f"fidelity {fidelity:.6f}"
            )

        except Exception as exc:
            print(
                f"Candidate {meta['candidate']} failed:",
                exc
            )


# ------------------------------------------------------------
# Make sure we received results
# ------------------------------------------------------------

if not all_results:
    print()
    print("No hardware results were obtained.")
    print("DONE")
    raise SystemExit


# ------------------------------------------------------------
# ACTUAL HARDWARE RANKING
# ------------------------------------------------------------

print()
print("=" * 90)
print("ACTUAL HARDWARE RANKING")
print("=" * 90)

hardware_sorted = sorted(
    all_results,
    key=lambda x: x["fidelity"],
    reverse=True
)

print(
    f"{'backend':<16}"
    f"{'candidate':>10}"
    f"{'mapping':>22}"
    f"{'risk':>12}"
    f"{'fidelity':>12}"
)

for r in hardware_sorted:

    print(
        f"{r['backend']:<16}"
        f"{r['candidate']:>10}"
        f"{str(r['mapping']):>22}"
        f"{r['total_risk']:>12.6f}"
        f"{r['fidelity']:>12.6f}"
    )


# ------------------------------------------------------------
# BEST ACTUAL CANDIDATE PER BACKEND
# ------------------------------------------------------------

print()
print("=" * 90)
print("BEST ACTUAL CANDIDATE PER BACKEND")
print("=" * 90)

available_backends = sorted(
    set(r["backend"] for r in all_results)
)

for backend_name in available_backends:

    candidates = [
        r for r in all_results
        if r["backend"] == backend_name
    ]

    best = max(
        candidates,
        key=lambda x: x["fidelity"]
    )

    print()
    print("Backend:", best["backend"])
    print("Candidate:", best["candidate"])
    print("Mapping:", best["mapping"])
    print("Calibration risk:", f"{best['total_risk']:.6f}")
    print("Actual GHZ fidelity:", f"{best['fidelity']:.6f}")


# ------------------------------------------------------------
# CALIBRATIONCOMPASS PREDICTION VS ACTUAL
# ------------------------------------------------------------

predicted_best = min(
    all_results,
    key=lambda x: x["total_risk"]
)

actual_best = max(
    all_results,
    key=lambda x: x["fidelity"]
)

print()
print("=" * 90)
print("CALIBRATIONCOMPASS VS ACTUAL HARDWARE")
print("=" * 90)

print()
print("CalibrationCompass prediction:")
print("Backend:", predicted_best["backend"])
print("Candidate:", predicted_best["candidate"])
print("Mapping:", predicted_best["mapping"])
print("Risk:", f"{predicted_best['total_risk']:.6f}")

print()
print("Actual best:")
print("Backend:", actual_best["backend"])
print("Candidate:", actual_best["candidate"])
print("Mapping:", actual_best["mapping"])
print("Fidelity:", f"{actual_best['fidelity']:.6f}")

match = (
    predicted_best["backend"] == actual_best["backend"]
    and predicted_best["candidate"] == actual_best["candidate"]
)

print()
print("Prediction matches actual best:", match)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

Path("results").mkdir(exist_ok=True)

output_file = Path(
    "results/live_hardware_validation.csv"
)

fieldnames = [
    "backend",
    "candidate",
    "mapping",
    "readout_risk",
    "gate_risk",
    "total_risk",
    "two_qubit_count",
    "depth",
    "fidelity",
    "counts_000",
    "counts_111",
    "job_id"
]

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in all_results:
        writer.writerow(row)


print()
print("Results saved to:")
print(output_file)

print()
print("DONE")