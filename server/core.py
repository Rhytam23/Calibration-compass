"""Streamlit-free calibration-risk analysis used by the API server.

Mirrors the logic in dashboard.py: transpile a circuit with several
seeds per backend, score each candidate from live IBM calibration data,
and rank by combined risk.
"""

import math
from datetime import datetime, timezone

from qiskit import transpile

DEFAULT_BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]

SEEDS = [11, 22, 33, 44, 55, 66]

NON_GATE_OPS = {"barrier", "delay", "reset"}


def analyze_candidate(compiled, props):
    used_qubits = set()
    measured_qubits = set()
    gate_errors = []
    two_q_edges = []
    two_q_errors = []
    gate_count = 0
    two_q_count = 0

    for item in compiled.data:
        name = item.operation.name.lower()
        physical = [compiled.find_bit(q).index for q in item.qubits]

        if name == "measure":
            measured_qubits.update(physical)
            continue

        if name in NON_GATE_OPS:
            continue

        used_qubits.update(physical)
        gate_count += 1

        if len(physical) == 2:
            two_q_count += 1
            edge = tuple(sorted(physical))
            if edge not in two_q_edges:
                two_q_edges.append(edge)

        try:
            error = props.gate_error(name, physical)
        except Exception:
            error = None

        if error is not None:
            gate_errors.append(error)
            if len(physical) == 2:
                two_q_errors.append(error)

    targets = measured_qubits or used_qubits
    readout_errors = []

    for q in sorted(targets):
        try:
            error = props.readout_error(q)
        except Exception:
            error = None
        if error is not None:
            readout_errors.append(error)

    readout_success = 1.0
    for error in readout_errors:
        readout_success *= 1.0 - error

    gate_success = 1.0
    for error in gate_errors:
        gate_success *= 1.0 - error

    readout_loss = 1.0 - readout_success
    gate_loss = 1.0 - gate_success

    return {
        "physical_qubits": sorted(used_qubits),
        "measured_qubits": sorted(measured_qubits),
        "readout_loss": readout_loss,
        "max_readout": max(readout_errors) if readout_errors else 0.0,
        "gate_loss": gate_loss,
        "gate_error_sum": sum(gate_errors),
        "two_q_gate_error_sum": sum(two_q_errors),
        "combined_risk": 1.0 - (1.0 - readout_loss) * (1.0 - gate_loss),
        "gate_count": gate_count,
        "two_q_gates": two_q_count,
        "two_q_edges": [list(e) for e in two_q_edges],
        "depth": compiled.depth(),
    }


def confidence_label(margin):
    if margin is None or math.isnan(margin):
        return "Single option"
    if margin >= 0.010:
        return "Strong preference"
    if margin >= 0.003:
        return "Moderate preference"
    return "Close call"


def analyze_backends(service, circuit, backend_names):
    """Return (candidates, status_rows). Errors are reported, not hidden."""

    candidates = []
    status_rows = []

    for backend_name in backend_names:
        try:
            backend = service.backend(backend_name)
            status = backend.status()

            status_rows.append({
                "backend": backend_name,
                "status": status.status_msg,
                "operational": status.operational,
                "pending_jobs": status.pending_jobs,
            })

            if not status.operational:
                continue

            if circuit.num_qubits > backend.num_qubits:
                status_rows[-1]["status"] = (
                    f"Circuit needs {circuit.num_qubits} qubits; "
                    f"backend has {backend.num_qubits}"
                )
                continue

            props = backend.properties(refresh=True)
            calibration_time = props.last_update_date

        except Exception as exc:
            status_rows.append({
                "backend": backend_name,
                "status": f"Error: {exc}",
                "operational": False,
                "pending_jobs": None,
            })
            continue

        for seed in SEEDS:
            try:
                compiled = transpile(
                    circuit,
                    backend=backend,
                    optimization_level=3,
                    layout_method="sabre",
                    routing_method="sabre",
                    seed_transpiler=seed,
                )

                features = analyze_candidate(compiled, props)

                age = None
                if calibration_time is not None:
                    stamp = calibration_time
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    age = (
                        datetime.now(timezone.utc) - stamp
                    ).total_seconds() / 60.0

                candidates.append({
                    "backend": backend_name,
                    "candidate": seed,
                    "calibration_time": str(calibration_time),
                    "calibration_age_minutes": age,
                    **features,
                })

            except Exception as exc:
                status_rows.append({
                    "backend": backend_name,
                    "status": f"Candidate {seed} failed: {exc}",
                    "operational": True,
                    "pending_jobs": None,
                })

    candidates.sort(key=lambda c: c["combined_risk"])
    return candidates, status_rows
