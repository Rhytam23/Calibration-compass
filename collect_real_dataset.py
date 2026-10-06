import ast
import csv
import math
from datetime import timezone
from pathlib import Path

import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


print("=" * 90)
print("CALIBRATIONCOMPASS - REAL HARDWARE DATASET")
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


def make_circuits():

    circuits = []

    # Bell state
    bell = QuantumCircuit(3)
    bell.h(0)
    bell.cx(0, 1)

    circuits.append(
        ("Bell", bell)
    )

    # 3-qubit ring
    ring = QuantumCircuit(3)
    ring.h(0)
    ring.cx(0, 1)
    ring.cx(1, 2)
    ring.cx(2, 0)

    circuits.append(
        ("Ring", ring)
    )

    return circuits


def hellinger_fidelity(ideal_probs, counts):

    total = sum(counts.values())

    if total <= 0:
        return 0.0

    measured = {
        str(key).replace(" ", ""): value / total
        for key, value in counts.items()
    }

    keys = set(ideal_probs) | set(measured)

    coefficient = 0.0

    for key in keys:

        coefficient += math.sqrt(
            ideal_probs.get(key, 0.0)
            * measured.get(key, 0.0)
        )

    return coefficient ** 2


def build_features(compiled, props):

    used_qubits = set()
    measured_qubits = set()

    total_gate_error = 0.0
    total_gate_length = 0.0
    max_gate_error = 0.0

    two_q_gate_error = 0.0
    two_q_gate_length = 0.0
    max_two_q_error = 0.0

    two_q_count = 0
    one_q_count = 0
    gate_count = 0

    for item in compiled.data:

        operation = item.operation
        qargs = item.qubits
        name = operation.name.lower()

        physical = [
            compiled.find_bit(q).index
            for q in qargs
        ]

        for q in physical:
            used_qubits.add(q)

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

        gate_count += 1

        if len(physical) == 2:
            two_q_count += 1
        else:
            one_q_count += 1

        try:

            error = props.gate_error(
                name,
                physical
            )

            if error is not None:

                total_gate_error += error

                max_gate_error = max(
                    max_gate_error,
                    error
                )

                if len(physical) == 2:

                    two_q_gate_error += error

                    max_two_q_error = max(
                        max_two_q_error,
                        error
                    )

        except Exception:
            pass

        try:

            length = props.gate_length(
                name,
                physical
            )

            if length is not None:

                total_gate_length += length

                if len(physical) == 2:
                    two_q_gate_length += length

        except Exception:
            pass

    readout_errors = []
    t1_values = []
    t2_values = []

    for q in sorted(measured_qubits):

        try:

            value = props.readout_error(q)

            if value is not None:
                readout_errors.append(value)

        except Exception:
            pass

    for q in sorted(used_qubits):

        try:

            value = props.t1(q)

            if value is not None:
                t1_values.append(value)

        except Exception:
            pass

        try:

            value = props.t2(q)

            if value is not None:
                t2_values.append(value)

        except Exception:
            pass

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

    readout_sum = sum(readout_errors)

    readout_product_loss = 0.0

    if readout_errors:

        success_product = 1.0

        for value in readout_errors:
            success_product *= (1.0 - value)

        readout_product_loss = 1.0 - success_product

    avg_t1 = (
        sum(t1_values) / len(t1_values)
        if t1_values
        else math.nan
    )

    avg_t2 = (
        sum(t2_values) / len(t2_values)
        if t2_values
        else math.nan
    )

    min_t1 = (
        min(t1_values)
        if t1_values
        else math.nan
    )

    min_t2 = (
        min(t2_values)
        if t2_values
        else math.nan
    )

    return {
        "mapping": sorted(measured_qubits),
        "active_physical_qubits": str(
            sorted(used_qubits)
        ),
        "depth": compiled.depth(),
        "gate_count": gate_count,
        "1Q_gates": one_q_count,
        "2Q_gates": two_q_count,
        "readout_sum": readout_sum,
        "avg_readout": avg_readout,
        "max_readout": max_readout,
        "readout_product_loss": readout_product_loss,
        "gate_error_sum": total_gate_error,
        "max_gate_error": max_gate_error,
        "2Q_gate_error_sum": two_q_gate_error,
        "max_2Q_gate_error": max_two_q_error,
        "gate_length_sum": total_gate_length,
        "2Q_gate_length_sum": two_q_gate_length,
        "avg_t1": avg_t1,
        "avg_t2": avg_t2,
        "min_t1": min_t1,
        "min_t2": min_t2
    }


new_rows = []


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

        props = backend.properties(
            refresh=True
        )

        calibration_time = (
            props.last_update_date
        )

        print(
            "Calibration time:",
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

    circuits = make_circuits()

    compiled_circuits = []
    metadata = []

    for circuit_name, base_circuit in circuits:

        for seed in SEEDS:

            # Measure the logical qubits BEFORE transpiling so the
            # result register has one bit per logical qubit (3 bits),
            # matching the ideal distribution. Measuring after
            # transpile would add a 156-bit register and give
            # fidelity 0.
            logical = base_circuit.copy()
            logical.measure_all()

            compiled = transpile(
                logical,
                backend=backend,
                optimization_level=3,
                layout_method="sabre",
                routing_method="sabre",
                seed_transpiler=seed
            )

            measured = compiled

            ideal_state = Statevector.from_instruction(
                base_circuit
            )

            ideal_probs = {
                str(key).replace(" ", ""):
                float(value)
                for key, value
                in ideal_state.probabilities_dict().items()
                if float(value) > 1e-12
            }

            features = build_features(
                compiled,
                props
            )

            compiled_circuits.append(
                measured
            )

            metadata.append({
                "circuit_type": circuit_name,
                "candidate": seed,
                "backend": backend_name,
                "original_qubits": 3,
                "ideal_probs": ideal_probs,
                **features,
                "calibration_time":
                    calibration_time.isoformat(),
                "pending_jobs_at_submission":
                    status.pending_jobs
            })

    print(
        "Submitting",
        len(compiled_circuits),
        "candidates..."
    )

    try:

        sampler = SamplerV2(
            mode=backend
        )

        job = sampler.run(
            compiled_circuits,
            shots=SHOTS
        )

        print(
            "Job ID:",
            job.job_id()
        )

        print(
            "Waiting for results..."
        )

        result = job.result()

    except Exception as exc:

        print(
            "Execution failed:"
        )

        print(exc)

        continue

    job_time = job.creation_date

    for i, meta in enumerate(metadata):

        counts = (
            result[i]
            .data.meas
            .get_counts()
        )

        fidelity = hellinger_fidelity(
            meta["ideal_probs"],
            counts
        )

        calibration_age_minutes = math.nan

        if (
            job_time is not None
            and calibration_time is not None
        ):

            try:

                job_time_local = job_time

                calibration_time_local = (
                    calibration_time
                )

                if (
                    job_time_local.tzinfo
                    is None
                ):

                    job_time_local = (
                        job_time_local.replace(
                            tzinfo=timezone.utc
                        )
                    )

                if (
                    calibration_time_local.tzinfo
                    is None
                ):

                    calibration_time_local = (
                        calibration_time_local.replace(
                            tzinfo=timezone.utc
                        )
                    )

                calibration_age_minutes = (
                    job_time_local
                    - calibration_time_local
                ).total_seconds() / 60.0

            except Exception:
                pass

        row = {
            key: value
            for key, value in meta.items()
            if key != "ideal_probs"
        }

        row["calibration_age_minutes"] = (
            calibration_age_minutes
        )

        row["fidelity"] = fidelity

        row["counts_000"] = counts.get(
            "000",
            0
        )

        row["counts_001"] = counts.get(
            "001",
            0
        )

        row["counts_010"] = counts.get(
            "010",
            0
        )

        row["counts_011"] = counts.get(
            "011",
            0
        )

        row["counts_100"] = counts.get(
            "100",
            0
        )

        row["counts_101"] = counts.get(
            "101",
            0
        )

        row["counts_110"] = counts.get(
            "110",
            0
        )

        row["counts_111"] = counts.get(
            "111",
            0
        )

        row["job_id"] = job.job_id()

        new_rows.append(row)

        print(
            f"{meta['circuit_type']:<8} "
            f"candidate {meta['candidate']:>2} | "
            f"mapping "
            f"{str(meta['mapping']):<18} | "
            f"fidelity {fidelity:.6f}"
        )


# ------------------------------------------------------------
# Reuse existing GHZ results
# ------------------------------------------------------------

output_rows = []

validation_file = Path(
    "results/live_hardware_validation.csv"
)

analysis_file = Path(
    "results/live_calibration_analysis.csv"
)

if (
    validation_file.exists()
    and analysis_file.exists()
):

    ghz_validation = pd.read_csv(
        validation_file
    )

    ghz_analysis = pd.read_csv(
        analysis_file
    )

    ghz = ghz_validation.merge(
        ghz_analysis[
            [
                "backend",
                "candidate",
                "avg_readout",
                "max_readout",
                "gate_error",
                "gate_length",
                "avg_t1",
                "avg_t2"
            ]
        ],
        on=[
            "backend",
            "candidate"
        ],
        how="left"
    )

    for _, r in ghz.iterrows():

        mapping_text = r["mapping"]

        try:

            mapping = ast.literal_eval(
                str(mapping_text)
            )

        except Exception:

            mapping = []

        output_rows.append({

            "circuit_type": "GHZ",

            "candidate":
                int(r["candidate"]),

            "backend":
                r["backend"],

            "original_qubits":
                3,

            "active_physical_qubits":
                str(mapping),

            "mapping":
                mapping,

            "depth":
                r.get(
                    "depth",
                    math.nan
                ),

            "gate_count":
                math.nan,

            "1Q_gates":
                1,

            "2Q_gates":
                r.get(
                    "2Q_gates",
                    2
                ),

            "readout_sum":
                r["avg_readout"] * 3,

            "avg_readout":
                r["avg_readout"],

            "max_readout":
                r["max_readout"],

            "readout_product_loss":
                math.nan,

            "gate_error_sum":
                r["gate_error"],

            "max_gate_error":
                math.nan,

            "2Q_gate_error_sum":
                math.nan,

            "max_2Q_gate_error":
                math.nan,

            "gate_length_sum":
                r["gate_length"],

            "2Q_gate_length_sum":
                math.nan,

            "avg_t1":
                r["avg_t1"],

            "avg_t2":
                r["avg_t2"],

            "min_t1":
                math.nan,

            "min_t2":
                math.nan,

            "calibration_time":
                math.nan,

            "pending_jobs_at_submission":
                math.nan,

            "calibration_age_minutes":
                math.nan,

            "fidelity":
                r["fidelity"],

            "counts_000":
                r["counts_000"],

            "counts_001":
                math.nan,

            "counts_010":
                math.nan,

            "counts_011":
                math.nan,

            "counts_100":
                math.nan,

            "counts_101":
                math.nan,

            "counts_110":
                math.nan,

            "counts_111":
                r["counts_111"],

            "job_id":
                r["job_id"]
        })


# Add new Bell + Ring results

output_rows.extend(
    new_rows
)


# ------------------------------------------------------------
# Save complete dataset
# ------------------------------------------------------------

output_path = Path(
    "results/real_hardware_dataset.csv"
)

output_path.parent.mkdir(
    exist_ok=True
)

fieldnames = sorted({
    key
    for row in output_rows
    for key in row
})

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        output_rows
    )


print()
print("=" * 90)
print("REAL HARDWARE DATASET SUMMARY")
print("=" * 90)

print(
    "Total rows:",
    len(output_rows)
)

print(
    "New Bell/Ring rows:",
    len(new_rows)
)

print(
    "Saved:",
    output_path
)

print()
print("DONE")