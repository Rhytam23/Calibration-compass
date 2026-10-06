import os as _os

# Resolve "results/..." relative to the repo root regardless of CWD.
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "."))

import os
from qiskit_ibm_runtime import QiskitRuntimeService
import pandas as pd
import numpy as np

print("=" * 80)
print("CALIBRATIONCOMPASS - LIVE IBM CALIBRATION")
print("=" * 80)

service = QiskitRuntimeService()

backends = service.backends(
    simulator=False,
    operational=True
)

rows = []

for backend in backends:

    print(f"\nReading {backend.name}...")

    props = backend.properties(refresh=True)

    if props is None:
        print("No calibration properties available.")
        continue

    readout = []
    t1 = []
    t2 = []

    for q in range(backend.num_qubits):

        try:
            r = props.readout_error(q)
            if r is not None:
                readout.append(r)
        except Exception:
            pass

        try:
            value = props.t1(q)
            if value is not None:
                t1.append(value)
        except Exception:
            pass

        try:
            value = props.t2(q)
            if value is not None:
                t2.append(value)
        except Exception:
            pass

    rows.append({
        "backend": backend.name,
        "qubits": backend.num_qubits,
        "avg_readout_error": np.mean(readout) if readout else np.nan,
        "max_readout_error": np.max(readout) if readout else np.nan,
        "avg_t1": np.mean(t1) if t1 else np.nan,
        "avg_t2": np.mean(t2) if t2 else np.nan,
        "calibration_time": props.last_update_date,
    })


df = pd.DataFrame(rows)

print("\nLIVE CALIBRATION SUMMARY")
print("=" * 80)

print(
    df.to_string(index=False)
)

os.makedirs("results", exist_ok=True)

out = (
    "results/live_ibm_calibration.csv"
)

df.to_csv(
    out,
    index=False
)

print(f"\nSaved: {out}")
print("\nDONE")