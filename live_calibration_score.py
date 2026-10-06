from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService

print("=" * 80)
print("CALIBRATIONCOMPASS - LIVE CALIBRATION SCORE")
print("=" * 80)

service = QiskitRuntimeService()

backend_name = input(
    "Backend (ibm_fez / ibm_kingston / ibm_marrakesh): "
).strip().lower()

if backend_name not in [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh"
]:
    print("Invalid backend name.")
    raise SystemExit

backend = service.backend(backend_name)
props = backend.properties(refresh=True)

# 3-qubit GHZ circuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()

seeds = [11, 22, 33, 44, 55, 66]

results = []

for seed in seeds:

    compiled = transpile(
        qc,
        backend=backend,
        optimization_level=3,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed
    )

    # Get final physical mapping
    mapping = compiled.layout.final_index_layout()
    physical_qubits = [int(mapping[i]) for i in range(3)]

    readout_risk = 0.0
    gate_risk = 0.0
    two_qubit_count = 0

    # Readout calibration risk
    for q in physical_qubits:
        try:
            readout_risk += props.readout_error(q)
        except Exception:
            pass

    # Gate calibration risk
    for item in compiled.data:

        operation = item.operation
        qargs = item.qubits

        name = operation.name.lower()

        if name in ["measure", "barrier", "delay", "reset"]:
            continue

        physical_indices = [compiled.find_bit(q).index for q in qargs]

        if len(physical_indices) == 2:
            two_qubit_count += 1

        try:
            gate_risk += props.gate_error(
                name,
                physical_indices
            )
        except Exception:
            pass

    total_risk = readout_risk + gate_risk

    results.append({
        "candidate": seed,
        "mapping": physical_qubits,
        "readout_risk": readout_risk,
        "gate_risk": gate_risk,
        "total_risk": total_risk,
        "depth": compiled.depth(),
        "2Q_gates": two_qubit_count
    })

# Sort by lowest total risk
results.sort(key=lambda x: x["total_risk"])

print()
print("=" * 80)
print("CANDIDATE CALIBRATION RANKING")
print("=" * 80)

print(
    f"{'candidate':>10} "
    f"{'mapping':>20} "
    f"{'readout':>12} "
    f"{'gate':>12} "
    f"{'total':>12}"
)

for r in results:
    print(
        f"{r['candidate']:>10} "
        f"{str(r['mapping']):>20} "
        f"{r['readout_risk']:>12.6f} "
        f"{r['gate_risk']:>12.6f} "
        f"{r['total_risk']:>12.6f}"
    )

best = results[0]

print()
print("=" * 80)
print("RECOMMENDED CANDIDATE")
print("=" * 80)

print("Candidate:", best["candidate"])
print("Mapping:", best["mapping"])
print("Total risk:", f"{best['total_risk']:.6f}")
print("Readout risk:", f"{best['readout_risk']:.6f}")
print("Gate risk:", f"{best['gate_risk']:.6f}")

print()
print("Reason:")
print(
    "This candidate has the lowest combined calibration risk "
    "(readout error + gate error) among the tested mappings."
)

print()
print("DONE")