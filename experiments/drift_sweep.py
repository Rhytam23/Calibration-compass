from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider
from qiskit.transpiler import InstructionProperties


# ============================================================
# 1. Create the GHZ circuit
# ============================================================

qc = QuantumCircuit(3, 3)

qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

qc.measure([0, 1, 2], [0, 1, 2])


# ============================================================
# 2. Run one backend
# ============================================================

def run_backend(backend):

    transpiled = transpile(
        qc,
        backend=backend,
        initial_layout=[0, 1, 2],
        optimization_level=1
    )

    simulator = AerSimulator.from_backend(backend)

    job = simulator.run(
        transpiled,
        shots=2000
    )

    result = job.result()
    counts = result.get_counts()

    success = (
        counts.get("000", 0) +
        counts.get("111", 0)
    ) / 2000

    return success


# ============================================================
# 3. Test different drift levels
# ============================================================

drift_factors = [1, 2, 4, 8, 16]

print("=" * 70)
print("CALIBRATIONCOMPASS - DRIFT SWEEP")
print("=" * 70)

print("\nFez qubit 1 will gradually become worse.")
print("We will see when another backend becomes better.\n")


# ============================================================
# 4. Run each drift level
# ============================================================

for factor in drift_factors:

    # Create fresh backends every time.
    # This prevents one experiment from affecting another.
    sherbrooke = fake_provider.FakeSherbrooke()
    torino = fake_provider.FakeTorino()
    fez = fake_provider.FakeFez()

    # --------------------------------------------------------
    # Change Fez qubit 1 readout error
    # --------------------------------------------------------

    properties = fez.target["measure"][(1,)]

    old_error = properties.error

    if old_error is None:
        raise RuntimeError(
            "Fez qubit 1 has no measure error to scale"
        )

    new_error = min(
        old_error * factor,
        0.50
    )

    fez.target.update_instruction_properties(
        instruction="measure",
        qargs=(1,),
        properties=InstructionProperties(
            duration=properties.duration,
            error=new_error
        )
    )

    # --------------------------------------------------------
    # Run the three backends
    # --------------------------------------------------------

    sherbrooke_score = run_backend(sherbrooke)
    torino_score = run_backend(torino)
    fez_score = run_backend(fez)

    scores = {
        "Sherbrooke": sherbrooke_score,
        "Torino": torino_score,
        "Fez": fez_score
    }

    winner = max(scores, key=scores.get)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("-" * 70)

    print(f"Drift factor: {factor}x")
    print(f"Fez qubit 1 readout error: {new_error:.4f}")

    print(f"Sherbrooke: {sherbrooke_score:.4f}")
    print(f"Torino:     {torino_score:.4f}")
    print(f"Fez:        {fez_score:.4f}")

    print(f"WINNER: {winner}")


print("\n" + "=" * 70)
print("DRIFT SWEEP COMPLETE")
print("=" * 70)