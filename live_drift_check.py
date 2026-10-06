
from qiskit_ibm_runtime import QiskitRuntimeService


print("=" * 90)
print("CALIBRATIONCOMPASS - LIVE CALIBRATION DRIFT CHECK")
print("=" * 90)

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="open-instance"
)

jobs = {
    "ibm_fez": "db2cdcfr11fs7396hcc0",
    "ibm_kingston": "db2cdfs2ljfc73d48s7g",
    "ibm_marrakesh": "db2cdl7r11fs7396hcmg"
}

for backend_name, job_id in jobs.items():

    print()
    print("-" * 90)
    print("Backend:", backend_name)
    print("Job ID:", job_id)
    print("-" * 90)

    try:
        job = service.job(job_id)

        print("Job status:", job.status())
        print("Job creation time:", job.creation_date)

    except Exception as exc:
        print("Could not retrieve job:")
        print(exc)
        continue

    try:
        backend = service.backend(backend_name)
        props = backend.properties(refresh=True)

        print("Current calibration time:", props.last_update_date)

        if job.creation_date is not None:

            job_time = job.creation_date
            calibration_time = props.last_update_date

            if job_time.tzinfo is None:
                job_time = job_time.replace(
                    tzinfo=calibration_time.tzinfo
                )

            difference = (
                calibration_time - job_time
            ).total_seconds() / 60

            print(
                "Calibration time - job time:",
                f"{difference:.2f} minutes"
            )

            if difference > 0:
                print(
                    "Calibration snapshot is newer than the job."
                )
            elif difference < 0:
                print(
                    "Job was created after the current "
                    "calibration snapshot."
                )
            else:
                print(
                    "Calibration and job timestamps are nearly identical."
                )

    except Exception as exc:
        print("Could not read calibration information:")
        print(exc)


print()
print("=" * 90)
print("CONCLUSION")
print("=" * 90)

print(
    "This check compares the hardware job timestamp with the "
    "latest available calibration timestamp."
)

print()
print("DONE")