# CalibrationCompass

Research code for choosing the best qubit mapping (transpile candidate) and
backend on IBM quantum hardware, comparing plain ESP (estimated success
probability) against learned pairwise rankers.

## Setup
    pip install -r requirements.txt
Live scripts need an IBM Quantum account saved via `QiskitRuntimeService.save_account`.

## Layout
- `calibrationcompass.py` – pairwise XGBoost demo on `results/candidate_features_v2.csv` (fake-backend data).
- `dashboard.py` – Streamlit app scoring candidates with live IBM calibration (`streamlit run dashboard.py`).
- `experiments/` – benchmarks, ablations and ranker experiments (run from the repo root).
- `collect_real_dataset.py`, `build_real_dataset_v2.py`, `rebuild_real_dataset_clean.py`, `fix_bell_dataset.py` – real-hardware dataset construction. `results/real_hardware_dataset_final.csv` is the canonical dataset (Bell fidelity uses the 2 active qubits).
- `train_real_model.py`, `train_final_real_model.py` – models on the real-hardware dataset.
- `live_*.py`, `validate_live_selection.py` – live calibration/mapping tools.
