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

## Deployment (Vercel frontend + Render API)
The heavy Qiskit/IBM work runs on Render; Vercel serves only a small static page.

**Backend (Render)** – `server/` (FastAPI) with `render.yaml`:
1. Render → New → Blueprint → select this repo (uses `render.yaml`).
2. Set secrets: `IBM_QUANTUM_TOKEN` (required), `ALLOWED_ORIGINS` (your Vercel URL, e.g. `https://my-app.vercel.app`), optional `API_KEY`.
3. Check `https://<service>.onrender.com/health`; API docs at `/docs`.

**Frontend (Vercel)** – `frontend/`:
1. Edit `frontend/config.js` and set `window.CC_API_URL` to the Render URL.
2. Vercel → New Project → import repo → set *Root Directory* to `frontend` (no build command).

Endpoints: `GET /health`, `GET /backends`, `POST /analyze` `{"qasm": "...", "backends": ["ibm_fez"]}`.
Never put the IBM token in the frontend. Free Render instances sleep when idle (first request is slow).
The Streamlit dashboard (`dashboard.py`) is unchanged and can still be run locally.
