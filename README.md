# 🧭 CalibrationCompass

Choose the best IBM quantum backend and qubit mapping for a circuit using **live calibration data**.
CalibrationCompass transpiles a circuit several ways, scores each candidate by readout and gate error
on the device's current calibration, and recommends the lowest-risk option. The repository also holds
the research code (ESP baselines vs. learned pairwise rankers) behind the approach.

[MIT licensed](LICENSE)

## Contents
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [API](#api)
- [Deployment](#deployment-vercel-frontend--render-api)
- [Repository layout](#repository-layout)
- [Research findings and caveats](#research-findings-and-caveats)
- [Contributing, security, license](#contributing-security-license)

## How it works
1. Parse an OpenQASM 2.0 circuit.
2. For each selected backend, transpile with 6 seeds (SABRE layout/routing, optimization level 3).
3. Using live calibration, compute readout loss, gate loss and a combined risk per candidate.
4. Rank candidates; the risk margin to the runner-up gives a confidence label
   (Strong ≥ 0.010, Moderate ≥ 0.003, otherwise Close call).

The confidence label measures separation between candidates, **not** the probability of hardware success.

## Quick start
```bash
git clone https://github.com/Rhytam23/Calibration-compass.git
cd Calibration-compass
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Live features need an IBM Quantum Platform account (API key). Copy `.env.example` to `.env` and fill it in,
or save an account with `QiskitRuntimeService.save_account`.

| Task | Command |
|---|---|
| Streamlit dashboard (live) | `streamlit run dashboard.py` |
| Offline pairwise demo | `python calibrationcompass.py` |
| API server locally | `pip install -r server/requirements.txt && uvicorn server.app:app --reload` |
| Any experiment | `python experiments/<script>.py` (works from any directory) |

## API
`POST /analyze`
```json
{"qasm": "OPENQASM 2.0; ...", "backends": ["ibm_fez", "ibm_kingston"]}
```
Returns `recommendation`, `confidence`, `risk_margin`, `candidates` and `backend_status`.
Other endpoints: `GET /health`, `GET /backends`, interactive docs at `/docs`.

Server environment variables (see `.env.example`):

| Variable | Required | Purpose |
|---|---|---|
| `IBM_QUANTUM_TOKEN` | yes | IBM Quantum Platform API key |
| `IBM_INSTANCE` | no | IBM instance (default `open-instance`) |
| `ALLOWED_ORIGINS` | for browsers | Comma-separated allowed origins (your Vercel URL) |
| `API_KEY` | no | If set, clients must send `X-API-Key` |
| `RATE_LIMIT_PER_MIN` | no | Per-IP limit on `/analyze` (default 6) |

## Deployment (Vercel frontend + Render API)
Heavy Qiskit/IBM work runs on Render; Vercel serves only a small static page.

**Backend (Render)** — `server/` + `render.yaml`
1. Render → New → Blueprint → select this repo.
2. Set `IBM_QUANTUM_TOKEN`, `ALLOWED_ORIGINS` and (optionally) `API_KEY` in the Environment tab.
3. Verify `https://<service>.onrender.com/health`.

**Frontend (Vercel)** — `frontend/`
1. Set `window.CC_API_URL` in `frontend/config.js` to the Render URL.
2. Vercel → New Project → import repo → Root Directory `frontend` (no build command).

Never put the IBM token in the frontend. Free Render instances sleep when idle, so the first request is slow.

## Repository layout
| Path | What it is |
|---|---|
| `server/` | FastAPI service (`app.py`) and Streamlit-free analysis core (`core.py`) |
| `frontend/` | Static page for Vercel |
| `dashboard.py` | Streamlit app using live IBM calibration |
| `calibrationcompass.py` | Pairwise XGBoost demo on `results/candidate_features_v2.csv` |
| `experiments/` | Benchmarks, ablations and ranker experiments |
| `collect_real_dataset.py`, `build_real_dataset_v2.py`, `rebuild_real_dataset_clean.py`, `fix_bell_dataset.py` | Real-hardware dataset construction |
| `train_real_model.py`, `train_final_real_model.py` | Models on the real-hardware dataset |
| `live_*.py`, `validate_live_selection.py` | Live calibration and mapping tools |
| `results/` | CSV outputs. `real_hardware_dataset_final.csv` is the canonical real-hardware dataset |

## Research findings and caveats
- Evaluation sets are small (30 circuit/backend decisions on the main benchmark), so differences of a few
  decisions are suggestive, not conclusive.
- Thresholds and blend weights are learned from leave-one-circuit-out scores to avoid in-sample tuning.
- Simulators are seeded for reproducibility. Hardware results depend on calibration at run time.
- Some live scripts use the current calibration rather than the calibration at job time.

## Contributing, security, license
See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Released under the [MIT License](LICENSE).
