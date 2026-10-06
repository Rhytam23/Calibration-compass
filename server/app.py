"""FastAPI service for CalibrationCompass (deploy on Render).

Environment variables
---------------------
IBM_QUANTUM_TOKEN   IBM Quantum Platform API key (required for /analyze)
IBM_INSTANCE        IBM instance CRN/name (optional, default "open-instance")
ALLOWED_ORIGINS     Comma-separated allowed browser origins, e.g.
                    "https://my-app.vercel.app" (default: none -> no CORS)
API_KEY             If set, requests must send header "X-API-Key: <value>"
RATE_LIMIT_PER_MIN  Max /analyze calls per client IP per minute (default 6)
"""

import os
import threading
import time
from collections import defaultdict, deque
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .core import DEFAULT_BACKENDS, analyze_backends, confidence_label

MAX_QASM_CHARS = 100_000

app = FastAPI(title="CalibrationCompass API", version="1.0.0")

origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class AnalyzeRequest(BaseModel):
    qasm: str = Field(..., description="OpenQASM 2.0 circuit")
    backends: list[str] | None = Field(
        None, description="IBM backend names; defaults to the Heron trio"
    )


def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def rate_limit(request: Request):
    limit = int(os.environ.get("RATE_LIMIT_PER_MIN", "6"))
    ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    now = time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)


@lru_cache(maxsize=1)
def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise HTTPException(
            status_code=503, detail="IBM_QUANTUM_TOKEN is not configured"
        )

    return QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=os.environ.get("IBM_INSTANCE", "open-instance"),
    )


@app.get("/")
def root():
    return {"service": "calibrationcompass", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/backends", dependencies=[Depends(require_api_key)])
def backends():
    return {"default": DEFAULT_BACKENDS}


@app.post(
    "/analyze",
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def analyze(req: AnalyzeRequest):
    from qiskit.qasm2 import loads as qasm2_loads

    if len(req.qasm) > MAX_QASM_CHARS:
        raise HTTPException(status_code=413, detail="QASM too large")

    try:
        circuit = qasm2_loads(req.qasm)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"QASM parsing failed: {exc}")

    if circuit.num_qubits == 0:
        raise HTTPException(status_code=400, detail="Circuit has no qubits")

    names = req.backends or DEFAULT_BACKENDS
    if len(names) > 5 or not all(
        isinstance(n, str) and n.startswith("ibm_") for n in names
    ):
        raise HTTPException(
            status_code=400, detail="Provide up to 5 IBM backend names (ibm_*)"
        )

    candidates, status = analyze_backends(get_service(), circuit, names)

    if not candidates:
        raise HTTPException(
            status_code=422,
            detail={"message": "No usable candidates", "status": status},
        )

    best = candidates[0]
    margin = (
        candidates[1]["combined_risk"] - best["combined_risk"]
        if len(candidates) > 1
        else None
    )

    return {
        "recommendation": best,
        "confidence": confidence_label(margin),
        "risk_margin": margin,
        "circuit": {
            "qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "operations": sum(circuit.count_ops().values()),
        },
        "candidates": candidates,
        "backend_status": status,
    }
