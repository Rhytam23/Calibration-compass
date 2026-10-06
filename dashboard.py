import math
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from qiskit import transpile
from qiskit.qasm2 import loads as qasm2_loads
from qiskit_ibm_runtime import QiskitRuntimeService


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="CalibrationCompass",
    page_icon="🧭",
    layout="wide"
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }

    .recommendation {
        padding: 1.3rem;
        border-radius: 15px;
        border: 1px solid rgba(100, 150, 255, 0.35);
        background: rgba(80, 120, 220, 0.08);
    }

    .recommendation-label {
        font-size: 0.75rem;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .recommendation-main {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    .confidence {
        font-size: 1.3rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONFIG
# ============================================================

BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh"
]

SEEDS = [
    11, 22, 33, 44, 55, 66
]


# ============================================================
# IBM SERVICE
# ============================================================

@st.cache_resource
def get_service():

    return QiskitRuntimeService(
        channel="ibm_quantum_platform",
        instance="open-instance"
    )


# ============================================================
# CANDIDATE ANALYSIS
# ============================================================

def analyze_candidate(
    compiled,
    props
):

    used_qubits = set()
    measured_qubits = set()

    gate_errors = []
    two_q_errors = []
    two_q_edges = []

    gate_count = 0
    two_q_count = 0

    for item in compiled.data:

        operation = item.operation
        qargs = item.qubits

        name = operation.name.lower()

        physical = [
            compiled.find_bit(q).index
            for q in qargs
        ]

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

        for q in physical:
            used_qubits.add(q)

        gate_count += 1

        if len(physical) == 2:

            two_q_count += 1

            edge = tuple(
                sorted(physical)
            )

            if edge not in two_q_edges:
                two_q_edges.append(edge)

        try:

            error = props.gate_error(
                name,
                physical
            )

        except Exception:

            error = None

        if error is not None:

            gate_errors.append(error)

            if len(physical) == 2:
                two_q_errors.append(error)

    # --------------------------------------------------------
    # Readout
    # --------------------------------------------------------

    readout_targets = (
        measured_qubits
        if measured_qubits
        else used_qubits
    )

    readout_errors = []

    for q in sorted(readout_targets):

        try:

            error = props.readout_error(q)

        except Exception:

            error = None

        if error is not None:
            readout_errors.append(error)

    if readout_errors:

        readout_success = 1.0

        for error in readout_errors:

            readout_success *= (
                1.0 - error
            )

        readout_loss = (
            1.0
            -
            readout_success
        )

        max_readout = max(
            readout_errors
        )

    else:

        readout_loss = 0.0
        max_readout = 0.0

    # --------------------------------------------------------
    # Gate
    # --------------------------------------------------------

    gate_success = 1.0

    for error in gate_errors:

        gate_success *= (
            1.0 - error
        )

    gate_loss = (
        1.0
        -
        gate_success
    )

    # --------------------------------------------------------
    # Combined risk
    # --------------------------------------------------------

    combined_risk = (
        1.0
        -
        (
            (1.0 - readout_loss)
            *
            (1.0 - gate_loss)
        )
    )

    return {
        "physical_qubits":
            sorted(used_qubits),

        "measured_qubits":
            sorted(measured_qubits),

        "readout_loss":
            readout_loss,

        "max_readout":
            max_readout,

        "gate_loss":
            gate_loss,

        "gate_error_sum":
            sum(gate_errors),

        "2Q_gate_error_sum":
            sum(two_q_errors),

        "combined_risk":
            combined_risk,

        "gate_count":
            gate_count,

        "2Q_gates":
            two_q_count,

        "2Q_edges":
            two_q_edges,

        "depth":
            compiled.depth()
    }


# ============================================================
# LIVE ANALYSIS
# ============================================================

def analyze_backends(
    circuit,
    backend_names
):

    service = get_service()

    results = []
    status_rows = []

    for backend_name in backend_names:

        try:

            backend = service.backend(
                backend_name
            )

            status = backend.status()

            status_rows.append({
                "Backend":
                    backend_name,

                "Status":
                    status.status_msg,

                "Operational":
                    status.operational,

                "Pending jobs":
                    status.pending_jobs
            })

            if not status.operational:
                continue

            if (
                circuit.num_qubits
                >
                backend.num_qubits
            ):
                continue

            props = backend.properties(
                refresh=True
            )

            calibration_time = (
                props.last_update_date
            )

        except Exception as exc:

            status_rows.append({
                "Backend":
                    backend_name,

                "Status":
                    f"Error: {exc}",

                "Operational":
                    False,

                "Pending jobs":
                    None
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
                    seed_transpiler=seed
                )

                features = analyze_candidate(
                    compiled,
                    props
                )

                if calibration_time is not None:

                    now = datetime.now(
                        timezone.utc
                    )

                    calibration_timestamp = (
                        calibration_time
                    )

                    if (
                        calibration_timestamp.tzinfo
                        is None
                    ):

                        calibration_timestamp = (
                            calibration_timestamp.replace(
                                tzinfo=timezone.utc
                            )
                        )

                    age_minutes = (
                        now
                        -
                        calibration_timestamp
                    ).total_seconds() / 60.0

                else:

                    age_minutes = math.nan

                results.append({
                    "backend":
                        backend_name,

                    "candidate":
                        seed,

                    "calibration_time":
                        str(calibration_time),

                    "calibration_age":
                        age_minutes,

                    **features
                })

            except Exception as exc:

                status_rows.append({
                    "Backend":
                        backend_name,

                    "Status":
                        f"Candidate {seed} failed: {exc}",

                    "Operational":
                        True,

                    "Pending jobs":
                        None
                })

                continue

    return (
        pd.DataFrame(results),
        pd.DataFrame(status_rows)
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧭 CalibrationCompass</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Live calibration-aware quantum backend and mapping selection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT
# ============================================================

st.subheader("1. Quantum Circuit")

default_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];"""

qasm_text = st.text_area(
    "OpenQASM 2.0",
    value=default_qasm,
    height=240
)


# ============================================================
# BACKENDS
# ============================================================

st.subheader("2. IBM Quantum Backends")

selected_backends = st.multiselect(
    "Select backends",
    BACKENDS,
    default=BACKENDS
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

run = st.button(
    "🧭 Analyze with Live Calibration",
    type="primary",
    width="stretch"
)


if run:

    # --------------------------------------------------------
    # Parse circuit
    # --------------------------------------------------------

    try:

        circuit = qasm2_loads(
            qasm_text
        )

    except Exception as exc:

        st.error(
            f"QASM parsing failed: {exc}"
        )

        st.stop()

    if circuit.num_qubits == 0:

        st.error(
            "The circuit contains no qubits."
        )

        st.stop()

    if not selected_backends:

        st.error(
            "Select at least one backend."
        )

        st.stop()


    # --------------------------------------------------------
    # Circuit summary
    # --------------------------------------------------------

    st.subheader(
        "3. Circuit Analysis"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Logical qubits",
        circuit.num_qubits
    )

    c2.metric(
        "Original depth",
        circuit.depth()
    )

    c3.metric(
        "Operations",
        sum(
            circuit.count_ops().values()
        )
    )

    with st.expander(
        "Show circuit"
    ):

        st.code(
            circuit.draw(
                output="text"
            )
        )


    # --------------------------------------------------------
    # Live analysis
    # --------------------------------------------------------

    with st.spinner(
        "Refreshing calibration data and evaluating candidates..."
    ):

        results_df, status_df = (
            analyze_backends(
                circuit,
                selected_backends
            )
        )


    # --------------------------------------------------------
    # Backend status
    # --------------------------------------------------------

    st.subheader(
        "4. Backend Status"
    )

    if not status_df.empty:

        st.dataframe(
            status_df,
            width="stretch",
            hide_index=True
        )


    if results_df.empty:

        st.error(
            "No usable candidates were generated."
        )

        st.stop()


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        "combined_risk"
    ).reset_index(
        drop=True
    )


    best = results_df.iloc[0]

    if len(results_df) > 1:

        second = results_df.iloc[1]

        margin = (
            second["combined_risk"]
            -
            best["combined_risk"]
        )

    else:

        second = None
        margin = math.nan


    # ========================================================
    # CONFIDENCE
    # ========================================================

    # These are deliberately risk-margin based.
    # They indicate separation between candidates,
    # NOT probability of hardware success.

    if math.isnan(margin):

        confidence_label = "Single option"

    elif margin >= 0.010:

        confidence_label = "Strong preference"

    elif margin >= 0.003:

        confidence_label = "Moderate preference"

    else:

        confidence_label = "Close call"


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.subheader(
        "5. CalibrationCompass Recommendation"
    )

    st.markdown(
        f"""
        <div class="recommendation">

        <div class="recommendation-label">
        Recommended execution target
        </div>

        <div class="recommendation-main">
        {best["backend"]} · Candidate {int(best["candidate"])}
        </div>

        <div>
        Physical qubits: {best["physical_qubits"]}
        </div>

        <div>
        Combined calibration risk:
        <b>{best["combined_risk"]:.6f}</b>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Confidence row
    # --------------------------------------------------------

    st.markdown(
        "### Recommendation confidence"
    )

    cc1, cc2, cc3 = st.columns(3)

    cc1.metric(
        "Assessment",
        confidence_label
    )

    if not math.isnan(margin):

        cc2.metric(
            "Risk margin",
            f"{margin:.6f}"
        )

    else:

        cc2.metric(
            "Risk margin",
            "N/A"
        )

    cc3.metric(
        "Candidates tested",
        len(results_df)
    )


    if confidence_label == "Close call":

        st.warning(
            "Close call: the top candidates have very similar "
            "calibration risk. Treat the recommendation as a "
            "preference, not a guarantee."
        )

    elif confidence_label == "Moderate preference":

        st.info(
            "Moderate preference: the recommended candidate "
            "has a noticeable calibration-risk advantage."
        )

    elif confidence_label == "Strong preference":

        st.success(
            "Strong preference: the recommended candidate has "
            "a clear calibration-risk advantage."
        )


    # --------------------------------------------------------
    # Key metrics
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Readout loss",
        f"{best['readout_loss']:.5f}"
    )

    k2.metric(
        "Gate loss",
        f"{best['gate_loss']:.5f}"
    )

    k3.metric(
        "2Q gates",
        int(best["2Q_gates"])
    )

    k4.metric(
        "Compiled depth",
        int(best["depth"])
    )


    # ========================================================
    # WHY
    # ========================================================

    st.markdown(
        "### Why this candidate?"
    )

    st.write(
        f"CalibrationCompass selected **{best['backend']} "
        f"candidate {int(best['candidate'])}** because it has "
        f"the lowest combined live calibration risk among the "
        f"{len(results_df)} candidates tested."
    )

    st.write(
        f"Physical qubits: "
        f"{best['physical_qubits']}"
    )

    st.write(
        f"Exact 2Q interactions: "
        f"{best['2Q_edges']}"
    )

    st.write(
        f"Calibration age: "
        f"{best['calibration_age']:.1f} minutes"
    )


    # ========================================================
    # RANKING
    # ========================================================

    st.subheader(
        "6. Candidate Ranking"
    )

    ranking_columns = [
        "backend",
        "candidate",
        "physical_qubits",
        "readout_loss",
        "gate_loss",
        "combined_risk",
        "2Q_gates",
        "depth",
        "calibration_age"
    ]

    ranking_df = results_df[
        ranking_columns
    ].copy()

    ranking_df["candidate"] = (
        ranking_df["candidate"].astype(int)
    )

    ranking_df.columns = [
        "Backend",
        "Candidate",
        "Physical qubits",
        "Readout loss",
        "Gate loss",
        "Combined risk",
        "2Q gates",
        "Depth",
        "Calibration age (min)"
    ]

    st.dataframe(
        ranking_df,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # BEST PER BACKEND
    # ========================================================

    st.subheader(
        "7. Best Candidate Per Backend"
    )

    best_per_backend = (
        results_df
        .loc[
            results_df.groupby(
                "backend"
            )["combined_risk"].idxmin()
        ]
        .sort_values(
            "combined_risk"
        )
    )

    backend_display = (
        best_per_backend[
            [
                "backend",
                "candidate",
                "physical_qubits",
                "combined_risk",
                "readout_loss",
                "gate_loss"
            ]
        ]
        .copy()
    )

    backend_display.columns = [
        "Backend",
        "Candidate",
        "Physical qubits",
        "Combined risk",
        "Readout loss",
        "Gate loss"
    ]

    st.dataframe(
        backend_display,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # RISK CHART
    # ========================================================

    st.subheader(
        "8. Calibration Risk"
    )

    chart_df = results_df[
        [
            "backend",
            "candidate",
            "combined_risk"
        ]
    ].copy()

    chart_df["label"] = (
        chart_df["backend"]
        +
        " · C"
        +
        chart_df["candidate"]
        .astype(int)
        .astype(str)
    )

    chart_df = chart_df.set_index(
        "label"
    )

    st.bar_chart(
        chart_df["combined_risk"]
    )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "Technical details"
    ):

        st.write(
            "Calibration timestamp:",
            best["calibration_time"]
        )

        st.write(
            "Average T1:",
            best["avg_t1"]
            if "avg_t1" in best
            else "Not calculated"
        )

        st.write(
            "Average T2:",
            best["avg_t2"]
            if "avg_t2" in best
            else "Not calculated"
        )

        st.write(
            "Total gate error:",
            best["gate_error_sum"]
        )

        st.write(
            "2Q gate error:",
            best["2Q_gate_error_sum"]
        )

        st.write(
            "Maximum readout error:",
            best["max_readout"]
        )


    # ========================================================
    # LIMITATION
    # ========================================================

    st.info(
        "CalibrationCompass is a live calibration-risk "
        "recommender. It does not guarantee the highest "
        "execution fidelity on hardware."
    )

else:

    st.info(
        "Paste an OpenQASM 2.0 circuit and click "
        "'Analyze with Live Calibration'."
    )