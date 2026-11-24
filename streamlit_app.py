"""Streamlit UI for Longitudinal Clinical Intelligence.
Synthetic data only. Do not enter real patient information.
"""

import os
import time
from collections import deque
from threading import Lock

import streamlit as st

from agent import build_agent, run_agent
from config import has_api_key
from data_store import load_patients
from observability import flush, log_feedback
from safety import DISCLAIMER
from tools import CITE_RE


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Longitudinal Clinical Intelligence",
    page_icon="🏥",
    layout="wide",
)

MAX_QUESTION_CHARS = 300
PER_SESSION_PER_HOUR = int(os.getenv("PER_SESSION_PER_HOUR", "10"))
GLOBAL_PER_DAY = int(os.getenv("GLOBAL_PER_DAY", "200"))

PATIENTS = load_patients()

EXAMPLES = [
    ("P001", "How has her kidney function (eGFR) changed over the follow-up?"),
    ("P001", "Did her HbA1c improve after empagliflozin was started?"),
    ("P001", "What was her most recent urine albumin-creatinine ratio?"),
    ("P002", "How did potassium change after spironolactone was started, and what was done about it?"),
    ("P003", "How has her TSH changed, and what could explain the 2026 rise?"),
    ("P001", "What dose of metformin should I prescribe now?"),
]

# Shared, process-local resources
@st.cache_resource
def get_graph():
    # Build once per application process.
    return build_agent()


_lock = Lock()
_day_hits = deque()


# --------------------------------------------------
# Rate limiting
# --------------------------------------------------

def allowed():
    """Demo-level limits: per Streamlit session and process-wide daily cap."""

    now = time.time()

    if "request_times" not in st.session_state:
        st.session_state.request_times = []

    # Per-session hourly limit
    hits = [
        t for t in st.session_state.request_times
        if now - t <= 3600
    ]

    if len(hits) >= PER_SESSION_PER_HOUR:
        return f"Rate limit reached: maximum {PER_SESSION_PER_HOUR} questions per session per hour."

    # Process-wide daily limit
    with _lock:
        while _day_hits and now - _day_hits[0] > 86400:
            _day_hits.popleft()

        if len(_day_hits) >= GLOBAL_PER_DAY:
            return "Daily demo limit reached. Please try again tomorrow."

        _day_hits.append(now)

    hits.append(now)
    st.session_state.request_times = hits

    return None


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def format_args(args):
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def build_result(res):
    """Format agent output for display."""

    cited = set(CITE_RE.findall(res["answer"]))

    evidence = "\n".join(
        f"- {'✅' if k in cited else '▫️'} **{k}**: {v}"
        for k, v in res["evidence"].items()
    ) or "_No evidence retrieved._"

    trace = "\n".join(
        f"{i}. `{t['tool']}({format_args(t['args'])})`"
        for i, t in enumerate(res["trace"], 1)
    ) or "_No tools called._"

    trace += (
        f"\n\nLLM steps: {res['llm_steps']}"
        f" · Verifier revisions: {res['revisions']}"
        f" · Latency: {res['latency_s']}s"
    )

    verification = {
        "ok": "✅ Verified: citations valid, numbers grounded, no directive language.",
        "unverified": "⚠️ Not fully verified: " + "; ".join(res["flags"]),
        "emergency": "🚨 Emergency short-circuit (no model call).",
    }[res["status"]]

    if res["status"] == "ok" and res["abstained"]:
        verification = "✅ Abstained: the record does not contain this information."

    if res["revisions"] and res["status"] == "ok":
        verification += (
            f" (Self-corrected after: {'; '.join(res['flags'])})"
        )

    return {
        "answer": res["answer"],
        "evidence": evidence,
        "trace": trace,
        "verification": verification,
        "run_id": str(res["run_id"]),
    }


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "patient_id" not in st.session_state:
    st.session_state.patient_id = "P001"

if "question" not in st.session_state:
    st.session_state.question = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = ""


def load_example():
    selected = st.session_state.example_choice

    if selected != "Select an example...":
        index = int(selected.split(" | ", 1)[0]) - 1
        patient_id, question = EXAMPLES[index]

        st.session_state.patient_id = patient_id
        st.session_state.question = question


# --------------------------------------------------
# Main UI
# --------------------------------------------------

st.title("🏥 Longitudinal Clinical Intelligence")
st.subheader("Agentic RAG for longitudinal clinical-record review")

st.warning(
    "Synthetic patients only. Not medical advice. "
    "Do not enter real patient information."
)

st.markdown(
    """
    A LangGraph agent reviews one patient's longitudinal timeline
    (laboratory results, medications, and clinical notes), retrieves
    supporting evidence, cites its sources, and abstains when the
    record does not contain the requested information.

    Answers are checked by a verification step. The application
    supports LangSmith tracing and a separate RAGAS evaluation pipeline.
    """
)

# Example questions
st.markdown("### Try an example")

example_options = ["Select an example..."] + [
    f"{i + 1} | {pid}: {question}"
    for i, (pid, question) in enumerate(EXAMPLES)
]

st.selectbox(
    "Example questions",
    options=example_options,
    key="example_choice",
    on_change=load_example,
)

# Patient selector
patient_options = list(PATIENTS.keys())

def patient_label(pid):
    p = PATIENTS[pid]
    return f"{pid} · {p['name']} (synthetic) · {p['age']}{p['sex']}"

st.selectbox(
    "Patient",
    options=patient_options,
    format_func=patient_label,
    key="patient_id",
)

# Question input
with st.form("question_form"):
    st.text_area(
        "Question",
        placeholder="How has her kidney function changed?",
        max_chars=MAX_QUESTION_CHARS,
        key="question",
    )

    submitted = st.form_submit_button(
        "Ask",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Agent execution
# --------------------------------------------------

if submitted:

    patient_id = st.session_state.patient_id
    question = (st.session_state.question or "").strip()

    st.session_state.last_result = None
    st.session_state.feedback_message = ""

    if patient_id not in PATIENTS:
        st.error("Please choose a valid patient.")

    elif not question:
        st.warning("Please enter a question.")

    elif len(question) > MAX_QUESTION_CHARS:
        st.error(
            f"Please keep questions under {MAX_QUESTION_CHARS} characters."
        )

    elif not has_api_key():
        st.error(
            "Server is missing its model API key. "
            "Configure GOOGLE_API_KEY or OPENAI_API_KEY."
        )

    else:
        limit_message = allowed()

        if limit_message:
            st.warning(limit_message)

        else:
            try:
                with st.spinner(
                    "Reviewing clinical records and verifying evidence..."
                ):
                    res = run_agent(
                        get_graph(),
                        question,
                        patient_id,
                        tags=["posit-connect"],
                    )

                st.session_state.last_result = build_result(res)

            except Exception as e:
                print(f"Agent error: {e!r}")

                error_text = repr(e)

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "quota" in error_text.lower()
                ):
                    st.error(
                        "The model API quota is busy. "
                        "Please try again shortly."
                    )
                else:
                    st.error(
                        "Something went wrong while answering. "
                        "Please check the application logs."
                    )


# --------------------------------------------------
# Display answer and evidence
# --------------------------------------------------

result = st.session_state.last_result

if result:

    st.divider()

    st.markdown("### Answer")
    st.markdown(result["answer"])

    st.caption(DISCLAIMER)

    st.markdown("### Verification")
    st.markdown(result["verification"])

    with st.expander("Evidence retrieved", expanded=False):
        st.markdown(result["evidence"])

    with st.expander("Agent execution trace", expanded=False):
        st.markdown(result["trace"])

    st.caption(f"LangSmith run ID: {result['run_id']}")

    # --------------------------------------------------
    # User feedback
    # --------------------------------------------------

    st.markdown("### Was this answer helpful?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Helpful", use_container_width=True):
            flush()
            ok = log_feedback(
                result["run_id"],
                "user_helpful",
                1.0,
            )

            st.session_state.feedback_message = (
                "Thanks for your feedback."
                if ok
                else "Feedback could not be stored. Check LangSmith tracing settings."
            )

    with col2:
        if st.button("👎 Not helpful", use_container_width=True):
            flush()
            ok = log_feedback(
                result["run_id"],
                "user_helpful",
                0.0,
            )

            st.session_state.feedback_message = (
                "Thanks for your feedback."
                if ok
                else "Feedback could not be stored. Check LangSmith tracing settings."
            )

    if st.session_state.feedback_message:
        st.info(st.session_state.feedback_message)
