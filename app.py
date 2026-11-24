"""Gradio UI (Hugging Face Spaces). Synthetic data only.

Secrets (Space Settings -> Variables and secrets): GOOGLE_API_KEY (free: aistudio.google.com/apikey), and optionally
LANGSMITH_API_KEY + variables LANGSMITH_TRACING=true, LANGSMITH_PROJECT=clinical-agentic-rag.
"""
import os
import time
from collections import defaultdict, deque
from threading import Lock

import gradio as gr

from agent import build_agent, run_agent
from config import has_api_key
from data_store import load_patients
from observability import flush, log_feedback
from safety import DISCLAIMER
from tools import CITE_RE

PER_IP_PER_HOUR = int(os.getenv("PER_IP_PER_HOUR", "10"))
GLOBAL_PER_DAY = int(os.getenv("GLOBAL_PER_DAY", "200"))  # protects the shared free-tier quota
MAX_QUESTION_CHARS = 300

PATIENTS = load_patients()
CHOICES = [(f"{pid} · {p['name']} (synthetic) · {p['age']}{p['sex']}", pid) for pid, p in PATIENTS.items()]
EXAMPLES = [
    ["P001", "How has her kidney function (eGFR) changed over the follow-up?"],
    ["P001", "Did her HbA1c improve after empagliflozin was started?"],
    ["P001", "What was her most recent urine albumin-creatinine ratio?"],
    ["P002", "How did potassium change after spironolactone was started, and what was done about it?"],
    ["P003", "How has her TSH changed, and what could explain the 2026 rise?"],
    ["P001", "What dose of metformin should I prescribe now?"],
]

_graph, _lock = None, Lock()
_ip_hits: dict[str, deque] = defaultdict(deque)
_day_hits: deque = deque()


def get_graph():
    global _graph
    with _lock:
        if _graph is None:  # built lazily: indexes the notes with the embedding API on first use
            _graph = build_agent()
    return _graph


def allowed(ip: str) -> str | None:
    now = time.time()
    with _lock:
        while _day_hits and now - _day_hits[0] > 86400:
            _day_hits.popleft()
        hits = _ip_hits[ip]
        while hits and now - hits[0] > 3600:
            hits.popleft()
        if len(_day_hits) >= GLOBAL_PER_DAY:
            return "Daily demo limit reached. Please try again tomorrow."
        if len(hits) >= PER_IP_PER_HOUR:
            return f"Rate limit: max {PER_IP_PER_HOUR} questions per hour."
        hits.append(now)
        _day_hits.append(now)
    return None


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def ask(patient_id: str, question: str, request: gr.Request):
    """Returns (answer, evidence, tool trace, verification, run_id, feedback status)."""
    blank = ("", "", "", "", None, "")
    question = (question or "").strip()
    if patient_id not in PATIENTS:
        return ("Please choose a patient.",) + blank[1:]
    if not question:
        return ("Please enter a question.",) + blank[1:]
    if len(question) > MAX_QUESTION_CHARS:
        return (f"Please keep questions under {MAX_QUESTION_CHARS} characters.",) + blank[1:]
    if not has_api_key():
        return ("Server is missing GOOGLE_API_KEY (add it under Space Settings > Secrets).",) + blank[1:]
    headers = getattr(request, "headers", None) or {}
    ip = (headers.get("x-forwarded-for", "") or getattr(request.client, "host", "unknown")).split(",")[0].strip()
    if (msg := allowed(ip)):
        return (msg,) + blank[1:]

    try:
        res = run_agent(get_graph(), question, patient_id, tags=["hf-space"])
    except Exception as e:  # friendly UI; full error is in the Space logs
        print(f"agent error: {e!r}")
        text = repr(e)
        if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
            return ("The free Gemini quota is busy right now. Please try again in a minute.",) + blank[1:]
        return ("Something went wrong while answering. Please try again.",) + blank[1:]

    answer = f"{res['answer']}\n\n---\n*{DISCLAIMER}*"
    cited = set(CITE_RE.findall(res["answer"]))
    evidence = "\n".join(f"- {'✅' if k in cited else '▫️'} **{k}**: {v}" for k, v in res["evidence"].items()) \
        or "_No evidence retrieved._"
    trace = "\n".join(f"{i}. `{t['tool']}({_fmt_args(t['args'])})`" for i, t in enumerate(res["trace"], 1)) \
        or "_No tools called._"
    trace += f"\n\nLLM steps: {res['llm_steps']} · verifier revisions: {res['revisions']} · {res['latency_s']}s"
    verification = {"ok": "✅ Verified: citations valid, numbers grounded, no directive language.",
                    "unverified": "⚠️ Not fully verified: " + "; ".join(res["flags"]),
                    "emergency": "🚨 Emergency short-circuit (no model call)."}[res["status"]]
    if res["status"] == "ok" and res["abstained"]:
        verification = "✅ Abstained: the record does not contain this information."
    if res["revisions"] and res["status"] == "ok":
        verification += f" (self-corrected after: {'; '.join(res['flags'])})"
    return answer, evidence, trace, verification, str(res["run_id"]), ""


def feedback(run_id: str | None, score: int):
    if not run_id:
        return "Ask a question first."
    flush()
    ok = log_feedback(run_id, "user_helpful", float(score))
    return "Thanks for the feedback." if ok else "Feedback noted (LangSmith tracing is off, so it was not stored)."


with gr.Blocks(title="Longitudinal Clinical Intelligence (synthetic demo)") as demo:
    gr.Markdown("# Longitudinal Clinical Intelligence: agentic RAG\n"
                "**Synthetic patients only. Not medical advice; do not enter real patient data.** "
                "A LangGraph agent reads one patient's timeline (labs, medications, notes), cites evidence, abstains when the "
                "record is silent, and is checked by a verifier. Evaluated with RAGAS, traced in LangSmith.")
    with gr.Row():
        patient = gr.Dropdown(CHOICES, value="P001", label="Patient (synthetic)", scale=1)
        question = gr.Textbox(label="Question", placeholder="How has her kidney function changed?", scale=3)
    go = gr.Button("Ask", variant="primary")
    gr.Examples(EXAMPLES, inputs=[patient, question])
    answer = gr.Markdown(label="Answer")
    verification = gr.Markdown()
    with gr.Accordion("Evidence (✅ = cited in the answer)", open=False):
        evidence = gr.Markdown()
    with gr.Accordion("Agent trace (tools called)", open=False):
        trace = gr.Markdown()
    run_state = gr.State(None)
    with gr.Row():
        up, down = gr.Button("👍 Helpful"), gr.Button("👎 Not helpful")
        fb = gr.Markdown()
    go.click(ask, [patient, question], [answer, evidence, trace, verification, run_state, fb])
    question.submit(ask, [patient, question], [answer, evidence, trace, verification, run_state, fb])
    up.click(lambda r: feedback(r, 1), run_state, fb)
    down.click(lambda r: feedback(r, 0), run_state, fb)

if __name__ == "__main__":
    user, pw = os.getenv("APP_USER"), os.getenv("APP_PASSWORD")  # optional login
    demo.launch(auth=(user, pw) if user and pw else None)
