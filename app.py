"""Gradio UI for Hugging Face Spaces. Wraps the LangGraph pipeline from graph.py.

Secrets (Space Settings -> Variables and secrets): GOOGLE_API_KEY (free from aistudio.google.com/apikey), and optionally
LANGSMITH_API_KEY, LANGSMITH_TRACING=true, LANGSMITH_PROJECT. Never commit a .env file.
"""
import os
import time
from collections import defaultdict, deque
from threading import Lock

import gradio as gr

from graph import build_graph
from rag import has_api_key  # loads .env locally; on Spaces, secrets arrive as env vars

PER_IP_PER_HOUR = int(os.getenv("PER_IP_PER_HOUR", "10"))
GLOBAL_PER_DAY = int(os.getenv("GLOBAL_PER_DAY", "300"))  # hard cap on your OpenAI spend
MAX_QUESTION_CHARS = 300

_graph = None
_lock = Lock()
_ip_hits: dict[str, deque] = defaultdict(deque)
_day_hits: deque = deque()

def get_graph():
    """Build lazily so the Space boots even if the key is missing, and shows a clear error instead."""
    global _graph
    with _lock:
        if _graph is None:
            _graph = build_graph()
    return _graph

def allowed(ip: str) -> str | None:
    """Return an error message if over a limit, else None (and record the hit)."""
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

def ask(question: str, request: gr.Request):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    if len(question) > MAX_QUESTION_CHARS:
        return f"Please keep questions under {MAX_QUESTION_CHARS} characters.", ""
    if not has_api_key():
        return "Server is missing GOOGLE_API_KEY (add it under Space Settings > Secrets).", ""

    headers = getattr(request, "headers", None) or {}
    ip = (headers.get("x-forwarded-for", "") or getattr(request.client, "host", "unknown")).split(",")[0].strip()
    if (msg := allowed(ip)):
        return msg, ""
    try:
        out = get_graph().invoke(
            {"question": question},
            config={"run_name": "rag_graph", "tags": ["hf-space"], "metadata": {"env": "prod"}},
        )
    except Exception as e:  # keep the UI friendly; the full error is in the Space logs
        print(f"pipeline error: {e!r}")
        text = repr(e)
        if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
            return "The free Gemini quota is busy right now. Please try again in a minute.", ""
        return "Something went wrong while answering. Please try again.", ""

    sources = ", ".join(sorted({d.metadata["source"] for d in out["docs"]})) or "none"
    details = f"Sources: {sources}\nQuery rewrites: {out.get('retries', 0)}"
    return out["answer"], details

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(label="Question", placeholder="Which plans support password-protected links?"),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Details")],
    title="Acme Cloud Storage docs bot",
    description="Corrective RAG built with LangChain + LangGraph, traced in LangSmith, evaluated with RAGAS.",
    examples=[
        "How much does the Pro plan cost?",
        "How long are deleted files kept on the Free plan?",
        "Is SAML single sign-on supported?",
        "Is there a Linux desktop app?",
    ],
    flagging_mode="never",
)
if __name__ == "__main__":
    user, pw = os.getenv("APP_USER"), os.getenv("APP_PASSWORD")  # optional login for private demos
    demo.launch(auth=(user, pw) if user and pw else None)
