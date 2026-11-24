"""LangSmith helpers: PHI-redacting tracer, per-run config, feedback logging."""
import os

from safety import redact_payload

_tracer = None
_client = None


def tracing_on() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def get_client():
    global _client
    if _client is None and tracing_on():
        from langsmith import Client

        _client = Client()
    return _client


def get_tracer():
    """Tracer whose client redacts inputs/outputs before anything leaves the process (REDACT_TRACES=false to disable)."""
    global _tracer
    if not tracing_on() or os.getenv("REDACT_TRACES", "true").lower() != "true":
        return None
    if _tracer is None:
        from langchain_core.tracers.langchain import LangChainTracer
        from langsmith import Client

        _tracer = LangChainTracer(client=Client(hide_inputs=redact_payload, hide_outputs=redact_payload))
    return _tracer


def run_config(run_id=None, tags=None, metadata=None) -> dict:
    cfg = {"run_name": "clinical_agent", "tags": list(tags or []), "metadata": dict(metadata or {})}
    if run_id:
        cfg["run_id"] = run_id
    tracer = get_tracer()
    if tracer:
        cfg["callbacks"] = [tracer]
    return cfg


def flush() -> None:
    """Make sure traces are uploaded before attaching feedback to them."""
    for c in (getattr(get_tracer(), "client", None), get_client()):
        if c is not None and hasattr(c, "flush"):
            try:
                c.flush()
            except Exception as e:  # observability must never break the app
                print(f"langsmith flush failed: {e!r}")


def log_feedback(run_id, key: str, score: float, comment: str | None = None) -> bool:
    client = get_client()
    if client is None or run_id is None:
        return False
    try:
        client.create_feedback(run_id=run_id, key=key, score=score, comment=comment)
        return True
    except Exception as e:
        print(f"feedback failed ({key}): {e!r}")
        return False
