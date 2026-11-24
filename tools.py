"""Agent tools. Every tool returns lines like '[LAB:eGFR:2026-01-22] ...' so answers can cite verifiable evidence ids.

`patient_id` is an InjectedToolArg: it is hidden from the LLM's tool schema and always set by the graph from the
selected patient, so the model cannot read another patient's record (even if prompted to).
"""
import re
from datetime import date
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool

from data_store import get_store

EVIDENCE_RE = re.compile(r"^\[((?:LAB|MED|COND|NOTE|GL):[^\]]+)\]\s*(.*)$", re.MULTILINE)
CITE_RE = re.compile(r"\[((?:LAB|MED|COND|NOTE|GL):[^\]]+)\]")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ALIASES = {
    "egfr": "eGFR", "gfr": "eGFR", "kidney function": "eGFR", "renal function": "eGFR",
    "hba1c": "HbA1c", "a1c": "HbA1c", "glycated hemoglobin": "HbA1c", "hemoglobin a1c": "HbA1c",
    "potassium": "Potassium", "k": "Potassium", "creatinine": "Creatinine", "bnp": "BNP",
    "tsh": "TSH", "weight": "Weight", "sbp": "SBP", "systolic": "SBP", "systolic blood pressure": "SBP",
    "dbp": "DBP", "diastolic": "DBP", "diastolic blood pressure": "DBP", "blood pressure": "SBP",
    "lvef": "LVEF", "ejection fraction": "LVEF",
}


def _patient(patient_id: str) -> dict:
    p = get_store().patients.get(patient_id)
    if not p:
        raise ValueError(f"unknown patient {patient_id}")
    return p


def _years(d: str) -> float:
    return date.fromisoformat(d).toordinal() / 365.25


def _slope(points: list) -> float | None:
    """Least-squares slope in units per year."""
    if len(points) < 2:
        return None
    xs = [_years(d) for d, _ in points]
    ys = [v for _, v in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


@tool
def get_lab_trend(test: str, patient_id: Annotated[str, InjectedToolArg],
                  start_date: str | None = None, end_date: str | None = None) -> str:
    """Get the time series of a lab or vital sign for the current patient, with first/last values, change and trend.

    Args:
        test: e.g. eGFR, HbA1c, creatinine, potassium, BNP, TSH, weight, SBP, DBP, LVEF.
        start_date: optional ISO date YYYY-MM-DD (inclusive).
        end_date: optional ISO date YYYY-MM-DD (inclusive).
    """
    obs = _patient(patient_id)["observations"]
    name = ALIASES.get(test.strip().lower(), test.strip())
    key = next((k for k in obs if k.lower() == name.lower()), None)
    if key is None:
        return f"No test named '{test}' in this patient's record. Available tests: {', '.join(sorted(obs))}."
    for d in (start_date, end_date):
        if d and not ISO.match(d):
            return "Invalid date; use ISO format YYYY-MM-DD."
    pts = [(d, v) for d, v in obs[key]["values"]
           if (not start_date or d >= start_date) and (not end_date or d <= end_date)]
    if not pts:
        return f"No {key} results in the requested date range."
    unit = obs[key]["unit"]
    lines = [f"{key} ({unit}): {len(pts)} result(s) from {pts[0][0]} to {pts[-1][0]}"]
    lines += [f"[LAB:{key}:{d}] {key} {v} {unit} on {d}" for d, v in pts]
    if len(pts) >= 2:
        (d0, v0), (d1, v1) = pts[0], pts[-1]
        change = round(v1 - v0, 2)
        pct = f"{(v1 - v0) / v0 * 100:+.1f}%" if v0 else "n/a"
        slope = _slope(pts)
        direction = "stable" if abs(change) < 1e-9 else ("increasing" if change > 0 else "decreasing")
        span_days = (date.fromisoformat(d1) - date.fromisoformat(d0)).days
        rate = f"; trend {slope:+.2f} {unit} per year" if slope is not None and span_days >= 365 else ""  # no extrapolating short spans
        lines.append(f"Summary: first {v0} ({d0}), last {v1} ({d1}); change {change:+g} ({pct}){rate}; overall {direction}.")
    return "\n".join(lines)


@tool
def get_medication_history(patient_id: Annotated[str, InjectedToolArg], drug: str | None = None) -> str:
    """List the current patient's medications with doses, start/stop dates and reasons (dose changes are separate rows).

    Args:
        drug: optional drug name filter (substring match), e.g. 'metformin'.
    """
    meds = _patient(patient_id)["medications"]
    if drug:
        meds = [m for m in meds if drug.strip().lower() in m["name"].lower()]
    if not meds:
        return f"No medication matching '{drug}' in this patient's record."
    out = []
    for m in sorted(meds, key=lambda m: (m["name"], m["start"])):
        stop = f"stopped {m['stop']}" if m["stop"] else "ongoing"
        out.append(f"[MED:{m['name']}:{m['start']}] {m['name']} {m['dose']}, started {m['start']}, {stop} ({m['reason']})")
    return "\n".join(out)


@tool
def get_conditions(patient_id: Annotated[str, InjectedToolArg]) -> str:
    """List the current patient's documented conditions (problem list) with dates."""
    out = []
    for c in _patient(patient_id)["conditions"]:
        slug = re.sub(r"[^a-z0-9]+", "-", c["name"].lower()).strip("-")
        out.append(f"[COND:{slug}] {c['name']}, documented {c['date']} ({c['status']})")
    return "\n".join(out)


@tool
def search_notes(query: str, patient_id: Annotated[str, InjectedToolArg], k: int = 3) -> str:
    """Semantic search over the current patient's clinical notes (for reasons, context, counselling, plans).

    Args:
        query: what to look for, e.g. 'why was metformin reduced'.
        k: number of notes to return (1-5).
    """
    k = max(1, min(int(k), 5))
    docs = get_store().note_index.similarity_search(query, k=k, filter=lambda d: d.metadata["patient_id"] == patient_id)
    if not docs:
        return "No matching notes for this patient."
    return "\n".join(f"[{d.metadata['id']}] ({d.metadata['type']}) {d.page_content}" for d in
                     sorted(docs, key=lambda d: d.metadata["date"]))


@tool
def search_guidelines(query: str, k: int = 2) -> str:
    """Search short illustrative reference snippets (general medical knowledge, not patient data).

    Args:
        query: topic, e.g. 'metformin kidney function'.
        k: number of snippets to return (1-3).
    """
    k = max(1, min(int(k), 3))
    docs = get_store().guideline_index.similarity_search(query, k=k)
    return "\n".join(f"[{d.metadata['id']}] {d.page_content}" for d in docs) or "No matching reference snippets."


TOOLS = [get_lab_trend, get_medication_history, get_conditions, search_notes, search_guidelines]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
