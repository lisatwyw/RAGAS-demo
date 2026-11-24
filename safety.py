"""Guardrails: emergency short-circuit, directive-language detection, numeric grounding, and trace redaction."""
import re

from tools import CITE_RE

EMERGENCY_TERMS = re.compile(
    r"\b(chest pain|can'?t breathe|cannot breathe|difficulty breathing|suicid\w*|overdos\w*|stroke symptoms|"
    r"unresponsive|severe bleeding|anaphyla\w*)\b", re.I)
PRESENT_MARKERS = re.compile(r"\b(right now|currently|at the moment|now|immediately|urgent\w*|is having|having)\b", re.I)
EMERGENCY_MESSAGE = ("This may be an emergency. This tool is a record-review assistant and cannot help with urgent care. "
                     "Contact emergency services or the on-call clinician immediately.")

DIRECTIVE_PATTERNS = [
    r"\b(i|we)\s+(would\s+|strongly\s+)?(recommend|suggest|advise)\b",
    r"\byou\s+(should|must|need to)\s+(prescribe|increase|decrease|reduce|stop|start|discontinue|switch|titrate|give|administer|order)\b",
    r"\b(should|must)\s+be\s+(increased|decreased|reduced|stopped|started|discontinued|switched|titrated)\b",
    r"\brecommended\s+(dose|dosage)\b",
]
DIRECTIVE_RE = re.compile("|".join(DIRECTIVE_PATTERNS), re.I)

DISCLAIMER = "Synthetic demo data. Decision support for record review only; not medical advice."


def is_emergency(question: str) -> bool:
    return bool(EMERGENCY_TERMS.search(question) and PRESENT_MARKERS.search(question))


def directive_language(text: str) -> bool:
    return bool(DIRECTIVE_RE.search(text))


def unsupported_numbers(answer: str, tool_text: str) -> list[str]:
    """Numbers in the answer (excluding citation ids and small integers) that never appear in tool output."""
    body = CITE_RE.sub("", answer)
    nums = set(re.findall(r"\d+(?:\.\d+)?", body))
    allowed = set(re.findall(r"\d+(?:\.\d+)?", tool_text))
    for n in list(allowed):  # tolerate rounding of derived values, e.g. 47.4% -> 47
        try:
            f = float(n)
            allowed |= {str(round(f)), f"{f:.1f}", f"{f:.2f}"}
        except ValueError:
            pass
    return sorted(n for n in nums if n not in allowed and not (n.isdigit() and int(n) <= 10))


# ---- trace redaction (regex demo; real PHI needs a proper de-identification service) ----
_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\+?1[ -.]?)?\(?\d{3}\)?[ -.]\d{3}[ -.]\d{4}\b"), "[PHONE]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\bMRN[:\s#]*[A-Z0-9-]+", re.I), "MRN [REDACTED]"),
    (re.compile(r"\bSYN-\d+\b"), "[MRN]"),
]
_NAMES: list[re.Pattern] = []


def _names() -> list[re.Pattern]:
    if not _NAMES:
        from data_store import load_patients

        for p in load_patients().values():
            _NAMES.append(re.compile(re.escape(p["name"]), re.I))
    return _NAMES


def redact(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    for pat in _names():
        text = pat.sub("[NAME]", text)
    return text


def redact_payload(obj):
    """Recursively redact strings in dict/list payloads (used as LangSmith hide_inputs/hide_outputs)."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_payload(v) for v in obj]
    return obj
