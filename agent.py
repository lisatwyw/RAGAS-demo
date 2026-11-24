"""Agentic RAG as a LangGraph state machine.

START -> guard --emergency--> END
           |
           v
         agent <---------------+
           |  tool calls?      |
           +--yes--> tools ----+      (patient_id injected from state: no cross-patient access)
           |
           no
           v
        verify --issues & revisions left--> agent (with feedback)
           |
           v
          END   (answer + evidence + flags)

verify is deterministic: valid citations, numeric grounding against tool output, no directive language.
"""
import time
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import MAX_AGENT_STEPS, MAX_REVISIONS, get_llm
from data_store import set_store
from observability import run_config
from safety import EMERGENCY_MESSAGE, directive_language, is_emergency, unsupported_numbers
from tools import CITE_RE, EVIDENCE_RE, TOOLS, TOOLS_BY_NAME

NOT_FOUND = "NOT FOUND IN RECORD"

SYSTEM_PROMPT = """You are a record-review assistant for clinicians. You answer questions about ONE patient's \
longitudinal record (synthetic demo data). The patient is fixed; your tools are already scoped to that patient.

Rules:
1. Look facts up with tools; never answer from memory. Use get_lab_trend for values/trends, get_medication_history \
for drugs/doses/dates, get_conditions for the problem list, search_notes for reasons and context, and \
search_guidelines only for general reference. For "how did X change after Y" questions, combine a trend with the \
medication or note timeline.
2. Cite every factual statement with evidence ids exactly as returned by tools, e.g. [LAB:eGFR:2026-01-22]. Never invent ids.
3. If the record does not contain the answer, reply starting exactly with "NOT FOUND IN RECORD:" and state what you \
checked. Do not guess, and never use another patient's data.
4. Do NOT diagnose, recommend treatments, or advise doses. Summarize what the record and reference snippets say, and \
state that clinical judgment is required when the question asks for a decision.
5. Be concise (under 150 words). Write dates in ISO format (YYYY-MM-DD) and keep units."""


class AgentState(TypedDict, total=False):
    question: str
    patient_id: str
    messages: Annotated[list, add_messages]
    steps: int
    revisions: int
    needs_revision: bool
    evidence: dict      # evidence id -> text
    tool_text: str      # all tool output (used for numeric grounding)
    tools_called: list
    trace: list         # [{"tool":..., "args":...}]
    answer: str
    abstained: bool
    status: str         # ok | unverified | emergency
    flags: list


def msg_text(m) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def build_agent(llm=None, store=None):
    llm = llm or get_llm()
    if store is not None:
        set_store(store)
    llm_tools = llm.bind_tools(TOOLS)

    def guard(state: AgentState) -> AgentState:
        if is_emergency(state["question"]):
            return {"answer": EMERGENCY_MESSAGE, "status": "emergency", "flags": ["emergency"], "abstained": False,
                    "evidence": {}, "tools_called": [], "trace": [], "steps": 0, "revisions": 0}
        return {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(state["question"])],
                "steps": 0, "revisions": 0, "evidence": {}, "tool_text": "", "tools_called": [], "trace": [],
                "flags": [], "status": "ok"}

    def agent_node(state: AgentState) -> AgentState:
        steps = state.get("steps", 0)
        if steps >= MAX_AGENT_STEPS:  # budget exhausted: force a final answer without tools
            ai = llm.invoke(state["messages"] + [HumanMessage(
                "Tool budget exhausted. Answer now using only the evidence already gathered.")])
        else:
            ai = llm_tools.invoke(state["messages"])
        return {"messages": [ai], "steps": steps + 1}

    def tools_node(state: AgentState) -> AgentState:
        evidence, called = dict(state["evidence"]), list(state["tools_called"])
        trace, tool_text, out = list(state["trace"]), state["tool_text"], []
        for call in state["messages"][-1].tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                content = f"Unknown tool '{call['name']}'."
            else:
                try:  # patient_id is always taken from state, overriding anything the model sent
                    content = str(tool.invoke({**call["args"], "patient_id": state["patient_id"]}))
                except Exception as e:
                    content = f"Tool error: {e}"
            called.append(call["name"])
            trace.append({"tool": call["name"], "args": call["args"]})
            for eid, text in EVIDENCE_RE.findall(content):
                evidence[eid] = text
            tool_text += "\n" + content
            out.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": out, "evidence": evidence, "tools_called": called, "trace": trace, "tool_text": tool_text}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) and state["steps"] <= MAX_AGENT_STEPS else "verify"

    def verify(state: AgentState) -> AgentState:
        answer = msg_text(state["messages"][-1]).strip()
        abstained = answer.upper().startswith(NOT_FOUND)
        issues = []
        if not abstained:
            cited = CITE_RE.findall(answer)
            bad = sorted({c for c in cited if c not in state["evidence"]})
            if not cited:
                issues.append("the answer has no evidence citations")
            if bad:
                issues.append(f"cited ids were never returned by a tool: {bad}")
            nums = unsupported_numbers(answer, state["tool_text"])
            if nums:
                issues.append(f"numbers not found in tool output: {nums}")
        if directive_language(answer):
            issues.append("it contains treatment/dose recommendation language")
        flags = list(state.get("flags", [])) + issues
        if issues and state["revisions"] < MAX_REVISIONS:
            feedback = ("Your draft failed verification: " + "; ".join(issues) + ". Revise using only evidence returned by "
                        "tools (call tools again if needed), cite valid ids, and avoid recommendations. If the record "
                        f"lacks the answer, start with '{NOT_FOUND}:'.")
            return {"messages": [HumanMessage(feedback)], "revisions": state["revisions"] + 1,
                    "needs_revision": True, "flags": flags}
        if issues:
            answer += "\n\n⚠ Verification warnings: " + "; ".join(issues) + "."
        return {"answer": answer, "abstained": abstained, "needs_revision": False, "flags": flags,
                "status": "unverified" if issues else "ok"}

    g = StateGraph(AgentState)
    g.add_node("guard", guard)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("verify", verify)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", lambda s: END if s.get("status") == "emergency" else "agent", {END: END, "agent": "agent"})
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "verify": "verify"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("verify", lambda s: "agent" if s.get("needs_revision") else END, {"agent": "agent", END: END})
    return g.compile()


def run_agent(graph, question: str, patient_id: str, run_id=None, tags=None) -> dict:
    run_id = run_id or uuid.uuid4()
    t0 = time.time()
    out = graph.invoke({"question": question, "patient_id": patient_id},
                       config=run_config(run_id, tags, {"patient_id": patient_id}))
    return {
        "run_id": run_id, "question": question, "patient_id": patient_id,
        "answer": out["answer"], "status": out["status"], "abstained": out.get("abstained", False),
        "evidence": out.get("evidence", {}), "tools_called": out.get("tools_called", []),
        "trace": out.get("trace", []), "flags": out.get("flags", []),
        "llm_steps": out.get("steps", 0), "revisions": out.get("revisions", 0),
        "latency_s": round(time.time() - t0, 2),
    }
