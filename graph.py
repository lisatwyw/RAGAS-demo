"""LangGraph layer: a small corrective-RAG graph.

START -> retrieve -> grade --relevant--> generate -> END
                       |
                       +--not relevant (retries left)--> rewrite -> retrieve ...
                       +--not relevant (no retries)----> generate (says "I don't know")
"""
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag import build_answer_chain, build_retriever, format_docs, get_llm

MAX_REWRITES = 1
NO_ANSWER = "I don't know based on the available documentation."


class RAGState(TypedDict, total=False):
    question: str
    query: str            # possibly rewritten search query
    docs: list[Document]
    relevant: bool
    retries: int
    answer: str


class Grade(BaseModel):
    relevant: bool = Field(description="True if the context contains the answer to the question")


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Decide whether the context contains enough information to answer the question."),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite the question as a short keyword-style search query for a product documentation index. "
                   "Return only the query."),
        ("human", "{question}"),
    ]
)


def build_graph(llm=None, retriever=None):
    llm = llm or get_llm()
    retriever = retriever or build_retriever()
    answer_chain = build_answer_chain(llm)
    grader = GRADE_PROMPT | llm.with_structured_output(Grade)
    rewriter = REWRITE_PROMPT | llm | StrOutputParser()

    def retrieve(state: RAGState) -> RAGState:
        return {"docs": retriever.invoke(state.get("query") or state["question"])}

    def grade(state: RAGState) -> RAGState:
        result = grader.invoke({"question": state["question"], "context": format_docs(state["docs"])})
        return {"relevant": result.relevant}

    def rewrite(state: RAGState) -> RAGState:
        return {
            "query": rewriter.invoke({"question": state["question"]}),
            "retries": state.get("retries", 0) + 1,
        }

    def generate(state: RAGState) -> RAGState:
        if not state.get("relevant"):
            return {"answer": NO_ANSWER}
        return {"answer": answer_chain.invoke({"context": format_docs(state["docs"]), "question": state["question"]})}

    def route_after_grade(state: RAGState) -> str:
        if state["relevant"] or state.get("retries", 0) >= MAX_REWRITES:
            return "generate"
        return "rewrite"

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()
