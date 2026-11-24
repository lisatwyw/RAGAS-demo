"""Ask one question. Traces appear in LangSmith automatically when LANGSMITH_TRACING=true."""
import sys

from graph import build_graph

if __name__ == "__main__":
    graph = build_graph()
    if "--diagram" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        sys.exit(0)
    question = " ".join(sys.argv[1:]) or "How much storage does the Pro plan include?"
    out = graph.invoke(
        {"question": question},
        config={"run_name": "rag_graph", "tags": ["demo"], "metadata": {"entrypoint": "main.py"}},
    )
    print(f"\nQ: {question}\nA: {out['answer']}")
    print("Sources:", sorted({d.metadata['source'] for d in out["docs"]}))
    print(f"Rewrites: {out.get('retries', 0)}")
