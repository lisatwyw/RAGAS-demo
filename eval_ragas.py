"""RAGAS 0.4.x evaluation of the LangGraph pipeline, with scores pushed back to LangSmith as feedback."""
import asyncio
import inspect
import math
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

import pandas as pd  # noqa: E402
from langchain_core.tracers.langchain import wait_for_all_tracers  # noqa: E402
from langsmith import Client  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from ragas import SingleTurnSample  # noqa: E402
from ragas.embeddings.base import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from graph import build_graph  # noqa: E402
from rag import EMBED_MODEL  # noqa: E402

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

# (question, reference answer). The last one is deliberately unanswerable from the docs.
EVAL_SET = [
    ("How much does the Pro plan cost and how much storage does it include?",
     "The Pro plan costs $8 per month and includes 1 TB of storage."),
    ("How long are deleted files kept in the trash on the Free plan?",
     "Deleted files stay in the trash for 30 days on the Free plan."),
    ("Which plans support password-protected share links?",
     "Password-protected links are available on the Pro and Team plans only."),
    ("Is SAML single sign-on supported?",
     "Yes, SAML SSO is available on the Team plan only."),
    ("Is there a Linux desktop app?",
     "The documentation does not say whether a Linux desktop app exists."),
]


def build_metrics():
    client = AsyncOpenAI(max_retries=5)
    llm = llm_factory(JUDGE_MODEL, client=client, max_tokens=4096)
    emb = embedding_factory("openai", model=EMBED_MODEL, client=client)
    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecisionWithReference(llm=llm),
        ContextRecall(llm=llm),
    ]


def run_pipeline(graph) -> list[dict]:
    rows = []
    for question, reference in EVAL_SET:
        run_id = uuid.uuid4()  # lets us attach scores to this exact LangSmith trace later
        out = graph.invoke(
            {"question": question},
            config={"run_id": run_id, "run_name": "rag_graph", "tags": ["ragas-eval"]},
        )
        rows.append({
            "run_id": run_id,
            "sample": SingleTurnSample(
                user_input=question,
                retrieved_contexts=[d.page_content for d in out["docs"]],
                response=out["answer"],
                reference=reference,
            ),
        })
    return rows


async def score_all(rows, metrics) -> None:
    sem = asyncio.Semaphore(4)

    async def score_one(row, metric):
        # each metric accepts different fields, so pass only what its ascore() asks for
        wanted = inspect.signature(metric.ascore).parameters
        inputs = {f: getattr(row["sample"], f) for f in wanted}
        async with sem:
            try:
                row[metric.name] = (await metric.ascore(**inputs)).value
            except Exception as e:  # one failed judge call shouldn't sink the run
                print(f"{metric.name} failed on {row['sample'].user_input!r}: {e}")
                row[metric.name] = math.nan

    await asyncio.gather(*(score_one(r, m) for r in rows for m in metrics))


def push_feedback(rows, metric_names) -> None:
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true" or not os.getenv("LANGSMITH_API_KEY"):
        print("LangSmith tracing not enabled; skipping feedback upload.")
        return
    wait_for_all_tracers()  # make sure the traces exist before attaching feedback
    client = Client()
    for row in rows:
        for name in metric_names:
            if not math.isnan(row[name]):
                client.create_feedback(run_id=row["run_id"], key=f"ragas_{name}", score=row[name])
    print("Scores attached to LangSmith traces as feedback (keys: ragas_*).")


def main() -> None:
    graph = build_graph()
    metrics = build_metrics()
    names = [m.name for m in metrics]

    rows = run_pipeline(graph)
    asyncio.run(score_all(rows, metrics))

    df = pd.DataFrame([{"question": r["sample"].user_input[:50], **{n: r[n] for n in names}} for r in rows])
    pd.set_option("display.width", 200)
    print("\n", df.round(3).to_string(index=False))
    print("\nMeans:\n", df[names].mean().round(3).to_string())
    df.to_csv("ragas_scores.csv", index=False)
    push_feedback(rows, names)


if __name__ == "__main__":
    main()
