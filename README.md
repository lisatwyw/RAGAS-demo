---
title: Report
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Corrective RAG with LangChain, LangGraph, LangSmith, RAGAS
---

# RAG stack demo: LangChain + LangGraph + LangSmith + RAGAS

A tiny support-docs Q&A bot ("Acme Cloud Storage") that shows how the four tools fit together.

| Tool | Role here | Where |
|---|---|---|
| **LangChain** | Loaders, splitter, embeddings, vector store, prompt/LCEL chain, structured output | `rag.py` |
| **LangGraph** | Corrective-RAG control flow: retrieve → grade → (rewrite → retrieve) → generate | `graph.py` |
| **LangSmith** | Tracing every node/LLM call (env vars only) + RAGAS scores attached to traces as feedback | `.env`, `eval_ragas.py` |
| **RAGAS** | Faithfulness, answer relevancy, context precision, context recall | `eval_ragas.py` |
| **Gradio** | Web UI for Hugging Face Spaces | `app.py` |

```
START → retrieve → grade ─relevant──────────────→ generate → END
           ↑          └─not relevant, retries left→ rewrite ─┘
                      └─not relevant, none left ──→ generate ("I don't know")
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt gradio
cp .env.example .env        # add OPENAI_API_KEY and LANGSMITH_API_KEY

python main.py "Which plans have password-protected links?"   # CLI
python app.py                                                  # web UI at http://127.0.0.1:7860
python main.py --diagram                                       # graph as Mermaid
python eval_ragas.py                                           # RAGAS scores -> ragas_scores.csv
```

## Deploy to Hugging Face Spaces (free)

1. Create a Space at https://huggingface.co/new-space → SDK **Gradio**, hardware **CPU Basic**.
2. In the Space: **Settings → Variables and secrets → New secret**, add:
   - `OPENAI_API_KEY` (required)
   - `LANGSMITH_API_KEY`, plus variables `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=rag-stack-demo` (optional)
   - Optional variables: `PER_IP_PER_HOUR` (default 15), `GLOBAL_PER_DAY` (default 300), `APP_USER` + `APP_PASSWORD` to require a login
3. Push the files to the Space (the `.gitignore` keeps `.env` out):
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   # copy in: README.md app.py rag.py graph.py main.py eval_ragas.py requirements.txt .gitignore .env.example kb/
   git add . && git commit -m "Deploy RAG demo" && git push
   ```
4. Wait for the build. The Space installs `gradio` itself from `sdk_version`, so it is intentionally not in `requirements.txt`.

Notes for Spaces:
- Every question uses **your** OpenAI key. The rate limits in `app.py` cap spend; also set a monthly limit on the key in the OpenAI dashboard.
- Free CPU Spaces sleep after inactivity and rebuild the in-memory index (3 small docs) on wake-up.
- The per-IP limit is in memory, so it resets when the Space restarts. It's a guard rail, not a security boundary.
- If the build shows a "configuration error", check the YAML block at the top of this file.
- Production traces are tagged `hf-space` / `env=prod` in LangSmith, so you can filter them apart from eval runs.

## Notes

- **Cost:** ~5 eval questions × (graph calls + ~10 judge calls each) on `gpt-4o-mini`: a few cents.
- **LangSmith is optional:** set `LANGSMITH_TRACING=false` and everything still runs.
- **Pinned on purpose:** `ragas==0.4.3` and `langchain-community<0.4`. RAGAS changed its API twice recently
  (`evaluate()` is deprecated in 0.4; metrics now live in `ragas.metrics.collections` and are scored with `ascore()`).
- **Unanswerable question** ("Linux desktop app?") shows the "I don't know" path. RAGAS scores on refusals are noisy.
- **Judge noise:** with 5 samples, one judge verdict moves a mean by 0.2. Use 50-100 samples before trusting a threshold.
- **Swapping the LLM:** change `get_llm()` in `rag.py`. RAGAS answer relevancy also needs embeddings, which Anthropic doesn't offer.

## Extending

- Replace `InMemoryVectorStore` with Chroma/pgvector; swap `kb/*.md` for your docs.
- Add a `MemorySaver` checkpointer in `graph.py` for multi-turn chat.
- Use `langsmith.evaluate()` with a LangSmith dataset to run the same eval from the UI.
