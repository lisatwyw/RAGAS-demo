"""Provider/model configuration. Default: Google Gemini free tier (no credit card). LLM_PROVIDER=openai for paid OpenAI."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before langchain imports so LANGSMITH_* tracing vars are picked up

from langchain_core.rate_limiters import InMemoryRateLimiter  # noqa: E402

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
IS_GEMINI = PROVIDER == "gemini"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite" if IS_GEMINI else "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001" if IS_GEMINI else "text-embedding-3-small")
DATA_DIR = Path(__file__).parent / "data"

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "6"))   # LLM calls per question before forcing an answer
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "1"))       # verifier-triggered rewrites

# Gemini free tier is ~15 requests/min per model (Google changes this without notice); stay under it.
CHAT_RPM = float(os.getenv("CHAT_RPM", "12"))
_limiter = InMemoryRateLimiter(requests_per_second=CHAT_RPM / 60, max_bucket_size=1)


def google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def has_api_key() -> bool:
    return bool(google_api_key()) if IS_GEMINI else bool(os.getenv("OPENAI_API_KEY"))


def get_llm():
    if IS_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0, google_api_key=google_api_key(),
                                      max_retries=5, rate_limiter=_limiter)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def get_embeddings():
    if IS_GEMINI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=google_api_key())
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBED_MODEL)
