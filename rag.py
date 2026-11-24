"""LangChain layer: load docs, split, embed, retrieve, and build the answer chain (LCEL).

Default provider is Google Gemini (free tier, no credit card). Set LLM_PROVIDER=openai to use OpenAI instead.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # must run before langchain imports so LANGSMITH_* tracing vars are picked up

from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.rate_limiters import InMemoryRateLimiter  # noqa: E402
from langchain_core.vectorstores import InMemoryVectorStore  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # "gemini" (free) or "openai"
IS_GEMINI = PROVIDER == "gemini"

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite" if IS_GEMINI else "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001" if IS_GEMINI else "text-embedding-3-small")
KB_DIR = Path(__file__).parent / "kb"

# Free-tier Gemini allows roughly 15 requests/min per model (Google changes this without notice).
# Stay a bit under it; the limiter makes LangChain wait instead of failing with 429s.
CHAT_RPM = float(os.getenv("CHAT_RPM", "12"))
_limiter = InMemoryRateLimiter(requests_per_second=CHAT_RPM / 60, max_bucket_size=1)

def google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
def has_api_key() -> bool:
    return bool(google_api_key()) if IS_GEMINI else bool(os.getenv("OPENAI_API_KEY"))
    
def get_llm():
    if IS_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=CHAT_MODEL, temperature=0, google_api_key=google_api_key(),
            max_retries=5, rate_limiter=_limiter,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=CHAT_MODEL, temperature=0)

def get_embeddings():
    if IS_GEMINI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=google_api_key())
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBED_MODEL)

def build_retriever(k: int = 3):
    """Markdown files -> chunks -> in-memory vector store -> retriever."""
    docs = [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in sorted(KB_DIR.glob("*.md"))
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    store = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    return store.as_retriever(search_kwargs={"k": k})

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('source')}] {d.page_content}" for d in docs)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Answer using ONLY the context. Be concise. "
         "If the context does not contain the answer, say you don't know."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

def build_answer_chain(llm):
    """prompt | llm | parser: a plain LangChain (LCEL) chain."""
    return ANSWER_PROMPT | llm | StrOutputParser()
