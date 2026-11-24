"""LangChain layer: load docs, split, embed, retrieve, and build the answer chain (LCEL)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before langchain imports so LANGSMITH_* tracing vars are picked up

from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.vectorstores import InMemoryVectorStore  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
KB_DIR = Path(__file__).parent / "kb"


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def build_retriever(k: int = 3):
    """Markdown files -> chunks -> in-memory vector store -> retriever."""
    docs = [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in sorted(KB_DIR.glob("*.md"))
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))
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


def build_answer_chain(llm: ChatOpenAI):
    """prompt | llm | parser: a plain LangChain (LCEL) chain."""
    return ANSWER_PROMPT | llm | StrOutputParser()
