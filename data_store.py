"""Loads the synthetic record and builds vector indexes for notes (patient-filtered) and reference snippets."""
import json
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from config import DATA_DIR, get_embeddings


def load_patients() -> dict:
    return json.loads((DATA_DIR / "patients.json").read_text(encoding="utf-8"))


@dataclass
class Store:
    patients: dict
    note_index: InMemoryVectorStore
    guideline_index: InMemoryVectorStore


_STORE: Store | None = None


def build_store(embeddings=None) -> Store:
    embeddings = embeddings or get_embeddings()
    notes = json.loads((DATA_DIR / "notes.json").read_text(encoding="utf-8"))
    guides = json.loads((DATA_DIR / "guidelines.json").read_text(encoding="utf-8"))
    note_docs = [
        Document(page_content=n["text"],
                 metadata={"id": f"NOTE:{n['date']}", "patient_id": n["patient_id"], "date": n["date"], "type": n["type"]})
        for n in notes
    ]
    guide_docs = [Document(page_content=f"{g['title']}: {g['text']}", metadata={"id": f"GL:{g['id']}"}) for g in guides]
    return Store(
        patients=load_patients(),
        note_index=InMemoryVectorStore.from_documents(note_docs, embeddings),
        guideline_index=InMemoryVectorStore.from_documents(guide_docs, embeddings),
    )


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = build_store()
    return _STORE


def set_store(store: Store | None) -> None:
    global _STORE
    _STORE = store
