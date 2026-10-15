# measures similarity between a question and the indexed chunks, and returns the top-k most similar chunks, with metadata

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from src import config


@dataclass
class Retrieved:
    id: str
    text: str
    section_path: str
    type: str
    similarity: float   # 1 - cosine distance; higher is more similar

@lru_cache(maxsize=1)
def open_collection():
    """Open the persisted collection (cached) and make sure it matches config.py."""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        col = client.get_collection(config.COLLECTION_NAME)
    except Exception:
        raise SystemExit(
            f"Collection '{config.COLLECTION_NAME}' not found in {config.CHROMA_DIR}. "
            "Run `python -m src.index` first."
        )
    stored = col.metadata or {}
    if (
        stored.get("embedding_model") != config.EMBEDDING_MODEL
        or int(stored.get("embedding_dim", -1)) != config.EMBEDDING_DIM
    ):
        raise SystemExit(
            "Index and config use different embedding settings "
            f"({stored.get('embedding_model')}/{stored.get('embedding_dim')} vs "
            f"{config.EMBEDDING_MODEL}/{config.EMBEDDING_DIM}). Re-run `python -m src.index --rebuild`."
        )
    return col


def retrieve(
    question: str,
    k: int = config.TOP_K,
    include_endnotes: bool = False,
    collection=None,
    embed_query_fn: Callable[[str], list[float]] | None = None,
) -> list[Retrieved]:
    if collection is None:
        collection = open_collection()
    if embed_query_fn is None:
        from src.embeddings import embed_query as embed_query_fn

    where = None if include_endnotes else {"type": {"$ne": "endnote"}}
    result = collection.query(
        query_embeddings=[embed_query_fn(question)],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        Retrieved(
            id=cid,
            text=doc,
            section_path=meta.get("section_path", ""),
            type=meta.get("type", "text"),
            similarity=round(1.0 - dist, 4),
        )
        for cid, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
