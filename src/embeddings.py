# helper file for index.py ((ii) embeds the chunks and queries using Gemini embedding models)

from __future__ import annotations

from google.genai import types

from src import config
from src.chunking import Chunk
from src.gemini_client import gemini_retry, get_client


def format_document(chunk: Chunk) -> str:
    """The asymmetric 'document' format recommended for gemini-embedding-2 retrieval."""
    return f"title: {chunk.title} | text: {chunk.text}"


def format_query(question: str) -> str:
    """The matching 'query' format for question answering."""
    return f"task: question answering | query: {question.strip()}"


@gemini_retry
def _embed_texts(texts: list[str]) -> list[list[float]]:
    contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
    result = get_client().models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIM),
    )
    vectors = [list(e.values) for e in result.embeddings]
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings but got {len(vectors)}; "
            "inputs were probably aggregated into one vector."
        )
    if any(len(v) != config.EMBEDDING_DIM for v in vectors):
        raise RuntimeError(f"Unexpected embedding size (wanted {config.EMBEDDING_DIM}).")
    return vectors


def embed_documents(chunks: list[Chunk]) -> list[list[float]]:
    """Embed ONE batch of chunks (the caller decides batching and persists progress).
    gemini-embedding-2 L2-normalises truncated dimensions itself, so no manual
    normalisation is needed (that is only required for gemini-embedding-001)."""
    return _embed_texts([format_document(c) for c in chunks])


def embed_query(question: str) -> list[float]:
    """Embed one user question for similarity search."""
    return _embed_texts([format_query(question)])[0]
