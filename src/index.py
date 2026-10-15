# full stage_1 (indexing) pipeline : Chunk -> Embed -> Index (in ChromaDB).

from __future__ import annotations

import argparse
import time

from src import config
from src.chunking import chunk_markdown, print_stats

import chromadb

from src.embeddings import embed_documents


def get_collection(client):
    """Create/open the collection, recording and validating the embedding settings."""
    meta = {"embedding_model": config.EMBEDDING_MODEL, "embedding_dim": config.EMBEDDING_DIM}
    try:  # chromadb >= 1.0
        col = client.get_or_create_collection(
            config.COLLECTION_NAME, metadata=meta, configuration={"hnsw": {"space": "cosine"}}
        )
    except TypeError:  # older chromadb without the `configuration` argument
        col = client.get_or_create_collection(
            config.COLLECTION_NAME, metadata={**meta, "hnsw:space": "cosine"}
        )
    stored = col.metadata or {}
    if (
        stored.get("embedding_model") != config.EMBEDDING_MODEL
        or int(stored.get("embedding_dim", -1)) != config.EMBEDDING_DIM
    ):
        raise SystemExit(
            f"Collection was built with {stored.get('embedding_model')} / {stored.get('embedding_dim')} "
            f"but config says {config.EMBEDDING_MODEL} / {config.EMBEDDING_DIM}. "
            "Embedding spaces are incompatible. Re-run with --rebuild."
        )
    return col


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="chunk and print stats only")
    ap.add_argument("--rebuild", action="store_true", help="delete the collection first")
    ap.add_argument("--limit", type=int, help="embed at most N new chunks (smoke test)")
    args = ap.parse_args()

    # ---- 1. chunk ------------------------------------------------------------
    chunks = chunk_markdown(config.DATA_PATH)
    print_stats(chunks)
    if args.dry_run:
        print("Dry run: nothing embedded or stored.")
        return

    # Heavy imports happen only now, so --dry-run works with no dependencies beyond the stdlib.
  

    # ---- 2. open the store -----------------------------------------------------
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if args.rebuild:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            print("Existing collection deleted.")
        except Exception:  # collection didn't exist
            pass
    col = get_collection(client)

    existing = set(col.get(include=[])["ids"])
    wanted = {c.id for c in chunks}
    stale = existing - wanted
    if stale:
        print(f"Warning: {len(stale)} stored chunks no longer match the chunker output "
              "(chunking changed?). Use --rebuild for a clean index.")
    todo = [c for c in chunks if c.id not in existing]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(existing & wanted)} chunks already stored, {len(todo)} to embed.")

    # ---- 3. embed + store, batch by batch (progress is saved as we go) -----------
    bs = config.EMBED_BATCH_SIZE
    for start in range(0, len(todo), bs):
        batch = todo[start : start + bs]
        vectors = embed_documents(batch)
        col.upsert(
            ids=[c.id for c in batch],
            embeddings=vectors,
            documents=[c.text for c in batch],     # the raw chunk text the LLM will read later
            metadatas=[c.metadata for c in batch],
        )
        print(f"  stored {start + len(batch)}/{len(todo)}")
        time.sleep(config.PAUSE_BETWEEN_BATCHES_S)

    print(f"Done. Collection '{config.COLLECTION_NAME}' now holds {col.count()} chunks "
          f"({config.EMBEDDING_MODEL}, {config.EMBEDDING_DIM} dims) in {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
