"""Thin wrapper around ChromaDB.

Kept deliberately narrow (add/query only) so swapping in pgvector or another
store later means changing this one file, not the ingestion or query code
that calls it.
"""
import json

import chromadb

from backend.config import settings
from backend.models.schemas import Chunk

_client = chromadb.PersistentClient(path=settings.chroma_dir)
_collection = _client.get_or_create_collection(name="doc_chunks")


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    if not chunks:
        return

    _collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.content for c in chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "page": c.page,
                "type": c.type,
                "bbox": json.dumps(c.bbox) if c.bbox else "",
            }
            for c in chunks
        ],
    )


def query(doc_id: str, query_embedding: list[float], top_k: int) -> list[Chunk]:
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"doc_id": doc_id},
    )

    chunks: list[Chunk] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for id_, content, meta in zip(ids, docs, metas):
        bbox = json.loads(meta["bbox"]) if meta.get("bbox") else None
        chunks.append(
            Chunk(
                id=id_,
                doc_id=meta["doc_id"],
                page=meta["page"],
                type=meta["type"],
                content=content,
                bbox=bbox,
            )
        )

    return chunks


def document_page_count(doc_id: str) -> int:
    results = _collection.get(where={"doc_id": doc_id})
    metas = results.get("metadatas", [])
    pages = {m["page"] for m in metas} if metas else set()
    return max(pages) if pages else 0


def has_document(doc_id: str) -> bool:
    results = _collection.get(where={"doc_id": doc_id}, limit=1)
    return len(results.get("ids", [])) > 0