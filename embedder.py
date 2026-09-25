"""Embeds text (and image captions) into a shared vector space.

Everything — raw text chunks, table renderings, and image captions — goes
through this one embedder, so retrieval never has to reconcile two different
vector spaces (see README for why this beats a separate CLIP index).
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from backend.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]