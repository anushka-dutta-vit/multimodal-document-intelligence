"""Splits page text into overlapping chunks for embedding/retrieval.

Chunking is page-aware (never merges text across pages) so that every chunk
can carry a single, precise page-number citation.
"""
import re


def split_into_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter; good enough for chunk boundaries.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str, chunk_size: int = 800, overlap: int = 150
) -> list[str]:
    """Greedy sentence-packing chunker: fills each chunk up to chunk_size
    characters, then starts the next chunk `overlap` characters back so
    context isn't lost at chunk boundaries."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the tail of the last one
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks