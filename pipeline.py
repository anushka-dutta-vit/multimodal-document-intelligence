"""Orchestrates the full ingestion flow for one uploaded PDF:

parse -> chunk text / flatten tables / caption images -> embed -> index
"""
import uuid

from backend.config import settings
from backend.embeddings.captioner import caption_image, table_rows_to_text
from backend.embeddings.embedder import embed_texts
from backend.ingestion.chunker import chunk_text
from backend.ingestion.pdf_parser import parse_pdf
from backend.models.schemas import Chunk
from backend.retrieval.vector_store import add_chunks


def ingest_document(pdf_path: str, doc_id: str) -> tuple[int, int]:
    """Returns (page_count, chunks_indexed)."""
    pages = parse_pdf(pdf_path)
    chunks: list[Chunk] = []

    for page in pages:
        # 1. Text chunks
        for text_chunk in chunk_text(
            page.text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        ):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    page=page.page,
                    type="text",
                    content=text_chunk,
                )
            )

        # 2. Tables -> flattened text (no vision call needed, already structured)
        for table in page.tables:
            flattened = table_rows_to_text(table.rows)
            if not flattened.strip():
                continue
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    page=page.page,
                    type="table",
                    content=flattened,
                    bbox=table.bbox,
                )
            )

        # 3. Images -> Claude vision caption
        for image_region in page.images:
            caption = caption_image(image_region.image)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    page=page.page,
                    type="image",
                    content=caption,
                    bbox=image_region.bbox,
                )
            )

    if chunks:
        embeddings = embed_texts([c.content for c in chunks])
        add_chunks(chunks, embeddings)

    return len(pages), len(chunks)