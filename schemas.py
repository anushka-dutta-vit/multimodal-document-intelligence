"""Shared request/response models."""
from typing import Literal, Optional

from pydantic import BaseModel


class Chunk(BaseModel):
    """One retrievable unit: a text passage or a captioned image/table region."""

    id: str
    doc_id: str
    page: int
    type: Literal["text", "table", "image"]
    content: str  # raw text, or the generated caption for image/table
    bbox: Optional[list[float]] = None  # [x0, y0, x1, y1] in PDF points


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks_indexed: int


class QueryRequest(BaseModel):
    doc_id: str
    question: str
    top_k: Optional[int] = None


class Citation(BaseModel):
    page: int
    type: Literal["text", "table", "image"]
    bbox: Optional[list[float]] = None
    preview: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    pages: int