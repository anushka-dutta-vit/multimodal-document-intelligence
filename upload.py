import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import settings
from backend.ingestion.pipeline import ingest_document
from backend.models.schemas import UploadResponse

router = APIRouter()


def _slugify(filename: str) -> str:
    stem = Path(filename).stem
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-").lower()
    return slug or "document"


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = f"{_slugify(file.filename)}-{uuid.uuid4().hex[:8]}"
    dest_path = Path(settings.upload_dir) / f"{doc_id}.pdf"

    with dest_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    try:
        pages, chunks_indexed = ingest_document(str(dest_path), doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        pages=pages,
        chunks_indexed=chunks_indexed,
    )