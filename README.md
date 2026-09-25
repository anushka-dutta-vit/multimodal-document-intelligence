Multimodal Document Intelligence

Upload a PDF (digital or scanned, with text, tables, and charts/images) and ask natural-language questions about it. The system retrieves the most relevant text chunks and image regions, feeds them to Claude with vision, and returns a grounded answer with citations back to the exact page and region it came from.

Why this project

Most "RAG chatbot" projects only handle text. Real documents (10-Ks, research papers, scanned forms, slide decks) mix text, tables, and charts on the same page, and the answer to a question is often in a chart, not in a paragraph. This project builds a pipeline that treats all three as first-class citizens.

Architecture
                ┌─────────────┐
   PDF/Image →  │  Ingestion  │  → per-page text, tables, image crops
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  Chunking    │  → text chunks (page/bbox metadata)
                │  + Captioning│  → image crops → Claude vision → text caption
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  Embedding   │  → sentence-transformers (text + captions)
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │ Vector Store │  → ChromaDB, metadata = page, bbox, type
                └──────┬──────┘
                       │
   Question  →  ┌──────▼──────┐
                │  Retrieval   │  → top-k text + image chunks
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │ Answer Engine│  → Claude (context + cropped images) → answer
                └──────┬──────┘      with citations [p.3, region A]
                       │
                ┌──────▼──────┐
                │   Frontend   │  → renders PDF page, highlights cited bbox
                └─────────────┘
Key design decisions (talk about these in interviews)
Why ChromaDB, not pgvector/Pinecone: file-based, zero infra to run locally, easy to swap for pgvector later without changing the retrieval interface (vector_store.py is a thin abstraction on purpose).
Why caption images instead of CLIP embeddings: CLIP embeddings are hard to explain/debug and don't capture fine-grained chart semantics well. Generating a text caption of each chart/table with a vision model, then embedding that caption with the same text embedder as everything else, keeps retrieval in a single embedding space and makes results explainable (you can literally read what the system "thinks" an image contains).
Why hybrid text+table extraction: pdfplumber table detection runs first; anything it can't parse as a clean table falls back to being treated as an image region and gets captioned instead of silently dropped.
OCR fallback: PDFs with no extractable text layer (scans) are rasterized and run through Tesseract; the ingestion layer picks native-text vs. OCR per page automatically.
Stack
Backend: FastAPI, Python 3.11
PDF parsing: pdfplumber (text + tables), pdf2image + pytesseract (OCR)
Embeddings: sentence-transformers (all-MiniLM-L6-v2)
Vector store: ChromaDB (persisted to disk)
Reasoning/vision/captioning: Claude API (claude-sonnet-4-6)
Frontend: single-file HTML/JS, PDF.js for rendering + bbox overlays
Setup
bash
git clone <your-repo-url>
cd multimodal-doc-intelligence
cp .env.example .env        # add your ANTHROPIC_API_KEY
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# system dependency for OCR + PDF rasterization
# macOS: brew install tesseract poppler
# Ubuntu: sudo apt-get install tesseract-ocr poppler-utils

uvicorn backend.main:app --reload

Or with Docker:

bash
docker compose up --build

Then open frontend/index.html in a browser (or serve it via any static server) and point it at http://localhost:8000.

API
Endpoint	Method	Description
/upload	POST	Upload a PDF, triggers ingestion pipeline
/documents	GET	List ingested documents
/query	POST	Ask a question, get an answer + citations
/health	GET	Health check
Example
bash
curl -X POST http://localhost:8000/upload -F "file=@report.pdf"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "report", "question": "What was Q3 revenue?"}'

Response:

json
{
  "answer": "Q3 revenue was $4.2M, up 12% QoQ.",
  "citations": [
    {"page": 12, "type": "image", "bbox": [72, 340, 480, 610], "preview": "bar chart titled 'Quarterly Revenue'"}
  ]
}
Roadmap / things to extend for a deeper resume story
Swap ChromaDB for pgvector + Postgres, add a migration path
Add re-ranking (cross-encoder) after initial retrieval
Add an eval harness: a small QA dataset per doc + retrieval precision@k
Batch ingestion + background job queue (Celery/RQ) for large PDFs
Auth + per-user document isolation