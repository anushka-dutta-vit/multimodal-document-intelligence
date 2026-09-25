from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import query, upload

app = FastAPI(
    title="Multimodal Document Intelligence",
    description="Upload a PDF, ask questions, get grounded answers with citations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(query.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}