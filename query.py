from fastapi import APIRouter, HTTPException

from backend.models.schemas import QueryRequest, QueryResponse
from backend.qa.answer_engine import answer_question
from backend.retrieval.vector_store import has_document

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    if not has_document(request.doc_id):
        raise HTTPException(
            status_code=404,
            detail=f"No indexed document found for doc_id '{request.doc_id}'. Upload it first.",
        )

    answer, citations = answer_question(
        doc_id=request.doc_id, question=request.question, top_k=request.top_k
    )
    return QueryResponse(answer=answer, citations=citations)