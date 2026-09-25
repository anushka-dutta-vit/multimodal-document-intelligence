"""Retrieval-augmented answer generation with citations.

Retrieves top-k chunks (text, table, and image captions all live in the same
index — see vector_store.py), builds a labeled context block, and asks
Claude to answer strictly from that context, citing which chunk(s) it used.
"""
import re

from anthropic import Anthropic

from backend.config import settings
from backend.embeddings.embedder import embed_query
from backend.models.schemas import Chunk, Citation
from backend.retrieval.vector_store import query as vector_query

_client = Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """You are a document Q&A assistant. You are given numbered \
context passages extracted from a document (some are raw text, some are \
table data, some are captions of charts/images). Answer the user's question \
using ONLY the information in these passages.

Rules:
- If the answer isn't in the context, say so plainly — do not guess.
- Be concise and direct.
- At the end of your answer, on a new line, output which passage numbers \
you actually used, in this exact format: CITED: [1, 3]
"""


def _build_context(chunks: list[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        label = {"text": "Text", "table": "Table", "image": "Image caption"}[c.type]
        lines.append(f"[{i}] (page {c.page}, {label}): {c.content}")
    return "\n\n".join(lines)


def _parse_cited_indices(answer: str) -> list[int]:
    match = re.search(r"CITED:\s*\[([\d,\s]*)\]", answer)
    if not match:
        return []
    return [int(n) for n in re.findall(r"\d+", match.group(1))]


def answer_question(doc_id: str, question: str, top_k: int | None = None) -> tuple[str, list[Citation]]:
    k = top_k or settings.top_k
    q_embedding = embed_query(question)
    chunks = vector_query(doc_id=doc_id, query_embedding=q_embedding, top_k=k)

    if not chunks:
        return "I couldn't find any indexed content for this document.", []

    context = _build_context(chunks)

    response = _client.messages.create(
        model=settings.claude_model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    raw_answer = " ".join(b.text for b in response.content if b.type == "text").strip()
    cited_indices = _parse_cited_indices(raw_answer)
    clean_answer = re.sub(r"\n?CITED:\s*\[[\d,\s]*\]\s*$", "", raw_answer).strip()

    citations = []
    for idx in cited_indices:
        if 1 <= idx <= len(chunks):
            c = chunks[idx - 1]
            preview = c.content[:140] + ("..." if len(c.content) > 140 else "")
            citations.append(
                Citation(page=c.page, type=c.type, bbox=c.bbox, preview=preview)
            )

    # Fall back to showing all retrieved sources if the model didn't cite explicitly
    if not citations:
        for c in chunks[:3]:
            preview = c.content[:140] + ("..." if len(c.content) > 140 else "")
            citations.append(
                Citation(page=c.page, type=c.type, bbox=c.bbox, preview=preview)
            )

    return clean_answer, citations