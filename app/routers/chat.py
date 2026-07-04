import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Meeting, Chunk
from app.schemas.chat import ChatRequest, ChatResponse, Citation, ChatTurn
from app.services.embeddings import get_embedding
from app.services.faiss_store import get_faiss_store
from app.config import settings

router = APIRouter(tags=["Chat"])

TOP_K = 5
GROQ_MODEL = "llama-3.3-70b-versatile"


def _rewrite_query_with_history(query: str, history: list[ChatTurn]) -> str:
    """Turns a follow-up question into a standalone one using recent
    history. Falls back to the raw query if there's no history or no
    Groq key configured."""
    if not history or not settings.groq_api_key:
        return query

    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)

    recent = history[-3:]
    transcript = "\n".join(f"{t.role}: {t.content}" for t in recent)
    prompt = (
        "Given this recent conversation, rewrite the final user question as a "
        "standalone question that doesn't depend on prior context. "
        "Return ONLY the rewritten question, nothing else.\n\n"
        f"{transcript}\nuser: {query}"
    )
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten or query
    except Exception:
        return query


def _generate_answer(query: str, chunks: list[Chunk]) -> dict:
    """Calls Groq with retrieved chunks, constraining citations to
    chunk ids that were actually retrieved — the LLM cannot invent one."""
    allowed_ids = [c.id for c in chunks]

    if not settings.groq_api_key:
        return {
            "answer": "(GROQ_API_KEY not set — configure it in .env to get real answers.) "
                      f"Retrieved {len(chunks)} relevant chunk(s) for: '{query}'",
            "cited_chunk_ids": allowed_ids[:1],
        }

    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)

    context_blocks = [
        f"[chunk_id={c.id}] (meeting_id={c.meeting_id})\n{c.chunk_text}"
        for c in chunks
    ]
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You answer questions about company meetings using ONLY the provided "
        "meeting excerpts. Cite only chunk_id values that appear in the excerpts "
        "below — never invent a chunk_id. If the excerpts don't contain the "
        "answer, say so explicitly. Respond ONLY with JSON matching this schema: "
        '{"answer": string, "cited_chunk_ids": [int, ...]}. '
        f"Allowed chunk_id values: {allowed_ids}"
    )

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {query}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
        )
        parsed = json.loads(resp.choices[0].message.content)
        cited = [cid for cid in parsed.get("cited_chunk_ids", []) if cid in allowed_ids]
        return {"answer": parsed.get("answer", ""), "cited_chunk_ids": cited}
    except Exception as e:
        return {"answer": f"LLM call failed: {e}", "cited_chunk_ids": []}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    # --- Step 11: retrieval ---
    standalone_query = _rewrite_query_with_history(req.query, req.chat_history)

    query_vector = get_embedding(standalone_query)
    store = get_faiss_store()
    results = store.search(query_vector, top_k=TOP_K)  # [(chunk_id, distance), ...]

    if not results:
        return ChatResponse(
            answer="No meetings have been indexed yet.",
            citations=[],
            retrieved_chunk_ids=[],
        )

    retrieved_ids = [cid for cid, _ in results]
    chunks_query = db.query(Chunk).filter(Chunk.id.in_(retrieved_ids))
    if req.meeting_id is not None:
        chunks_query = chunks_query.filter(Chunk.meeting_id == req.meeting_id)
    chunks = chunks_query.all()

    if not chunks:
        return ChatResponse(
            answer="No relevant content found for this meeting filter.",
            citations=[],
            retrieved_chunk_ids=retrieved_ids,
        )

    # --- Step 12: generation ---
    result = _generate_answer(standalone_query, chunks)

    chunks_by_id = {c.id: c for c in chunks}
    meetings_by_id = {}
    citations = []
    for cid in result["cited_chunk_ids"]:
        c = chunks_by_id.get(cid)
        if not c:
            continue
        if c.meeting_id not in meetings_by_id:
            meetings_by_id[c.meeting_id] = db.query(Meeting).get(c.meeting_id)
        m = meetings_by_id[c.meeting_id]
        citations.append(Citation(
            chunk_id=c.id,
            meeting_id=c.meeting_id,
            meeting_title=m.title if m else "Unknown",
            source_text=c.chunk_text[:300],
        ))

    return ChatResponse(
        answer=result["answer"],
        citations=citations,
        retrieved_chunk_ids=retrieved_ids,  # logged for eval even if not all cited
    )