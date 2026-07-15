import json
import re
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Meeting, Chunk
from app.schemas.chat import ChatRequest, ChatResponse, Citation, ChatTurn
from app.services.embeddings import get_embedding
from app.services.faiss_store import get_faiss_store
from app.config import settings

router = APIRouter(tags=["Chat"])

TOP_K = 5
# When a meeting_id/project_id filter is active, FAISS's top-k search
# still happens globally FIRST, then results get filtered down — so a
# small project's relevant chunks could get crowded out by unrelated
# projects before filtering ever runs. Fetch a wider candidate pool
# whenever a filter is active, then cap at TOP_K after filtering.
TOP_K_WHEN_FILTERED = 25
GROQ_MODEL = "llama-3.3-70b-versatile"

FOLLOW_UP_SIGNAL_WORDS = {
    "it", "that", "this", "they", "them", "those", "these",
    "he", "she", "him", "her", "further", "also", "too", "again",
}


def _looks_like_follow_up(query: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", query.lower())
    if len(words) <= 4:
        return True
    return any(w in FOLLOW_UP_SIGNAL_WORDS for w in words)


def _rewrite_query_with_history(query: str, history: list[ChatTurn]) -> str:
    if not history or not settings.groq_api_key:
        return query
    if not _looks_like_follow_up(query):
        return query

    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)

    recent = history[-3:]
    transcript = "\n".join(f"{t.role}: {t.content}" for t in recent)
    prompt = (
        "Given this recent conversation, rewrite the final user question as a "
        "standalone question that doesn't depend on prior context. "
        "If the question is already standalone, return it unchanged. "
        "Return ONLY the rewritten question, nothing else — no preamble, "
        "no quotes, no explanation.\n\n"
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


def _retrieve_chunks(db: Session, query: str, meeting_id: int = None, project_id: int = None):
    """Shared retrieval logic used by both /chat and /chat/stream.
    Returns (chunks, retrieved_ids). Widens the FAISS candidate pool
    whenever a filter is active, since filtering happens AFTER the
    vector search, not before."""
    is_filtered = meeting_id is not None or project_id is not None
    top_k = TOP_K_WHEN_FILTERED if is_filtered else TOP_K

    query_vector = get_embedding(query)
    store = get_faiss_store()
    results = store.search(query_vector, top_k=top_k)

    if not results:
        return [], []

    retrieved_ids = [cid for cid, _ in results]
    chunks_query = db.query(Chunk).filter(Chunk.id.in_(retrieved_ids))

    if meeting_id is not None:
        chunks_query = chunks_query.filter(Chunk.meeting_id == meeting_id)
    elif project_id is not None:
        chunks_query = chunks_query.join(Meeting, Chunk.meeting_id == Meeting.id).filter(
            Meeting.project_id == project_id
        )

    chunks = chunks_query.all()

    # Preserve FAISS's relevance order, then cap at TOP_K even though
    # we fetched a wider pool — the wider pool was only to survive
    # filtering, not to send more context to the LLM than necessary.
    chunks_by_id = {c.id: c for c in chunks}
    ordered = [chunks_by_id[cid] for cid in retrieved_ids if cid in chunks_by_id][:TOP_K]
    return ordered, retrieved_ids


def _generate_answer(query: str, chunks: list[Chunk]) -> dict:
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
    standalone_query = _rewrite_query_with_history(req.query, req.chat_history)
    chunks, retrieved_ids = _retrieve_chunks(db, standalone_query, req.meeting_id, req.project_id)

    if not retrieved_ids:
        return ChatResponse(answer="No meetings have been indexed yet.", citations=[], retrieved_chunk_ids=[])
    if not chunks:
        return ChatResponse(
            answer="No relevant content found for this filter.",
            citations=[],
            retrieved_chunk_ids=retrieved_ids,
        )

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

    return ChatResponse(answer=result["answer"], citations=citations, retrieved_chunk_ids=retrieved_ids)


def _stream_answer_tokens(query: str, chunks: list[Chunk]):
    allowed_ids = [c.id for c in chunks]

    if not settings.groq_api_key:
        yield (
            "(GROQ_API_KEY not set — configure it in .env to get real answers.) "
            f"Retrieved {len(chunks)} relevant chunk(s) for: '{query}'"
        )
        return

    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)

    context_blocks = [
        f"[chunk_id={c.id}] (meeting_id={c.meeting_id})\n{c.chunk_text}"
        for c in chunks
    ]
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You answer questions about company meetings using ONLY the provided "
        "meeting excerpts. When you state a fact from an excerpt, cite it "
        "inline immediately after, like [chunk_id]. Only use chunk_id values "
        "that appear in the excerpts below — never invent one. If the "
        "excerpts don't contain the answer, say so explicitly, in plain "
        "text, no citation needed for that. "
        f"Allowed chunk_id values: {allowed_ids}"
    )

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {query}"},
            ],
            max_tokens=800,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"\n\n[LLM call failed: {e}]"


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    standalone_query = _rewrite_query_with_history(req.query, req.chat_history)
    chunks, retrieved_ids = _retrieve_chunks(db, standalone_query, req.meeting_id, req.project_id)

    if not retrieved_ids:
        def _empty():
            yield "No meetings have been indexed yet."
        return StreamingResponse(_empty(), media_type="text/plain")

    if not chunks:
        def _none_relevant():
            yield "No relevant content found for this filter."
        return StreamingResponse(_none_relevant(), media_type="text/plain")

    return StreamingResponse(_stream_answer_tokens(standalone_query, chunks), media_type="text/plain")