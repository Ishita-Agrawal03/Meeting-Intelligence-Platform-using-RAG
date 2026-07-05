# AI Meeting Intelligence Platform

A retrieval-augmented generation (RAG) platform that ingests meeting transcripts and notes, builds a continuously growing, searchable knowledge base, and answers natural-language questions with cited, source-grounded answers — instead of just summarizing one meeting at a time.

## Why this exists

Most meeting tools give you a transcript, a summary, and a recording — three things that get forgotten the moment the next meeting starts. This platform treats every uploaded meeting as part of one growing organizational memory, so questions like *"Why did we choose PostgreSQL?"* or *"What tasks are assigned to Priya?"* can be answered months later, across every meeting ever uploaded — not just the most recent one.

## Demo

A Streamlit UI is included for demoing without touching raw API docs — upload a transcript, ask a question, see the cited answer with sources.

*(screenshot here)*

## Features

- **Multi-format ingestion** — `.txt`, `.pdf`, `.docx`
- **Dual-path chunking** — auto-detects speaker-labeled transcripts vs. plain prose notes, and chunks each differently (turn-preserving for transcripts, paragraph-based for notes), with configurable overlap
- **Semantic search** — FAISS (`IndexIDMap`) over `sentence-transformers` embeddings, so chunk IDs map directly to database rows with no separate translation table
- **Grounded chat** — retrieves relevant chunks, then constrains the LLM (Groq / Llama 3.3) to cite *only* chunk IDs it was actually given, preventing fabricated sources
- **Structured extraction** — automatically pulls a summary, decisions, and action items (owner + deadline) from every uploaded meeting, using JSON-mode LLM output rather than free-text parsing
- **Multi-meeting knowledge base** — new uploads are added incrementally; existing meetings are never re-processed
- **Evaluation harness** — a benchmark suite that runs against the live `/chat` endpoint and logs retrieval accuracy and answer correctness to a separate database, so quality can be tracked across changes over time

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python, FastAPI |
| Database | SQLAlchemy + SQLite |
| Vector store | FAISS (`IndexIDMap`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim, pinned) |
| LLM | Groq (Llama 3.3 70B) |
| Frontend | Streamlit |

## Architecture

```
Upload (.txt / .pdf / .docx)
        │
        ▼
  Text extraction
        │
        ▼
  Detect: transcript or notes?
        │
   ┌────┴────┐
   ▼         ▼
Speaker-   Paragraph-
aware      based
chunking   chunking
   │         │
   └────┬────┘
        ▼
   Embed chunks (sentence-transformers)
        │
        ▼
   Store: chunk text → SQLite
          vector + id  → FAISS
        │
        ▼
   Structured extraction (Groq, JSON mode)
   → summary, decisions, tasks saved to SQLite
```

```
User question
        │
        ▼
   Embed question
        │
        ▼
   FAISS search → top-k chunk IDs
        │
        ▼
   Fetch chunk text from SQLite
        │
        ▼
   Groq generates answer, citing ONLY
   chunk IDs it was actually given
        │
        ▼
   Answer + citations returned
```

## Database schema

| Table | Purpose |
|---|---|
| `meetings` | title, project, agenda, transcript path, status, AI-generated summary |
| `participants` | who attended each meeting (join table) |
| `chunks` | retrievable text units; `id` doubles as the FAISS vector ID |
| `tasks` | extracted action items (owner, task, deadline, source chunk) |
| `decisions` | extracted decisions (decision text, source chunk) |

Evaluation data lives in a **separate** `evaluation.db` — it's development/research metadata, not application state, so it's kept independent and can be wiped/rebuilt without touching real meeting data.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file:
```
DATABASE_URL=sqlite:///meetings.db
GROQ_API_KEY=your_groq_key_here
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

Run the backend:
```bash
uvicorn app.main:app --reload
```

Run the UI (in a second terminal):
```bash
streamlit run streamlit_app.py
```

API docs available at `http://127.0.0.1:8000/docs`.

## Running the evaluation suite

```bash
cd eval
python run_eval.py
```

This runs a benchmark question set against the live `/chat` endpoint and logs retrieval accuracy and answer correctness to `evaluation.db`, so results can be compared across runs after tuning chunking, prompts, or retrieval parameters.

**Latest result:** 100% retrieval accuracy, 100% answer correctness on a 7-question benchmark against a real 6-participant project-kickoff transcript. *(Benchmark is intentionally being expanded to more meetings and harder edge cases — treat this as an early result, not a final claim.)*

## Known limitations

These are deliberate scope decisions for an MVP, not oversights:

- **Single-process only** — FAISS is loaded in memory per process; running multiple `uvicorn` workers would cause concurrent writes to race on the same index file. Fine for a single-user demo; would need a dedicated vector DB (pgvector, Qdrant) for multi-worker production use.
- **No participant name normalization** — "Rahul" and "Rahul S." would currently be treated as different people. Deferred until there's an actual feature (cross-meeting person search) that needs it.
- **Extracted tasks/decisions cite the first chunk of a meeting**, not necessarily the exact chunk they came from — a placeholder for finer-grained traceability.
- **No audio/video ingestion yet** — Whisper transcription is planned but not implemented.
- **Deleting a meeting doesn't clean up its FAISS vectors** — the underlying vectors remain in the index (harmless, but not reclaimed).
- **No authentication or multi-tenant isolation** — out of scope for a single-user portfolio project.

## Roadmap

- Whisper-based audio/video transcription
- Chunk-level traceability for extracted tasks/decisions
- Participant name normalization / entity resolution across meetings
- Timeline and participant-based filtered retrieval
- Expanded evaluation benchmark across multiple meetings and edge cases

## Project structure

```
app/
  main.py                    FastAPI entrypoint
  config.py                  Settings (.env-backed)
  db/
    database.py               Engine/session setup
    models.py                 Meeting, Chunk, Participant, Task, Decision
  routers/
    meetings.py                CRUD + upload + tasks/decisions/participants
    chat.py                     RAG chat endpoint
  services/
    extraction.py               Text extraction (.txt/.pdf/.docx)
    chunking.py                 Dual-path chunking
    embeddings.py               Embedding model wrapper (pinned)
    faiss_store.py               FAISS wrapper (IndexIDMap, save/load)
    structured_extraction.py     Summary/decisions/tasks via Groq JSON mode
  schemas/
    meeting.py, chat.py         Pydantic request/response models
eval/
  eval_db.py                  Separate evaluation database
  benchmark.json               Benchmark questions
  run_eval.py                  Eval runner
scripts/
  test_extraction.py           Standalone extraction test
  test_chunking.py             Standalone chunking test
  test_embedding.py            Standalone embedding test
  test_faiss.py                Standalone FAISS test
streamlit_app.py               Frontend
requirements.txt
```
