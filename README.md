# AI Meeting Intelligence Platform

A retrieval-augmented generation (RAG) platform that turns uploaded meeting transcripts, notes, and audio/video recordings into a continuously growing, project-scoped knowledge base — answering natural-language questions with cited, source-grounded, streamed answers instead of just summarizing one meeting at a time.

**🔴 Live demo:**
- Frontend: `https://meeting-intelligence-platform-using-rag.streamlit.app/`
- Backend API docs: https://meeting-intelligence-platform-using-rag.onrender.com/docs

*(Note: the backend runs on Render's free tier — the first request after a period of inactivity may take 30–60s while it wakes up. Uploaded data does not persist across backend restarts on the free tier; see Known Limitations.)*

## Why this exists

Most meeting tools give you a transcript, a summary, and a recording — three things that get forgotten the moment the next meeting starts. This platform treats every uploaded meeting as part of one growing project memory, so questions like *"Why did we choose PostgreSQL?"* or *"What tasks are assigned to Priya?"* can be answered weeks later, across every meeting ever uploaded to that project.

## Interface

Navigation mirrors Claude/ChatGPT: a sidebar lists your projects (like a chat history list). Clicking one loads that project's full workspace — Chat, Tasks, Decisions, and Participants — scoped to every meeting inside it, with no dropdowns needed anywhere in the main content. Uploading a new meeting to an active project adds to its knowledge base, the same way returning to an old chat thread and continuing it would.

## Features

- **Multi-format ingestion** — `.txt`, `.pdf`, `.docx`, and audio/video (`.mp3`, `.wav`, `.mp4`, `.m4a`, `.webm`) via Groq-hosted Whisper
- **Background processing for audio/video** — transcription runs asynchronously via FastAPI `BackgroundTasks`, so uploads return immediately instead of blocking on a slow operation
- **Dual-path chunking** — auto-detects speaker-labeled transcripts vs. plain prose notes using a stoplist + repetition-based speaker filter (not just regex shape), chunking each differently with configurable overlap; oversized single-block notes fall back to sentence-level splitting
- **Project-scoped semantic search** — FAISS (`IndexIDMap`) over `sentence-transformers` embeddings; retrieval widens its candidate pool automatically when a project/meeting filter is active, since filtering happens after the global vector search
- **Streaming, grounded chat** — retrieves relevant chunks, then streams the LLM's answer token-by-token, with inline citations constrained to only the chunk IDs actually retrieved — the model cannot cite or fabricate a source it wasn't given
- **Query rewriting with cost-aware gating** — follow-up questions ("what about that?") are rewritten into standalone queries using recent chat history, but only when a lightweight heuristic detects the question actually depends on prior context — skipping the extra LLM call otherwise
- **Structured extraction** — automatically pulls a summary, decisions, and action items (owner + deadline) from every uploaded meeting using JSON-mode LLM output; participant names are auto-detected from real speaker patterns (not extracted via LLM, so it's free)
- **Evaluation harness** — a benchmark suite that runs against the live `/chat` endpoint (project-scoped, mirroring real usage) and logs retrieval accuracy and answer correctness to a separate database, tracking quality across changes over time — includes adversarial no-answer questions to catch hallucination
- **Containerized** — full Docker + Docker Compose setup with persistent named volumes, deployed live on Render (backend) and Streamlit Community Cloud (frontend)

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python, FastAPI |
| Database | SQLAlchemy + SQLite |
| Vector store | FAISS (`IndexIDMap`, `IndexFlatL2`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim, pinned) |
| LLM | Groq (Llama 3.3 70B / 3.1 8B) |
| Speech-to-text | Groq-hosted Whisper (`whisper-large-v3`) |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Deployment | Render (backend), Streamlit Community Cloud (frontend) |

## Architecture

**Upload pipeline:**
```
Upload (.txt / .pdf / .docx / .mp3 / .wav / .mp4 / .m4a)
        │
        ▼
  Audio/video?  ──Yes──▶  Return immediately, status="transcribing"
        │                  Background: Whisper → same pipeline below
        No
        ▼
  Text extraction
        │
        ▼
  Detect: transcript or notes? (repetition-filtered speaker detection)
        │
   ┌────┴────┐
   ▼         ▼
Speaker-   Paragraph/sentence-
aware      based chunking
chunking   (with oversized-block fallback)
   │         │
   └────┬────┘
        ▼
  Embed chunks → store: text → SQLite, vector+id → FAISS
        │
        ▼
  Auto-detect participants (free — reuses chunking's speaker detection)
        │
        ▼
  Structured extraction (Groq, JSON mode) → summary, decisions, tasks
```

**Chat pipeline:**
```
User question (scoped to active project)
        │
        ▼
  Follow-up? (heuristic) → optionally rewrite using chat history
        │
        ▼
  Embed question → FAISS search (widened pool if project-scoped)
        │
        ▼
  Filter to project's chunks → fetch real text from SQLite
        │
        ▼
  Stream LLM answer, citing ONLY retrieved chunk IDs inline
        │
        ▼
  Tokens streamed to UI as they're generated
```

## Database schema

| Table | Purpose |
|---|---|
| `projects` | Top-level container — the unit shown in the sidebar |
| `meetings` | Individual uploads; belongs to one project; title, agenda, status, AI-generated summary |
| `chunks` | Retrievable text units; `id` doubles as the FAISS vector ID |
| `participants` | Per-meeting attendee names (auto-detected or manually entered) |
| `tasks` | Extracted action items (owner, task, deadline, source chunk) |
| `decisions` | Extracted decisions (decision text, source chunk) |

Evaluation data lives in a **separate** `evaluation.db` — development/research metadata, not application state, kept independent so it can be wiped and rebuilt without touching real project data.

## Setup — local (Python venv)

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
python -m streamlit run streamlit_app.py
```

## Setup — Docker

```bash
docker compose up --build
```

Builds and runs both containers, backend on `:8000`, frontend on `:8501`. Data persists in named volumes across `docker compose down` / `up` cycles. The backend Dockerfile installs a CPU-only `torch` build explicitly (avoids ~2GB of unused CUDA downloads); the frontend uses its own minimal `requirements-frontend.txt` since it only makes HTTP calls to the backend and needs no ML libraries at all.

## Running the evaluation suite

```bash
cd eval
python run_eval.py --base-url http://127.0.0.1:8000
# or against the live deployment:
python run_eval.py --base-url https://meeting-intelligence-platform-using-rag.onrender.com
```

Requires two real projects to already exist (created via the UI), matching the names referenced in `eval/benchmark.json`: `"Smart Customer Support AI"` and `"PostgreSQL Architecture Decision"`. Runs each question through the live `/chat` endpoint scoped to its real project (matching how the actual UI behaves), checks both retrieval accuracy (was the right project cited?) and answer correctness separately, and logs every run to `evaluation.db` for trend tracking across iterations.

**Latest result:** 100% retrieval accuracy and 100% answer correctness across a 12-question benchmark spanning two projects and adversarial no-answer cases, on both `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` — consistent across two model sizes, not a single lucky run.

## Known limitations

Deliberate scope decisions for a portfolio project, not oversights:

- **Render free-tier storage is ephemeral** — the deployed backend's SQLite database and FAISS index reset whenever the free service restarts/redeploys. Fine for demos; a persistent deployment would need a paid disk or a hosted database.
- **Whisper transcription has no speaker labels** — audio/video uploads are classified as "notes," not "transcript," so participants won't auto-populate for them; add participants manually at project/meeting creation instead. Real diarization (e.g. `pyannote.audio`) is a deliberately deferred, heavier addition.
- **Causal ("why") questions are answered less reliably than direct ("what did X say") questions on the same underlying content** — observed directly in evaluation: identical retrieved chunk, different question phrasing, different correctness. LLM-side inconsistency under a strict "don't answer unless certain" prompt, not a retrieval failure.
- **Deleting a project doesn't clean up its FAISS vectors** — orphaned vectors remain in the index (harmless, not reclaimed).
- **Extracted tasks/decisions cite the first chunk of a meeting**, not necessarily the exact chunk they came from — a placeholder for finer-grained traceability.
- **No participant name normalization** — "Rahul" and "Rahul S." would currently be treated as different people. Deferred until a feature (cross-meeting person search) actually needs it.
- **Single-process only** — FAISS is loaded in memory per process; multiple `uvicorn` workers would race writing the same index file. Fine for single-user use; would need a dedicated vector DB (pgvector, Qdrant) for multi-worker production use.
- **No authentication or multi-tenant isolation** — out of scope for a single-user portfolio project.

## Roadmap

- Persistent hosted deployment (paid disk or managed Postgres + object storage)
- Real speaker diarization for audio/video
- Chunk-level traceability for extracted tasks/decisions
- Participant name normalization / entity resolution across meetings
- Timeline and participant-based filtered retrieval

## Project structure

```
app/
  main.py                        FastAPI entrypoint
  config.py                      Settings (.env-backed)
  db/
    database.py                   Engine/session setup
    models.py                     Project, Meeting, Chunk, Participant, Task, Decision
  routers/
    projects.py                    Primary API: create/list projects, upload (auto-creates meeting), scoped tasks/decisions/participants
    meetings.py                    Legacy per-meeting endpoints (kept for API completeness)
    chat.py                        RAG chat (/chat) + streaming (/chat/stream), project/meeting scoping
  services/
    extraction.py                  Text extraction (.txt/.pdf/.docx)
    transcription.py               Audio/video transcription via Groq Whisper
    chunking.py                    Dual-path chunking + speaker detection
    embeddings.py                  Embedding model wrapper (pinned)
    faiss_store.py                 FAISS wrapper (IndexIDMap, save/load)
    structured_extraction.py       Summary/decisions/tasks via Groq JSON mode
    pipeline.py                    Shared processing pipeline (sync + background paths)
  schemas/
    meeting.py, chat.py            Pydantic request/response models
eval/
  eval_db.py                      Separate evaluation database
  benchmark.json                   Benchmark questions (project-scoped)
  run_eval.py                      Eval runner (with rate-limit retry/backoff)
scripts/
  test_extraction.py, test_chunking.py, test_embedding.py, test_faiss.py   Standalone component tests
streamlit_app.py                  Frontend — sidebar project navigation + 4-tab workspace
Dockerfile                        Backend image (CPU-only torch)
Dockerfile.streamlit              Frontend image (minimal deps)
docker-compose.yml                Local orchestration with persistent volumes
requirements.txt                  Backend dependencies
requirements-frontend.txt         Frontend-only dependencies
```
