"""
Shared processing pipeline — extraction/transcription -> chunking ->
embedding -> FAISS -> participants -> structured extraction -> status.

Factored out so both the synchronous (text/pdf/docx) and background
(audio/video) upload paths call the SAME logic instead of maintaining
two separate copies that could silently drift apart.
"""
import os
import time
from pathlib import Path
from sqlalchemy.orm import Session

from app.db.models import Meeting, Chunk, Participant, Task, Decision
from app.services.extraction import extract_text
from app.services.transcription import transcribe_audio
from app.services.chunking import chunk_document, detect_source_type, detect_speakers
from app.services.embeddings import get_embeddings
from app.services.faiss_store import get_faiss_store
from app.services.structured_extraction import extract_structured_info

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AUDIO_VIDEO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mpga", ".mpeg", ".mp4", ".webm"}


def save_uploaded_file(meeting_id: int, filename: str, file_bytes: bytes) -> str:
    """Saves raw upload bytes to disk with retry logic (OneDrive can
    intermittently lock a newly-written file mid-sync). Returns the
    filepath, or raises RuntimeError after exhausting retries."""
    safe_name = f"{meeting_id}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)

    last_error = None
    for _ in range(3):
        try:
            with open(filepath, "wb") as buffer:
                buffer.write(file_bytes)
            return filepath
        except PermissionError as e:
            last_error = e
            time.sleep(0.5)

    raise RuntimeError(
        f"Could not write file after retries (likely OneDrive sync lock): {last_error}. "
        "Try pausing OneDrive sync or moving the project outside OneDrive."
    )


def is_audio_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in AUDIO_VIDEO_EXTENSIONS


def process_meeting_text(db: Session, meeting: Meeting, raw_text: str) -> dict:
    """Runs chunking -> embedding -> FAISS -> participants -> structured
    extraction for a meeting whose raw text is already available (either
    extracted from a document, or transcribed from audio/video).
    Updates meeting.status to 'ready' or 'failed' as it goes. Returns a
    dict of counts for the API response."""
    if not raw_text.strip():
        meeting.status = "failed"
        db.commit()
        raise ValueError("No extractable text found.")

    detected_type = detect_source_type(raw_text)
    chunk_results = chunk_document(raw_text, source_type=detected_type)

    if not chunk_results:
        meeting.status = "failed"
        db.commit()
        raise ValueError("Chunking produced no chunks.")

    chunk_rows = []
    for cr in chunk_results:
        row = Chunk(
            meeting_id=meeting.id,
            chunk_text=cr.text,
            chunk_type=cr.chunk_type,
            speakers=cr.speakers,
            position=cr.position,
        )
        db.add(row)
        chunk_rows.append(row)
    db.commit()
    for row in chunk_rows:
        db.refresh(row)

    vectors = get_embeddings([r.chunk_text for r in chunk_rows])
    chunk_ids = [r.id for r in chunk_rows]
    store = get_faiss_store()
    store.add(chunk_ids, vectors)

    # Speaker auto-detection only fires for "transcript"-type text.
    # Whisper output has no speaker labels, so this normally won't
    # populate participants for audio/video — add them manually.
    if detected_type == "transcript":
        detected_speakers = detect_speakers(raw_text)
        existing_names = {
            p.person_name for p in
            db.query(Participant).filter(Participant.meeting_id == meeting.id).all()
        }
        for name in detected_speakers - existing_names:
            db.add(Participant(meeting_id=meeting.id, person_name=name))
        db.commit()

    extracted = extract_structured_info(raw_text)
    meeting.summary = extracted["summary"] or None
    first_chunk_id = chunk_rows[0].id if chunk_rows else None

    for decision_text in extracted["decisions"]:
        db.add(Decision(meeting_id=meeting.id, decision=decision_text, source_chunk_id=first_chunk_id))

    for task_item in extracted["tasks"]:
        db.add(Task(
            meeting_id=meeting.id,
            owner=task_item.get("owner"),
            task=task_item.get("task", ""),
            deadline=task_item.get("deadline"),
            source_chunk_id=first_chunk_id,
        ))

    meeting.source_type = detected_type
    meeting.status = "ready"
    db.commit()

    return {
        "source_type": detected_type,
        "chunks_created": len(chunk_results),
        "decisions_found": len(extracted["decisions"]),
        "tasks_found": len(extracted["tasks"]),
    }


def process_meeting_document(db: Session, meeting: Meeting, filepath: str) -> dict:
    """Synchronous path for text/pdf/docx: extract text, then run the
    shared pipeline. Raises on failure (caller should turn into a 400)."""
    try:
        raw_text = extract_text(Path(filepath))
    except Exception as e:
        meeting.status = "failed"
        db.commit()
        raise ValueError(f"Failed to extract text: {e}")

    return process_meeting_text(db, meeting, raw_text)


def process_meeting_audio_background(meeting_id: int, filepath: str):
    """Background path for audio/video: transcribe, then run the shared
    pipeline. Uses its OWN database session since the original request's
    session is already closed by the time a background task runs."""
    from app.db.database import SessionLocal
    db = SessionLocal()
    try:
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if meeting is None:
            return

        meeting.status = "processing"
        db.commit()

        raw_text = transcribe_audio(Path(filepath))
        process_meeting_text(db, meeting, raw_text)

    except Exception as e:
        print(f"Background audio processing failed for meeting {meeting_id}: {e}")
        try:
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            if meeting:
                meeting.status = "failed"
                db.commit()
        except Exception:
            pass
    finally:
        db.close()