"""
Shared processing pipeline — extraction/transcription -> chunking ->
embedding -> FAISS -> participants -> structured extraction -> status.
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
    intermittently lock a newly-written file mid-sync).

    filename is sanitized via Path(...).name FIRST — stripping any
    directory components — before being used to build the save path.
    Without this, a filename containing path separators (e.g.
    "../../etc/something") could plausibly cause the file to be
    written outside UPLOAD_FOLDER entirely."""
    safe_filename = Path(filename).name
    safe_name = f"{meeting_id}_{safe_filename}"
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


def delete_uploaded_file(filepath: str):
    """Removes the raw uploaded file from disk. Safe to call even if
    the file is already missing — a meeting record can exist with no
    file (e.g. failed before upload completed)."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError as e:
            print(f"Could not delete file {filepath}: {e}")


def is_audio_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in AUDIO_VIDEO_EXTENSIONS


def process_meeting_text(db: Session, meeting: Meeting, raw_text: str) -> dict:
    """Runs chunking -> embedding -> FAISS -> participants -> structured
    extraction. Updates meeting.status to 'ready' or 'failed'. Raises
    ValueError on failure so BOTH the sync and background callers can
    catch the same exception type and handle it identically — this
    fixes a real inconsistency where the sync path let embedding/FAISS
    failures propagate as raw uncaught exceptions (chunks committed to
    SQLite with no vectors, meeting stuck at "processing" forever),
    while the background path already caught everything broadly."""
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

    # Embedding + FAISS indexing — this is the section that previously
    # had NO error handling on the sync path. If get_embeddings() or
    # store.add() throws here, the except block below catches it,
    # marks the meeting failed, and re-raises as ValueError — instead
    # of leaving committed chunks with no vectors and no status update.
    try:
        vectors = get_embeddings([r.chunk_text for r in chunk_rows])
        chunk_ids = [r.id for r in chunk_rows]
        store = get_faiss_store()
        store.add(chunk_ids, vectors)
    except Exception as e:
        meeting.status = "failed"
        db.commit()
        raise ValueError(f"Embedding/indexing failed: {e}")

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
    """Synchronous path for text/pdf/docx."""
    try:
        raw_text = extract_text(Path(filepath))
    except Exception as e:
        meeting.status = "failed"
        db.commit()
        raise ValueError(f"Failed to extract text: {e}")

    return process_meeting_text(db, meeting, raw_text)


def process_meeting_audio_background(meeting_id: int, filepath: str):
    """Background path for audio/video."""
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