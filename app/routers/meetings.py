import os
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Meeting, Chunk, Participant, Task, Decision
from app.schemas.meeting import MeetingCreate
from app.services.extraction import extract_text
from app.services.chunking import chunk_document, detect_source_type, detect_speakers
from app.services.embeddings import get_embeddings
from app.services.faiss_store import get_faiss_store
from app.services.structured_extraction import extract_structured_info

router = APIRouter(prefix="/meetings", tags=["Meetings"])

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# Create Meeting
# -----------------------------
@router.post("/")
def create_meeting(meeting: MeetingCreate, db: Session = Depends(get_db)):
    new_meeting = Meeting(
        title=meeting.title,
        project=meeting.project,
        agenda=meeting.agenda,
    )

    db.add(new_meeting)
    db.commit()
    db.refresh(new_meeting)

    for name in meeting.participants:
        name = name.strip()
        if name:
            db.add(Participant(meeting_id=new_meeting.id, person_name=name))
    db.commit()

    return {
        "message": "Meeting created successfully",
        "id": new_meeting.id,
    }


# -----------------------------
# Get All Meetings
# -----------------------------
@router.get("/")
def get_meetings(db: Session = Depends(get_db)):
    meetings = db.query(Meeting).order_by(Meeting.id.desc()).all()
    return meetings


# -----------------------------
# Get One Meeting
# -----------------------------
@router.get("/{meeting_id}")
def get_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()

    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return meeting


# -----------------------------
# Delete Meeting
# -----------------------------
@router.delete("/{meeting_id}")
def delete_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()

    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")

    db.delete(meeting)
    db.commit()

    return {"message": "Meeting deleted successfully"}


# -----------------------------
# Tasks / Decisions / Participants
# -----------------------------
@router.get("/tasks/all")
def list_tasks(owner: str = None, db: Session = Depends(get_db)):
    query = db.query(Task)
    if owner:
        query = query.filter(Task.owner.ilike(f"%{owner}%"))
    return query.all()


@router.get("/decisions/all")
def list_decisions(db: Session = Depends(get_db)):
    return db.query(Decision).all()


@router.get("/{meeting_id}/participants")
def get_participants(meeting_id: int, db: Session = Depends(get_db)):
    return db.query(Participant).filter(Participant.meeting_id == meeting_id).all()


# -----------------------------
# Upload Transcript
# -----------------------------
@router.post("/{meeting_id}/upload")
async def upload_transcript(
    meeting_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()

    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")

    filename = f"{meeting_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # OneDrive can briefly lock a newly-written file while syncing,
    # causing an intermittent PermissionError that has nothing to do
    # with the code itself. Retry a few times before giving up.
    file_bytes = file.file.read()
    last_error = None
    for attempt in range(3):
        try:
            with open(filepath, "wb") as buffer:
                buffer.write(file_bytes)
            last_error = None
            break
        except PermissionError as e:
            last_error = e
            time.sleep(0.5)

    if last_error is not None:
        meeting.status = "failed"
        db.commit()
        raise HTTPException(
            500,
            f"Could not write file after retries (likely OneDrive sync lock): {last_error}. "
            "Try pausing OneDrive sync or moving the project outside OneDrive.",
        )

    meeting.transcript_path = filepath
    meeting.status = "processing"
    db.commit()

    # --- extraction ---
    try:
        raw_text = extract_text(Path(filepath))
    except Exception as e:
        meeting.status = "failed"
        db.commit()
        raise HTTPException(400, f"Failed to extract text: {e}")

    if not raw_text.strip():
        meeting.status = "failed"
        db.commit()
        raise HTTPException(400, "No extractable text found in file.")

    # --- chunking ---
    detected_type = detect_source_type(raw_text)
    chunk_results = chunk_document(raw_text, source_type=detected_type)

    if not chunk_results:
        meeting.status = "failed"
        db.commit()
        raise HTTPException(400, "Chunking produced no chunks.")

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
        db.refresh(row)  # populates row.id, needed before embedding

    # --- embedding + FAISS indexing ---
    # chunk.id is reused as the FAISS vector id, so no separate
    # mapping table is needed between "FAISS position" and "chunk row".
    vectors = get_embeddings([r.chunk_text for r in chunk_rows])
    chunk_ids = [r.id for r in chunk_rows]
    store = get_faiss_store()
    store.add(chunk_ids, vectors)

    # --- auto-populate participants from detected speakers ---
    # Free — reuses the same real-speaker detection already computed
    # during chunking, no extra Groq call. Skips names that were
    # already added manually at meeting-creation time, so re-uploading
    # doesn't create duplicate rows for the same person.
    if detected_type == "transcript":
        detected_speakers = detect_speakers(raw_text)
        existing_names = {
            p.person_name for p in
            db.query(Participant).filter(Participant.meeting_id == meeting.id).all()
        }
        for name in detected_speakers - existing_names:
            db.add(Participant(meeting_id=meeting.id, person_name=name))
        db.commit()

    # --- structured extraction: summary, decisions, tasks ---
    # Runs on the whole meeting's text, not per-chunk, so it isn't
    # confused by decisions/tasks that span a chunk boundary.
    # Best-effort: if it fails or no Groq key is set, the rest of the
    # pipeline (chat/retrieval) still works fine without it.
    extracted = extract_structured_info(raw_text)
    meeting.summary = extracted["summary"] or None

    # naive mapping: attach every extracted item to the FIRST chunk,
    # since we don't yet trace which chunk a specific sentence came
    # from. Good enough for MVP traceability; a future improvement
    # would match each item back to its originating chunk directly.
    first_chunk_id = chunk_rows[0].id if chunk_rows else None

    for decision_text in extracted["decisions"]:
        db.add(Decision(
            meeting_id=meeting.id,
            decision=decision_text,
            source_chunk_id=first_chunk_id,
        ))

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
        "message": "Transcript uploaded, extracted, chunked, and indexed successfully",
        "file": filename,
        "source_type": detected_type,
        "chunks_created": len(chunk_results),
        "decisions_found": len(extracted["decisions"]),
        "tasks_found": len(extracted["tasks"]),
    }