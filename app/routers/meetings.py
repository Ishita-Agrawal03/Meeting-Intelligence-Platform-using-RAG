import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Meeting, Chunk
from app.schemas.meeting import MeetingCreate
from app.services.extraction import extract_text
from app.services.chunking import chunk_document, detect_source_type
from app.services.embeddings import get_embeddings
from app.services.faiss_store import get_faiss_store

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
    )

    db.add(new_meeting)
    db.commit()
    db.refresh(new_meeting)

    return {
        "message": "Meeting created successfully",
        "id": new_meeting.id,
    }


# -----------------------------
# Get All Meetings
# -----------------------------
@router.get("/")
def get_meetings(db: Session = Depends(get_db)):
    meetings = db.query(Meeting).all()
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

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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

    meeting.source_type = detected_type
    meeting.status = "ready"
    db.commit()

    return {
        "message": "Transcript uploaded, extracted, chunked, and indexed successfully",
        "file": filename,
        "source_type": detected_type,
        "chunks_created": len(chunk_results),
    }