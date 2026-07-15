from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Meeting, Participant, Task, Decision
from app.schemas.meeting import MeetingCreate
from app.services.pipeline import (
    save_uploaded_file,
    is_audio_video,
    process_meeting_document,
    process_meeting_audio_background,
)

router = APIRouter(prefix="/meetings", tags=["Meetings"])


# -----------------------------
# Create Meeting (legacy path — kept for API completeness;
# the primary flow is now POST /projects/{project_id}/upload)
# -----------------------------
@router.post("/")
def create_meeting(meeting: MeetingCreate, db: Session = Depends(get_db)):
    new_meeting = Meeting(
        title=meeting.title,
        project_id=meeting.project_id,
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

    return {"message": "Meeting created successfully", "id": new_meeting.id}


@router.get("/")
def get_meetings(db: Session = Depends(get_db)):
    return db.query(Meeting).order_by(Meeting.id.desc()).all()


@router.get("/{meeting_id}")
def get_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.delete("/{meeting_id}")
def delete_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    db.delete(meeting)
    db.commit()
    return {"message": "Meeting deleted successfully"}


# -----------------------------
# Tasks / Decisions / Participants (global, all meetings —
# project-scoped versions live in /projects/{id}/tasks etc.)
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
# Upload Transcript (legacy path — kept for API completeness;
# the primary flow is now POST /projects/{project_id}/upload)
# -----------------------------
@router.post("/{meeting_id}/upload")
async def upload_transcript(
    meeting_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")

    file_bytes = await file.read()
    try:
        filepath = save_uploaded_file(meeting.id, file.filename, file_bytes)
    except RuntimeError as e:
        meeting.status = "failed"
        db.commit()
        raise HTTPException(500, str(e))

    meeting.transcript_path = filepath
    db.commit()

    if is_audio_video(file.filename):
        meeting.status = "transcribing"
        db.commit()
        background_tasks.add_task(process_meeting_audio_background, meeting.id, filepath)
        return {
            "message": "Audio/video uploaded — transcribing in the background.",
            "file": file.filename,
            "status": "transcribing",
            "note": "Poll GET /meetings/{id} for status. Participants won't "
                    "auto-populate for audio/video — add manually if needed.",
        }

    meeting.status = "processing"
    db.commit()
    try:
        result = process_meeting_document(db, meeting, filepath)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "message": "Transcript uploaded, extracted, chunked, and indexed successfully",
        "file": file.filename,
        **result,
    }