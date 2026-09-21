import asyncio
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Meeting, Participant, Task, Decision
from app.schemas.meeting import MeetingCreate
from app.services.pipeline import (
    save_uploaded_file,
    delete_uploaded_file,
    is_audio_video,
    process_meeting_document,
    process_meeting_audio_background,
)

router = APIRouter(prefix="/meetings", tags=["Meetings"])


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

    # Clean up the raw uploaded file from disk BEFORE deleting the DB
    # row — previously the file was left behind forever with no
    # cleanup lifecycle at all, since only the database row was removed.
    delete_uploaded_file(meeting.transcript_path)

    db.delete(meeting)
    db.commit()
    return {"message": "Meeting deleted successfully"}


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

    # save_uploaded_file() is a blocking, synchronous function (it does
    # real disk I/O and a time.sleep() retry loop). Calling it directly
    # inside an `async def` route would block the entire event loop —
    # stalling EVERY other concurrent request, not just this one — for
    # up to 1.5s if a retry is needed. asyncio.to_thread() runs it on a
    # separate thread instead, so the event loop stays free.
    try:
        filepath = await asyncio.to_thread(save_uploaded_file, meeting.id, file.filename, file_bytes)
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