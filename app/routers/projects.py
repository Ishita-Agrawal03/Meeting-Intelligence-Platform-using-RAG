from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Project, Meeting, Chunk, Task, Decision, Participant
from app.schemas.meeting import ProjectCreate
from app.services.pipeline import (
    save_uploaded_file,
    is_audio_video,
    process_meeting_document,
    process_meeting_audio_background,
)

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.post("/")
def create_or_get_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    """Idempotent: if a project with this exact name already exists,
    returns it instead of erroring — matches 'pick an old project or
    start a new one' with a single action, no separate existence check
    needed on the frontend."""
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "Project name is required.")

    existing = db.query(Project).filter(Project.name == name).first()
    if existing:
        return {"id": existing.id, "name": existing.name, "created": False}

    project = Project(name=name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return {"id": project.id, "name": project.name, "created": True}


@router.get("/")
def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.id.desc()).all()
    result = []
    for p in projects:
        meeting_count = db.query(Meeting).filter(Meeting.project_id == p.id).count()
        result.append({"id": p.id, "name": p.name, "meeting_count": meeting_count})
    return result


@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(404, "Project not found")
    db.delete(project)  # cascades to meetings -> chunks/tasks/decisions/participants
    db.commit()
    return {"message": "Project deleted successfully"}


@router.get("/{project_id}/meetings")
def list_project_meetings(project_id: int, db: Session = Depends(get_db)):
    return db.query(Meeting).filter(Meeting.project_id == project_id).order_by(Meeting.id.desc()).all()


@router.post("/{project_id}/upload")
async def upload_to_project(
    project_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """The primary upload path: creates a new meeting under this project
    AND processes the file in one call, so the frontend never has to
    separately 'create a meeting' before uploading to it."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(404, "Project not found")

    meeting = Meeting(title=file.filename, project_id=project_id, status="pending")
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

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
            "meeting_id": meeting.id,
            "status": "transcribing",
            "note": "Participants won't auto-populate for audio/video — "
                    "Whisper transcription has no speaker labels.",
        }

    meeting.status = "processing"
    db.commit()
    try:
        result = process_meeting_document(db, meeting, filepath)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "message": "Uploaded, processed, and indexed successfully.",
        "meeting_id": meeting.id,
        "status": "ready",
        **result,
    }


@router.get("/{project_id}/tasks")
def list_project_tasks(project_id: int, owner: str = None, db: Session = Depends(get_db)):
    query = (
        db.query(Task)
        .join(Meeting, Task.meeting_id == Meeting.id)
        .filter(Meeting.project_id == project_id)
    )
    if owner:
        query = query.filter(Task.owner.ilike(f"%{owner}%"))
    return query.all()


@router.get("/{project_id}/decisions")
def list_project_decisions(project_id: int, db: Session = Depends(get_db)):
    return (
        db.query(Decision)
        .join(Meeting, Decision.meeting_id == Meeting.id)
        .filter(Meeting.project_id == project_id)
        .all()
    )


@router.get("/{project_id}/participants")
def list_project_participants(project_id: int, db: Session = Depends(get_db)):
    """Deduped across every meeting in the project — the same person
    appearing in multiple meetings shows up once."""
    rows = (
        db.query(Participant)
        .join(Meeting, Participant.meeting_id == Meeting.id)
        .filter(Meeting.project_id == project_id)
        .all()
    )
    seen = {}
    for r in rows:
        seen[r.person_name] = r.person_name
    return [{"person_name": name} for name in seen.values()]