from typing import List, Optional
from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str


class MeetingCreate(BaseModel):
    title: str
    project_id: int
    agenda: Optional[str] = None
    participants: List[str] = []  # names, e.g. ["Rahul", "Ishita"]