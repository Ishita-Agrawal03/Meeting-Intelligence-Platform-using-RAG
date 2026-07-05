from typing import List, Optional
from pydantic import BaseModel


class MeetingCreate(BaseModel):
    title: str
    project: str
    agenda: Optional[str] = None
    participants: List[str] = []  # names, e.g. ["Rahul", "Ishita"]