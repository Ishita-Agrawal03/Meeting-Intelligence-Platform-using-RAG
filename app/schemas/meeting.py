from pydantic import BaseModel


class MeetingCreate(BaseModel):
    title: str
    project: str