from typing import List, Optional
from pydantic import BaseModel


class ChatTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    meeting_id: Optional[int] = None    # restrict to one meeting
    project_id: Optional[int] = None    # restrict to all meetings in a project
    chat_history: List[ChatTurn] = []


class Citation(BaseModel):
    chunk_id: int
    meeting_id: int
    meeting_title: str
    project_id: int
    project_name: str
    source_text: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_chunk_ids: List[int]