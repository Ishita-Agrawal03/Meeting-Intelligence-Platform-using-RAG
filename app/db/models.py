from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.database import Base


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    project = Column(String)
    transcript_path = Column(String)
    source_type = Column(String)
    status = Column(String, default="pending")
    summary = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    chunks = relationship("Chunk", back_populates="meeting", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=False)
    chunk_text = Column(String, nullable=False)
    chunk_type = Column(String)
    speakers = Column(String)
    position = Column(Integer, default=0)
    meeting = relationship("Meeting", back_populates="chunks")