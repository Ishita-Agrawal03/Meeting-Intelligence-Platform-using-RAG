from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.database import Base


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    project = Column(String)
    agenda = Column(String)
    transcript_path = Column(String)
    source_type = Column(String)
    status = Column(String, default="pending")
    summary = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship("Chunk", back_populates="meeting", cascade="all, delete-orphan")
    participants = relationship("Participant", back_populates="meeting", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="meeting", cascade="all, delete-orphan")
    decisions = relationship("Decision", back_populates="meeting", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=False)
    chunk_text = Column(String, nullable=False)
    chunk_type = Column(String)   # "transcript" or "notes"
    speakers = Column(String)     # comma-separated names, "" if none
    position = Column(Integer, default=0)

    meeting = relationship("Meeting", back_populates="chunks")


class Participant(Base):
    __tablename__ = "participants"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=False)
    person_name = Column(String, nullable=False, index=True)

    meeting = relationship("Meeting", back_populates="participants")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=False)
    owner = Column(String)
    task = Column(String, nullable=False)
    deadline = Column(String)
    status = Column(String, default="pending")  # pending / completed
    source_chunk_id = Column(Integer, ForeignKey("chunks.id"))

    meeting = relationship("Meeting", back_populates="tasks")


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=False)
    decision = Column(String, nullable=False)
    source_chunk_id = Column(Integer, ForeignKey("chunks.id"))

    meeting = relationship("Meeting", back_populates="decisions")