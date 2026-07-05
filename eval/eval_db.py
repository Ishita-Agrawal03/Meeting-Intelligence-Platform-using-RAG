"""
Evaluation database — deliberately SEPARATE from meetings.db.

Reasoning (from earlier design discussion): evaluation data is
development/research metadata, not application state. Nothing in
the live app ever reads from this — deleting it wouldn't break the
product. Keeping it in its own file means you can wipe and rebuild
it anytime without touching real meeting data, and it never needs
to follow meetings.db if that ever migrates to a different database.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func

EVAL_DATABASE_URL = "sqlite:///evaluation.db"

engine = create_engine(EVAL_DATABASE_URL, connect_args={"check_same_thread": False})
EvalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
EvalBase = declarative_base()


class EvaluationRun(EvalBase):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_at = Column(DateTime(timezone=True), server_default=func.now())
    retrieval_hit_rate = Column(Float)
    answer_accuracy = Column(Float)
    notes = Column(String)

    results = relationship("EvaluationResult", back_populates="run", cascade="all, delete-orphan")


class EvaluationResult(EvalBase):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False)
    question = Column(String, nullable=False)
    expected_meeting_title = Column(String)
    retrieved_chunk_ids = Column(String)  # stored as comma-separated ids
    retrieval_correct = Column(Boolean)
    answer_correct = Column(Boolean)
    answer_text = Column(String)

    run = relationship("EvaluationRun", back_populates="results")


def init_eval_db():
    EvalBase.metadata.create_all(bind=engine)