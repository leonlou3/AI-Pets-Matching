from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    memories: Mapped[list["MemoryItem"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="MemoryItem.created_at",
    )
    pet_profiles: Mapped[list["PetProfile"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="PetProfile.version",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    session: Mapped[InterviewSession] = relationship(back_populates="messages")


class MemoryItem(Base):
    __tablename__ = "memory_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    content: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), index=True)
    memory_key: Mapped[str] = mapped_column(String(128), index=True)
    importance: Mapped[int] = mapped_column(Integer, default=3)
    confidence: Mapped[float] = mapped_column(Float)
    evidence: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(32), default="interview")
    conflicts_with_id: Mapped[str | None] = mapped_column(
        ForeignKey("memory_items.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32), default="candidate", index=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    session: Mapped[InterviewSession] = relationship(back_populates="memories")


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    agent_name: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    provider: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(32))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)


class PetProfile(Base):
    __tablename__ = "pet_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    name: Mapped[str] = mapped_column(String(64))
    species: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(128))
    core_traits: Mapped[list[str]] = mapped_column(JSON)
    relationship_style: Mapped[str] = mapped_column(Text)
    communication_style: Mapped[str] = mapped_column(Text)
    strengths: Mapped[list[str]] = mapped_column(JSON)
    easily_misunderstood_as: Mapped[str] = mapped_column(Text)
    learning_topics: Mapped[list[str]] = mapped_column(JSON)
    declaration: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    session: Mapped[InterviewSession] = relationship(back_populates="pet_profiles")
    feedback: Mapped[list["VerificationFeedback"]] = relationship(
        back_populates="pet_profile",
        cascade="all, delete-orphan",
    )


class VerificationFeedback(Base):
    __tablename__ = "verification_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    pet_profile_id: Mapped[str] = mapped_column(
        ForeignKey("pet_profiles.id", ondelete="CASCADE"),
        index=True,
    )
    rating: Mapped[str] = mapped_column(String(32))
    correction: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    pet_profile: Mapped[PetProfile] = relationship(back_populates="feedback")
