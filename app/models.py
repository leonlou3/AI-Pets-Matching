from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    nickname: Mapped[str] = mapped_column(String(64))
    gender: Mapped[str] = mapped_column(String(16))
    orientation: Mapped[str] = mapped_column(String(32))
    city: Mapped[str] = mapped_column(String(64), index=True)
    birth_year: Mapped[int] = mapped_column(Integer)
    seeking_genders: Mapped[list[str]] = mapped_column(JSON)
    seeking_min_age: Mapped[int] = mapped_column(Integer)
    seeking_max_age: Mapped[int] = mapped_column(Integer)
    accept_remote: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)


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


class MatchRun(Base):
    __tablename__ = "match_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    eligible_users: Mapped[int] = mapped_column(Integer, default=0)
    pairs_considered: Mapped[int] = mapped_column(Integer, default=0)
    pairs_passed_hard_filter: Mapped[int] = mapped_column(Integer, default=0)
    pairs_passed_compatibility: Mapped[int] = mapped_column(Integer, default=0)
    pairs_dialogued: Mapped[int] = mapped_column(Integer, default=0)
    recommendations_created: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    pairs: Mapped[list["MatchPair"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class MatchPair(Base):
    __tablename__ = "match_pairs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("match_runs.id", ondelete="CASCADE"),
        index=True,
    )
    user_a_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    user_b_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    # Furthest funnel stage reached: hard_filter / compatibility / dialogue /
    # judgement / recommended
    stage: Mapped[str] = mapped_column(String(32), index=True)
    outcome: Mapped[str] = mapped_column(String(32), index=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    compatibility_score: Mapped[float] = mapped_column(Float, default=0)
    shared_memory_keys: Mapped[list[str]] = mapped_column(JSON, default=list)
    dialogue_transcript: Mapped[list[dict]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    run: Mapped[MatchRun] = relationship(back_populates="pairs")
    verdicts: Mapped[list["JudgeVerdict"]] = relationship(
        back_populates="pair",
        cascade="all, delete-orphan",
    )


class JudgeVerdict(Base):
    __tablename__ = "judge_verdicts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    pair_id: Mapped[str] = mapped_column(
        ForeignKey("match_pairs.id", ondelete="CASCADE"),
        index=True,
    )
    # Direction "a_for_b" means: is user A a good match for user B.
    direction: Mapped[str] = mapped_column(String(16))
    decision: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    hard_conflicts: Mapped[list[str]] = mapped_column(JSON, default=list)
    fit_evidence: Mapped[list[str]] = mapped_column(JSON, default=list)
    risks: Mapped[list[str]] = mapped_column(JSON, default=list)
    uncertainties: Mapped[list[str]] = mapped_column(JSON, default=list)
    icebreaker_suggestion: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    pair: Mapped[MatchPair] = relationship(back_populates="verdicts")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    pair_id: Mapped[str] = mapped_column(
        ForeignKey("match_pairs.id", ondelete="CASCADE"),
        index=True,
    )
    user_a_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    user_b_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    reason: Mapped[str] = mapped_column(Text)
    icebreaker_suggestion: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
