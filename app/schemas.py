from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MemoryStatus = Literal[
    "candidate",
    "confirmed",
    "corrected",
    "rejected",
    "deleted",
    "superseded",
]


class CreateInterviewRequest(BaseModel):
    owner_id: str = Field(min_length=1, max_length=128)


class InterviewCreated(BaseModel):
    id: str
    owner_id: str
    status: str

    model_config = ConfigDict(from_attributes=True)


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class ExtractedMemory(BaseModel):
    content: str = Field(min_length=1, max_length=1000)
    memory_key: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[a-z0-9_.-]+$",
        description="Stable semantic key, for example relationship.distance",
    )
    category: Literal[
        "objective_fact",
        "relationship_goal",
        "hard_boundary",
        "preference",
        "value",
        "lifestyle",
        "communication",
        "uncertain",
    ]
    confidence: float = Field(ge=0, le=1)
    importance: int = Field(default=3, ge=1, le=5)
    evidence: str = Field(min_length=1, max_length=1000)


class InterviewProgress(BaseModel):
    covered_categories: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    focus_topic: str | None = None


class AgentTurnOutput(BaseModel):
    reply: str = Field(min_length=1, max_length=4000)
    candidate_memories: list[ExtractedMemory] = Field(default_factory=list, max_length=8)
    progress: InterviewProgress = Field(default_factory=InterviewProgress)


class MemoryView(BaseModel):
    id: str
    content: str
    category: str
    memory_key: str
    importance: int
    confidence: float
    evidence: str
    source_type: str
    conflicts_with_id: str | None
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageView(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgentMetrics(BaseModel):
    run_id: str
    model_name: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    estimated_cost: float


class TurnResponse(BaseModel):
    reply: str
    candidate_memories: list[MemoryView]
    progress: InterviewProgress
    metrics: AgentMetrics


class InterviewDetail(BaseModel):
    id: str
    owner_id: str
    status: str
    messages: list[MessageView]
    memories: list[MemoryView]


class ReviewMemoryRequest(BaseModel):
    action: Literal["confirm", "correct", "reject", "delete"]
    corrected_content: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def require_correction(self) -> "ReviewMemoryRequest":
        if self.action == "correct" and not self.corrected_content:
            raise ValueError("corrected_content is required when action is correct")
        if self.action != "correct" and self.corrected_content is not None:
            raise ValueError("corrected_content is only allowed when action is correct")
        return self


class HatchingReadiness(BaseModel):
    ready: bool
    confirmed_memory_count: int
    covered_categories: list[str]
    missing_categories: list[str]
    message: str


PetSpecies = Literal[
    "cat",
    "dog",
    "fox",
    "rabbit",
    "otter",
    "deer",
    "owl",
    "capybara",
    "red_panda",
    "penguin",
]


class PetProfileOutput(BaseModel):
    name: str = Field(min_length=1, max_length=20)
    species: PetSpecies
    title: str = Field(min_length=1, max_length=40)
    core_traits: list[str] = Field(min_length=3, max_length=3)
    relationship_style: str = Field(min_length=1, max_length=500)
    communication_style: str = Field(min_length=1, max_length=500)
    strengths: list[str] = Field(min_length=1, max_length=4)
    easily_misunderstood_as: str = Field(min_length=1, max_length=500)
    learning_topics: list[str] = Field(default_factory=list, max_length=5)
    declaration: str = Field(min_length=1, max_length=200)


class PetProfileView(PetProfileOutput):
    id: str
    session_id: str
    version: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HatchPetResponse(BaseModel):
    pet: PetProfileView
    readiness: HatchingReadiness
    metrics: AgentMetrics


class VerificationRequest(BaseModel):
    rating: Literal["accurate", "partly_accurate", "inaccurate"]
    correction: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def require_feedback_for_low_rating(self) -> "VerificationRequest":
        if self.rating != "accurate" and not self.correction:
            raise ValueError("correction is required unless rating is accurate")
        return self


class CorrectionMemoryOutput(BaseModel):
    candidate_memories: list[ExtractedMemory] = Field(default_factory=list, max_length=8)


class VerificationResponse(BaseModel):
    feedback_id: str
    pet_status: str
    candidate_memories: list[MemoryView]
    metrics: AgentMetrics | None = None
