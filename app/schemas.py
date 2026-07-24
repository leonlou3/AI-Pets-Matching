from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MemoryStatus = Literal[
    "candidate",
    "confirmed",
    "corrected",
    "rejected",
    "deleted",
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
    evidence: str = Field(min_length=1, max_length=1000)


class AgentTurnOutput(BaseModel):
    reply: str = Field(min_length=1, max_length=4000)
    candidate_memories: list[ExtractedMemory] = Field(default_factory=list, max_length=8)


class MemoryView(BaseModel):
    id: str
    content: str
    category: str
    confidence: float
    evidence: str
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
