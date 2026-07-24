from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent_service import AgentProcessingError, InterviewAgentService
from app.models import InterviewSession, MemoryItem
from app.schemas import (
    AgentMetrics,
    CreateInterviewRequest,
    InterviewCreated,
    InterviewDetail,
    MemoryView,
    ReviewMemoryRequest,
    SendMessageRequest,
    TurnResponse,
)


router = APIRouter()


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    async for session in request.app.state.database.session():
        yield session


def get_agent_service(request: Request) -> InterviewAgentService:
    return request.app.state.agent_service


async def owner_header(
    x_owner_id: str = Header(min_length=1, max_length=128),
) -> str:
    # This is an explicit development seam. Production must replace it with
    # a verified user id from WeChat login or another authentication provider.
    return x_owner_id


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/v1/interviews",
    response_model=InterviewCreated,
    status_code=status.HTTP_201_CREATED,
)
async def create_interview(
    request: CreateInterviewRequest,
    db: AsyncSession = Depends(get_db),
) -> InterviewSession:
    session = InterviewSession(owner_id=request.owner_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.post(
    "/v1/interviews/{session_id}/messages",
    response_model=TurnResponse,
)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    owner_id: str = Depends(owner_header),
    db: AsyncSession = Depends(get_db),
    agent_service: InterviewAgentService = Depends(get_agent_service),
) -> TurnResponse:
    session = await db.scalar(
        select(InterviewSession).where(
            InterviewSession.id == session_id,
            InterviewSession.owner_id == owner_id,
        )
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Interview session not found")

    try:
        turn = await agent_service.process_turn(db, session, request.content)
    except AgentProcessingError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    return TurnResponse(
        reply=turn.output.reply,
        candidate_memories=[MemoryView.model_validate(item) for item in turn.memories],
        metrics=AgentMetrics(
            run_id=turn.run.id,
            model_name=turn.run.model_name,
            prompt_version=turn.run.prompt_version,
            input_tokens=turn.run.input_tokens,
            output_tokens=turn.run.output_tokens,
            latency_ms=turn.run.latency_ms,
            estimated_cost=turn.run.estimated_cost,
        ),
    )


@router.get(
    "/v1/interviews/{session_id}",
    response_model=InterviewDetail,
)
async def get_interview(
    session_id: str,
    owner_id: str = Depends(owner_header),
    db: AsyncSession = Depends(get_db),
) -> InterviewDetail:
    session = await db.scalar(
        select(InterviewSession)
        .where(
            InterviewSession.id == session_id,
            InterviewSession.owner_id == owner_id,
        )
        .options(
            selectinload(InterviewSession.messages),
            selectinload(InterviewSession.memories),
        )
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Interview session not found")
    return InterviewDetail(
        id=session.id,
        owner_id=session.owner_id,
        status=session.status,
        messages=session.messages,
        memories=session.memories,
    )


@router.patch("/v1/memories/{memory_id}", response_model=MemoryView)
async def review_memory(
    memory_id: str,
    request: ReviewMemoryRequest,
    owner_id: str = Depends(owner_header),
    db: AsyncSession = Depends(get_db),
) -> MemoryItem:
    memory = await db.scalar(
        select(MemoryItem)
        .join(InterviewSession)
        .where(
            MemoryItem.id == memory_id,
            InterviewSession.owner_id == owner_id,
        )
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    if memory.status == "deleted":
        raise HTTPException(status_code=409, detail="Memory has been deleted")

    status_by_action = {
        "confirm": "confirmed",
        "correct": "corrected",
        "reject": "rejected",
        "delete": "deleted",
    }
    if request.action == "correct":
        memory.content = request.corrected_content or memory.content
    memory.status = status_by_action[request.action]
    await db.commit()
    await db.refresh(memory)
    return memory
