from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent_service import AgentProcessingError, InterviewAgentService
from app.matching_service import MatchingService
from app.memory_service import (
    apply_memory_review,
    calculate_hatching_readiness,
)
from app.models import (
    AgentRun,
    InterviewSession,
    MatchPair,
    MatchRun,
    MemoryItem,
    PetProfile,
    Recommendation,
    User,
)
from app.pet_service import PetAgentService
from app.schemas import (
    AgentMetrics,
    CreateInterviewRequest,
    CreateUserRequest,
    HatchPetResponse,
    HatchingReadiness,
    InterviewCreated,
    InterviewDetail,
    MatchPairView,
    MatchRunDetail,
    MatchRunSummary,
    MemoryView,
    PetProfileView,
    RecommendationView,
    ReviewMemoryRequest,
    SendMessageRequest,
    TurnResponse,
    UserView,
    VerificationRequest,
    VerificationResponse,
)


router = APIRouter()


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    async for session in request.app.state.database.session():
        yield session


def get_agent_service(request: Request) -> InterviewAgentService:
    return request.app.state.agent_service


def get_pet_service(request: Request) -> PetAgentService:
    return request.app.state.pet_service


def get_matching_service(request: Request) -> MatchingService:
    return request.app.state.matching_service


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
        progress=turn.output.progress,
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

    reviewed_memory = await apply_memory_review(db, memory, request)
    await db.commit()
    await db.refresh(reviewed_memory)
    return reviewed_memory


@router.get(
    "/v1/interviews/{session_id}/hatching-readiness",
    response_model=HatchingReadiness,
)
async def get_hatching_readiness(
    session_id: str,
    owner_id: str = Depends(owner_header),
    db: AsyncSession = Depends(get_db),
) -> HatchingReadiness:
    session = await db.scalar(
        select(InterviewSession)
        .where(
            InterviewSession.id == session_id,
            InterviewSession.owner_id == owner_id,
        )
        .options(selectinload(InterviewSession.memories))
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Interview session not found")
    return calculate_hatching_readiness(session.memories)


@router.post(
    "/v1/interviews/{session_id}/pet",
    response_model=HatchPetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def hatch_pet(
    session_id: str,
    owner_id: str = Depends(owner_header),
    db: AsyncSession = Depends(get_db),
    pet_service: PetAgentService = Depends(get_pet_service),
) -> HatchPetResponse:
    session = await db.scalar(
        select(InterviewSession)
        .where(
            InterviewSession.id == session_id,
            InterviewSession.owner_id == owner_id,
        )
        .options(selectinload(InterviewSession.memories))
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Interview session not found")

    readiness = calculate_hatching_readiness(session.memories)
    if not readiness.ready:
        raise HTTPException(
            status_code=409,
            detail=readiness.model_dump(),
        )

    try:
        result = await pet_service.hatch(db, session, session.memories)
    except AgentProcessingError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    return HatchPetResponse(
        pet=PetProfileView.model_validate(result.pet),
        readiness=readiness,
        metrics=_agent_metrics(result.run),
    )


@router.get(
    "/v1/interviews/{session_id}/pet",
    response_model=PetProfileView,
)
async def get_current_pet(
    session_id: str,
    owner_id: str = Depends(owner_header),
    db: AsyncSession = Depends(get_db),
) -> PetProfile:
    pet = await db.scalar(
        select(PetProfile)
        .join(InterviewSession)
        .where(
            PetProfile.session_id == session_id,
            PetProfile.status.in_(["active", "needs_revision"]),
            InterviewSession.owner_id == owner_id,
        )
        .order_by(PetProfile.version.desc())
    )
    if pet is None:
        raise HTTPException(status_code=404, detail="Active pet profile not found")
    return pet


@router.post(
    "/v1/pets/{pet_id}/verification",
    response_model=VerificationResponse,
)
async def verify_pet(
    pet_id: str,
    request: VerificationRequest,
    owner_id: str = Depends(owner_header),
    db: AsyncSession = Depends(get_db),
    pet_service: PetAgentService = Depends(get_pet_service),
) -> VerificationResponse:
    pet = await db.scalar(
        select(PetProfile)
        .join(InterviewSession)
        .where(
            PetProfile.id == pet_id,
            InterviewSession.owner_id == owner_id,
        )
    )
    if pet is None:
        raise HTTPException(status_code=404, detail="Pet profile not found")
    if pet.status == "superseded":
        raise HTTPException(
            status_code=409,
            detail="Superseded pet profiles cannot be verified",
        )

    try:
        result = await pet_service.verify(db, pet, request)
    except AgentProcessingError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    return VerificationResponse(
        feedback_id=result.feedback.id,
        pet_status=result.pet.status,
        candidate_memories=[
            MemoryView.model_validate(item) for item in result.memories
        ],
        metrics=_agent_metrics(result.run) if result.run else None,
    )


def _agent_metrics(run: AgentRun) -> AgentMetrics:
    return AgentMetrics(
        run_id=run.id,
        model_name=run.model_name,
        prompt_version=run.prompt_version,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        latency_ms=run.latency_ms,
        estimated_cost=run.estimated_cost,
    )


@router.post(
    "/v1/users",
    response_model=UserView,
    status_code=status.HTTP_201_CREATED,
)
async def create_user(
    request: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
) -> User:
    user = User(**request.model_dump())
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/v1/users", response_model=list[UserView])
async def list_users(db: AsyncSession = Depends(get_db)) -> list[User]:
    return list((await db.scalars(select(User).order_by(User.created_at))).all())


@router.post(
    "/v1/matching/rounds",
    response_model=MatchRunSummary,
    status_code=status.HTTP_201_CREATED,
)
async def run_matching_round(
    db: AsyncSession = Depends(get_db),
    matching_service: MatchingService = Depends(get_matching_service),
) -> MatchRun:
    try:
        return await matching_service.run_round(db)
    except AgentProcessingError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc


@router.get("/v1/matching/rounds/{run_id}", response_model=MatchRunDetail)
async def get_matching_round(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> MatchRunDetail:
    run = await db.scalar(
        select(MatchRun)
        .where(MatchRun.id == run_id)
        .options(selectinload(MatchRun.pairs).selectinload(MatchPair.verdicts))
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Match run not found")

    pair_ids = [pair.id for pair in run.pairs]
    recommendations = list(
        (
            await db.scalars(
                select(Recommendation).where(Recommendation.pair_id.in_(pair_ids))
            )
        ).all()
    ) if pair_ids else []

    return MatchRunDetail(
        id=run.id,
        status=run.status,
        eligible_users=run.eligible_users,
        pairs_considered=run.pairs_considered,
        pairs_passed_hard_filter=run.pairs_passed_hard_filter,
        pairs_passed_compatibility=run.pairs_passed_compatibility,
        pairs_dialogued=run.pairs_dialogued,
        recommendations_created=run.recommendations_created,
        created_at=run.created_at,
        pairs=[MatchPairView.model_validate(pair) for pair in run.pairs],
        recommendations=[
            RecommendationView.model_validate(item) for item in recommendations
        ],
    )


@router.get(
    "/v1/users/{user_id}/recommendations",
    response_model=list[RecommendationView],
)
async def list_user_recommendations(
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[Recommendation]:
    return list(
        (
            await db.scalars(
                select(Recommendation)
                .where(
                    (Recommendation.user_a_id == user_id)
                    | (Recommendation.user_b_id == user_id)
                )
                .order_by(Recommendation.created_at.desc())
            )
        ).all()
    )
