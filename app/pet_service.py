import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_service import AgentProcessingError
from app.config import Settings
from app.memory_service import store_candidate_memories
from app.model_gateway import GatewayError, GatewayResult, ModelGateway
from app.models import (
    AgentRun,
    InterviewSession,
    MemoryItem,
    PetProfile,
    VerificationFeedback,
)
from app.safety import redact_sensitive_text
from app.schemas import (
    CorrectionMemoryOutput,
    PetProfileOutput,
    VerificationRequest,
)


PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"
PET_PROMPT_VERSION = "v1"
VERIFICATION_PROMPT_VERSION = "v1"


@dataclass
class HatchedPet:
    pet: PetProfile
    run: AgentRun


@dataclass
class VerifiedPet:
    feedback: VerificationFeedback
    pet: PetProfile
    memories: list[MemoryItem]
    run: AgentRun | None


class PetAgentService:
    def __init__(self, gateway: ModelGateway, settings: Settings) -> None:
        self.gateway = gateway
        self.settings = settings
        self.pet_prompt = (
            PROMPTS_ROOT / "pet_generator" / f"{PET_PROMPT_VERSION}.txt"
        ).read_text(encoding="utf-8")
        self.verification_prompt = (
            PROMPTS_ROOT
            / "verification_memory"
            / f"{VERIFICATION_PROMPT_VERSION}.txt"
        ).read_text(encoding="utf-8")

    async def hatch(
        self,
        db: AsyncSession,
        session: InterviewSession,
        memories: list[MemoryItem],
    ) -> HatchedPet:
        started = perf_counter()
        messages = [
            {"role": "system", "content": self.pet_prompt},
            {
                "role": "system",
                "content": (
                    "输出 JSON Schema："
                    + json.dumps(
                        PetProfileOutput.model_json_schema(),
                        ensure_ascii=False,
                    )
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    [
                        {
                            "memory_key": item.memory_key,
                            "category": item.category,
                            "content": redact_sensitive_text(item.content),
                            "importance": item.importance,
                        }
                        for item in memories
                        if item.status in {"confirmed", "corrected"}
                    ],
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            result = await self.gateway.complete(messages)
            output = PetProfileOutput.model_validate_json(result.content)
        except GatewayError as exc:
            await self._record_failure(
                db,
                session.id,
                "pet_generator",
                PET_PROMPT_VERSION,
                started,
                exc.code,
            )
            raise AgentProcessingError(exc.code, "模型服务暂时不可用") from exc
        except ValidationError as exc:
            await self._record_failure(
                db,
                session.id,
                "pet_generator",
                PET_PROMPT_VERSION,
                started,
                "invalid_structured_output",
            )
            raise AgentProcessingError(
                "invalid_structured_output",
                "模型返回的宠物画像格式无效",
            ) from exc

        latest_version = (
            await db.scalar(
                select(func.max(PetProfile.version)).where(
                    PetProfile.session_id == session.id
                )
            )
            or 0
        )
        await db.execute(
            update(PetProfile)
            .where(
                PetProfile.session_id == session.id,
                PetProfile.status.in_(["active", "needs_revision"]),
            )
            .values(status="superseded")
        )
        pet = PetProfile(
            session_id=session.id,
            version=latest_version + 1,
            **output.model_dump(),
        )
        run = self._successful_run(
            session.id,
            "pet_generator",
            PET_PROMPT_VERSION,
            started,
            result,
        )
        db.add_all([pet, run])
        await db.commit()
        await db.refresh(pet)
        await db.refresh(run)
        return HatchedPet(pet=pet, run=run)

    async def verify(
        self,
        db: AsyncSession,
        pet: PetProfile,
        request: VerificationRequest,
    ) -> VerifiedPet:
        feedback = VerificationFeedback(
            pet_profile_id=pet.id,
            rating=request.rating,
            correction=request.correction,
        )
        db.add(feedback)
        pet.status = (
            "active" if request.rating == "accurate" else "needs_revision"
        )

        if not request.correction:
            await db.commit()
            await db.refresh(feedback)
            return VerifiedPet(
                feedback=feedback,
                pet=pet,
                memories=[],
                run=None,
            )

        started = perf_counter()
        messages = [
            {"role": "system", "content": self.verification_prompt},
            {
                "role": "system",
                "content": (
                    "输出 JSON Schema："
                    + json.dumps(
                        CorrectionMemoryOutput.model_json_schema(),
                        ensure_ascii=False,
                    )
                ),
            },
            {
                "role": "user",
                "content": redact_sensitive_text(request.correction),
            },
        ]
        try:
            result = await self.gateway.complete(messages)
            output = CorrectionMemoryOutput.model_validate_json(result.content)
        except GatewayError as exc:
            await self._record_failure(
                db,
                pet.session_id,
                "verification_memory",
                VERIFICATION_PROMPT_VERSION,
                started,
                exc.code,
            )
            raise AgentProcessingError(exc.code, "模型服务暂时不可用") from exc
        except ValidationError as exc:
            await self._record_failure(
                db,
                pet.session_id,
                "verification_memory",
                VERIFICATION_PROMPT_VERSION,
                started,
                "invalid_structured_output",
            )
            raise AgentProcessingError(
                "invalid_structured_output",
                "模型返回的纠正记忆格式无效",
            ) from exc

        memories = await store_candidate_memories(
            db,
            pet.session_id,
            output.candidate_memories,
            source_type="verification",
        )
        run = self._successful_run(
            pet.session_id,
            "verification_memory",
            VERIFICATION_PROMPT_VERSION,
            started,
            result,
        )
        db.add(run)
        await db.commit()
        await db.refresh(feedback)
        await db.refresh(pet)
        for memory in memories:
            await db.refresh(memory)
        await db.refresh(run)
        return VerifiedPet(
            feedback=feedback,
            pet=pet,
            memories=memories,
            run=run,
        )

    def _successful_run(
        self,
        session_id: str,
        agent_name: str,
        prompt_version: str,
        started: float,
        result: GatewayResult,
    ) -> AgentRun:
        input_cost = (
            result.usage.input_tokens
            * self.settings.model_input_price_per_million
            / 1_000_000
        )
        output_cost = (
            result.usage.output_tokens
            * self.settings.model_output_price_per_million
            / 1_000_000
        )
        return AgentRun(
            session_id=session_id,
            agent_name=agent_name,
            status="success",
            provider=self.gateway.provider_name,
            model_name=result.model_name,
            prompt_version=prompt_version,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            latency_ms=int((perf_counter() - started) * 1000),
            estimated_cost=input_cost + output_cost,
        )

    async def _record_failure(
        self,
        db: AsyncSession,
        session_id: str,
        agent_name: str,
        prompt_version: str,
        started: float,
        error_code: str,
    ) -> None:
        db.add(
            AgentRun(
                session_id=session_id,
                agent_name=agent_name,
                status="failed",
                provider=self.gateway.provider_name,
                model_name=self.settings.model_name,
                prompt_version=prompt_version,
                latency_ms=int((perf_counter() - started) * 1000),
                error_code=error_code,
            )
        )
        await db.commit()
