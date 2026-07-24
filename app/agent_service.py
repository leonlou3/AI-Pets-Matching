import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.memory_service import store_candidate_memories
from app.model_gateway import GatewayError, GatewayResult, ModelGateway
from app.models import AgentRun, InterviewSession, MemoryItem, Message
from app.safety import redact_sensitive_text
from app.schemas import AgentTurnOutput


PROMPT_VERSION = "v2"
PROMPT_PATH = (
    Path(__file__).resolve().parent
    / "prompts"
    / "interviewer"
    / f"{PROMPT_VERSION}.txt"
)


class AgentProcessingError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ProcessedTurn:
    output: AgentTurnOutput
    memories: list[MemoryItem]
    run: AgentRun


class InterviewAgentService:
    def __init__(self, gateway: ModelGateway, settings: Settings) -> None:
        self.gateway = gateway
        self.settings = settings
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    async def process_turn(
        self,
        db: AsyncSession,
        session: InterviewSession,
        user_content: str,
    ) -> ProcessedTurn:
        started = perf_counter()
        user_message = Message(
            session_id=session.id,
            role="user",
            content=user_content,
        )
        db.add(user_message)
        await db.flush()

        statement = (
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.created_at.desc())
            .limit(12)
        )
        recent_messages = list((await db.scalars(statement)).all())
        recent_messages.reverse()
        confirmed_memories = list(
            (
                await db.scalars(
                    select(MemoryItem).where(
                        MemoryItem.session_id == session.id,
                        MemoryItem.status.in_(["confirmed", "corrected"]),
                    )
                )
            ).all()
        )

        schema_json = json.dumps(
            AgentTurnOutput.model_json_schema(),
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "system",
                "content": f"输出 JSON Schema：{schema_json}",
            },
            {
                "role": "system",
                "content": (
                    "主人已经确认的记忆："
                    + json.dumps(
                        [
                            {
                                "memory_key": memory.memory_key,
                                "category": memory.category,
                                "content": redact_sensitive_text(memory.content),
                                "importance": memory.importance,
                            }
                            for memory in confirmed_memories
                        ],
                        ensure_ascii=False,
                    )
                ),
            },
            *[
                {
                    "role": message.role,
                    "content": redact_sensitive_text(message.content),
                }
                for message in recent_messages
            ],
        ]

        try:
            result = await self.gateway.complete(messages)
            output = AgentTurnOutput.model_validate_json(result.content)
        except GatewayError as exc:
            await self._record_failure(
                db,
                session.id,
                started,
                exc.code,
            )
            raise AgentProcessingError(exc.code, "模型服务暂时不可用") from exc
        except ValidationError as exc:
            await self._record_failure(
                db,
                session.id,
                started,
                "invalid_structured_output",
            )
            raise AgentProcessingError(
                "invalid_structured_output",
                "模型返回的数据格式无效",
            ) from exc

        assistant_message = Message(
            session_id=session.id,
            role="assistant",
            content=output.reply,
        )
        db.add(assistant_message)

        memories = await store_candidate_memories(
            db,
            session.id,
            output.candidate_memories,
            source_type="interview",
        )

        run = self._successful_run(session.id, started, result)
        db.add(run)
        await db.commit()
        for memory in memories:
            await db.refresh(memory)
        await db.refresh(run)
        return ProcessedTurn(output=output, memories=memories, run=run)

    def _successful_run(
        self,
        session_id: str,
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
            agent_name="interviewer",
            status="success",
            provider=self.gateway.provider_name,
            model_name=result.model_name,
            prompt_version=PROMPT_VERSION,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            latency_ms=int((perf_counter() - started) * 1000),
            estimated_cost=input_cost + output_cost,
        )

    async def _record_failure(
        self,
        db: AsyncSession,
        session_id: str,
        started: float,
        error_code: str,
    ) -> None:
        db.add(
            AgentRun(
                session_id=session_id,
                agent_name="interviewer",
                status="failed",
                provider=self.gateway.provider_name,
                model_name=self.settings.model_name,
                prompt_version=PROMPT_VERSION,
                latency_ms=int((perf_counter() - started) * 1000),
                error_code=error_code,
            )
        )
        await db.commit()
