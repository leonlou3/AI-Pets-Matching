import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_service import AgentProcessingError
from app.config import Settings
from app.model_gateway import GatewayError, GatewayResult, ModelGateway
from app.models import (
    AgentRun,
    InterviewSession,
    JudgeVerdict,
    MatchPair,
    MatchRun,
    MemoryItem,
    PetProfile,
    Recommendation,
    User,
)
from app.safety import redact_sensitive_text
from app.schemas import DialogueTurnOutput, JudgeOutput

PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"
DIALOGUE_PROMPT_VERSION = "v1"
JUDGE_PROMPT_VERSION = "v1"

ACTIVE_MEMORY_STATUSES = ("confirmed", "corrected")
USABLE_PET_STATUSES = ("active", "needs_revision")


@dataclass
class EligibleUser:
    user: User
    pet: PetProfile
    memories: list[MemoryItem]

    def desensitized_memories(self) -> list[dict]:
        return [
            {
                "memory_key": item.memory_key,
                "category": item.category,
                "content": redact_sensitive_text(item.content),
                "importance": item.importance,
            }
            for item in self.memories
        ]


@dataclass
class PairEvaluation:
    pair: MatchPair
    recommendation: Recommendation | None = None
    verdicts: list[JudgeVerdict] = field(default_factory=list)


def current_age(birth_year: int) -> int:
    return datetime.now(UTC).year - birth_year


def hard_filter(a: User, b: User) -> tuple[bool, str | None]:
    """Deterministic code-level filter. Model output is never involved here."""
    if a.gender not in b.seeking_genders:
        return False, f"{b.nickname} 不寻找 {a.gender} 用户"
    if b.gender not in a.seeking_genders:
        return False, f"{a.nickname} 不寻找 {b.gender} 用户"

    age_a, age_b = current_age(a.birth_year), current_age(b.birth_year)
    if not (a.seeking_min_age <= age_b <= a.seeking_max_age):
        return False, f"{b.nickname} 的年龄不在 {a.nickname} 的接受范围内"
    if not (b.seeking_min_age <= age_a <= b.seeking_max_age):
        return False, f"{a.nickname} 的年龄不在 {b.nickname} 的接受范围内"

    if a.city != b.city and not (a.accept_remote and b.accept_remote):
        return False, "双方不在同一城市，且至少一方不接受异地"
    return True, None


def compatibility(
    a_memories: list[MemoryItem],
    b_memories: list[MemoryItem],
) -> tuple[float, list[str]]:
    """Importance-weighted overlap of confirmed memory topics."""
    a_keys = {item.memory_key: item.importance for item in a_memories}
    b_keys = {item.memory_key: item.importance for item in b_memories}
    shared = sorted(set(a_keys) & set(b_keys))
    union = set(a_keys) | set(b_keys)
    if not union:
        return 0.0, []

    shared_weight = sum(max(a_keys[key], b_keys[key]) for key in shared)
    union_weight = sum(
        max(a_keys.get(key, 0), b_keys.get(key, 0)) for key in union
    )
    return round(shared_weight / union_weight, 4), shared


class MatchingService:
    def __init__(self, gateway: ModelGateway, settings: Settings) -> None:
        self.gateway = gateway
        self.settings = settings
        self.dialogue_prompt = (
            PROMPTS_ROOT / "pet_dialogue" / f"{DIALOGUE_PROMPT_VERSION}.txt"
        ).read_text(encoding="utf-8")
        self.judge_prompt = (
            PROMPTS_ROOT / "judge" / f"{JUDGE_PROMPT_VERSION}.txt"
        ).read_text(encoding="utf-8")

    async def eligible_users(self, db: AsyncSession) -> list[EligibleUser]:
        users = list(
            (
                await db.scalars(
                    select(User).where(User.status == "active").order_by(User.created_at)
                )
            ).all()
        )
        eligible: list[EligibleUser] = []
        for user in users:
            pet = await db.scalar(
                select(PetProfile)
                .join(InterviewSession)
                .where(
                    InterviewSession.owner_id == user.id,
                    PetProfile.status.in_(USABLE_PET_STATUSES),
                )
                .order_by(PetProfile.version.desc())
            )
            if pet is None:
                continue
            memories = list(
                (
                    await db.scalars(
                        select(MemoryItem)
                        .join(InterviewSession)
                        .where(
                            InterviewSession.owner_id == user.id,
                            MemoryItem.status.in_(ACTIVE_MEMORY_STATUSES),
                        )
                        .order_by(MemoryItem.created_at)
                    )
                ).all()
            )
            if not memories:
                continue
            eligible.append(EligibleUser(user=user, pet=pet, memories=memories))
        return eligible

    async def run_round(self, db: AsyncSession) -> MatchRun:
        participants = await self.eligible_users(db)
        run = MatchRun(eligible_users=len(participants))
        db.add(run)
        await db.flush()

        for side_a, side_b in combinations(participants, 2):
            run.pairs_considered += 1
            evaluation = await self._evaluate_pair(db, run, side_a, side_b)
            if evaluation.pair.stage != "hard_filter":
                run.pairs_passed_hard_filter += 1
            if evaluation.pair.stage in {"dialogue", "judgement", "recommended"}:
                run.pairs_passed_compatibility += 1
            if evaluation.pair.dialogue_transcript:
                run.pairs_dialogued += 1
            if evaluation.recommendation is not None:
                run.recommendations_created += 1

        await db.commit()
        await db.refresh(run)
        return run

    async def _evaluate_pair(
        self,
        db: AsyncSession,
        run: MatchRun,
        side_a: EligibleUser,
        side_b: EligibleUser,
    ) -> PairEvaluation:
        pair = MatchPair(
            run_id=run.id,
            user_a_id=side_a.user.id,
            user_b_id=side_b.user.id,
            stage="hard_filter",
            outcome="rejected",
        )
        db.add(pair)
        await db.flush()

        passed, reason = hard_filter(side_a.user, side_b.user)
        if not passed:
            pair.failure_reason = reason
            await db.flush()
            return PairEvaluation(pair=pair)

        score, shared_keys = compatibility(side_a.memories, side_b.memories)
        pair.stage = "compatibility"
        pair.compatibility_score = score
        pair.shared_memory_keys = shared_keys
        if (
            len(shared_keys) < self.settings.matching_min_shared_memory_keys
            or score < self.settings.matching_min_compatibility_score
        ):
            pair.failure_reason = (
                f"共同确认主题不足（{len(shared_keys)} 个，得分 {score}）"
            )
            await db.flush()
            return PairEvaluation(pair=pair)

        pair.stage = "dialogue"
        transcript = await self._run_dialogue(db, side_a, side_b)
        pair.dialogue_transcript = transcript

        pair.stage = "judgement"
        verdicts: list[JudgeVerdict] = []
        decisions: dict[str, JudgeOutput] = {}
        for direction, owner, candidate in (
            ("b_for_a", side_a, side_b),
            ("a_for_b", side_b, side_a),
        ):
            output = await self._judge(db, owner, candidate, shared_keys, transcript)
            decisions[direction] = output
            verdict = JudgeVerdict(
                pair_id=pair.id,
                direction=direction,
                decision=output.decision,
                confidence=output.confidence,
                hard_conflicts=output.hard_conflicts,
                fit_evidence=output.fit_evidence,
                risks=output.risks,
                uncertainties=output.uncertainties,
                icebreaker_suggestion=output.icebreaker_suggestion,
            )
            db.add(verdict)
            verdicts.append(verdict)

        both_passed = all(
            output.decision == "pass"
            and output.confidence >= self.settings.matching_judge_min_confidence
            for output in decisions.values()
        )
        if not both_passed:
            failed = [
                direction
                for direction, output in decisions.items()
                if output.decision != "pass"
                or output.confidence < self.settings.matching_judge_min_confidence
            ]
            pair.failure_reason = f"裁判未通过方向：{', '.join(failed)}"
            await db.flush()
            return PairEvaluation(pair=pair, verdicts=verdicts)

        pair.stage = "recommended"
        pair.outcome = "recommended"
        evidence = decisions["b_for_a"].fit_evidence + decisions["a_for_b"].fit_evidence
        recommendation = Recommendation(
            pair_id=pair.id,
            user_a_id=side_a.user.id,
            user_b_id=side_b.user.id,
            reason="；".join(dict.fromkeys(evidence)) or "双方裁判均通过",
            icebreaker_suggestion=decisions["b_for_a"].icebreaker_suggestion
            or decisions["a_for_b"].icebreaker_suggestion,
        )
        db.add(recommendation)
        await db.flush()
        return PairEvaluation(
            pair=pair,
            recommendation=recommendation,
            verdicts=verdicts,
        )

    async def _run_dialogue(
        self,
        db: AsyncSession,
        side_a: EligibleUser,
        side_b: EligibleUser,
    ) -> list[dict]:
        transcript: list[dict] = []
        for round_no in range(1, self.settings.matching_dialogue_rounds + 1):
            for speaker_label, speaker in (("pet_a", side_a), ("pet_b", side_b)):
                output = await self._dialogue_turn(
                    db,
                    speaker,
                    transcript,
                    round_no,
                )
                transcript.append(
                    {
                        "round_no": round_no,
                        "speaker": speaker_label,
                        "utterance": redact_sensitive_text(output.utterance),
                    }
                )
                if output.wants_to_end:
                    return transcript
        return transcript

    async def _dialogue_turn(
        self,
        db: AsyncSession,
        speaker: EligibleUser,
        transcript: list[dict],
        round_no: int,
    ) -> DialogueTurnOutput:
        payload = {
            "pet_name": speaker.pet.name,
            "round_no": round_no,
            "max_rounds": self.settings.matching_dialogue_rounds,
            "own_memories": speaker.desensitized_memories(),
            "transcript": transcript,
        }
        content = await self._call_agent(
            db,
            agent_name="pet_dialogue",
            prompt=self.dialogue_prompt,
            prompt_version=DIALOGUE_PROMPT_VERSION,
            schema=DialogueTurnOutput,
            payload=payload,
        )
        return content

    async def _judge(
        self,
        db: AsyncSession,
        owner: EligibleUser,
        candidate: EligibleUser,
        shared_keys: list[str],
        transcript: list[dict],
    ) -> JudgeOutput:
        payload = {
            "owner_memories": owner.desensitized_memories(),
            "candidate_memories": candidate.desensitized_memories(),
            "shared_memory_keys": shared_keys,
            "transcript": transcript,
        }
        return await self._call_agent(
            db,
            agent_name="judge",
            prompt=self.judge_prompt,
            prompt_version=JUDGE_PROMPT_VERSION,
            schema=JudgeOutput,
            payload=payload,
        )

    async def _call_agent(
        self,
        db: AsyncSession,
        agent_name: str,
        prompt: str,
        prompt_version: str,
        schema,
        payload: dict,
    ):
        started = perf_counter()
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "system",
                "content": (
                    "输出 JSON Schema："
                    + json.dumps(schema.model_json_schema(), ensure_ascii=False)
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            result = await self.gateway.complete(messages)
            output = schema.model_validate_json(result.content)
        except GatewayError as exc:
            self._record_run(db, agent_name, prompt_version, started, error=exc.code)
            raise AgentProcessingError(exc.code, "模型服务暂时不可用") from exc
        except ValidationError as exc:
            self._record_run(
                db,
                agent_name,
                prompt_version,
                started,
                error="invalid_structured_output",
            )
            raise AgentProcessingError(
                "invalid_structured_output",
                "模型返回的数据格式无效",
            ) from exc

        self._record_run(db, agent_name, prompt_version, started, result=result)
        return output

    def _record_run(
        self,
        db: AsyncSession,
        agent_name: str,
        prompt_version: str,
        started: float,
        result: GatewayResult | None = None,
        error: str | None = None,
    ) -> None:
        if result is not None:
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
            db.add(
                AgentRun(
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
            )
        else:
            db.add(
                AgentRun(
                    agent_name=agent_name,
                    status="failed",
                    provider=self.gateway.provider_name,
                    model_name=self.settings.model_name,
                    prompt_version=prompt_version,
                    latency_ms=int((perf_counter() - started) * 1000),
                    error_code=error,
                )
            )
