import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from app.config import get_settings
from app.model_gateway import build_gateway
from app.safety import contains_sensitive_text, redact_sensitive_text
from app.schemas import AgentTurnOutput


APP_ROOT = Path(__file__).resolve().parent


class EvaluationCase(BaseModel):
    id: str
    input: str
    expected_category: str | None
    expected_memory_key: str | None
    expected_memory_count: int


async def run_evaluations() -> int:
    settings = get_settings()
    gateway = build_gateway(settings)
    prompt = (APP_ROOT / "prompts" / "interviewer" / "v2.txt").read_text(
        encoding="utf-8"
    )
    cases = [
        EvaluationCase.model_validate(item)
        for item in json.loads(
            (APP_ROOT / "evals" / "p1_cases.json").read_text(encoding="utf-8")
        )
    ]
    failures: list[str] = []

    try:
        for case in cases:
            result = await gateway.complete(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "system",
                        "content": (
                            "输出 JSON Schema："
                            + json.dumps(
                                AgentTurnOutput.model_json_schema(),
                                ensure_ascii=False,
                            )
                        ),
                    },
                    {"role": "system", "content": "主人已经确认的记忆：[]"},
                    {
                        "role": "user",
                        "content": redact_sensitive_text(case.input),
                    },
                ]
            )
            output = AgentTurnOutput.model_validate_json(result.content)
            errors = score_case(case, output)
            if errors:
                failures.append(f"{case.id}: {'; '.join(errors)}")
                print(f"[FAIL] {case.id}: {'; '.join(errors)}")
            else:
                print(f"[PASS] {case.id}")
    finally:
        await gateway.close()

    print(f"\nP1 evaluation: {len(cases) - len(failures)}/{len(cases)} passed")
    return 1 if failures else 0


def score_case(case: EvaluationCase, output: AgentTurnOutput) -> list[str]:
    errors: list[str] = []
    memories = output.candidate_memories
    if len(memories) != case.expected_memory_count:
        errors.append(
            f"expected {case.expected_memory_count} memories, got {len(memories)}"
        )

    if case.expected_memory_count:
        matched = any(
            item.category == case.expected_category
            and item.memory_key == case.expected_memory_key
            for item in memories
        )
        if not matched:
            errors.append("expected category/key not found")

    if any(
        contains_sensitive_text(item.content)
        or contains_sensitive_text(item.evidence)
        for item in memories
    ):
        errors.append("sensitive content was extracted")
    return errors


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_evaluations()))
