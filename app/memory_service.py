from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MemoryItem
from app.safety import contains_sensitive_text
from app.schemas import ExtractedMemory, HatchingReadiness, ReviewMemoryRequest


CORE_HATCHING_CATEGORIES = {
    "relationship_goal",
    "preference",
    "value",
    "lifestyle",
    "communication",
}
ACTIVE_MEMORY_STATUSES = {"confirmed", "corrected"}


async def store_candidate_memories(
    db: AsyncSession,
    session_id: str,
    candidates: list[ExtractedMemory],
    source_type: str,
) -> list[MemoryItem]:
    stored: list[MemoryItem] = []
    for candidate in candidates:
        if contains_sensitive_text(candidate.content) or contains_sensitive_text(
            candidate.evidence
        ):
            continue

        existing = list(
            (
                await db.scalars(
                    select(MemoryItem)
                    .where(
                        MemoryItem.session_id == session_id,
                        MemoryItem.memory_key == candidate.memory_key,
                        MemoryItem.status.notin_(["rejected", "deleted", "superseded"]),
                    )
                    .order_by(MemoryItem.created_at.desc())
                )
            ).all()
        )
        if any(item.content == candidate.content for item in existing):
            continue

        conflict = next(
            (item for item in existing if item.status in ACTIVE_MEMORY_STATUSES),
            None,
        )
        memory = MemoryItem(
            session_id=session_id,
            content=candidate.content,
            memory_key=candidate.memory_key,
            category=candidate.category,
            confidence=candidate.confidence,
            importance=candidate.importance,
            evidence=candidate.evidence,
            source_type=source_type,
            conflicts_with_id=conflict.id if conflict else None,
        )
        db.add(memory)
        stored.append(memory)

    await db.flush()
    return stored


async def apply_memory_review(
    db: AsyncSession,
    memory: MemoryItem,
    review: ReviewMemoryRequest,
) -> None:
    status_by_action = {
        "confirm": "confirmed",
        "correct": "corrected",
        "reject": "rejected",
        "delete": "deleted",
    }
    if review.action == "correct":
        memory.content = review.corrected_content or memory.content

    if review.action in {"confirm", "correct"} and memory.conflicts_with_id:
        conflicting = await db.get(MemoryItem, memory.conflicts_with_id)
        if conflicting and conflicting.status in ACTIVE_MEMORY_STATUSES:
            conflicting.status = "superseded"

    memory.status = status_by_action[review.action]


def calculate_hatching_readiness(memories: list[MemoryItem]) -> HatchingReadiness:
    active = [item for item in memories if item.status in ACTIVE_MEMORY_STATUSES]
    covered = sorted({item.category for item in active} & CORE_HATCHING_CATEGORIES)
    missing = sorted(CORE_HATCHING_CATEGORIES - set(covered))
    ready = len(active) >= 4 and len(covered) >= 3

    if ready:
        message = "已具备生成初生宠物的基础信息。"
    else:
        message = (
            "至少需要 4 条已确认记忆，并覆盖关系目标、偏好、价值观、"
            "生活方式、沟通方式中的 3 类。"
        )

    return HatchingReadiness(
        ready=ready,
        confirmed_memory_count=len(active),
        covered_categories=covered,
        missing_categories=missing,
        message=message,
    )
