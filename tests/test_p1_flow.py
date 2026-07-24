import httpx
import pytest

from app.config import Settings
from app.main import create_app
from app.model_gateway import MockModelGateway


async def add_and_confirm_memory(
    client: httpx.AsyncClient,
    session_id: str,
    content: str,
) -> dict:
    turn = await client.post(
        f"/v1/interviews/{session_id}/messages",
        headers={"X-Owner-ID": "owner-1"},
        json={"content": content},
    )
    assert turn.status_code == 200
    memory = turn.json()["candidate_memories"][0]
    confirmed = await client.patch(
        f"/v1/memories/{memory['id']}",
        headers={"X-Owner-ID": "owner-1"},
        json={"action": "confirm"},
    )
    assert confirmed.status_code == 200
    return confirmed.json()


@pytest.mark.asyncio
async def test_interview_hatching_and_reverse_verification_flow(tmp_path) -> None:
    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'p1.db'}",
        model_provider="mock",
        model_name="mock-model",
    )
    app = create_app(settings=settings, gateway=MockModelGateway())

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/v1/interviews",
                json={"owner_id": "owner-1"},
            )
            session_id = created.json()["id"]

            too_early = await client.post(
                f"/v1/interviews/{session_id}/pet",
                headers={"X-Owner-ID": "owner-1"},
            )
            assert too_early.status_code == 409
            assert too_early.json()["detail"]["ready"] is False

            await add_and_confirm_memory(
                client,
                session_id,
                "我希望寻找能认真发展的长期关系",
            )
            await add_and_confirm_memory(
                client,
                session_id,
                "我最看重彼此诚实坦诚",
            )
            await add_and_confirm_memory(
                client,
                session_id,
                "我每天都需要一些独处时间",
            )
            await add_and_confirm_memory(
                client,
                session_id,
                "发生矛盾时我希望直接沟通",
            )

            readiness = await client.get(
                f"/v1/interviews/{session_id}/hatching-readiness",
                headers={"X-Owner-ID": "owner-1"},
            )
            assert readiness.status_code == 200
            assert readiness.json()["ready"] is True
            assert readiness.json()["confirmed_memory_count"] == 4

            hatched = await client.post(
                f"/v1/interviews/{session_id}/pet",
                headers={"X-Owner-ID": "owner-1"},
            )
            assert hatched.status_code == 201
            pet = hatched.json()["pet"]
            assert pet["name"] == "小栖"
            assert len(pet["core_traits"]) == 3
            assert hatched.json()["metrics"]["prompt_version"] == "v1"

            verification = await client.post(
                f"/v1/pets/{pet['id']}/verification",
                headers={"X-Owner-ID": "owner-1"},
                json={
                    "rating": "partly_accurate",
                    "correction": "我不是慢热，只是在表达感情前需要确认安全感",
                },
            )
            assert verification.status_code == 200
            verification_payload = verification.json()
            assert verification_payload["pet_status"] == "needs_revision"
            assert (
                verification_payload["candidate_memories"][0]["source_type"]
                == "verification"
            )
            assert verification_payload["metrics"]["prompt_version"] == "v1"


@pytest.mark.asyncio
async def test_confirming_conflicting_memory_supersedes_old_value(tmp_path) -> None:
    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'conflict.db'}",
        model_provider="mock",
    )
    app = create_app(settings=settings, gateway=MockModelGateway())

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/v1/interviews",
                json={"owner_id": "owner-1"},
            )
            session_id = created.json()["id"]
            old_memory = await add_and_confirm_memory(
                client,
                session_id,
                "我不接受异地恋",
            )

            new_turn = await client.post(
                f"/v1/interviews/{session_id}/messages",
                headers={"X-Owner-ID": "owner-1"},
                json={"content": "如果有明确结束时间，我现在可以接受异地恋"},
            )
            new_memory = new_turn.json()["candidate_memories"][0]
            assert new_memory["conflicts_with_id"] == old_memory["id"]

            confirmed = await client.patch(
                f"/v1/memories/{new_memory['id']}",
                headers={"X-Owner-ID": "owner-1"},
                json={"action": "confirm"},
            )
            assert confirmed.status_code == 200

            detail = await client.get(
                f"/v1/interviews/{session_id}",
                headers={"X-Owner-ID": "owner-1"},
            )
            status_by_id = {
                memory["id"]: memory["status"]
                for memory in detail.json()["memories"]
            }
            assert status_by_id[old_memory["id"]] == "superseded"
            assert status_by_id[new_memory["id"]] == "confirmed"
