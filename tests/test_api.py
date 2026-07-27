import json

import httpx
import pytest
from sqlalchemy import func, select

from app.config import Settings
from app.main import create_app
from app.model_gateway import GatewayResult, GatewayUsage, MockModelGateway
from app.models import AgentRun


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        model_provider="mock",
        model_name="mock-model",
    )


@pytest.mark.asyncio
async def test_complete_candidate_memory_confirmation_flow(settings: Settings) -> None:
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
            assert created.status_code == 201
            session_id = created.json()["id"]

            turn = await client.post(
                f"/v1/interviews/{session_id}/messages",
                headers={"X-Owner-ID": "owner-1"},
                json={"content": "我不接受异地恋"},
            )
            assert turn.status_code == 200
            payload = turn.json()
            assert payload["candidate_memories"][0]["status"] == "candidate"
            assert payload["candidate_memories"][0]["category"] == "hard_boundary"
            assert payload["metrics"]["prompt_version"] == "v2"
            assert payload["metrics"]["input_tokens"] > 0

            memory_id = payload["candidate_memories"][0]["id"]
            confirmed = await client.patch(
                f"/v1/memories/{memory_id}",
                headers={"X-Owner-ID": "owner-1"},
                json={"action": "confirm"},
            )
            assert confirmed.status_code == 200
            assert confirmed.json()["status"] == "confirmed"

            detail = await client.get(
                f"/v1/interviews/{session_id}",
                headers={"X-Owner-ID": "owner-1"},
            )
            assert detail.status_code == 200
            assert [item["role"] for item in detail.json()["messages"]] == [
                "user",
                "assistant",
            ]
            assert detail.json()["memories"][0]["status"] == "confirmed"

            hidden = await client.get(
                f"/v1/interviews/{session_id}",
                headers={"X-Owner-ID": "someone-else"},
            )
            assert hidden.status_code == 404

        async with app.state.database.sessions() as db:
            successful_runs = await db.scalar(
                select(func.count())
                .select_from(AgentRun)
                .where(AgentRun.status == "success")
            )
            assert successful_runs == 1


class RecordingGateway:
    provider_name = "recording"

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def complete(self, messages: list[dict[str, str]]) -> GatewayResult:
        self.messages = messages
        return GatewayResult(
            content=json.dumps(
                {
                    "reply": "信息已安全记录。",
                    "candidate_memories": [],
                },
                ensure_ascii=False,
            ),
            model_name="recording-model",
            usage=GatewayUsage(),
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_sensitive_identifiers_are_redacted_before_model_call(
    settings: Settings,
) -> None:
    gateway = RecordingGateway()
    app = create_app(settings=settings, gateway=gateway)

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
            response = await client.post(
                f"/v1/interviews/{session_id}/messages",
                headers={"X-Owner-ID": "owner-1"},
                json={"content": "我的手机号是13812345678"},
            )

    assert response.status_code == 200
    model_input = gateway.messages[-1]["content"]
    assert "13812345678" not in model_input
    assert "[敏感信息已隐藏]" in model_input
