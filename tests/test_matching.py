import json

import httpx
import pytest
from sqlalchemy import func, select

from app.config import Settings
from app.main import create_app
from app.matching_service import compatibility, hard_filter
from app.model_gateway import MockModelGateway
from app.models import AgentRun, MemoryItem, User


def make_user(**overrides) -> User:
    defaults = dict(
        nickname="测试",
        gender="female",
        orientation="heterosexual",
        city="上海",
        birth_year=1995,
        seeking_genders=["male"],
        seeking_min_age=25,
        seeking_max_age=40,
        accept_remote=False,
        status="active",
    )
    defaults.update(overrides)
    return User(**defaults)


def make_memory(memory_key: str, importance: int = 4) -> MemoryItem:
    return MemoryItem(
        session_id="s",
        content=memory_key,
        memory_key=memory_key,
        category="preference",
        confidence=0.9,
        importance=importance,
        evidence=memory_key,
        status="confirmed",
    )


class TestHardFilter:
    def test_mutual_gender_seeking_required(self) -> None:
        a = make_user(gender="female", seeking_genders=["male"])
        b = make_user(nickname="同性", gender="female", seeking_genders=["male"])
        passed, reason = hard_filter(a, b)
        assert passed is False
        assert "不寻找" in reason

    def test_age_must_fit_both_directions(self) -> None:
        a = make_user(birth_year=1996, seeking_min_age=28, seeking_max_age=38)
        b = make_user(
            nickname="年长",
            gender="male",
            seeking_genders=["female"],
            birth_year=1985,
            seeking_min_age=25,
            seeking_max_age=40,
        )
        passed, reason = hard_filter(a, b)
        assert passed is False
        assert "年龄" in reason

    def test_different_city_requires_mutual_remote(self) -> None:
        a = make_user(city="上海", accept_remote=True)
        b = make_user(
            nickname="北京",
            gender="male",
            seeking_genders=["female"],
            city="北京",
            accept_remote=False,
        )
        passed, reason = hard_filter(a, b)
        assert passed is False
        assert "异地" in reason

        b.accept_remote = True
        passed, reason = hard_filter(a, b)
        assert passed is True
        assert reason is None

    def test_compatible_pair_passes(self) -> None:
        a = make_user()
        b = make_user(
            nickname="对方",
            gender="male",
            seeking_genders=["female"],
            birth_year=1993,
        )
        passed, reason = hard_filter(a, b)
        assert passed is True
        assert reason is None


class TestCompatibility:
    def test_no_memories_scores_zero(self) -> None:
        score, shared = compatibility([], [])
        assert score == 0
        assert shared == []

    def test_shared_topics_increase_score(self) -> None:
        a = [make_memory("relationship.goal"), make_memory("values.honesty")]
        b = [make_memory("relationship.goal"), make_memory("lifestyle.solo_time")]
        score, shared = compatibility(a, b)
        assert shared == ["relationship.goal"]
        assert 0 < score < 1

    def test_identical_topics_score_one(self) -> None:
        a = [make_memory("relationship.goal"), make_memory("values.honesty")]
        b = [make_memory("relationship.goal"), make_memory("values.honesty")]
        score, shared = compatibility(a, b)
        assert score == 1
        assert len(shared) == 2


class TestMockJudge:
    def test_children_conflict_fails(self) -> None:
        payload = MockModelGateway._judge_payload(
            json,
            json.dumps(
                {
                    "owner_memories": [
                        {"content": "我想要孩子"},
                        {"content": "我想找长期关系"},
                    ],
                    "candidate_memories": [
                        {"content": "我不想要孩子"},
                        {"content": "我想找长期关系"},
                    ],
                    "shared_memory_keys": ["relationship.goal"],
                },
                ensure_ascii=False,
            ),
        )
        assert payload["decision"] == "fail"
        assert payload["hard_conflicts"]

    def test_too_few_memories_is_insufficient_evidence(self) -> None:
        payload = MockModelGateway._judge_payload(
            json,
            json.dumps(
                {
                    "owner_memories": [{"content": "我想找长期关系"}],
                    "candidate_memories": [
                        {"content": "我想找长期关系"},
                        {"content": "我看重诚实"},
                    ],
                    "shared_memory_keys": [],
                },
                ensure_ascii=False,
            ),
        )
        assert payload["decision"] == "insufficient_evidence"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'matching.db'}",
        model_provider="mock",
    )


TRAINING_MESSAGES = [
    "我希望寻找能认真发展的长期关系",
    "我最看重两个人彼此诚实",
    "我每天都需要一些独处时间",
    "发生矛盾时我希望直接沟通",
]


async def onboard(client: httpx.AsyncClient, profile: dict, messages: list[str]) -> dict:
    created = await client.post("/v1/users", json=profile)
    assert created.status_code == 201
    user = created.json()
    headers = {"X-Owner-ID": user["id"]}

    interview = await client.post("/v1/interviews", json={"owner_id": user["id"]})
    session_id = interview.json()["id"]
    for message in messages:
        turn = await client.post(
            f"/v1/interviews/{session_id}/messages",
            headers=headers,
            json={"content": message},
        )
        assert turn.status_code == 200
        for memory in turn.json()["candidate_memories"]:
            confirmed = await client.patch(
                f"/v1/memories/{memory['id']}",
                headers=headers,
                json={"action": "confirm"},
            )
            assert confirmed.status_code == 200

    hatched = await client.post(
        f"/v1/interviews/{session_id}/pet",
        headers=headers,
    )
    assert hatched.status_code == 201
    return user


@pytest.mark.asyncio
async def test_user_creation_and_validation(settings: Settings) -> None:
    app = create_app(settings=settings, gateway=MockModelGateway())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            valid = await client.post(
                "/v1/users",
                json={
                    "nickname": "小雨",
                    "gender": "female",
                    "orientation": "heterosexual",
                    "city": "上海",
                    "birth_year": 1996,
                    "seeking_genders": ["male"],
                    "seeking_min_age": 28,
                    "seeking_max_age": 38,
                    "accept_remote": False,
                },
            )
            assert valid.status_code == 201

            invalid_age_range = await client.post(
                "/v1/users",
                json={
                    "nickname": "错误",
                    "gender": "male",
                    "orientation": "heterosexual",
                    "city": "上海",
                    "birth_year": 1996,
                    "seeking_genders": ["female"],
                    "seeking_min_age": 40,
                    "seeking_max_age": 30,
                },
            )
            assert invalid_age_range.status_code == 422

            listed = await client.get("/v1/users")
            assert listed.status_code == 200
            assert len(listed.json()) == 1


@pytest.mark.asyncio
async def test_matching_round_end_to_end(settings: Settings) -> None:
    app = create_app(settings=settings, gateway=MockModelGateway())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://t",
            timeout=60,
        ) as client:
            user_a = await onboard(
                client,
                {
                    "nickname": "小雨",
                    "gender": "female",
                    "orientation": "heterosexual",
                    "city": "上海",
                    "birth_year": 1996,
                    "seeking_genders": ["male"],
                    "seeking_min_age": 28,
                    "seeking_max_age": 38,
                    "accept_remote": False,
                },
                TRAINING_MESSAGES,
            )
            user_b = await onboard(
                client,
                {
                    "nickname": "阿哲",
                    "gender": "male",
                    "orientation": "heterosexual",
                    "city": "上海",
                    "birth_year": 1994,
                    "seeking_genders": ["female"],
                    "seeking_min_age": 26,
                    "seeking_max_age": 34,
                    "accept_remote": False,
                },
                TRAINING_MESSAGES,
            )
            # No interview and no pet: must be excluded from matching.
            ineligible = await client.post(
                "/v1/users",
                json={
                    "nickname": "路人",
                    "gender": "male",
                    "orientation": "heterosexual",
                    "city": "上海",
                    "birth_year": 1993,
                    "seeking_genders": ["female"],
                    "seeking_min_age": 25,
                    "seeking_max_age": 40,
                },
            )
            assert ineligible.status_code == 201

            round_response = await client.post("/v1/matching/rounds")
            assert round_response.status_code == 201
            summary = round_response.json()
            assert summary["eligible_users"] == 2
            assert summary["pairs_considered"] == 1
            assert summary["recommendations_created"] == 1

            detail = await client.get(f"/v1/matching/rounds/{summary['id']}")
            assert detail.status_code == 200
            payload = detail.json()

            pair = payload["pairs"][0]
            assert pair["stage"] == "recommended"
            assert pair["outcome"] == "recommended"
            assert len(pair["shared_memory_keys"]) >= 2
            assert pair["dialogue_transcript"]
            directions = sorted(v["direction"] for v in pair["verdicts"])
            assert directions == ["a_for_b", "b_for_a"]
            assert all(v["decision"] == "pass" for v in pair["verdicts"])

            recommendation = payload["recommendations"][0]
            assert recommendation["icebreaker_suggestion"]

            for user in (user_a, user_b):
                listed = await client.get(
                    f"/v1/users/{user['id']}/recommendations"
                )
                assert listed.status_code == 200
                assert len(listed.json()) == 1

        async with app.state.database.sessions() as db:
            dialogue_runs = await db.scalar(
                select(func.count())
                .select_from(AgentRun)
                .where(AgentRun.agent_name == "pet_dialogue")
            )
            judge_runs = await db.scalar(
                select(func.count())
                .select_from(AgentRun)
                .where(AgentRun.agent_name == "judge")
            )
            assert dialogue_runs >= 2
            assert judge_runs == 2


@pytest.mark.asyncio
async def test_children_conflict_blocks_recommendation(settings: Settings) -> None:
    app = create_app(settings=settings, gateway=MockModelGateway())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://t",
            timeout=60,
        ) as client:
            await onboard(
                client,
                {
                    "nickname": "想要",
                    "gender": "female",
                    "orientation": "heterosexual",
                    "city": "上海",
                    "birth_year": 1996,
                    "seeking_genders": ["male"],
                    "seeking_min_age": 25,
                    "seeking_max_age": 40,
                },
                TRAINING_MESSAGES + ["我未来几年内想要孩子"],
            )
            await onboard(
                client,
                {
                    "nickname": "不要",
                    "gender": "male",
                    "orientation": "heterosexual",
                    "city": "上海",
                    "birth_year": 1994,
                    "seeking_genders": ["female"],
                    "seeking_min_age": 25,
                    "seeking_max_age": 40,
                },
                TRAINING_MESSAGES + ["我不想要孩子"],
            )

            round_response = await client.post("/v1/matching/rounds")
            summary = round_response.json()
            assert summary["recommendations_created"] == 0

            detail = await client.get(f"/v1/matching/rounds/{summary['id']}")
            pair = detail.json()["pairs"][0]
            assert pair["stage"] == "judgement"
            assert pair["outcome"] == "rejected"
            assert any(
                v["decision"] == "fail" and v["hard_conflicts"]
                for v in pair["verdicts"]
            )


@pytest.mark.asyncio
async def test_full_demo_flow() -> None:
    from app.demo import run_demo

    detail = await run_demo()
    assert detail["eligible_users"] == 10
    assert detail["pairs_considered"] == 45
    assert detail["recommendations_created"] >= 1
    assert (
        detail["pairs_passed_hard_filter"]
        >= detail["pairs_passed_compatibility"]
        >= detail["recommendations_created"]
    )
    assert len(detail["recommendations"]) == detail["recommendations_created"]
