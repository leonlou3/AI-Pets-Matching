"""End-to-end demo: 10 users register, train pets, and get matched.

Run with:

    python -m app.demo

The demo uses the deterministic mock model by default, so it costs no tokens.
Set MODEL_PROVIDER=openai_compatible in .env to run it against a real model.
"""

import asyncio
from pathlib import Path

import httpx
from sqlalchemy import func, select

from app.config import Settings
from app.main import create_app
from app.models import AgentRun

DEMO_DB_PATH = Path("demo.db")

# Each entry: profile fields + the interview messages used to train the pet.
DEMO_USERS = [
    {
        "profile": {
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
        "messages": [
            "我希望寻找能认真发展的长期关系",
            "我最看重两个人彼此诚实",
            "我每天都需要一些独处时间",
            "发生矛盾时我希望直接沟通",
            "我未来几年内想要孩子",
        ],
    },
    {
        "profile": {
            "nickname": "阿哲",
            "gender": "male",
            "orientation": "heterosexual",
            "city": "上海",
            "birth_year": 1994,
            "seeking_genders": ["female"],
            "seeking_min_age": 26,
            "seeking_max_age": 32,
            "accept_remote": False,
        },
        "messages": [
            "我在认真寻找可以走向结婚的长期关系",
            "我很看重坦诚相待",
            "我需要规律的独处充电时间",
            "有分歧时我倾向直接沟通解决",
            "我想要孩子",
        ],
    },
    {
        "profile": {
            "nickname": "婷婷",
            "gender": "female",
            "orientation": "heterosexual",
            "city": "上海",
            "birth_year": 1997,
            "seeking_genders": ["male"],
            "seeking_min_age": 28,
            "seeking_max_age": 40,
            "accept_remote": False,
        },
        "messages": [
            "我想找长期稳定的关系",
            "我看重诚实",
            "争吵之后我希望能好好沟通复盘",
            "我不想要孩子",
        ],
    },
    {
        "profile": {
            "nickname": "大伟",
            "gender": "male",
            "orientation": "heterosexual",
            "city": "北京",
            "birth_year": 1990,
            "seeking_genders": ["female"],
            "seeking_min_age": 25,
            "seeking_max_age": 35,
            "accept_remote": True,
        },
        "messages": [
            "我在找认真的长期关系",
            "我最在意诚实",
            "我平时需要独处放空",
            "遇到矛盾我会主动沟通",
        ],
    },
    {
        "profile": {
            "nickname": "琳琳",
            "gender": "female",
            "orientation": "heterosexual",
            "city": "北京",
            "birth_year": 1995,
            "seeking_genders": ["male"],
            "seeking_min_age": 30,
            "seeking_max_age": 40,
            "accept_remote": False,
        },
        "messages": [
            "我希望认真发展一段长期关系",
            "诚实对我最重要",
            "我需要自己的独处空间",
            "我想要孩子",
        ],
    },
    {
        "profile": {
            "nickname": "老陈",
            "gender": "male",
            "orientation": "heterosexual",
            "city": "上海",
            "birth_year": 1985,
            "seeking_genders": ["female"],
            "seeking_min_age": 30,
            "seeking_max_age": 40,
            "accept_remote": True,
        },
        "messages": [
            "我喜欢安静简单的生活",
            "我作息很规律",
            "我看重坦诚",
            "发生分歧时我习惯先冷静再沟通",
        ],
    },
    {
        "profile": {
            "nickname": "娜娜",
            "gender": "female",
            "orientation": "heterosexual",
            "city": "深圳",
            "birth_year": 1999,
            "seeking_genders": ["male"],
            "seeking_min_age": 27,
            "seeking_max_age": 40,
            "accept_remote": True,
        },
        "messages": [
            "我喜欢热闹的城市生活",
            "这个问题我还没想好",
            "要不要孩子以后再看",
            "争吵后我需要一点自己的空间再沟通",
        ],
    },
    {
        "profile": {
            "nickname": "军军",
            "gender": "male",
            "orientation": "heterosexual",
            "city": "上海",
            "birth_year": 1996,
            "seeking_genders": ["female"],
            "seeking_min_age": 26,
            "seeking_max_age": 31,
            "accept_remote": False,
        },
        "messages": [
            "我想认真谈一段长期感情",
            "我在意对方是否诚实",
            "有问题我喜欢当面沟通",
            "我也需要独处的时间",
        ],
    },
    {
        "profile": {
            "nickname": "苏苏",
            "gender": "female",
            "orientation": "heterosexual",
            "city": "上海",
            "birth_year": 1993,
            "seeking_genders": ["male"],
            "seeking_min_age": 30,
            "seeking_max_age": 45,
            "accept_remote": True,
        },
        "messages": [
            "我在找愿意走进婚姻的长期关系",
            "我特别看重坦诚",
            "争吵后我希望及时沟通修复",
            "我喜欢旅行和尝试新餐厅",
            "我未来想要孩子",
        ],
    },
    {
        "profile": {
            "nickname": "浩浩",
            "gender": "male",
            "orientation": "heterosexual",
            "city": "上海",
            "birth_year": 1992,
            "seeking_genders": ["female"],
            "seeking_min_age": 28,
            "seeking_max_age": 36,
            "accept_remote": True,
        },
        "messages": [
            "我希望建立长期稳定的关系",
            "我看重彼此坦诚",
            "遇到分歧我会先沟通",
            "我不想要孩子",
        ],
    },
]


async def onboard_user(client: httpx.AsyncClient, entry: dict) -> dict:
    """Register a user, train the pet through interview, and hatch it."""
    created = await client.post("/v1/users", json=entry["profile"])
    created.raise_for_status()
    user = created.json()
    headers = {"X-Owner-ID": user["id"]}

    interview = await client.post("/v1/interviews", json={"owner_id": user["id"]})
    interview.raise_for_status()
    session_id = interview.json()["id"]

    for message in entry["messages"]:
        turn = await client.post(
            f"/v1/interviews/{session_id}/messages",
            headers=headers,
            json={"content": message},
        )
        turn.raise_for_status()
        for memory in turn.json()["candidate_memories"]:
            confirmed = await client.patch(
                f"/v1/memories/{memory['id']}",
                headers=headers,
                json={"action": "confirm"},
            )
            confirmed.raise_for_status()

    hatched = await client.post(
        f"/v1/interviews/{session_id}/pet",
        headers=headers,
    )
    hatched.raise_for_status()
    pet = hatched.json()["pet"]
    print(
        f"  {user['nickname']}（{user['city']}）已孵化宠物："
        f"{pet['name']}（{pet['species']}）"
    )
    return user


async def run_demo() -> dict:
    if DEMO_DB_PATH.exists():
        DEMO_DB_PATH.unlink()

    settings = Settings(database_url=f"sqlite+aiosqlite:///{DEMO_DB_PATH}")
    app = create_app(settings=settings)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://demo",
            timeout=120,
        ) as client:
            print("== 第一步：10 位用户注册并训练宠物 ==")
            users = [await onboard_user(client, entry) for entry in DEMO_USERS]
            nickname_by_id = {user["id"]: user["nickname"] for user in users}

            print("\n== 第二步：运行匹配轮 ==")
            round_response = await client.post("/v1/matching/rounds")
            round_response.raise_for_status()
            summary = round_response.json()

            detail_response = await client.get(
                f"/v1/matching/rounds/{summary['id']}"
            )
            detail_response.raise_for_status()
            detail = detail_response.json()

        print("\n== 匹配漏斗 ==")
        print(f"  可参与匹配用户：{detail['eligible_users']}")
        print(f"  评估候选对：{detail['pairs_considered']}")
        print(f"  通过硬条件过滤：{detail['pairs_passed_hard_filter']}")
        print(f"  通过兼容度初筛：{detail['pairs_passed_compatibility']}")
        print(f"  完成宠物对谈：{detail['pairs_dialogued']}")
        print(f"  生成双向推荐：{detail['recommendations_created']}")

        print("\n== 未通过的候选对（部分）==")
        shown = 0
        for pair in detail["pairs"]:
            if pair["outcome"] == "recommended" or shown >= 8:
                continue
            a = nickname_by_id[pair["user_a_id"]]
            b = nickname_by_id[pair["user_b_id"]]
            print(f"  {a} × {b}｜阶段 {pair['stage']}｜{pair['failure_reason']}")
            shown += 1

        print("\n== 推荐结果 ==")
        for recommendation in detail["recommendations"]:
            a = nickname_by_id[recommendation["user_a_id"]]
            b = nickname_by_id[recommendation["user_b_id"]]
            print(f"  {a} ❤ {b}")
            print(f"    推荐理由：{recommendation['reason']}")
            print(f"    破冰建议：{recommendation['icebreaker_suggestion']}")

        async with app.state.database.sessions() as db:
            totals = (
                await db.execute(
                    select(
                        func.count(AgentRun.id),
                        func.sum(AgentRun.input_tokens),
                        func.sum(AgentRun.output_tokens),
                        func.sum(AgentRun.estimated_cost),
                    ).where(AgentRun.status == "success")
                )
            ).one()

        print("\n== 模型调用成本 ==")
        print(f"  成功调用次数：{totals[0]}")
        print(f"  输入 tokens：{totals[1] or 0}")
        print(f"  输出 tokens：{totals[2] or 0}")
        print(f"  估算费用：{round(totals[3] or 0, 6)} 元")

        return detail


if __name__ == "__main__":
    asyncio.run(run_demo())
