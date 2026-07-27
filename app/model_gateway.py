import asyncio
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings


class GatewayError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GatewayUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class GatewayResult:
    content: str
    model_name: str
    usage: GatewayUsage


class ModelGateway(Protocol):
    provider_name: str

    async def complete(self, messages: list[dict[str, str]]) -> GatewayResult: ...

    async def close(self) -> None: ...


class MockModelGateway:
    """Deterministic local model used for demos and automated tests."""

    provider_name = "mock"

    async def complete(self, messages: list[dict[str, str]]) -> GatewayResult:
        import json

        user_message = messages[-1]["content"]
        instructions = "\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )

        if "初生宠物生成 Agent" in instructions:
            payload = {
                "name": "小栖",
                "species": "cat",
                "title": "温柔而有边界的观察家",
                "core_traits": ["真诚", "独立", "重视沟通"],
                "relationship_style": "慢慢建立信任，一旦确认就认真投入。",
                "communication_style": "愿意直接表达，也重视彼此冷静整理的空间。",
                "strengths": ["边界清晰", "关系目标认真", "能够倾听"],
                "easily_misunderstood_as": "刚认识时的谨慎有时会被误解为冷淡。",
                "learning_topics": ["冲突后的修复偏好", "理想的共同生活节奏"],
                "declaration": "我会替你认真了解，但把最终选择留给你。",
            }
        elif "宠物画像纠正提取 Agent" in instructions:
            payload = {
                "candidate_memories": [
                    {
                        "content": user_message,
                        "memory_key": "verification.profile_correction",
                        "category": "uncertain",
                        "confidence": 1,
                        "importance": 4,
                        "evidence": user_message,
                    }
                ]
            }
        else:
            if "[敏感信息已隐藏]" in user_message:
                payload = {
                    "reply": "这类身份信息不需要记录，我们继续聊你的关系期待吧。",
                    "candidate_memories": [],
                    "progress": {
                        "covered_categories": [],
                        "missing_topics": ["关系目标", "价值观", "生活方式", "沟通方式"],
                        "focus_topic": "关系目标",
                    },
                }
                content = json.dumps(payload, ensure_ascii=False)
                return GatewayResult(
                    content=content,
                    model_name="mock-model",
                    usage=GatewayUsage(
                        input_tokens=1,
                        output_tokens=max(len(content) // 2, 1),
                    ),
                )
            if "不接受异地" in user_message or "不能异地" in user_message:
                category = "hard_boundary"
                memory_key = "relationship.distance"
                memory = "不接受异地恋"
            elif "异地" in user_message:
                category = "preference"
                memory_key = "relationship.distance"
                memory = user_message
            elif "长期" in user_message or "结婚" in user_message:
                category = "relationship_goal"
                memory_key = "relationship.goal"
                memory = user_message
            elif "诚实" in user_message or "坦诚" in user_message:
                category = "value"
                memory_key = "values.honesty"
                memory = user_message
            elif "独处" in user_message or "作息" in user_message:
                category = "lifestyle"
                memory_key = "lifestyle.solo_time"
                memory = user_message
            elif "沟通" in user_message or "争吵" in user_message:
                category = "communication"
                memory_key = "communication.conflict_style"
                memory = user_message
            elif "没想好" in user_message or "不知道" in user_message:
                category = "uncertain"
                memory_key = "uncertain.open_question"
                memory = user_message
            else:
                category = "preference"
                memory_key = "preference.general"
                memory = user_message

            payload = {
                "reply": "我记下了。能说说这项要求背后的原因或实际经历吗？",
                "candidate_memories": [
                    {
                        "content": memory,
                        "memory_key": memory_key,
                        "category": category,
                        "confidence": 0.9,
                        "importance": 4,
                        "evidence": user_message,
                    }
                ],
                "progress": {
                    "covered_categories": [category],
                    "missing_topics": ["价值观", "生活方式", "沟通方式"],
                    "focus_topic": category,
                },
            }

        content = json.dumps(payload, ensure_ascii=False)
        return GatewayResult(
            content=content,
            model_name="mock-model",
            usage=GatewayUsage(
                input_tokens=max(len(user_message) // 2, 1),
                output_tokens=max(len(content) // 2, 1),
            ),
        )

    async def close(self) -> None:
        return None


class OpenAICompatibleGateway:
    provider_name = "openai_compatible"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.model_base_url.rstrip("/") + "/",
            timeout=settings.model_timeout_seconds,
            headers={
                "Authorization": (
                    f"Bearer {settings.model_api_key.get_secret_value()}"
                ),
                "Content-Type": "application/json",
            },
        )

    async def complete(self, messages: list[dict[str, str]]) -> GatewayResult:
        attempts = self.settings.model_max_retries + 1
        last_error: GatewayError | None = None

        for attempt in range(attempts):
            try:
                response = await self.client.post(
                    "chat/completions",
                    json={
                        "model": self.settings.model_name,
                        "messages": messages,
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"},
                    },
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise GatewayError(
                        "provider_transient_error",
                        f"Model provider returned HTTP {response.status_code}",
                    )
                if response.is_error:
                    raise GatewayError(
                        "provider_request_error",
                        f"Model provider returned HTTP {response.status_code}",
                    )

                payload = response.json()
                usage = payload.get("usage") or {}
                return GatewayResult(
                    content=payload["choices"][0]["message"]["content"],
                    model_name=payload.get("model", self.settings.model_name),
                    usage=GatewayUsage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                    ),
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = GatewayError("provider_unavailable", str(exc))
            except (KeyError, TypeError, ValueError) as exc:
                raise GatewayError(
                    "invalid_provider_response",
                    "Model provider returned an invalid response",
                ) from exc
            except GatewayError as exc:
                if exc.code == "provider_request_error":
                    raise
                last_error = exc

            if attempt < attempts - 1:
                await asyncio.sleep(0.25 * (2**attempt))

        raise last_error or GatewayError(
            "provider_unavailable",
            "Model provider is unavailable",
        )

    async def close(self) -> None:
        await self.client.aclose()


def build_gateway(settings: Settings) -> ModelGateway:
    if settings.model_provider == "mock":
        return MockModelGateway()
    return OpenAICompatibleGateway(settings)
