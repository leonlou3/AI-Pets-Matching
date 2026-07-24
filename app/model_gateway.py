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
        if "不接受异地" in user_message or "不能异地" in user_message:
            category = "hard_boundary"
            memory = "不接受异地恋"
        else:
            category = "uncertain"
            memory = user_message

        content = json.dumps(
            {
                "reply": "我记下了。能说说这项要求背后的原因或实际经历吗？",
                "candidate_memories": [
                    {
                        "content": memory,
                        "category": category,
                        "confidence": 0.9,
                        "evidence": user_message,
                    }
                ],
            },
            ensure_ascii=False,
        )
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
