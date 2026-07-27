import re


_SENSITIVE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
)


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in _SENSITIVE_PATTERNS:
        redacted = pattern.sub("[敏感信息已隐藏]", redacted)
    return redacted


def contains_sensitive_text(text: str) -> bool:
    return "[敏感信息已隐藏]" in text or any(
        pattern.search(text) for pattern in _SENSITIVE_PATTERNS
    )
