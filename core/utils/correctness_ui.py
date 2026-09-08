"""Small presentation helpers for governed correctness results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def execution_is_disabled(result: Mapping[str, Any] | None) -> bool:
    """Disable execution only for an explicit governed fail-closed result."""
    return isinstance(result, Mapping) and result.get("executable") is False


def structured_validation_messages(result: Mapping[str, Any] | None) -> list[str]:
    """Render only structured validation fields, never a raw top-level error."""
    if not isinstance(result, Mapping):
        return []
    validation = result.get("validation")
    if not isinstance(validation, Mapping):
        return []
    issues = validation.get("issues")
    if not isinstance(issues, (list, tuple)):
        return []

    messages: list[str] = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        code = issue.get("code")
        message = issue.get("message")
        if isinstance(code, str) and code and isinstance(message, str) and message:
            messages.append(f"[{code}] {message}")
    return messages


def public_failure_messages(result: Mapping[str, Any] | None) -> list[str]:
    """Prefer stable structured issues and provide a non-null legacy fallback."""
    structured = structured_validation_messages(result)
    if structured:
        return structured
    error = result.get("error") if isinstance(result, Mapping) else None
    return [error if isinstance(error, str) and error else "未知错误"]


__all__ = [
    "execution_is_disabled",
    "public_failure_messages",
    "structured_validation_messages",
]
