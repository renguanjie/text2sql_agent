"""Small presentation helpers for governed correctness results."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_PUBLIC_ISSUE_MESSAGES = {
    "AMBIGUOUS_COLUMN": "字段引用存在歧义",
    "CORRECTNESS_UNAVAILABLE": "SQL 正确性校验暂时不可用",
    "DIMENSION_NOT_ALLOWED": "查询维度不被允许",
    "EXECUTION_CONTEXT_REQUIRED": "执行前需要完整的治理上下文",
    "EXPLAIN_FAILED": "数据库预检失败",
    "GENERATION_FAILED": "SQL 生成暂时不可用，请稍后重试",
    "HISTORY_CONTEXT_MISMATCH": "执行历史与当前治理上下文不匹配",
    "INVALID_JOIN": "查询连接关系不被允许",
    "METRIC_MISMATCH": "查询指标与语义定义不一致",
    "POLICY_DENIED": "字段不允许直接查询",
    "QUERY_SCOPE_DENIED": "SQL 查询范围不被允许",
    "REPAIR_FAILED": "SQL 修复失败",
    "SEMANTIC_CONTENT_MISMATCH": "SQL 生成内容与当前语义内容不一致",
    "SEMANTIC_VALIDATION_FAILED": "SQL 未通过语义校验",
    "SEMANTIC_VERSION_MISMATCH": "SQL 生成版本与当前语义版本不一致",
    "SQL_LENGTH_EXCEEDED": "SQL 长度超过安全限制",
    "SQL_VALIDATION_FAILED": "SQL 未通过基础校验",
    "SYNTAX_ERROR": "SQL 语法校验失败",
    "TIME_DIMENSION_REQUIRED": "查询缺少必需时间维度",
    "UNKNOWN_COLUMN": "查询引用未知字段",
    "UNKNOWN_TABLE": "查询引用未知表",
    "UNSAFE_SQL": "SQL 未通过只读安全校验",
    "UNSUPPORTED_SQL": "SQL 类型不受支持",
}

_PUBLIC_LEGACY_ERRORS = frozenset(
    {
        "信息不足以生成 SQL",
        "无法从响应中提取 SQL",
        "LLM 服务调用失败，请检查模型配置后重试",
        "SQL 修复服务调用失败",
        "无法修复 SQL",
        "SQL 生成暂时不可用，请稍后重试",
        "MySQL 客户端未连接",
        "执行记录状态更新失败",
        "语义治理模式下请重新生成 SQL",
        "历史记录不存在",
    }
)
_PUBLIC_EXECUTION_ERROR_RE = re.compile(
    r"^查询执行失败，请联系管理员并提供错误编号 [0-9a-f]{32}$"
)


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
        if isinstance(code, str) and code in _PUBLIC_ISSUE_MESSAGES:
            messages.append(f"[{code}] {_PUBLIC_ISSUE_MESSAGES[code]}")
    return messages


def public_failure_messages(result: Mapping[str, Any] | None) -> list[str]:
    """Prefer stable structured issues and provide a non-null legacy fallback."""
    structured = structured_validation_messages(result)
    if structured:
        return structured
    error = result.get("error") if isinstance(result, Mapping) else None
    if isinstance(error, str) and (
        error in _PUBLIC_LEGACY_ERRORS or _PUBLIC_EXECUTION_ERROR_RE.fullmatch(error)
    ):
        return [error]
    return ["未知错误"]


__all__ = [
    "execution_is_disabled",
    "public_failure_messages",
    "structured_validation_messages",
]
