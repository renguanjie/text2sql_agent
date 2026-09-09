"""
日志脱敏工具模块
提供统一的日志脱敏函数，防止敏感信息泄露
"""
import hashlib
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal
from uuid import uuid4

from loguru import logger


def truncate_log(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断日志文本，防止过长的内容写入日志

    Args:
        text: 原始文本
        max_length: 最大长度（默认 100 字符）
        suffix: 截断后缀（默认 "..."）

    Returns:
        截断后的文本

    Examples:
        >>> truncate_log("SELECT * FROM users WHERE id = 1", max_length=20)
        'SELECT * FROM users ...'
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def sanitize_sql(sql: str, max_length: int = 80) -> str:
    """
    脱敏 SQL 查询，截断并移除多余的空白

    Args:
        sql: SQL 查询字符串
        max_length: 最大长度（默认 80 字符）

    Returns:
        脱敏后的 SQL 字符串

    Examples:
        >>> sanitize_sql("SELECT * FROM users WHERE password = 'secret123'")
        'SELECT * FROM users WHERE password = ...'
    """
    if not sql:
        return sql
    try:
        import sqlglot
        from sqlglot import exp

        statement = sqlglot.parse_one(sql)
        statement = statement.transform(
            lambda node: exp.Placeholder() if isinstance(node, exp.Literal) else node
        )
        sanitized = statement.sql()
    except Exception:
        import re

        sanitized = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "?", sql)
        sanitized = re.sub(r"\b\d+(?:\.\d+)?\b", "?", sanitized)
    sanitized = ' '.join(sanitized.split())
    # 截断
    return truncate_log(sanitized, max_length)


def sanitize_query(query: str, max_length: int = 60) -> str:
    """
    脱敏 Cypher 或其他查询语言，截断并移除多余的空白

    Args:
        query: 查询字符串
        max_length: 最大长度（默认 60 字符）

    Returns:
        脱敏后的查询字符串
    """
    if not query:
        return query
    # 移除多余的空白字符
    sanitized = ' '.join(query.split())
    # 截断
    return truncate_log(sanitized, max_length)


def text_fingerprint(text: str) -> str:
    """返回不可逆短摘要，用于关联日志而不记录原始用户内容。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def log_safe_exception(
    stage: str,
    exc: Exception,
    *,
    sql: str | None = None,
    fingerprint_source: str | None = None,
    level: Literal["error", "warning"] = "error",
) -> str:
    """Log fixed exception metadata without traceback or attacker-controlled text."""
    error_id = uuid4().hex
    source = fingerprint_source if fingerprint_source is not None else sql
    if source is None:
        source = f"{stage}:{type(exc).__name__}"
    message = (
        "operation_failure error_id={} stage={} exception_type={} fingerprint={}"
    )
    args = [error_id, stage, type(exc).__name__, text_fingerprint(source)]
    log = logger.warning if level == "warning" else logger.error
    log(message, *args)
    return error_id


_AUTHORIZATION_RE = re.compile(
    r"\bauthorization\s*[:=]\s*(?:bearer\s+)?[^\s,;]+",
    flags=re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bbearer\s+[^\s,;]+", flags=re.IGNORECASE)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:password|passwd|api[_-]?key|access[_-]?token|token|secret)"
    r"\s*[:=]\s*[^\s,;]+",
    flags=re.IGNORECASE,
)


def sanitize_log_message(message: Any) -> str:
    """Remove common credential forms from a value before it reaches a log sink."""
    text = str(message)
    text = _AUTHORIZATION_RE.sub("[REDACTED_CREDENTIAL]", text)
    text = _BEARER_RE.sub("[REDACTED_CREDENTIAL]", text)
    return _SECRET_ASSIGNMENT_RE.sub("[REDACTED_CREDENTIAL]", text)


def safe_log_format(record: Mapping[str, Any]) -> str:
    """Loguru formatter that omits exception payloads and sanitizes the message."""
    extra = record.get("extra")
    if isinstance(extra, dict):
        extra["safe_message"] = sanitize_log_message(record.get("message", ""))
    return (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
        "{name}:{function}:{line} - {extra[safe_message]}\n"
    )


def make_safe_streamlit_sink(writer: Callable[[str], Any]) -> Callable[[Any], None]:
    """Wrap a UI writer so accidental raw messages are credential-scrubbed."""
    def sink(message: Any) -> None:
        writer(sanitize_log_message(message).rstrip("\n"))

    return sink


def mask_sensitive(text: str, pattern: str = r'(password|api_key|token|secret)\s*[=:]\s*\S+',
                   replacement: str = r'\1=***') -> str:
    """
    掩码敏感信息（如密码、API Key 等）

    Args:
        text: 原始文本
        pattern: 正则表达式模式（默认匹配常见的敏感字段）
        replacement: 替换文本

    Returns:
        掩码后的文本

    Examples:
        >>> mask_sensitive("api_key=sk-123456")
        'api_key=***'
    """
    masked = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return sanitize_log_message(masked)
