"""
日志脱敏工具模块
提供统一的日志脱敏函数，防止敏感信息泄露
"""
from typing import Optional


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
    # 移除多余的空白字符
    sanitized = ' '.join(sql.split())
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
    import re
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)
