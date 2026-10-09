"""语义治理设置面板的纯函数辅助层。

不依赖 Streamlit，因此可被单元测试直接覆盖。取值范围与跨字段规则刻意对齐
`core/config_validation.py`：`reload_governance` 会把后端的一切失败收敛成
"语义治理配置无效"（见 `core/app_context.py` 的 `_build_governance_components`），
所以具体原因必须在这一层给出。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

# 与 core/semantic/registry.py 的 _DATASOURCE 一致；tests 会断言两者不漂移
DATASOURCE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")

# 对齐 core/config_validation.py 的 _INTEGER_CONFIG_LIMITS
INT_LIMITS: dict[str, tuple[str, int, int]] = {
    "semantic_context_max_chars": ("SEMANTIC_CONTEXT_MAX_CHARS", 1, 100_000),
    "semantic_context_max_items": ("SEMANTIC_CONTEXT_MAX_ITEMS", 1, 10_000),
    "sql_max_length": ("SQL_MAX_LENGTH", 1, 100_000),
    "sql_query_max_rows": ("SQL_QUERY_MAX_ROWS", 1, 100_000),
    "sql_query_timeout_ms": ("SQL_QUERY_TIMEOUT_MS", 1, 3_600_000),
    "sql_explain_timeout_ms": ("SQL_EXPLAIN_TIMEOUT_MS", 1, 3_600_000),
    "sql_explain_max_estimated_rows": (
        "SQL_EXPLAIN_MAX_ESTIMATED_ROWS",
        0,
        1_000_000_000,
    ),
    "max_prepared_requests": ("MAX_PREPARED_REQUESTS", 1, 4_096),
    "sql_repair_max_attempts": ("SQL_REPAIR_MAX_ATTEMPTS", 0, 1),
}

# 对齐 core/config_validation.py 的 _FLOAT_CONFIG_LIMITS
FLOAT_LIMITS: dict[str, tuple[str, float, float]] = {
    "prepared_request_ttl_seconds": (
        "PREPARED_REQUEST_TTL_SECONDS",
        0.0,
        3_600.0,
    ),
    "governed_history_binding_ttl_seconds": (
        "GOVERNED_HISTORY_BINDING_TTL_SECONDS",
        0.0,
        3_600.0,
    ),
}

BOOL_KEYS: tuple[str, ...] = (
    "semantic_governance_enabled",
    "sql_explain_enabled",
    "sql_scope_enforcement",
)

# 只读展示：语义治理第一阶段仅支持 mysql（设计文档 §5.1），不提供方言切换，
# 避免给出"可以改"的错觉；模型目录涉及路径解析，交由 .env 配置。
DISPLAY_ONLY_KEYS: tuple[str, ...] = ("semantic_model_dir", "sql_dialect")

EDITABLE_KEYS: frozenset[str] = (
    frozenset(INT_LIMITS)
    | frozenset(FLOAT_LIMITS)
    | frozenset(BOOL_KEYS)
    | {"semantic_datasource", "sql_blocked_tables"}
)


def _is_strict_int(value: Any) -> bool:
    """判断是否为 int，且不接受 bool 子类。"""
    return isinstance(value, int) and not isinstance(value, bool)


def parse_blocked_tables(text: str) -> list[str]:
    """把逗号/空白分隔的文本解析为去重且保序的表名列表。"""
    if not isinstance(text, str):
        return []
    tables: list[str] = []
    for chunk in re.split(r"[,\s]+", text):
        name = chunk.strip()
        if name and name not in tables:
            tables.append(name)
    return tables


def format_blocked_tables(tables: Sequence[str] | None) -> str:
    """把表名列表还原为逗号分隔文本，用于回填输入框。"""
    if not tables:
        return ""
    return ", ".join(str(table) for table in tables)


def validate_updates(
    current: Mapping[str, Any], updates: Mapping[str, Any]
) -> list[str]:
    """校验「当前配置 + 本次改动」合并后的结果，返回中文错误列表。

    空列表表示可以提交给 `reload_governance`。规则与
    `core.config_validation.validate_runtime_config` 保持一致。
    """
    merged: dict[str, Any] = dict(current)
    merged.update(updates)
    errors: list[str] = []

    for key, (public_key, minimum, maximum) in INT_LIMITS.items():
        if key not in merged:
            continue
        value = merged[key]
        if not _is_strict_int(value) or not minimum <= value <= maximum:
            errors.append(f"{public_key} 必须是 {minimum} 到 {maximum} 的整数")

    for key, (public_key, minimum, maximum) in FLOAT_LIMITS.items():
        if key not in merged:
            continue
        value = merged[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not minimum < float(value) <= maximum
        ):
            errors.append(f"{public_key} 必须是大于 {minimum} 且不超过 {maximum} 的数值")

    datasource = merged.get("semantic_datasource")
    if datasource is not None and (
        not isinstance(datasource, str)
        or DATASOURCE_PATTERN.fullmatch(datasource) is None
    ):
        errors.append(
            "SEMANTIC_DATASOURCE 只能包含字母、数字、下划线和连字符，"
            "需以字母或数字开头，且不超过 128 字符"
        )

    blocked_tables = merged.get("sql_blocked_tables")
    if blocked_tables is not None:
        if isinstance(blocked_tables, (str, bytes)) or not isinstance(
            blocked_tables, Sequence
        ):
            errors.append("SQL_BLOCKED_TABLES 必须是表名列表")
        elif any(not isinstance(table, str) or not table.strip() for table in blocked_tables):
            errors.append("SQL_BLOCKED_TABLES 的每一项都必须是非空表名")

    governance_enabled = bool(merged.get("semantic_governance_enabled", False))
    explain_enabled = bool(merged.get("sql_explain_enabled", False))

    if explain_enabled and not governance_enabled:
        errors.append("启用 MySQL EXPLAIN 预检前，必须先启用语义治理")

    if governance_enabled or explain_enabled:
        dialect = merged.get("sql_dialect", "mysql")
        if not isinstance(dialect, str) or dialect.casefold() != "mysql":
            enabled = ["语义治理"]
            if explain_enabled:
                enabled.append("EXPLAIN 预检")
            errors.append(
                f"启用{'与'.join(enabled)}时 SQL_DIALECT 仅支持 mysql，"
                "请在 .env 中修改后重启"
            )

    return errors


def _values_equal(left: Any, right: Any) -> bool:
    """比较配置值；区分 bool 与 int，并把 list/tuple 视为同类。"""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if left is None or right is None:
            return left == right
        return list(left) == list(right)
    return left == right


def diff_updates(
    current: Mapping[str, Any], desired: Mapping[str, Any]
) -> dict[str, Any]:
    """只挑出真正变化的键，避免无谓地重建整个组件图。"""
    changed: dict[str, Any] = {}
    for key, value in desired.items():
        if key not in EDITABLE_KEYS:
            continue
        if key not in current or not _values_equal(current[key], value):
            changed[key] = value
    return changed


def describe_status(
    config: Mapping[str, Any], semantic_version: str | None = None
) -> str:
    """生成一行治理状态摘要，用于面板顶部。"""
    governance = bool(config.get("semantic_governance_enabled", False))
    explain = bool(config.get("sql_explain_enabled", False))
    parts = [
        f"语义治理：{'已启用' if governance else '已关闭'}",
        f"EXPLAIN 预检：{'已启用' if explain else '已关闭'}",
    ]
    if governance:
        parts.append(
            f"语义版本：{semantic_version or '未加载'}"
            f"（数据源 {config.get('semantic_datasource', 'default')}）"
        )
        parts.append(
            f"纠错上限：{config.get('sql_repair_max_attempts', 1)} 次"
        )
    return " ｜ ".join(parts)


def active_semantic_version(registry: Any) -> str | None:
    """读取当前激活的语义快照版本；未加载或不可用时返回 None。"""
    if registry is None:
        return None
    try:
        return str(registry.get_snapshot().model_version)
    except Exception:
        return None


__all__ = [
    "BOOL_KEYS",
    "DATASOURCE_PATTERN",
    "DISPLAY_ONLY_KEYS",
    "EDITABLE_KEYS",
    "FLOAT_LIMITS",
    "INT_LIMITS",
    "active_semantic_version",
    "describe_status",
    "diff_updates",
    "format_blocked_tables",
    "parse_blocked_tables",
    "validate_updates",
]
