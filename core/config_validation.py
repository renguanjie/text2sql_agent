"""Strict, shared validation for environment and runtime configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from math import isfinite
from typing import Any


class RuntimeConfigurationError(ValueError):
    """Configuration error whose message contains keys, never supplied values."""


MAX_HISTORY_RECORDS = 10_000
MAX_SQL_LENGTH = 100_000


def _is_strict_int(value: Any) -> bool:
    """Return whether a value is an int without accepting bool subclasses."""
    return isinstance(value, int) and not isinstance(value, bool)


_INTEGER_CONFIG_LIMITS = {
    "mysql_port": ("MYSQL_PORT", 1, 65_535),
    "mysql_pool_size": ("MYSQL_POOL_SIZE", 1, 1_000),
    "mysql_max_overflow": ("MYSQL_MAX_OVERFLOW", 0, 1_000),
    "mysql_pool_timeout": ("MYSQL_POOL_TIMEOUT", 1, 3_600),
    "neo4j_max_pool_size": ("NEO4J_MAX_POOL_SIZE", 1, 1_000),
    "neo4j_connection_timeout": ("NEO4J_CONNECTION_TIMEOUT", 1, 3_600),
    "prompt_max_length": ("PROMPT_MAX_LENGTH", 1, 100_000),
    "retrieval_top_k": ("RETRIEVAL_TOP_K", 1, 100),
    "upload_max_bytes": ("UPLOAD_MAX_BYTES", 1, 1_073_741_824),
    "graph_max_nodes": ("GRAPH_MAX_NODES", 1, 1_000_000),
    "graph_max_edges": ("GRAPH_MAX_EDGES", 1, 10_000_000),
    "graph_max_depth": ("GRAPH_MAX_DEPTH", 1, 1_000),
    "graph_max_string_length": ("GRAPH_MAX_STRING_LENGTH", 1, 1_000_000),
    "faiss_embedding_dim": ("FAISS_EMBEDDING_DIM", 1, 100_000),
    "few_shot_top_k": ("FEW_SHOT_TOP_K", 1, 100),
    "sql_max_length": ("SQL_MAX_LENGTH", 1, MAX_SQL_LENGTH),
    "sql_query_max_rows": ("SQL_QUERY_MAX_ROWS", 1, 100_000),
    "sql_query_timeout_ms": ("SQL_QUERY_TIMEOUT_MS", 1, 3_600_000),
    "semantic_context_max_chars": ("SEMANTIC_CONTEXT_MAX_CHARS", 1, 100_000),
    "semantic_context_max_items": ("SEMANTIC_CONTEXT_MAX_ITEMS", 1, 10_000),
    "sql_explain_timeout_ms": ("SQL_EXPLAIN_TIMEOUT_MS", 1, 3_600_000),
    "sql_explain_max_estimated_rows": (
        "SQL_EXPLAIN_MAX_ESTIMATED_ROWS",
        0,
        1_000_000_000,
    ),
    "streamlit_server_port": ("STREAMLIT_SERVER_PORT", 1, 65_535),
    "max_prepared_requests": ("MAX_PREPARED_REQUESTS", 1, 4_096),
}

_FLOAT_CONFIG_LIMITS = {
    "bm25_k1": ("BM25_K1", 0, 100, False),
    "bm25_b": ("BM25_B", 0, 1, True),
    "bm25_weight": ("BM25_WEIGHT", 0, 1, True),
    "dense_weight": ("DENSE_WEIGHT", 0, 1, True),
    "few_shot_min_similarity": ("FEW_SHOT_MIN_SIMILARITY", 0, 1, True),
    "prepared_request_ttl_seconds": (
        "PREPARED_REQUEST_TTL_SECONDS",
        0,
        3_600,
        False,
    ),
    "governed_history_binding_ttl_seconds": (
        "GOVERNED_HISTORY_BINDING_TTL_SECONDS",
        0,
        3_600,
        False,
    ),
}


_BOOLEAN_CONFIG_KEYS = {
    "neo4j_enabled": "NEO4J_ENABLED",
    "faiss_enabled": "FAISS_ENABLED",
    "index_cache_enabled": "INDEX_CACHE_ENABLED",
    "few_shot_enabled": "FEW_SHOT_ENABLED",
    "sql_validation_enabled": "SQL_VALIDATION_ENABLED",
    "sql_scope_enforcement": "SQL_SCOPE_ENFORCEMENT",
    "semantic_governance_enabled": "SEMANTIC_GOVERNANCE_ENABLED",
    "sql_explain_enabled": "SQL_EXPLAIN_ENABLED",
    "streamlit_head": "STREAMLIT_HEAD",
    "history_auto_save": "HISTORY_AUTO_SAVE",
}


def parse_env_bool(
    key: str,
    default: bool,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Parse the documented ``true``/``false`` literals, case-insensitively."""
    source = os.environ if environ is None else environ
    raw = source.get(key)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeConfigurationError(
        f"{key} 必须使用布尔值 true 或 false"
    )


def parse_env_int(
    key: str,
    default: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Parse an integer environment value without echoing untrusted input."""
    source = os.environ if environ is None else environ
    raw = source.get(key)
    if raw is None:
        return default
    if not isinstance(raw, str):
        raise RuntimeConfigurationError(f"{key} 必须是整数")
    try:
        return int(raw.strip(), 10)
    except ValueError:
        raise RuntimeConfigurationError(f"{key} 必须是整数") from None


def parse_env_float(
    key: str,
    default: float,
    environ: Mapping[str, str] | None = None,
) -> float:
    """Parse a finite numeric environment value without echoing untrusted input."""
    source = os.environ if environ is None else environ
    raw = source.get(key)
    if raw is None:
        return default
    if not isinstance(raw, str):
        raise RuntimeConfigurationError(f"{key} 必须是数值")
    try:
        value = float(raw.strip())
    except ValueError:
        raise RuntimeConfigurationError(f"{key} 必须是数值") from None
    if not isfinite(value):
        raise RuntimeConfigurationError(f"{key} 必须是有限数值")
    return value


def _validate_integer(
    value: Any, *, key: str, minimum: int, maximum: int
) -> int:
    if not _is_strict_int(value) or not minimum <= value <= maximum:
        raise RuntimeConfigurationError(
            f"{key} 必须是 {minimum} 到 {maximum} 的整数"
        )
    return value


def _validate_number(
    value: Any,
    *,
    key: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value > maximum
        or (value < minimum if minimum_inclusive else value <= minimum)
    ):
        comparison = "不小于" if minimum_inclusive else "大于"
        raise RuntimeConfigurationError(
            f"{key} 必须是 {comparison} {minimum} 且不超过 {maximum} 的有限数值"
        )
    return float(value)


def validate_runtime_config(config: Mapping[str, Any]) -> None:
    """Apply one fail-closed startup and runtime configuration contract."""
    for internal_key, public_key in _BOOLEAN_CONFIG_KEYS.items():
        if internal_key in config and not isinstance(config[internal_key], bool):
            raise RuntimeConfigurationError(
                f"{public_key} 必须使用布尔值 true 或 false"
            )

    for internal_key, (public_key, minimum, maximum) in _INTEGER_CONFIG_LIMITS.items():
        if internal_key in config:
            _validate_integer(
                config[internal_key],
                key=public_key,
                minimum=minimum,
                maximum=maximum,
            )

    for internal_key, (
        public_key,
        minimum,
        maximum,
        minimum_inclusive,
    ) in _FLOAT_CONFIG_LIMITS.items():
        if internal_key in config:
            _validate_number(
                config[internal_key],
                key=public_key,
                minimum=minimum,
                maximum=maximum,
                minimum_inclusive=minimum_inclusive,
            )

    if "sql_repair_max_attempts" in config:
        value = config["sql_repair_max_attempts"]
        if not _is_strict_int(value) or value not in (0, 1):
            raise RuntimeConfigurationError(
                "SQL_REPAIR_MAX_ATTEMPTS 必须是 0 或 1 的整数"
            )

    if "history_max_records" in config:
        validate_history_max_records(
            config["history_max_records"], key="HISTORY_MAX_RECORDS"
        )

    if {"bm25_weight", "dense_weight"} <= config.keys():
        if abs(config["bm25_weight"] + config["dense_weight"] - 1.0) > 0.01:
            raise RuntimeConfigurationError("BM25_WEIGHT 与 DENSE_WEIGHT 的总和必须为 1.0")

    governance_enabled = config.get("semantic_governance_enabled", False)
    explain_enabled = config.get("sql_explain_enabled", False)
    if explain_enabled and not governance_enabled:
        raise RuntimeConfigurationError(
            "SQL_EXPLAIN_ENABLED=true 要求 "
            "SEMANTIC_GOVERNANCE_ENABLED=true"
        )

    if not (governance_enabled or explain_enabled):
        return

    dialect = config.get("sql_dialect", "mysql")
    if not isinstance(dialect, str) or dialect.casefold() != "mysql":
        enabled_keys = ["SEMANTIC_GOVERNANCE_ENABLED"]
        if explain_enabled:
            enabled_keys.append("SQL_EXPLAIN_ENABLED")
        raise RuntimeConfigurationError(
            f"SQL_DIALECT 在启用 {'/'.join(enabled_keys)} 时仅支持 mysql"
        )


def validate_history_max_records(value: Any, *, key: str) -> int:
    """Require DB pruning to remain enabled with a defensible upper bound."""
    if (
        not _is_strict_int(value)
        or not 1 <= value <= MAX_HISTORY_RECORDS
    ):
        raise RuntimeConfigurationError(
            f"{key} 必须是 1 到 {MAX_HISTORY_RECORDS} 的整数"
        )
    return value


__all__ = [
    "RuntimeConfigurationError",
    "MAX_HISTORY_RECORDS",
    "MAX_SQL_LENGTH",
    "parse_env_bool",
    "parse_env_float",
    "parse_env_int",
    "validate_history_max_records",
    "validate_runtime_config",
]
