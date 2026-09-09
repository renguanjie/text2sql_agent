"""Strict, shared validation for environment and runtime configuration."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any


class RuntimeConfigurationError(ValueError):
    """Configuration error whose message contains keys, never supplied values."""


MAX_HISTORY_RECORDS = 10_000


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


def validate_runtime_config(config: Mapping[str, Any]) -> None:
    """Apply the same fail-closed governance rules at startup and reload."""
    for internal_key, public_key in _BOOLEAN_CONFIG_KEYS.items():
        if internal_key in config and type(config[internal_key]) is not bool:
            raise RuntimeConfigurationError(
                f"{public_key} 必须使用布尔值 true 或 false"
            )

    if "history_max_records" in config:
        validate_history_max_records(
            config["history_max_records"], key="HISTORY_MAX_RECORDS"
        )

    if "governed_history_binding_ttl_seconds" in config:
        ttl = config["governed_history_binding_ttl_seconds"]
        if (
            isinstance(ttl, bool)
            or not isinstance(ttl, (int, float))
            or not 0 < ttl <= 3600
        ):
            raise RuntimeConfigurationError(
                "governed_history_binding_ttl_seconds 必须大于 0 且不超过 3600"
            )

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
        type(value) is not int
        or not 1 <= value <= MAX_HISTORY_RECORDS
    ):
        raise RuntimeConfigurationError(
            f"{key} 必须是 1 到 {MAX_HISTORY_RECORDS} 的整数"
        )
    return value


__all__ = [
    "RuntimeConfigurationError",
    "MAX_HISTORY_RECORDS",
    "parse_env_bool",
    "validate_history_max_records",
    "validate_runtime_config",
]
