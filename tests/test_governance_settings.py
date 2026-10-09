"""语义治理设置面板辅助层的单元测试。

重点保证两件事：
1. 面板暴露的键集合与 `reload_governance` 的白名单严格一致，避免前端送出被后端拒绝的键；
2. 取值范围与跨字段规则与 `core/config_validation.py` 不漂移，
   使前端能在请求后端前给出具体原因（后端只会返回泛化的"语义治理配置无效"）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.app_context import _GOVERNANCE_RELOAD_KEYS
from core.config_validation import _FLOAT_CONFIG_LIMITS, _INTEGER_CONFIG_LIMITS
from core.semantic import registry as semantic_registry
from core.utils.governance_settings import (
    BOOL_KEYS,
    DATASOURCE_PATTERN,
    DISPLAY_ONLY_KEYS,
    EDITABLE_KEYS,
    FLOAT_LIMITS,
    INT_LIMITS,
    active_semantic_version,
    describe_status,
    diff_updates,
    format_blocked_tables,
    parse_blocked_tables,
    validate_updates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG = {
    "semantic_governance_enabled": False,
    "semantic_model_dir": "semantic_models",
    "semantic_datasource": "default",
    "semantic_context_max_chars": 2400,
    "semantic_context_max_items": 24,
    "sql_repair_max_attempts": 1,
    "sql_explain_enabled": False,
    "sql_explain_timeout_ms": 3000,
    "sql_explain_max_estimated_rows": 0,
    "sql_dialect": "mysql",
    "sql_max_length": 2000,
    "sql_query_max_rows": 500,
    "sql_query_timeout_ms": 10000,
    "sql_scope_enforcement": True,
    "sql_blocked_tables": ["sql_generation_history", "user_feedback"],
    "max_prepared_requests": 256,
    "prepared_request_ttl_seconds": 120.0,
    "governed_history_binding_ttl_seconds": 900.0,
}


# ==================== 与后端契约的一致性 ====================


def test_exposed_keys_match_reload_whitelist_exactly():
    """面板键集合必须恰好等于 reload_governance 白名单，不多不少。"""
    assert EDITABLE_KEYS | set(DISPLAY_ONLY_KEYS) == set(_GOVERNANCE_RELOAD_KEYS)


def test_editable_and_display_only_keys_are_disjoint():
    """只读键不应同时可编辑，否则会出现控件与展示互相矛盾。"""
    assert EDITABLE_KEYS.isdisjoint(DISPLAY_ONLY_KEYS)


def test_panel_covers_every_whitelist_key():
    """白名单里的每个键都必须被面板覆盖，避免有配置项无法从前端调整。"""
    missing = set(_GOVERNANCE_RELOAD_KEYS) - EDITABLE_KEYS - set(DISPLAY_ONLY_KEYS)
    assert not missing, f"以下治理键未在面板中暴露：{sorted(missing)}"


@pytest.mark.parametrize(
    "key", sorted(set(INT_LIMITS) - {"sql_repair_max_attempts"})
)
def test_int_limits_match_config_validation(key):
    """整数取值范围必须与 core/config_validation 完全一致。"""
    public_key, minimum, maximum = INT_LIMITS[key]
    assert key in _INTEGER_CONFIG_LIMITS, f"{key} 不在后端整数限制表中"
    backend_public, backend_min, backend_max = _INTEGER_CONFIG_LIMITS[key]
    assert (public_key, minimum, maximum) == (
        backend_public,
        backend_min,
        backend_max,
    )


def test_repair_attempts_limit_matches_backend_special_case():
    """SQL_REPAIR_MAX_ATTEMPTS 在后端是专门的 0/1 规则，不走通用整数表。"""
    assert "sql_repair_max_attempts" not in _INTEGER_CONFIG_LIMITS
    assert INT_LIMITS["sql_repair_max_attempts"] == (
        "SQL_REPAIR_MAX_ATTEMPTS",
        0,
        1,
    )


@pytest.mark.parametrize("key", sorted(FLOAT_LIMITS))
def test_float_limits_match_config_validation(key):
    """浮点取值范围必须与 core/config_validation 完全一致。"""
    public_key, minimum, maximum = FLOAT_LIMITS[key]
    assert key in _FLOAT_CONFIG_LIMITS, f"{key} 不在后端浮点限制表中"
    backend_public, backend_min, backend_max, backend_min_inclusive = (
        _FLOAT_CONFIG_LIMITS[key]
    )
    assert (public_key, minimum, maximum) == (
        backend_public,
        backend_min,
        backend_max,
    )
    assert backend_min_inclusive is False, (
        f"{key} 的下界变为闭区间，validate_updates 的比较运算符需要同步调整"
    )


def test_datasource_pattern_matches_registry():
    """数据源正则必须与 SemanticRegistry 的校验一致，否则前端放行后端拒绝。"""
    assert DATASOURCE_PATTERN.pattern == semantic_registry._DATASOURCE.pattern


def test_bool_keys_are_declared_in_config_validation():
    """布尔键必须在后端的布尔键表中登记。"""
    from core.config_validation import _BOOLEAN_CONFIG_KEYS

    for key in BOOL_KEYS:
        assert key in _BOOLEAN_CONFIG_KEYS, f"{key} 未在后端布尔键表中登记"


# ==================== 禁用表解析 ====================


def test_parse_blocked_tables_splits_on_comma_and_whitespace():
    assert parse_blocked_tables("a, b\nc\t d ,, e") == ["a", "b", "c", "d", "e"]


def test_parse_blocked_tables_dedupes_while_preserving_order():
    assert parse_blocked_tables("b, a, b, a") == ["b", "a"]


def test_parse_blocked_tables_handles_empty_and_non_string():
    assert parse_blocked_tables("") == []
    assert parse_blocked_tables("   ") == []
    assert parse_blocked_tables(None) == []


def test_format_blocked_tables_roundtrip():
    tables = ["sql_generation_history", "user_feedback"]
    assert parse_blocked_tables(format_blocked_tables(tables)) == tables
    assert format_blocked_tables(None) == ""
    assert format_blocked_tables([]) == ""


# ==================== 跨字段规则 ====================


def test_valid_updates_produce_no_errors():
    assert validate_updates(BASE_CONFIG, {"semantic_context_max_items": 30}) == []


def test_explain_requires_governance():
    errors = validate_updates(BASE_CONFIG, {"sql_explain_enabled": True})
    assert len(errors) == 1
    assert "必须先启用语义治理" in errors[0]


def test_explain_with_governance_is_allowed():
    updates = {"semantic_governance_enabled": True, "sql_explain_enabled": True}
    assert validate_updates(BASE_CONFIG, updates) == []


def test_governance_rejects_non_mysql_dialect():
    config = dict(BASE_CONFIG, sql_dialect="postgresql")
    errors = validate_updates(config, {"semantic_governance_enabled": True})
    assert any("仅支持 mysql" in e for e in errors)


def test_dialect_check_is_case_insensitive():
    config = dict(BASE_CONFIG, sql_dialect="MySQL")
    assert validate_updates(config, {"semantic_governance_enabled": True}) == []


def test_disabling_governance_does_not_reintroduce_dialect_error():
    """关闭治理后非 mysql 方言应重新可用，错误必须基于合并后的配置判定。"""
    config = dict(
        BASE_CONFIG,
        sql_dialect="postgresql",
        semantic_governance_enabled=True,
        sql_explain_enabled=True,
    )
    updates = {"semantic_governance_enabled": False, "sql_explain_enabled": False}
    assert validate_updates(config, updates) == []


def test_repair_attempts_rejects_values_above_one():
    errors = validate_updates(BASE_CONFIG, {"sql_repair_max_attempts": 2})
    assert any("SQL_REPAIR_MAX_ATTEMPTS" in e for e in errors)


def test_repair_attempts_rejects_bool():
    """bool 是 int 子类，必须被严格整数校验拒绝。"""
    errors = validate_updates(BASE_CONFIG, {"sql_repair_max_attempts": True})
    assert any("SQL_REPAIR_MAX_ATTEMPTS" in e for e in errors)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("semantic_context_max_chars", 0),
        ("semantic_context_max_chars", 100_001),
        ("sql_query_max_rows", 0),
        ("sql_query_max_rows", 100_001),
        ("sql_explain_timeout_ms", 0),
        ("max_prepared_requests", 4_097),
        ("sql_max_length", 100_001),
    ],
)
def test_int_bounds_are_enforced(key, value):
    errors = validate_updates(BASE_CONFIG, {key: value})
    assert errors, f"{key}={value} 应当被拒绝"
    assert INT_LIMITS[key][0] in errors[0]


def test_sql_explain_max_estimated_rows_allows_zero():
    """0 表示不按估算行数拒绝，是合法取值而非下界违规。"""
    assert validate_updates(BASE_CONFIG, {"sql_explain_max_estimated_rows": 0}) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("prepared_request_ttl_seconds", 0.0),
        ("prepared_request_ttl_seconds", 3_601.0),
        ("governed_history_binding_ttl_seconds", -1.0),
    ],
)
def test_float_bounds_are_enforced(key, value):
    assert validate_updates(BASE_CONFIG, {key: value}), f"{key}={value} 应当被拒绝"


@pytest.mark.parametrize(
    "datasource",
    ["../etc/passwd", "-leading-dash", "has space", "a" * 129, "", "中文源"],
)
def test_datasource_rejects_unsafe_identifiers(datasource):
    errors = validate_updates(BASE_CONFIG, {"semantic_datasource": datasource})
    assert any("SEMANTIC_DATASOURCE" in e for e in errors)


@pytest.mark.parametrize("datasource", ["default", "neon-db", "Db_1", "a" * 128])
def test_datasource_accepts_safe_identifiers(datasource):
    assert validate_updates(BASE_CONFIG, {"semantic_datasource": datasource}) == []


def test_blocked_tables_rejects_non_sequence():
    errors = validate_updates(BASE_CONFIG, {"sql_blocked_tables": "a,b"})
    assert any("SQL_BLOCKED_TABLES" in e for e in errors)


def test_blocked_tables_rejects_empty_entries():
    errors = validate_updates(BASE_CONFIG, {"sql_blocked_tables": ["ok", "  "]})
    assert any("非空表名" in e for e in errors)


def test_multiple_errors_are_all_reported():
    """一次提交多处错误时应全部列出，避免用户逐次试错。"""
    errors = validate_updates(
        BASE_CONFIG,
        {
            "sql_explain_enabled": True,
            "sql_query_max_rows": 0,
            "semantic_datasource": "../x",
        },
    )
    assert len(errors) == 3


# ==================== 差异计算 ====================


def test_diff_returns_only_changed_keys():
    desired = dict(BASE_CONFIG, sql_query_max_rows=1000)
    assert diff_updates(BASE_CONFIG, desired) == {"sql_query_max_rows": 1000}


def test_diff_returns_empty_when_unchanged():
    assert diff_updates(BASE_CONFIG, dict(BASE_CONFIG)) == {}


def test_diff_distinguishes_bool_from_int():
    """True == 1 在 Python 中成立，必须按类型区分，否则会漏掉开关变更。"""
    current = dict(BASE_CONFIG, semantic_governance_enabled=False)
    desired = dict(current, semantic_governance_enabled=True)
    assert diff_updates(current, desired) == {"semantic_governance_enabled": True}

    # 配置里若因故存成 int 1，切到 True 也必须被识别为变更并重新发布
    int_typed = dict(BASE_CONFIG, semantic_governance_enabled=1)
    assert diff_updates(int_typed, dict(BASE_CONFIG, semantic_governance_enabled=True)) == {
        "semantic_governance_enabled": True
    }


def test_diff_treats_list_and_tuple_as_equal():
    """ctx.config 可能给出 tuple，控件给出 list，不应被判定为变更。"""
    current = dict(BASE_CONFIG, sql_blocked_tables=("a", "b"))
    desired = dict(BASE_CONFIG, sql_blocked_tables=["a", "b"])
    assert diff_updates(current, desired) == {}


def test_diff_detects_blocked_tables_change():
    desired = dict(BASE_CONFIG, sql_blocked_tables=["a"])
    assert diff_updates(BASE_CONFIG, desired) == {"sql_blocked_tables": ["a"]}


def test_diff_drops_keys_outside_whitelist():
    """非白名单键必须在到达后端前被剔除，否则 reload_governance 会整体拒绝。"""
    desired = dict(BASE_CONFIG, llm_model="gpt-4o", sql_query_max_rows=10)
    assert diff_updates(BASE_CONFIG, desired) == {"sql_query_max_rows": 10}


# ==================== 展示辅助 ====================


def test_describe_status_when_disabled():
    text = describe_status(BASE_CONFIG)
    assert "语义治理：已关闭" in text
    assert "EXPLAIN 预检：已关闭" in text
    assert "语义版本" not in text


def test_describe_status_when_enabled():
    config = dict(BASE_CONFIG, semantic_governance_enabled=True)
    text = describe_status(config, "1.1.0")
    assert "语义治理：已启用" in text
    assert "1.1.0" in text
    assert "default" in text


def test_describe_status_shows_unloaded_version():
    config = dict(BASE_CONFIG, semantic_governance_enabled=True)
    assert "未加载" in describe_status(config, None)


def test_active_semantic_version_reads_snapshot():
    class _Snapshot:
        model_version = "2.3.4"

    class _Registry:
        def get_snapshot(self):
            return _Snapshot()

    assert active_semantic_version(_Registry()) == "2.3.4"


def test_active_semantic_version_tolerates_unloaded_registry():
    class _Registry:
        def get_snapshot(self):
            raise semantic_registry.SemanticRegistryError("语义模型尚未加载")

    assert active_semantic_version(_Registry()) is None
    assert active_semantic_version(None) is None


# ==================== 默认语义模型可用性 ====================


def test_default_semantic_model_exists():
    """面板启用的数据源必须真的有对应的 YAML，否则保存会失败。"""
    assert (PROJECT_ROOT / "semantic_models" / "default.yaml").is_file()
