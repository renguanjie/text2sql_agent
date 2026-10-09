"""语义治理设置面板的 Streamlit AppTest 集成测试。

与 tests/test_governance_settings.py 的纯函数测试互补：这里真实渲染面板，
验证控件初值来自 ctx.config、保存只提交变化项、前置校验能拦住非法组合、
以及后端异常被正确呈现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.app_context import _GOVERNANCE_RELOAD_KEYS, ApplicationConfigurationError
from core.utils.governance_settings import EDITABLE_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("streamlit", reason="Streamlit 未安装")
from streamlit.testing.v1 import AppTest  # noqa: E402

BASE_CONFIG: dict[str, Any] = {
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


class FakeContext:
    """只实现面板用到的 ApplicationContext 表面。"""

    def __init__(self, config: dict[str, Any] | None = None, error: Exception | None = None):
        self._config = dict(config or BASE_CONFIG)
        self.semantic_registry = None
        self._error = error
        self.reload_calls: list[dict[str, Any]] = []

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def reload_governance(self, **updates: Any) -> None:
        if self._error is not None:
            raise self._error
        unknown = set(updates) - set(_GOVERNANCE_RELOAD_KEYS)
        if unknown:
            raise ApplicationConfigurationError(
                f"reload_governance 仅支持更新白名单键；收到 {sorted(unknown)}"
            )
        self.reload_calls.append(dict(updates))
        self._config.update(updates)


def _panel(ctx: FakeContext | None = None, tmp_path: Path | None = None) -> AppTest:
    """构造只渲染治理面板的最小脚本并首跑一次。"""
    script = (tmp_path or Path(".")) / "governance_panel_under_test.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from app.pages.governance_settings import show_governance_settings\n"
        "show_governance_settings()\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(str(script), default_timeout=60)
    if ctx is not None:
        at.session_state["app_context"] = ctx
    at.run()
    return at


def _save(at: AppTest) -> AppTest:
    at.button(key="btn_save_gov").click()
    at.run()
    return at


def _reset(at: AppTest) -> AppTest:
    at.button(key="btn_reset_gov").click()
    at.run()
    return at


@pytest.fixture()
def app(tmp_path):
    def _make(ctx: FakeContext | None = None) -> AppTest:
        return _panel(ctx, tmp_path)

    return _make


# ==================== 纠错次数选项映射 ====================


def test_repair_choice_roundtrip():
    from app.pages.governance_settings import (
        _REPAIR_ATTEMPT_CHOICES,
        repair_choice_index,
        repair_choice_value,
    )

    for attempts in (0, 1):
        choice = _REPAIR_ATTEMPT_CHOICES[repair_choice_index(attempts)]
        assert repair_choice_value(choice) == attempts


def test_repair_choices_cover_exactly_the_allowed_values():
    """设计文档规定纠错上限只能是 0 或 1，选项集合不得偏离。"""
    from app.pages.governance_settings import (
        _REPAIR_ATTEMPT_CHOICES,
        repair_choice_value,
    )

    assert sorted(repair_choice_value(c) for c in _REPAIR_ATTEMPT_CHOICES) == [0, 1]


# ==================== 渲染 ====================


def test_panel_renders_disabled_status(app):
    at = app(FakeContext())
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "语义治理：已关闭" in captions
    assert "EXPLAIN 预检：已关闭" in captions


def test_panel_renders_enabled_status_with_version(app):
    ctx = FakeContext(dict(BASE_CONFIG, semantic_governance_enabled=True))
    at = app(ctx)
    assert not at.exception
    assert "语义治理：已启用" in " ".join(c.value for c in at.caption)


def test_every_editable_key_has_a_widget(app):
    """白名单里每个可编辑键都必须有对应控件，否则该项无法从前端调整。"""
    at = app(FakeContext())
    assert not at.exception
    rendered = {
        key[len("cfg_gov_"):]
        for key in at.session_state.filtered_state
        if key.startswith("cfg_gov_")
    }
    assert rendered == set(EDITABLE_KEYS), (
        f"缺少控件：{sorted(set(EDITABLE_KEYS) - rendered)}；"
        f"多余控件：{sorted(rendered - set(EDITABLE_KEYS))}"
    )


def test_widgets_seed_from_context_config(app):
    ctx = FakeContext(dict(BASE_CONFIG, sql_query_max_rows=42, semantic_datasource="neon-db"))
    at = app(ctx)
    assert at.number_input(key="cfg_gov_sql_query_max_rows").value == 42
    assert at.text_input(key="cfg_gov_semantic_datasource").value == "neon-db"


def test_blocked_tables_render_as_comma_separated_text(app):
    at = app(FakeContext())
    value = at.text_area(key="cfg_gov_sql_blocked_tables").value
    assert value == "sql_generation_history, user_feedback"


# ==================== 保存链路 ====================


def test_save_without_edits_reports_no_change(app):
    ctx = FakeContext()
    at = _save(app(ctx))
    assert not at.exception
    assert ctx.reload_calls == []
    assert any("配置未发生变化" in s.value for s in at.success)


def test_save_submits_only_changed_keys(app):
    ctx = FakeContext()
    at = app(ctx)
    at.toggle(key="cfg_gov_semantic_governance_enabled").set_value(True)
    at.run()
    at = _save(at)
    assert not at.exception
    assert ctx.reload_calls == [{"semantic_governance_enabled": True}]
    assert any("已应用 1 项治理配置变更" in s.value for s in at.success)


def test_save_submits_multiple_changed_keys(app):
    ctx = FakeContext()
    at = app(ctx)
    at.number_input(key="cfg_gov_sql_query_max_rows").set_value(1000)
    at.number_input(key="cfg_gov_sql_explain_timeout_ms").set_value(5000)
    at.run()
    at = _save(at)
    assert not at.exception
    assert ctx.reload_calls == [
        {"sql_query_max_rows": 1000, "sql_explain_timeout_ms": 5000}
    ]


def test_save_submits_int_not_float_for_int_keys(app):
    """st.number_input 可能给出 float，提交给后端前必须收敛为 int。"""
    ctx = FakeContext()
    at = app(ctx)
    at.number_input(key="cfg_gov_semantic_context_max_items").set_value(30)
    at.run()
    at = _save(at)
    submitted = ctx.reload_calls[0]["semantic_context_max_items"]
    assert submitted == 30
    assert isinstance(submitted, int) and not isinstance(submitted, bool)


def test_save_submits_blocked_tables_as_list(app):
    ctx = FakeContext()
    at = app(ctx)
    at.text_area(key="cfg_gov_sql_blocked_tables").set_value("a, b\n c ,a")
    at.run()
    at = _save(at)
    assert not at.exception
    assert ctx.reload_calls == [{"sql_blocked_tables": ["a", "b", "c"]}]


def test_repair_attempts_choice_submits_int(app):
    """单选项是自描述文本，提交前必须还原为后端要求的 0/1 整数。"""
    ctx = FakeContext()
    at = app(ctx)
    at.radio(key="cfg_gov_sql_repair_max_attempts").set_value("0 — 校验失败即拒绝")
    at.run()
    at = _save(at)
    assert not at.exception
    submitted = ctx.reload_calls[0]["sql_repair_max_attempts"]
    assert submitted == 0
    assert isinstance(submitted, int) and not isinstance(submitted, bool)


def test_successful_save_marks_widgets_for_reseed(app):
    ctx = FakeContext()
    at = app(ctx)
    at.number_input(key="cfg_gov_sql_query_max_rows").set_value(777)
    at.run()
    at = _save(at)
    assert not at.exception
    assert at.session_state["_gov_reseed"] is True
    assert any("已应用 1 项" in s.value for s in at.success)


def test_reset_reseeds_widgets_to_new_active_config(app):
    """恢复默认后，控件必须在下一轮渲染时从新的 ctx.config 重新播种。"""
    from app.pages.governance_settings import governance_defaults

    defaults = governance_defaults()
    original = int(defaults["sql_query_max_rows"])
    ctx = FakeContext(dict(defaults, sql_query_max_rows=original + 1))

    at = app(ctx)
    assert at.number_input(key="cfg_gov_sql_query_max_rows").value == original + 1

    at = _reset(at)
    assert not at.exception
    assert ctx.reload_calls == [{"sql_query_max_rows": original}]

    at.run()
    assert at.number_input(key="cfg_gov_sql_query_max_rows").value == original


# ==================== 前置校验拦截 ====================


def test_non_mysql_dialect_blocks_enabling_governance(app):
    """后端只会回一句「语义治理配置无效」，具体原因必须由前端给出。"""
    ctx = FakeContext(dict(BASE_CONFIG, sql_dialect="postgresql"))
    at = app(ctx)
    at.toggle(key="cfg_gov_semantic_governance_enabled").set_value(True)
    at.run()
    at = _save(at)
    assert ctx.reload_calls == []
    errors = " ".join(e.value for e in at.error)
    assert "仅支持 mysql" in errors


def test_invalid_datasource_is_blocked(app):
    ctx = FakeContext()
    at = app(ctx)
    at.text_input(key="cfg_gov_semantic_datasource").set_value("../../etc/passwd")
    at.run()
    at = _save(at)
    assert ctx.reload_calls == []
    assert "SEMANTIC_DATASOURCE" in " ".join(e.value for e in at.error)


def test_out_of_range_rows_is_blocked(app):
    ctx = FakeContext()
    at = app(ctx)
    at.number_input(key="cfg_gov_sql_query_max_rows").set_value(999_999)
    at.run()
    at = _save(at)
    assert ctx.reload_calls == []
    assert "SQL_QUERY_MAX_ROWS" in " ".join(e.value for e in at.error)


def test_explain_toggle_is_disabled_until_governance_enabled(app):
    """EXPLAIN 依赖语义治理，UI 层应先禁用，而不是等保存时报错。"""
    at = app(FakeContext())
    toggle = at.toggle(key="cfg_gov_sql_explain_enabled")
    assert toggle.proto.disabled is True

    ctx = FakeContext(dict(BASE_CONFIG, semantic_governance_enabled=True))
    at = app(ctx)
    assert at.toggle(key="cfg_gov_sql_explain_enabled").proto.disabled is False


def test_explain_can_be_enabled_together_with_governance(app):
    ctx = FakeContext(dict(BASE_CONFIG, semantic_governance_enabled=True))
    at = app(ctx)
    at.toggle(key="cfg_gov_sql_explain_enabled").set_value(True)
    at.run()
    at = _save(at)
    assert not at.exception
    assert ctx.reload_calls == [{"sql_explain_enabled": True}]


# ==================== 后端异常 ====================


def test_backend_configuration_error_is_surfaced(app):
    ctx = FakeContext(error=ApplicationConfigurationError("语义治理配置无效"))
    at = app(ctx)
    at.number_input(key="cfg_gov_sql_query_max_rows").set_value(600)
    at.run()
    at = _save(at)
    assert "语义治理配置无效" in " ".join(e.value for e in at.error)


def test_unexpected_backend_error_does_not_leak_details(app):
    """非 ApplicationConfigurationError 的异常不得把原始消息抛给用户。"""
    ctx = FakeContext(error=RuntimeError("mysql://user:secret@db:3306/x"))
    at = app(ctx)
    at.number_input(key="cfg_gov_sql_query_max_rows").set_value(600)
    at.run()
    at = _save(at)
    text = " ".join(e.value for e in at.error)
    assert "secret" not in text
    assert "logs/app.log" in text


# ==================== 恢复默认 ====================


def test_reset_restores_env_defaults(app):
    from app.pages.governance_settings import governance_defaults

    defaults = governance_defaults()
    original = int(defaults["sql_query_max_rows"])
    ctx = FakeContext(dict(defaults, sql_query_max_rows=original + 1))
    at = _reset(app(ctx))
    assert not at.exception
    assert ctx.reload_calls == [{"sql_query_max_rows": original}]


def test_reset_with_matching_config_reports_no_change(app):
    from app.pages.governance_settings import governance_defaults

    ctx = FakeContext(governance_defaults())
    at = _reset(app(ctx))
    assert not at.exception
    assert ctx.reload_calls == []
    assert any("配置未发生变化" in s.value for s in at.success)
