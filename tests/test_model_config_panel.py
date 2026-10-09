"""模型配置面板的 Streamlit AppTest 集成测试。

覆盖两处修复：
- API Key 不再作为控件初值下发到浏览器（此前 `.env` 中的真实密钥会随 widget
  proto 序列化到客户端，`type="password"` 只遮蔽显示）；
- 「恢复默认」真正把 `.env` 配置重新应用到运行时，并让控件重新播种，
  此前它只清除显示用的覆盖键，界面与实际生效配置会分叉。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PY = PROJECT_ROOT / "app.py"

pytest.importorskip("streamlit", reason="Streamlit 未安装")
from streamlit.testing.v1 import AppTest  # noqa: E402

from config import (  # noqa: E402
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_PROVIDERS,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_PROVIDERS,
)
from core.app_context import ApplicationConfigurationError  # noqa: E402

SECRET = "sk-TEST-SECRET-do-not-leak-0123456789"

# 需要一个带 API Key 的提供商来复现密钥回显路径
KEY_PROVIDER = next(k for k, v in LLM_PROVIDERS.items() if v["needs_api_key"])
KEY_PROVIDER_LABEL = LLM_PROVIDERS[KEY_PROVIDER]["label"]
NOKEY_PROVIDER = next(k for k, v in LLM_PROVIDERS.items() if not v["needs_api_key"])
NOKEY_PROVIDER_LABEL = LLM_PROVIDERS[NOKEY_PROVIDER]["label"]


class FakeContext:
    """只实现 app.py 会触到的 ApplicationContext 表面。

    mysql_client / neo4j_client / sql_generator 一律为 None，使所有数据库分支
    短路，测试因此完全离线且不依赖 .env 里的 NEO4J_ENABLED。
    """

    def __init__(self, config: dict[str, Any]):
        self._config = dict(config)
        self.mysql_client = None
        self.neo4j_client = None
        self.sql_generator = None
        self.semantic_registry = None
        self.initialization_error = None
        self.llm_reload_calls: list[dict[str, Any]] = []
        self.embedding_reload_calls: list[dict[str, Any]] = []

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def ensure_initialized(self) -> bool:
        return True

    def reset(self) -> None:
        self._config = dict(self._config)

    def reload_llm(self, **updates: Any) -> None:
        self.llm_reload_calls.append(dict(updates))
        self._config.update(updates)

    def reload_embedding(self, **updates: Any) -> None:
        self.embedding_reload_calls.append(dict(updates))
        self._config.update(updates)


def _base_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_api_key": "",
        "llm_base_url": LLM_BASE_URL,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_api_key": "",
    }
    config.update(overrides)
    return config


def _launch(ctx: FakeContext) -> AppTest:
    at = AppTest.from_file(str(APP_PY), default_timeout=90)
    at.session_state["initialized"] = True
    at.session_state["app_context"] = ctx
    at.run()
    assert not at.exception, f"app.py 渲染异常：{at.exception}"
    return at


def _all_widget_values(at: AppTest) -> list[str]:
    values: list[str] = []
    for group in (at.text_input, at.text_area, at.selectbox, at.radio):
        for widget in group:
            values.append(str(widget.value))
    return values


def _assert_key_widget_rendered(at: AppTest) -> None:
    """断言面板确实按运行时配置渲染出了 API Key 控件。

    没有这个前置条件，"密钥没有出现在任何控件里"会因为控件根本没渲染而假绿：
    修复前面板读的是 .env 的 LLM_PROVIDER 而非 ctx.config，提供商对不上时
    整个 API Key 控件都不存在。
    """
    assert at.selectbox(key="cfg_llm_provider").value == KEY_PROVIDER_LABEL
    assert at.text_input(key="cfg_llm_api_key").proto.type == 1  # PASSWORD


# ==================== 面板必须反映运行时生效配置 ====================


def test_panel_reflects_runtime_config_not_env():
    """面板初值必须来自 ctx.config，而不是 .env 的默认值。"""
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_model="runtime-model")
    )
    at = _launch(ctx)
    assert at.selectbox(key="cfg_llm_provider").value == KEY_PROVIDER_LABEL
    assert at.text_input(key="cfg_llm_model").value == "runtime-model"


# ==================== A：密钥不得下发到浏览器 ====================


def test_api_key_widget_is_blank_even_when_a_key_is_active():
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_api_key=SECRET)
    )
    at = _launch(ctx)
    _assert_key_widget_rendered(at)
    assert at.text_input(key="cfg_llm_api_key").value == ""


def test_active_secret_never_reaches_any_widget_or_session_state():
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_api_key=SECRET)
    )
    at = _launch(ctx)
    _assert_key_widget_rendered(at)
    assert all(SECRET not in v for v in _all_widget_values(at))
    assert all(
        SECRET not in str(value)
        for value in at.session_state.filtered_state.values()
    )


def test_embedding_api_key_widget_is_blank_too():
    emb_provider = next(
        k for k, v in EMBEDDING_PROVIDERS.items() if v["needs_api_key"]
    )
    ctx = FakeContext(
        _base_config(embedding_provider=emb_provider, embedding_api_key=SECRET)
    )
    at = _launch(ctx)
    assert at.selectbox(key="cfg_emb_provider").value == (
        EMBEDDING_PROVIDERS[emb_provider]["label"]
    )
    assert at.text_input(key="cfg_emb_api_key").value == ""
    assert all(SECRET not in v for v in _all_widget_values(at))


def test_secret_still_absent_after_saving_with_blank_key():
    """留空保存后，原密钥只应进入服务端 ctx，不得残留在控件状态里。"""
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_api_key=SECRET)
    )
    at = _launch(ctx)
    _assert_key_widget_rendered(at)
    at.button(key="btn_save_llm").click()
    at.run()
    assert all(SECRET not in v for v in _all_widget_values(at))
    assert all(
        SECRET not in str(value)
        for value in at.session_state.filtered_state.values()
    )


# ==================== 留空即沿用 / 填写则替换 ====================


def test_blank_key_on_save_keeps_the_active_key():
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_api_key=SECRET)
    )
    at = _launch(ctx)
    at.button(key="btn_save_llm").click()
    at.run()
    assert not at.exception
    assert len(ctx.llm_reload_calls) == 1
    assert ctx.llm_reload_calls[0]["llm_api_key"] == SECRET


def test_entered_key_on_save_replaces_the_active_key():
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_api_key=SECRET)
    )
    at = _launch(ctx)
    at.text_input(key="cfg_llm_api_key").set_value("sk-brand-new-key")
    at.button(key="btn_save_llm").click()
    at.run()
    assert ctx.llm_reload_calls[0]["llm_api_key"] == "sk-brand-new-key"


def test_provider_without_key_sends_empty_key():
    """切到本地提供商时不应把上一个提供商的密钥继续带着走。"""
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_api_key=SECRET)
    )
    at = _launch(ctx)
    at.selectbox(key="cfg_llm_provider").set_value(NOKEY_PROVIDER_LABEL)
    at.run()
    at.button(key="btn_save_llm").click()
    at.run()
    assert ctx.llm_reload_calls[0]["llm_provider"] == NOKEY_PROVIDER
    assert ctx.llm_reload_calls[0]["llm_api_key"] == ""


def test_switching_provider_updates_recommended_model():
    """提供商联动默认模型的行为必须保留（改动前的既有正确行为）。"""
    ctx = FakeContext(_base_config())
    at = _launch(ctx)
    at.selectbox(key="cfg_llm_provider").set_value(KEY_PROVIDER_LABEL)
    at.run()
    assert at.text_input(key="cfg_llm_model").value == (
        LLM_PROVIDERS[KEY_PROVIDER]["default_model"]
    )


# ==================== B：恢复默认必须真正生效并同步界面 ====================


def test_reset_applies_env_defaults_to_runtime():
    """恢复默认不能再只是清显示状态；必须真的重载回 .env 的配置。"""
    other_model = "some-other-model"
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_model=other_model)
    )
    at = _launch(ctx)
    at.button(key="btn_reset_llm").click()
    at.run()
    assert not at.exception
    assert len(ctx.llm_reload_calls) == 1
    assert ctx.llm_reload_calls[0]["llm_provider"] == LLM_PROVIDER
    assert ctx.llm_reload_calls[0]["llm_model"] == LLM_MODEL


def test_reset_reseeds_widgets_to_the_active_config():
    """恢复默认后界面必须与实际生效配置一致，不再停留在旧选择上。"""
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_model="some-other-model")
    )
    at = _launch(ctx)
    assert at.selectbox(key="cfg_llm_provider").value == KEY_PROVIDER_LABEL

    at.button(key="btn_reset_llm").click()
    at.run()
    at.run()  # 下一轮渲染才执行重新播种

    assert at.selectbox(key="cfg_llm_provider").value == (
        LLM_PROVIDERS[LLM_PROVIDER]["label"]
    )
    assert at.text_input(key="cfg_llm_model").value == LLM_MODEL


def test_reset_no_longer_asks_user_to_refresh():
    ctx = FakeContext(_base_config(llm_provider=KEY_PROVIDER))
    at = _launch(ctx)
    at.button(key="btn_reset_llm").click()
    at.run()
    messages = [s.value for s in at.success]
    assert not any("请刷新页面" in m for m in messages)
    assert any("已恢复 .env 默认 LLM" in m for m in messages)


def test_embedding_reset_applies_env_defaults():
    ctx = FakeContext(_base_config(embedding_model="some-other-embedding"))
    at = _launch(ctx)
    at.button(key="btn_reset_emb").click()
    at.run()
    assert len(ctx.embedding_reload_calls) == 1
    assert ctx.embedding_reload_calls[0]["embedding_provider"] == EMBEDDING_PROVIDER
    assert ctx.embedding_reload_calls[0]["embedding_model"] == EMBEDDING_MODEL


# ==================== 两组配置的重新播种必须互不干扰 ====================


def test_saving_llm_does_not_discard_pending_embedding_edits():
    ctx = FakeContext(_base_config())
    at = _launch(ctx)
    at.text_input(key="cfg_emb_model").set_value("my-pending-embedding-model")
    at.run()
    at.button(key="btn_save_llm").click()
    at.run()
    at.run()  # 触发 LLM 侧的重新播种
    assert at.text_input(key="cfg_emb_model").value == "my-pending-embedding-model"


def test_saving_embedding_does_not_discard_pending_llm_edits():
    ctx = FakeContext(_base_config())
    at = _launch(ctx)
    at.text_input(key="cfg_llm_model").set_value("my-pending-llm-model")
    at.run()
    at.button(key="btn_save_emb").click()
    at.run()
    at.run()
    assert at.text_input(key="cfg_llm_model").value == "my-pending-llm-model"


# ==================== C：失败原因必须呈现给用户 ====================


class FailingContext(FakeContext):
    """reload 抛出指定异常的 ctx，用于验证错误呈现。"""

    def __init__(self, config: dict[str, Any], error: Exception):
        super().__init__(config)
        self._error = error

    def reload_llm(self, **updates: Any) -> None:
        raise self._error

    def reload_embedding(self, **updates: Any) -> None:
        raise self._error


def test_configuration_error_reason_is_surfaced():
    """后端给出的具体原因必须原样呈现，而不是被泛化文案吞掉。"""
    reason = "LLM_PROVIDER 指定的提供商需要非空的 LLM_API_KEY"
    ctx = FailingContext(
        _base_config(), ApplicationConfigurationError(reason)
    )
    at = _launch(ctx)
    at.button(key="btn_save_llm").click()
    at.run()
    messages = [e.value for e in at.error]
    assert any(reason in m for m in messages), messages


def test_only_one_error_box_is_rendered_on_failure():
    """修复前 ctx 缺失路径会连弹两个错误框；任何失败都只应有一个。"""
    ctx = FailingContext(
        _base_config(), ApplicationConfigurationError("配置无效")
    )
    at = _launch(ctx)
    at.button(key="btn_save_llm").click()
    at.run()
    assert len(at.error) == 1, [e.value for e in at.error]


def test_unexpected_error_does_not_leak_details():
    """非配置类异常只给日志指引，不得把连接串等细节带回浏览器。"""
    ctx = FailingContext(
        _base_config(),
        RuntimeError("cannot connect mysql://user:sup3rsecret@db:3306/x"),
    )
    at = _launch(ctx)
    at.button(key="btn_save_llm").click()
    at.run()
    text = " ".join(e.value for e in at.error)
    assert "sup3rsecret" not in text
    assert "logs/app.log" in text


def test_failed_save_leaves_active_config_untouched():
    ctx = FailingContext(_base_config(), ApplicationConfigurationError("配置无效"))
    at = _launch(ctx)
    at.text_input(key="cfg_llm_model").set_value("should-not-apply")
    at.button(key="btn_save_llm").click()
    at.run()
    assert ctx.config["llm_model"] != "should-not-apply"
    assert "_llm_reseed" not in at.session_state.filtered_state


def test_embedding_configuration_error_is_surfaced():
    reason = "EMBEDDING_PROVIDER 指定的提供商需要非空的 EMBEDDING_API_KEY"
    ctx = FailingContext(
        _base_config(), ApplicationConfigurationError(reason)
    )
    at = _launch(ctx)
    at.button(key="btn_save_emb").click()
    at.run()
    assert any(reason in e.value for e in at.error)


def test_reset_failure_explains_and_keeps_current_config():
    ctx = FailingContext(
        _base_config(llm_model="keep-me"), ApplicationConfigurationError("配置无效")
    )
    at = _launch(ctx)
    at.button(key="btn_reset_llm").click()
    at.run()
    text = " ".join(e.value for e in at.error)
    assert "恢复默认失败" in text
    assert "配置无效" in text


# ==================== E：Embedding 维度回落值 ====================


def test_embedding_dimensions_come_from_the_catalog():
    ctx = FakeContext(_base_config())
    at = _launch(ctx)
    at.button(key="btn_save_emb").click()
    at.run()
    call = ctx.embedding_reload_calls[0]
    expected = EMBEDDING_PROVIDERS[call["embedding_provider"]]["dimensions"]
    assert call["faiss_embedding_dim"] == expected


def test_no_stale_1536_embedding_dim_fallback():
    """旧代码回落到 1536，与 FAISS_EMBEDDING_DIM 默认 1024 冲突。"""
    source = APP_PY.read_text(encoding="utf-8")
    assert '"dimensions", 1536' not in source


# ==================== 状态卡片 ====================


def test_status_card_reflects_active_config_not_env():
    """系统信息卡片必须显示实际生效值（此前读的是已被移除的覆盖键）。"""
    ctx = FakeContext(
        _base_config(llm_provider=KEY_PROVIDER, llm_model="status-card-model")
    )
    at = _launch(ctx)
    body = " ".join(m.value for m in at.markdown)
    assert "status-card-model" in body
