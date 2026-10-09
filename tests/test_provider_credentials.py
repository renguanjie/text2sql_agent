"""提供商凭据校验（D）的单元测试。

修复前 `core/config_validation.py` 对 `llm_provider` / `llm_api_key` /
`embedding_provider` 零校验：规则只存在于 `config.validate_config()`，
仅在模块加载时针对 `.env` 生效。`reload_llm` → `_validate_candidate_config`
→ `validate_runtime_config` 这条热重载路径走不到它，所以 UI 上选了 OpenAI
却不填 Key 时，只会在更深处失败并被收敛成一句泛化提示。

这里同时锁住"不能收紧过头"：`create_llm` 对未知 provider 是有意的容错回退，
`reload_llm` 的原子性测试也依赖该语义，因此未知 provider 必须继续放行。
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import config as config_module
from core import provider_catalog
from core.app_context import ApplicationContext, ApplicationConfigurationError
from core.config_validation import (
    RuntimeConfigurationError,
    validate_runtime_config,
)
from core.provider_catalog import (
    EMBEDDING_PROVIDERS,
    LLM_PROVIDERS,
    provider_requires_api_key,
)

SECRET = "sk-super-secret-value-0123456789"
KEY_PROVIDER = next(k for k, v in LLM_PROVIDERS.items() if v["needs_api_key"])
NOKEY_PROVIDER = next(k for k, v in LLM_PROVIDERS.items() if not v["needs_api_key"])
EMB_KEY_PROVIDER = next(
    k for k, v in EMBEDDING_PROVIDERS.items() if v["needs_api_key"]
)
UNKNOWN_PROVIDER = "totally-unknown-provider"


# ==================== provider_requires_api_key ====================


def test_known_provider_needing_key_returns_true():
    assert provider_requires_api_key(LLM_PROVIDERS, KEY_PROVIDER) is True


def test_known_provider_not_needing_key_returns_false():
    assert provider_requires_api_key(LLM_PROVIDERS, NOKEY_PROVIDER) is False


@pytest.mark.parametrize("provider", [UNKNOWN_PROVIDER, "old-provider", "new"])
def test_unknown_provider_is_tolerated(provider):
    """未知 provider 必须放行，否则会破坏 create_llm 的容错回退与既有测试。"""
    assert provider_requires_api_key(LLM_PROVIDERS, provider) is False


@pytest.mark.parametrize("provider", [None, 123, "", ["openai"], {"a": 1}])
def test_non_string_provider_is_tolerated(provider):
    assert provider_requires_api_key(LLM_PROVIDERS, provider) is False


def test_embedding_catalog_uses_the_same_helper():
    assert provider_requires_api_key(EMBEDDING_PROVIDERS, EMB_KEY_PROVIDER) is True
    assert provider_requires_api_key(EMBEDDING_PROVIDERS, UNKNOWN_PROVIDER) is False


# ==================== 目录只有一份定义 ====================


def test_config_reexports_the_same_catalog_objects():
    """config 必须是重新导出而非副本，否则两份目录会各自漂移。"""
    assert config_module.LLM_PROVIDERS is LLM_PROVIDERS
    assert config_module.EMBEDDING_PROVIDERS is EMBEDDING_PROVIDERS
    assert config_module.LLM_PROVIDERS is provider_catalog.LLM_PROVIDERS


# ==================== validate_runtime_config ====================


@pytest.mark.parametrize(
    "cfg",
    [
        {},
        {"llm_provider": NOKEY_PROVIDER},
        {"llm_provider": KEY_PROVIDER, "llm_api_key": SECRET},
        {"llm_provider": UNKNOWN_PROVIDER},
        {"llm_provider": UNKNOWN_PROVIDER, "llm_api_key": ""},
        {"llm_provider": "old", "llm_model": "old-model"},
        {"embedding_provider": EMB_KEY_PROVIDER, "embedding_api_key": SECRET},
        {"llm_model": "gpt-4o"},
    ],
)
def test_accepts_valid_or_tolerated_configs(cfg):
    validate_runtime_config(cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        {"llm_provider": KEY_PROVIDER},
        {"llm_provider": KEY_PROVIDER, "llm_api_key": ""},
        {"llm_provider": KEY_PROVIDER, "llm_api_key": "   "},
        {"llm_provider": KEY_PROVIDER, "llm_api_key": None},
        {"llm_provider": KEY_PROVIDER, "llm_api_key": 12345},
        {"embedding_provider": EMB_KEY_PROVIDER},
        {"embedding_provider": EMB_KEY_PROVIDER, "embedding_api_key": ""},
    ],
)
def test_rejects_key_provider_without_usable_key(cfg):
    with pytest.raises(RuntimeConfigurationError):
        validate_runtime_config(cfg)


def test_error_message_names_both_keys():
    with pytest.raises(RuntimeConfigurationError) as exc:
        validate_runtime_config({"llm_provider": KEY_PROVIDER, "llm_api_key": ""})
    message = str(exc.value)
    assert "LLM_PROVIDER" in message
    assert "LLM_API_KEY" in message


def test_error_message_never_echoes_supplied_values():
    """RuntimeConfigurationError 的约定是只含键名、不含取值。"""
    with pytest.raises(RuntimeConfigurationError) as exc:
        validate_runtime_config(
            {"llm_provider": KEY_PROVIDER, "llm_api_key": 987654321}
        )
    assert "987654321" not in str(exc.value)


def test_embedding_error_message_names_both_keys():
    with pytest.raises(RuntimeConfigurationError) as exc:
        validate_runtime_config(
            {"embedding_provider": EMB_KEY_PROVIDER, "embedding_api_key": ""}
        )
    message = str(exc.value)
    assert "EMBEDDING_PROVIDER" in message
    assert "EMBEDDING_API_KEY" in message


def test_provider_credential_error_survives_other_rule_checks():
    """凭据规则不能因为其他键合法而被跳过。"""
    with pytest.raises(RuntimeConfigurationError, match="LLM_API_KEY"):
        validate_runtime_config(
            {
                "llm_provider": KEY_PROVIDER,
                "llm_api_key": "",
                "sql_dialect": "mysql",
                "sql_query_max_rows": 500,
                "semantic_governance_enabled": False,
            }
        )


# ==================== 热重载路径 ====================


def _prepared_context(config: dict[str, Any]) -> ApplicationContext:
    context = ApplicationContext(config)
    context._llm = object()
    context._sql_chain = object()
    context._sql_generator = SimpleNamespace(correctness_pipeline=None)
    context._embedding_factory = SimpleNamespace(clear_cache=lambda: None)
    context._knowledge_retriever = object()
    return context


def test_reload_llm_rejects_key_provider_without_key():
    original = {"llm_provider": NOKEY_PROVIDER, "llm_model": "local-model"}
    context = _prepared_context(dict(original))
    previous_llm = context.llm

    with pytest.raises(ApplicationConfigurationError, match="LLM_API_KEY"):
        context.reload_llm(llm_provider=KEY_PROVIDER, llm_api_key="")

    # 失败必须是原子的：配置与组件都保持原样
    assert context.config["llm_provider"] == NOKEY_PROVIDER
    assert context.llm is previous_llm


def test_reload_llm_error_is_application_configuration_error():
    """UI 依赖这个类型来决定是否可以直接展示消息。"""
    context = _prepared_context({"llm_provider": NOKEY_PROVIDER})
    with pytest.raises(ApplicationConfigurationError) as exc:
        context.reload_llm(llm_provider=KEY_PROVIDER, llm_api_key="")
    assert isinstance(exc.value, ValueError)
    assert SECRET not in str(exc.value)


def test_reload_llm_still_tolerates_unknown_provider(monkeypatch):
    """收紧凭据规则不得顺带收紧未知 provider 的容错语义。"""
    import core.llm_factory

    context = _prepared_context(
        {
            "llm_provider": "old-provider",
            "llm_model": "old-model",
            "semantic_governance_enabled": False,
            "sql_explain_enabled": False,
            "sql_dialect": "mysql",
        }
    )
    new_llm = object()
    new_chain = object()
    monkeypatch.setattr(core.llm_factory, "create_llm", lambda **_kwargs: new_llm)
    chain_module = ModuleType("core.chain.sql_chain")
    chain_module.SQLGenerationChain = lambda **_kwargs: new_chain
    monkeypatch.setitem(sys.modules, "core.chain.sql_chain", chain_module)
    monkeypatch.setattr(
        context,
        "_build_sql_generator",
        lambda **components: SimpleNamespace(**components),
    )

    context.reload_llm(llm_provider=UNKNOWN_PROVIDER, llm_model="m")

    assert context.config["llm_provider"] == UNKNOWN_PROVIDER


def test_reload_llm_accepts_key_provider_with_key(monkeypatch):
    import core.llm_factory

    context = _prepared_context(
        {
            "llm_provider": NOKEY_PROVIDER,
            "llm_model": "local-model",
            "semantic_governance_enabled": False,
            "sql_explain_enabled": False,
            "sql_dialect": "mysql",
        }
    )
    new_llm = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        core.llm_factory,
        "create_llm",
        lambda **kwargs: captured.update(kwargs) or new_llm,
    )
    chain_module = ModuleType("core.chain.sql_chain")
    chain_module.SQLGenerationChain = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "core.chain.sql_chain", chain_module)
    monkeypatch.setattr(
        context,
        "_build_sql_generator",
        lambda **components: SimpleNamespace(**components),
    )

    context.reload_llm(
        llm_provider=KEY_PROVIDER, llm_model="gpt-4o", llm_api_key=SECRET
    )

    assert context.config["llm_provider"] == KEY_PROVIDER
    assert captured["api_key"] == SECRET


def test_reload_embedding_rejects_key_provider_without_key():
    context = _prepared_context({"embedding_provider": "ollama"})
    with pytest.raises(ApplicationConfigurationError, match="EMBEDDING_API_KEY"):
        context.reload_embedding(
            embedding_provider=EMB_KEY_PROVIDER, embedding_api_key=""
        )
