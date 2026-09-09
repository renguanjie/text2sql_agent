from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_config_script(script: str, **updates: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "LLM_PROVIDER": "ollama",
            "EMBEDDING_PROVIDER": "ollama",
            "SEMANTIC_GOVERNANCE_ENABLED": "false",
            "SQL_EXPLAIN_ENABLED": "false",
            **updates,
        }
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "key",
    [
        "NEO4J_ENABLED",
        "FAISS_ENABLED",
        "INDEX_CACHE_ENABLED",
        "FEW_SHOT_ENABLED",
        "SQL_VALIDATION_ENABLED",
        "SQL_SCOPE_ENFORCEMENT",
        "SEMANTIC_GOVERNANCE_ENABLED",
        "SQL_EXPLAIN_ENABLED",
        "STREAMLIT_HEAD",
        "HISTORY_AUTO_SAVE",
    ],
)
def test_every_boolean_environment_key_rejects_unknown_spelling(key: str):
    secret_value = "ture-do-not-disclose"
    result = _run_config_script("import config", **{key: secret_value})

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert key in output
    assert secret_value not in output


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("TRUE", True), (" false ", False), ("False", False)],
)
def test_strict_boolean_parser_accepts_only_documented_literals(raw: str, expected: bool):
    from core.config_validation import parse_env_bool

    assert parse_env_bool("FEATURE_FLAG", False, {"FEATURE_FLAG": raw}) is expected


@pytest.mark.parametrize("raw", ["1", "0", "yes", "on", "ture", ""])
def test_strict_boolean_parser_error_names_key_without_echoing_value(raw: str):
    from core.config_validation import parse_env_bool

    with pytest.raises(ValueError) as error:
        parse_env_bool("FEATURE_FLAG", False, {"FEATURE_FLAG": raw})

    assert "FEATURE_FLAG" in str(error.value)
    assert raw not in str(error.value) if raw else True


def test_config_environment_reload_revalidates_boolean_values():
    secret_value = "reload-typo-secret"
    script = f"""
import importlib
import os
import config
os.environ['HISTORY_AUTO_SAVE'] = {secret_value!r}
try:
    importlib.reload(config)
except ValueError as exc:
    print(exc)
    raise SystemExit(17)
raise SystemExit(0)
"""
    result = _run_config_script(script)

    output = result.stdout + result.stderr
    assert result.returncode == 17
    assert "HISTORY_AUTO_SAVE" in output
    assert secret_value not in output


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("SQL_MAX_LENGTH", "100001"),
        ("SQL_QUERY_MAX_ROWS", "0"),
        ("SQL_QUERY_TIMEOUT_MS", "0"),
        ("PROMPT_MAX_LENGTH", "0"),
        ("PROMPT_MAX_LENGTH", "999"),
        ("PROMPT_MAX_LENGTH", "100001"),
        ("SEMANTIC_CONTEXT_MAX_CHARS", "0"),
        ("SEMANTIC_CONTEXT_MAX_ITEMS", "0"),
        ("SQL_REPAIR_MAX_ATTEMPTS", "2"),
        ("SQL_EXPLAIN_TIMEOUT_MS", "0"),
        ("SQL_EXPLAIN_MAX_ESTIMATED_ROWS", "-1"),
    ],
)
def test_runtime_limit_environment_rejects_invalid_documented_values(
    key: str, raw: str
):
    """Startup uses the same bounded numeric contract as runtime reloads."""
    result = _run_config_script("import config", **{key: raw})

    assert result.returncode != 0
    assert key in result.stdout + result.stderr


def test_integer_environment_parser_does_not_echo_invalid_value():
    secret_value = "not-an-integer-secret"
    result = _run_config_script("import config", SQL_QUERY_TIMEOUT_MS=secret_value)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "SQL_QUERY_TIMEOUT_MS" in output
    assert secret_value not in output


@pytest.mark.parametrize(
    ("config", "key"),
    [
        ({"sql_explain_timeout_ms": 0}, "SQL_EXPLAIN_TIMEOUT_MS"),
        ({"sql_explain_max_estimated_rows": -1}, "SQL_EXPLAIN_MAX_ESTIMATED_ROWS"),
        ({"sql_max_length": 100_001}, "SQL_MAX_LENGTH"),
        ({"sql_query_max_rows": 0}, "SQL_QUERY_MAX_ROWS"),
        ({"sql_query_timeout_ms": 0}, "SQL_QUERY_TIMEOUT_MS"),
        ({"prompt_max_length": 999}, "PROMPT_MAX_LENGTH"),
        ({"prompt_max_length": 100_001}, "PROMPT_MAX_LENGTH"),
        ({"semantic_context_max_chars": 0}, "SEMANTIC_CONTEXT_MAX_CHARS"),
        ({"semantic_context_max_items": 0}, "SEMANTIC_CONTEXT_MAX_ITEMS"),
        ({"sql_repair_max_attempts": 2}, "SQL_REPAIR_MAX_ATTEMPTS"),
        ({"max_prepared_requests": 4097}, "MAX_PREPARED_REQUESTS"),
        ({"prepared_request_ttl_seconds": 0}, "PREPARED_REQUEST_TTL_SECONDS"),
        (
            {"governed_history_binding_ttl_seconds": 3601},
            "GOVERNED_HISTORY_BINDING_TTL_SECONDS",
        ),
    ],
)
def test_runtime_numeric_validation_rejects_unsafe_limits(config: dict, key: str):
    """Reload validation must enforce every bounded runtime safety limit."""
    from core.config_validation import (
        RuntimeConfigurationError,
        validate_runtime_config,
    )

    with pytest.raises(RuntimeConfigurationError, match=key):
        validate_runtime_config(config)


@pytest.mark.parametrize("value", [True, "2000"])
def test_runtime_integer_validation_rejects_bool_and_string_values(value):
    """Only the environment parser may turn strings into runtime integers."""
    from core.config_validation import (
        RuntimeConfigurationError,
        validate_runtime_config,
    )

    with pytest.raises(RuntimeConfigurationError, match="SQL_MAX_LENGTH"):
        validate_runtime_config({"sql_max_length": value})


def test_prompt_length_minimum_is_shared_by_startup_and_reload_validation():
    """A 999-character prompt budget must not bypass startup or reload validation."""
    from core.app_context import ApplicationConfigurationError, ApplicationContext
    from core.config_validation import validate_runtime_config

    accepted = _run_config_script(
        "import config; print(config.PROMPT_MAX_LENGTH)",
        PROMPT_MAX_LENGTH="1000",
    )
    rejected = _run_config_script("import config", PROMPT_MAX_LENGTH="999")

    assert accepted.returncode == 0
    assert accepted.stdout.strip() == "1000"
    assert rejected.returncode != 0
    assert "PROMPT_MAX_LENGTH" in rejected.stdout + rejected.stderr
    validate_runtime_config({"prompt_max_length": 1000})
    with pytest.raises(ApplicationConfigurationError, match="PROMPT_MAX_LENGTH"):
        ApplicationContext._validate_candidate_config({"prompt_max_length": 999})


def test_postgres_remains_valid_when_governance_and_explain_are_disabled():
    result = _run_config_script(
        "import config; print(config.SQL_DIALECT)",
        SQL_DIALECT="postgres",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "postgres"


def test_governance_dialect_literal_does_not_accept_hidden_whitespace():
    result = _run_config_script(
        "import config",
        SQL_DIALECT=" mysql ",
        SEMANTIC_GOVERNANCE_ENABLED="true",
    )

    assert result.returncode != 0
    assert "SQL_DIALECT" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "flag",
    ["SEMANTIC_GOVERNANCE_ENABLED", "SQL_EXPLAIN_ENABLED"],
)
def test_non_mysql_dialect_rejects_governed_runtime(flag: str):
    updates = {
        "SQL_DIALECT": "postgres-secret-value",
        "SEMANTIC_GOVERNANCE_ENABLED": "true",
    }
    if flag == "SQL_EXPLAIN_ENABLED":
        updates["SQL_EXPLAIN_ENABLED"] = "true"
    result = _run_config_script("import config", **updates)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "SQL_DIALECT" in output
    assert flag in output
    assert "postgres-secret-value" not in output


def test_governance_reload_uses_strict_validation_and_keeps_live_state():
    from core.app_context import ApplicationConfigurationError, ApplicationContext

    context = ApplicationContext(
        {
            "semantic_governance_enabled": False,
            "sql_explain_enabled": False,
            "sql_dialect": "mysql",
        }
    )
    old_registry = object()
    old_pipeline = object()
    old_generator = object()
    context._semantic_registry = old_registry
    context._correctness_pipeline = old_pipeline
    context._sql_generator = old_generator

    with pytest.raises(ApplicationConfigurationError) as error:
        context.reload_governance(
            semantic_governance_enabled="ture-secret",
        )

    message = str(error.value)
    assert "SEMANTIC_GOVERNANCE_ENABLED" in message
    assert "ture-secret" not in message
    assert context.semantic_registry is old_registry
    assert context.correctness_pipeline is old_pipeline
    assert context.sql_generator is old_generator
    assert context.config["semantic_governance_enabled"] is False


def test_runtime_reload_rejects_non_mysql_governance_with_safe_error():
    from core.app_context import ApplicationConfigurationError, ApplicationContext

    context = ApplicationContext(
        {
            "semantic_governance_enabled": False,
            "sql_explain_enabled": False,
            "sql_dialect": "postgres-secret-value",
        }
    )

    with pytest.raises(ApplicationConfigurationError) as error:
        context.reload_governance(semantic_governance_enabled=True)

    message = str(error.value)
    assert "SQL_DIALECT" in message
    assert "SEMANTIC_GOVERNANCE_ENABLED" in message
    assert "postgres-secret-value" not in message
    assert context.config["semantic_governance_enabled"] is False


@pytest.mark.parametrize(
    ("updates", "hint"),
    [
        ({"llm_model": "candidate-model"}, "reload_llm"),
        ({"embedding_model": "candidate-model"}, "reload_embedding"),
        ({"retrieval_top_k": 8}, "完整重启"),
    ],
)
def test_governance_reload_rejects_cross_domain_updates_before_candidate_build(
    monkeypatch: pytest.MonkeyPatch, updates: dict, hint: str
):
    """Cross-domain keys must never publish a graph built around stale components."""
    import core.retrieval.bm25_tfidf
    from core.app_context import ApplicationConfigurationError, ApplicationContext

    context = ApplicationContext(
        {
            "semantic_governance_enabled": False,
            "sql_explain_enabled": False,
            "sql_dialect": "mysql",
        }
    )
    old_generator = object()
    context._llm = object()
    context._embedding_factory = object()
    context._sql_chain = object()
    context._sql_generator = old_generator
    context._build_governance_components = lambda _config: (None, None)
    monkeypatch.setattr(
        core.retrieval.bm25_tfidf,
        "KnowledgeRetriever",
        lambda **_kwargs: object(),
    )

    with pytest.raises(ApplicationConfigurationError, match=hint):
        context.reload_governance(**updates)

    assert context.sql_generator is old_generator
    assert context.config == {
        "semantic_governance_enabled": False,
        "sql_explain_enabled": False,
        "sql_dialect": "mysql",
    }


def test_governance_dialect_reload_rebuilds_the_sql_chain(monkeypatch):
    """A newly governed dialect cannot reuse a chain built for another dialect."""
    import core.chain.sql_chain
    import core.retrieval.bm25_tfidf
    from core.app_context import ApplicationContext

    old_chain = object()
    new_chain = object()
    candidate_registry = object()
    candidate_pipeline = type("Pipeline", (), {"registry": candidate_registry})()
    candidate_retriever = object()
    context = ApplicationContext(
        {
            "semantic_governance_enabled": False,
            "sql_explain_enabled": False,
            "sql_dialect": "postgres",
        }
    )
    context._llm = object()
    context._embedding_factory = object()
    context._sql_chain = old_chain
    context._build_governance_components = lambda _config: (
        candidate_registry,
        candidate_pipeline,
    )
    monkeypatch.setattr(
        core.chain.sql_chain,
        "SQLGenerationChain",
        lambda **_kwargs: new_chain,
    )
    monkeypatch.setattr(
        core.retrieval.bm25_tfidf,
        "KnowledgeRetriever",
        lambda **_kwargs: candidate_retriever,
    )
    monkeypatch.setattr(context, "_populate_retriever", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        context,
        "_build_sql_generator",
        lambda **components: type("Generator", (), components)(),
    )

    context.reload_governance(
        semantic_governance_enabled=True,
        sql_dialect="mysql",
    )

    assert context.sql_chain is new_chain
    assert context.sql_generator.sql_chain is new_chain


@pytest.mark.parametrize("raw", ["0", "-1", "10001"])
def test_history_capacity_environment_cannot_disable_or_exceed_pruning(raw: str):
    result = _run_config_script("import config", HISTORY_MAX_RECORDS=raw)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "HISTORY_MAX_RECORDS" in output


@pytest.mark.parametrize("limit", [0, -1, 10_001, True])
def test_mysql_client_rejects_unbounded_history_configuration(limit):
    from core.history.mysql_client import MySQLClient

    with pytest.raises(ValueError, match="history_max_records"):
        MySQLClient(
            host="localhost",
            port=3306,
            user="user",
            password="password",
            database="database",
            history_max_records=limit,
        )
