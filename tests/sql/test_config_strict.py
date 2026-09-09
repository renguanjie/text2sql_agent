from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

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


def test_postgres_remains_valid_when_governance_and_explain_are_disabled():
    result = _run_config_script(
        "import config; print(config.SQL_DIALECT)",
        SQL_DIALECT="postgres",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "postgres"


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
