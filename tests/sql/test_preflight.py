from contextlib import contextmanager
import logging
import re
from unittest.mock import Mock

import pytest
from loguru import logger


@contextmanager
def capture_loguru(caplog):
    caplog.set_level(logging.DEBUG)
    sink_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    try:
        yield
    finally:
        logger.remove(sink_id)


def test_mysql_explain_emits_hinted_select_without_session_mutation_or_limit():
    """Catches a preflight adapter that changes pooled state or inspected query scope."""
    from core.history.mysql_client import MySQLClient

    client = MySQLClient("localhost", 3306, "user", "password", "database")
    executed_sql = []

    class ResultRow:
        _mapping = {"rows": 7}

    class Connection:
        def execute(self, statement):
            executed_sql.append(str(statement))
            return [ResultRow()]

    @contextmanager
    def connection_context():
        yield Connection()

    client.get_connection = connection_context

    assert client.explain_query("SELECT id FROM users", timeout_ms=3000) == [{"rows": 7}]
    assert executed_sql == ["EXPLAIN SELECT /*+ MAX_EXECUTION_TIME(3000) */ id FROM users"]
    assert "SET SESSION" not in executed_sql[0]
    assert "LIMIT" not in executed_sql[0]


def test_mysql_explain_rejects_non_positive_timeout_before_database_execution():
    """Catches a preflight adapter that sends unbounded EXPLAIN work to MySQL."""
    from core.history.mysql_client import MySQLClient

    client = MySQLClient("localhost", 3306, "user", "password", "database")

    with pytest.raises(ValueError, match="timeout_ms 必须大于 0"):
        client.explain_query("SELECT 1", timeout_ms=0)


def test_mysql_explain_rejects_non_select_before_database_execution():
    """Catches a preflight adapter that lets writes reach EXPLAIN."""
    from core.history.mysql_client import MySQLClient

    client = MySQLClient("localhost", 3306, "user", "password", "database")

    with pytest.raises(ValueError, match="仅支持 SELECT 预检"):
        client.explain_query("UPDATE users SET active = 1", timeout_ms=3000)


def test_preflight_rejects_empty_plan():
    """Catches treating a missing MySQL query plan as a safe plan."""
    from core.sql.preflight import MySQLExplainPreflight

    client = Mock()
    client.explain_query.return_value = []

    result = MySQLExplainPreflight(client, timeout_ms=3000).check("SELECT 1")

    assert not result.valid
    assert result.issues[0].code == "EXPLAIN_FAILED"


def test_preflight_rejects_excessive_estimated_rows():
    """Catches accepting plans whose estimated scan exceeds the configured limit."""
    from core.sql.preflight import MySQLExplainPreflight

    client = Mock()
    client.explain_query.return_value = [{"rows": 900}, {"rows": 200}]

    result = MySQLExplainPreflight(
        client, timeout_ms=3000, max_estimated_rows=1000
    ).check("SELECT 1")

    assert result.estimated_rows == 1100
    assert not result.valid


def test_preflight_masks_database_exception():
    """Catches leaking database exception details through the validation response."""
    from core.sql.preflight import MySQLExplainPreflight

    client = Mock()
    client.explain_query.side_effect = RuntimeError("password=secret table=private")

    result = MySQLExplainPreflight(client, timeout_ms=3000).check("SELECT 1")

    assert "secret" not in result.issues[0].message
    assert result.issues[0].message == "数据库预检失败"


def test_preflight_sums_only_nonnegative_estimated_rows_when_limit_disabled():
    """Catches negative or zero-limit plan estimates being treated as a rejection."""
    from core.sql.preflight import MySQLExplainPreflight

    client = Mock()
    client.explain_query.return_value = [{"rows": -5}, {"rows": None}, {"rows": "12"}]

    result = MySQLExplainPreflight(client, timeout_ms=3000).check("SELECT 1")

    assert result.valid
    assert result.estimated_rows == 12


@pytest.mark.parametrize(
    "plan",
    ([{"rows": "not-a-number"}], [None]),
    ids=["non_numeric_rows", "non_mapping_row"],
)
def test_preflight_masks_malformed_plan_rows(plan):
    """Catches malformed MySQL plan rows escaping the fail-closed result contract."""
    from core.sql.preflight import MySQLExplainPreflight

    client = Mock()
    client.explain_query.return_value = plan

    result = MySQLExplainPreflight(client, timeout_ms=3000).check("SELECT 1")

    assert not result.valid
    assert result.issues[0].code == "EXPLAIN_FAILED"
    assert result.issues[0].message == "数据库预检失败"


def test_preflight_exception_log_contains_metadata_without_sensitive_text(caplog):
    """Catches database exception text or traceback leaking from preflight logs."""
    from core.sql.preflight import MySQLExplainPreflight

    client = Mock()
    client.explain_query.side_effect = RuntimeError(
        "password=secret SQL SELECT hidden database=private"
    )

    with capture_loguru(caplog):
        result = MySQLExplainPreflight(client, timeout_ms=3000).check(
            "SELECT visible FROM orders"
        )

    assert not result.valid
    assert "secret" not in caplog.text
    assert "hidden" not in caplog.text
    assert "private" not in caplog.text
    assert "Traceback" not in caplog.text
    assert "stage=mysql_explain" in caplog.text
    assert "exception_type=RuntimeError" in caplog.text
    assert re.search(r"error_id=[0-9a-f]{32}", caplog.text)
