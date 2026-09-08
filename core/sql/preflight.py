"""Fail-closed database query-plan preflight checks."""
from typing import Any

from loguru import logger

from ..semantic.models import PreflightResult, ValidationIssue
from ..utils.log_sanitizer import sanitize_sql


def failed_preflight(message: str, estimated_rows: int = 0) -> PreflightResult:
    """Create a stable preflight failure result for callers and users."""
    return PreflightResult(
        valid=False,
        estimated_rows=estimated_rows,
        issues=(
            ValidationIssue(
                code="EXPLAIN_FAILED",
                message=message,
                stage="preflight",
            ),
        ),
    )


class MySQLExplainPreflight:
    """Use MySQL EXPLAIN estimates to reject unsafe query plans before execution."""

    def __init__(
        self,
        client: Any,
        timeout_ms: int = 3000,
        max_estimated_rows: int = 0,
    ):
        self.client = client
        self.timeout_ms = timeout_ms
        self.max_estimated_rows = max_estimated_rows

    def check(self, sql: str) -> PreflightResult:
        """Return a fail-closed result from a MySQL EXPLAIN plan."""
        try:
            plan = self.client.explain_query(sql, self.timeout_ms)
        except Exception:
            logger.exception("MySQL EXPLAIN 失败：query={}", sanitize_sql(sql))
            return failed_preflight("数据库预检失败")

        if not plan:
            return failed_preflight("数据库未返回查询计划")

        estimated_rows = sum(max(0, int(row.get("rows") or 0)) for row in plan)
        if self.max_estimated_rows and estimated_rows > self.max_estimated_rows:
            return failed_preflight("预计扫描行数超过限制", estimated_rows)

        return PreflightResult(valid=True, estimated_rows=estimated_rows)
