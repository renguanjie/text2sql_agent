"""Fail-closed database query-plan preflight checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Event, Thread
from typing import Any, Generic, TypeVar, cast

from ..semantic.models import PreflightResult, ValidationIssue
from ..utils.log_sanitizer import log_safe_exception

_T = TypeVar("_T")
_DEADLINE_WORKERS = 4
_DEADLINE_SLOTS = BoundedSemaphore(_DEADLINE_WORKERS)


class ExplainDeadlineExceeded(TimeoutError):
    """Raised when EXPLAIN does not complete inside its wall-clock budget."""


@dataclass(slots=True)
class _DeadlineState(Generic[_T]):
    done: Event = field(default_factory=Event)
    cancelled: Event = field(default_factory=Event)
    result: _T | None = None
    error: BaseException | None = None


def run_with_wall_clock_deadline(
    operation: Callable[[Event], _T], timeout_ms: int
) -> _T:
    """Run blocking EXPLAIN work with a prompt and globally bounded deadline."""
    if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms <= 0:
        raise ValueError("timeout_ms 必须大于 0")
    if not _DEADLINE_SLOTS.acquire(blocking=False):
        raise ExplainDeadlineExceeded("EXPLAIN deadline worker capacity exhausted")

    state: _DeadlineState[_T] = _DeadlineState()

    def invoke() -> None:
        try:
            state.result = operation(state.cancelled)
        except BaseException as exc:
            state.error = exc
        finally:
            state.done.set()
            _DEADLINE_SLOTS.release()

    worker = Thread(
        target=invoke,
        name="mysql-explain-deadline",
        daemon=True,
    )
    try:
        worker.start()
    except BaseException:
        _DEADLINE_SLOTS.release()
        raise

    if not state.done.wait(timeout_ms / 1000):
        state.cancelled.set()
        raise ExplainDeadlineExceeded("EXPLAIN deadline exceeded")
    if state.error is not None:
        raise state.error
    return cast(_T, state.result)


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
        if (
            not isinstance(timeout_ms, int)
            or isinstance(timeout_ms, bool)
            or timeout_ms <= 0
        ):
            raise ValueError("timeout_ms 必须大于 0")
        if (
            not isinstance(max_estimated_rows, int)
            or isinstance(max_estimated_rows, bool)
            or max_estimated_rows < 0
        ):
            raise ValueError("max_estimated_rows 必须是非负整数")
        self.client = client
        self.timeout_ms = timeout_ms
        self.max_estimated_rows = max_estimated_rows

    def check(self, sql: str) -> PreflightResult:
        """Return a fail-closed result from a MySQL EXPLAIN plan."""
        try:
            if getattr(
                self.client, "explain_query_enforces_wall_clock_deadline", False
            ):
                plan = self.client.explain_query(sql, self.timeout_ms)
            else:
                plan = run_with_wall_clock_deadline(
                    lambda _cancelled: self.client.explain_query(
                        sql, self.timeout_ms
                    ),
                    self.timeout_ms,
                )
            if not plan:
                return failed_preflight("数据库未返回查询计划")

            estimated_rows = sum(max(0, int(row.get("rows") or 0)) for row in plan)
        except Exception as exc:
            log_safe_exception("mysql_explain", exc, sql=sql)
            return failed_preflight("数据库预检失败")

        if self.max_estimated_rows and estimated_rows > self.max_estimated_rows:
            return failed_preflight("预计扫描行数超过限制", estimated_rows)

        return PreflightResult(valid=True, estimated_rows=estimated_rows)


__all__ = [
    "ExplainDeadlineExceeded",
    "MySQLExplainPreflight",
    "failed_preflight",
    "run_with_wall_clock_deadline",
]
