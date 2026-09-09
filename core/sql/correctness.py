"""Fail-closed orchestration for governed SQL correctness checks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from hmac import compare_digest
from threading import Lock
from time import monotonic
from typing import Any, Literal

from ..history.identity import GovernedHistoryIdentity
from ..semantic.context import SemanticContextBuilder, SemanticPromptBundle
from ..semantic.models import (
    CorrectnessResult,
    PreflightResult,
    SemanticSnapshot,
    ValidationIssue,
    ValidationReport,
)
from ..semantic.registry import SemanticRegistry
from ..semantic.validator import SemanticSQLValidator
from ..utils.log_sanitizer import log_safe_exception
from .preflight import MySQLExplainPreflight
from .validator import validate_query_scope, validate_sql

_CORRECTNESS_UNAVAILABLE = ValidationIssue(
    code="CORRECTNESS_UNAVAILABLE",
    message="SQL 正确性校验暂时不可用",
    stage="semantic",
)
_INVALID_REPAIR = ValidationIssue(
    code="REPAIR_FAILED",
    message="SQL 修复未返回有效候选",
    stage="semantic",
)
_REPAIR_FAILED = ValidationIssue(
    code="REPAIR_FAILED",
    message="SQL 修复失败",
    stage="semantic",
)
_SQL_LENGTH_EXCEEDED = ValidationIssue(
    code="SQL_LENGTH_EXCEEDED",
    message="SQL 长度超过安全限制",
    stage="safety",
)


class _CorrectnessRequest:
    """One-shot identity token; authoritative state stays in its pipeline."""

    __slots__ = ()


class PreparedRequestCapacityError(RuntimeError):
    """Raised when no bounded prepared-capability slot is available."""


@dataclass(frozen=True, slots=True)
class _PreparedState:
    question: str = field(repr=False)
    snapshot: SemanticSnapshot = field(repr=False)
    semantic_context: SemanticPromptBundle | None = field(repr=False)
    expires_at: float = field(repr=False)


class CorrectnessPipeline:
    """Apply one pinned semantic snapshot to validation, repair, and authorization."""

    def __init__(
        self,
        registry: SemanticRegistry,
        context_builder: SemanticContextBuilder,
        semantic_validator: SemanticSQLValidator,
        preflight: MySQLExplainPreflight | None = None,
        max_repair_attempts: Literal[0, 1] = 1,
        blocked_tables: tuple[str, ...] = (),
        max_sql_length: int = 2000,
        max_prepared_requests: int = 256,
        prepared_request_ttl_seconds: float = 120.0,
        clock: Callable[[], float] = monotonic,
    ):
        if type(max_repair_attempts) is not int or max_repair_attempts not in (0, 1):
            raise ValueError("max_repair_attempts 必须是 0 或 1")
        if type(max_sql_length) is not int or max_sql_length <= 0:
            raise ValueError("max_sql_length 必须是正整数")
        if (
            type(max_prepared_requests) is not int
            or not 1 <= max_prepared_requests <= 4096
        ):
            raise ValueError("max_prepared_requests 必须是 1 到 4096 的整数")
        if (
            isinstance(prepared_request_ttl_seconds, bool)
            or not isinstance(prepared_request_ttl_seconds, (int, float))
            or not 0 < prepared_request_ttl_seconds <= 3600
        ):
            raise ValueError(
                "prepared_request_ttl_seconds 必须大于 0 且不超过 3600"
            )
        if not callable(clock):
            raise TypeError("clock 必须可调用")
        self.registry = registry
        self.context_builder = context_builder
        self.semantic_validator = semantic_validator
        self.preflight = preflight
        self.max_repair_attempts = max_repair_attempts
        self.blocked_tables = blocked_tables
        self.max_sql_length = max_sql_length
        self.max_prepared_requests = max_prepared_requests
        self.prepared_request_ttl_seconds = float(prepared_request_ttl_seconds)
        self._clock = clock
        self._prepared_requests: OrderedDict[
            _CorrectnessRequest, _PreparedState
        ] = OrderedDict()
        self._prepared_lock = Lock()
        self._authorized_history_identity: ContextVar[
            tuple[int, GovernedHistoryIdentity] | None
        ] = ContextVar("authorized_history_identity", default=None)

    def prepare_generation(self, question: str) -> _CorrectnessRequest:
        """Pin and render the immutable snapshot used by one generation request."""
        snapshot = self.registry.get_snapshot()
        context = self.context_builder.build(question, snapshot)
        return self._issue_request(
            question,
            snapshot,
            semantic_context=SemanticPromptBundle(context),
        )

    def discard_prepared(self, request: Any) -> bool:
        """Idempotently release an abandoned prepared generation request."""
        if not isinstance(request, _CorrectnessRequest):
            return False
        with self._prepared_lock:
            self._prune_prepared_locked(self._clock())
            return self._prepared_requests.pop(request, None) is not None

    def describe_prepared(self, request: Any) -> tuple[str, str, str]:
        """Return non-authoritative prompt metadata for a live issued token."""
        if not isinstance(request, _CorrectnessRequest):
            raise ValueError("prepared correctness request is invalid or expired")
        with self._prepared_lock:
            self._prune_prepared_locked(self._clock())
            state = self._prepared_requests.get(request)
        if state is None:
            raise ValueError("prepared correctness request is invalid or expired")
        return (
            state.semantic_context or "",
            state.snapshot.model_version,
            state.snapshot.content_sha256,
        )

    def describe_prepared_history_identity(
        self, request: Any
    ) -> GovernedHistoryIdentity:
        """Return the persisted identity of the exact snapshot pinned by a live token."""
        if not isinstance(request, _CorrectnessRequest):
            raise ValueError("prepared correctness request is invalid or expired")
        with self._prepared_lock:
            self._prune_prepared_locked(self._clock())
            state = self._prepared_requests.get(request)
        if state is None:
            raise ValueError("prepared correctness request is invalid or expired")
        return GovernedHistoryIdentity.from_snapshot(state.snapshot)

    def process(
        self,
        question: str,
        candidate_sql: str,
        repair: Callable[[str, list[dict], str], dict] | None = None,
        allowed_tables: tuple[str, ...] = (),
        allowed_schemas: tuple[str, ...] = (),
    ) -> CorrectnessResult:
        request = self._pin_request(question)
        if isinstance(request, CorrectnessResult):
            return request.model_copy(update={"sql": candidate_sql})
        return self.process_prepared(
            request,
            candidate_sql,
            repair,
            allowed_tables,
            allowed_schemas,
        )

    def process_prepared(
        self,
        request: _CorrectnessRequest,
        candidate_sql: str,
        repair: Callable[[str, list[dict], str], dict] | None = None,
        allowed_tables: tuple[str, ...] = (),
        allowed_schemas: tuple[str, ...] = (),
    ) -> CorrectnessResult:
        """Validate and repair using only a request's pinned immutable snapshot."""
        state = self._consume_request(request)
        if state is None:
            return self._error_result(candidate_sql, None, 0)
        initial = self._initial_validation(
            state,
            candidate_sql,
            allowed_tables,
            allowed_schemas,
        )
        if isinstance(initial, CorrectnessResult):
            return initial
        snapshot, validation = initial
        if validation.valid:
            return self._result(candidate_sql, snapshot, validation, 0)
        if self._has_issue(validation, "SQL_LENGTH_EXCEEDED"):
            return self._result(candidate_sql, snapshot, validation, 0)
        if repair is None or self.max_repair_attempts == 0:
            return self._result(candidate_sql, snapshot, validation, 0)

        context = self._build_repair_context(state, candidate_sql, validation)
        if isinstance(context, CorrectnessResult):
            return context

        try:
            repair_result = repair(
                candidate_sql,
                self._public_issues(validation),
                context,
            )
        except Exception as exc:
            self._log_internal_failure("repair", exc)
            return self._repair_error(
                candidate_sql, snapshot, validation, _REPAIR_FAILED
            )
        return self._complete_repair(
            state.question,
            candidate_sql,
            repair_result,
            snapshot,
            validation,
            allowed_tables,
            allowed_schemas,
        )

    async def process_async(
        self,
        question: str,
        candidate_sql: str,
        repair: Callable[[str, list[dict], str], Awaitable[dict]] | None = None,
        allowed_tables: tuple[str, ...] = (),
        allowed_schemas: tuple[str, ...] = (),
    ) -> CorrectnessResult:
        request = self._pin_request(question)
        if isinstance(request, CorrectnessResult):
            return request.model_copy(update={"sql": candidate_sql})
        return await self.process_async_prepared(
            request,
            candidate_sql,
            repair,
            allowed_tables,
            allowed_schemas,
        )

    async def process_async_prepared(
        self,
        request: _CorrectnessRequest,
        candidate_sql: str,
        repair: Callable[[str, list[dict], str], Awaitable[dict]] | None = None,
        allowed_tables: tuple[str, ...] = (),
        allowed_schemas: tuple[str, ...] = (),
    ) -> CorrectnessResult:
        """Async counterpart of :meth:`process_prepared`."""
        state = self._consume_request(request)
        if state is None:
            return self._error_result(candidate_sql, None, 0)
        initial = self._initial_validation(
            state,
            candidate_sql,
            allowed_tables,
            allowed_schemas,
        )
        if isinstance(initial, CorrectnessResult):
            return initial
        snapshot, validation = initial
        if validation.valid:
            return self._result(candidate_sql, snapshot, validation, 0)
        if self._has_issue(validation, "SQL_LENGTH_EXCEEDED"):
            return self._result(candidate_sql, snapshot, validation, 0)
        if repair is None or self.max_repair_attempts == 0:
            return self._result(candidate_sql, snapshot, validation, 0)

        context = self._build_repair_context(state, candidate_sql, validation)
        if isinstance(context, CorrectnessResult):
            return context

        try:
            repair_result = await repair(
                candidate_sql,
                self._public_issues(validation),
                context,
            )
        except Exception as exc:
            self._log_internal_failure("repair", exc)
            return self._repair_error(
                candidate_sql, snapshot, validation, _REPAIR_FAILED
            )
        return self._complete_repair(
            state.question,
            candidate_sql,
            repair_result,
            snapshot,
            validation,
            allowed_tables,
            allowed_schemas,
        )

    def authorize_execution(
        self,
        question: str,
        sql: str,
        expected_semantic_version: str,
        expected_content_sha256: str | None = None,
        allowed_tables: tuple[str, ...] = (),
        allowed_schemas: tuple[str, ...] = (),
    ) -> CorrectnessResult:
        """Authorize execution against the current semantic snapshot."""
        result, identity = self._authorize_execution_with_history_identity(
            question=question,
            sql=sql,
            expected_semantic_version=expected_semantic_version,
            expected_content_sha256=expected_content_sha256,
            allowed_tables=allowed_tables,
            allowed_schemas=allowed_schemas,
        )
        self._authorized_history_identity.set(
            (id(result), identity)
            if result.executable and identity is not None
            else None
        )
        return result

    def take_authorization_history_identity(
        self, authorization: CorrectnessResult
    ) -> GovernedHistoryIdentity | None:
        """Consume the identity paired with this context's latest authorization."""
        stored = self._authorized_history_identity.get()
        if stored is None or stored[0] != id(authorization):
            return None
        self._authorized_history_identity.set(None)
        return stored[1]

    def _authorize_execution_with_history_identity(
        self,
        question: str,
        sql: str,
        expected_semantic_version: str,
        expected_content_sha256: str | None = None,
        allowed_tables: tuple[str, ...] = (),
        allowed_schemas: tuple[str, ...] = (),
    ) -> tuple[CorrectnessResult, GovernedHistoryIdentity | None]:
        """Authorize execution and return the identity of the exact snapshot used."""
        try:
            snapshot = self.registry.get_snapshot()
        except Exception as exc:
            self._log_internal_failure("snapshot", exc)
            return self._error_result(sql, None, 0), None

        identity = GovernedHistoryIdentity.from_snapshot(snapshot)

        if snapshot.model_version != expected_semantic_version:
            validation = ValidationReport(
                valid=False,
                issues=(
                    ValidationIssue(
                        code="SEMANTIC_VERSION_MISMATCH",
                        message="SQL 生成版本与当前语义版本不一致",
                        stage="semantic",
                    ),
                ),
            )
            return self._result(sql, snapshot, validation, 0), identity

        if (
            not isinstance(expected_content_sha256, str)
            or not compare_digest(snapshot.content_sha256, expected_content_sha256)
        ):
            validation = ValidationReport(
                valid=False,
                issues=(
                    ValidationIssue(
                        code="SEMANTIC_CONTENT_MISMATCH",
                        message="SQL 生成内容与当前语义内容不一致",
                        stage="semantic",
                    ),
                ),
            )
            return self._result(sql, snapshot, validation, 0), identity

        try:
            validation = self._validate_once(
                question,
                sql,
                snapshot,
                run_preflight=True,
                allowed_tables=allowed_tables,
                allowed_schemas=allowed_schemas,
            )
        except Exception as exc:
            self._log_internal_failure("authorization", exc)
            return self._error_result(
                sql, snapshot.model_version, 0, snapshot.content_sha256
            ), identity
        return self._result(sql, snapshot, validation, 0), identity

    def _pin_request(
        self, question: str
    ) -> _CorrectnessRequest | CorrectnessResult:
        try:
            snapshot = self.registry.get_snapshot()
        except Exception as exc:
            self._log_internal_failure("snapshot", exc)
            return self._error_result("", None, 0)
        try:
            return self._issue_request(
                question,
                snapshot,
                semantic_context=None,
            )
        except Exception as exc:
            self._log_internal_failure("prepared_request", exc)
            return self._error_result(
                "", snapshot.model_version, 0, snapshot.content_sha256
            )

    def _issue_request(
        self,
        question: str,
        snapshot: SemanticSnapshot,
        *,
        semantic_context: SemanticPromptBundle | None,
    ) -> _CorrectnessRequest:
        request = _CorrectnessRequest()
        state = _PreparedState(
            question=question,
            snapshot=snapshot,
            semantic_context=semantic_context,
            expires_at=self._clock() + self.prepared_request_ttl_seconds,
        )
        with self._prepared_lock:
            self._prune_prepared_locked(self._clock())
            if len(self._prepared_requests) >= self.max_prepared_requests:
                raise PreparedRequestCapacityError(
                    "prepared correctness request capacity exceeded"
                )
            self._prepared_requests[request] = state
        return request

    def _prune_prepared_locked(self, now: float) -> None:
        expired = [
            request
            for request, state in self._prepared_requests.items()
            if state.expires_at <= now
        ]
        for request in expired:
            self._prepared_requests.pop(request, None)

    def _consume_request(self, request: Any) -> _PreparedState | None:
        if not isinstance(request, _CorrectnessRequest):
            return None
        with self._prepared_lock:
            self._prune_prepared_locked(self._clock())
            return self._prepared_requests.pop(request, None)

    def _initial_validation(
        self,
        state: _PreparedState,
        candidate_sql: str,
        allowed_tables: tuple[str, ...],
        allowed_schemas: tuple[str, ...],
    ) -> tuple[SemanticSnapshot, ValidationReport] | CorrectnessResult:
        snapshot = state.snapshot
        try:
            validation = self._validate_once(
                state.question,
                candidate_sql,
                snapshot,
                run_preflight=True,
                allowed_tables=allowed_tables,
                allowed_schemas=allowed_schemas,
            )
        except Exception as exc:
            self._log_internal_failure("validation", exc)
            return self._error_result(
                candidate_sql, snapshot.model_version, 0, snapshot.content_sha256
            )
        return snapshot, validation

    def _build_repair_context(
        self,
        state: _PreparedState,
        sql: str,
        validation: ValidationReport,
    ) -> SemanticPromptBundle | CorrectnessResult:
        snapshot = state.snapshot
        if state.semantic_context:
            return state.semantic_context
        try:
            return SemanticPromptBundle(
                self.context_builder.build(state.question, snapshot)
            )
        except Exception as exc:
            self._log_internal_failure("repair_context", exc)
            return self._repair_error(sql, snapshot, validation, _REPAIR_FAILED, 0)

    def _complete_repair(
        self,
        question: str,
        original_sql: str,
        repair_result: Any,
        snapshot: SemanticSnapshot,
        original_validation: ValidationReport,
        allowed_tables: tuple[str, ...],
        allowed_schemas: tuple[str, ...],
    ) -> CorrectnessResult:
        repaired_sql = self._repaired_sql(repair_result)
        if repaired_sql is None:
            return self._repair_error(
                original_sql, snapshot, original_validation, _INVALID_REPAIR
            )
        try:
            validation = self._validate_once(
                question,
                repaired_sql,
                snapshot,
                run_preflight=True,
                allowed_tables=allowed_tables,
                allowed_schemas=allowed_schemas,
            )
        except Exception as exc:
            self._log_internal_failure("repair_validation", exc)
            return self._error_result(
                repaired_sql, snapshot.model_version, 1, snapshot.content_sha256
            )
        return self._result(repaired_sql, snapshot, validation, 1)

    def _validate_once(
        self,
        question: str,
        sql: str,
        snapshot: SemanticSnapshot,
        run_preflight: bool,
        allowed_tables: tuple[str, ...] = (),
        allowed_schemas: tuple[str, ...] = (),
    ) -> ValidationReport:
        if isinstance(sql, str) and len(sql) > self.max_sql_length:
            return ValidationReport(valid=False, issues=(_SQL_LENGTH_EXCEEDED,))
        if not isinstance(sql, str) or not sql.strip():
            return ValidationReport(
                valid=False,
                issues=(
                    ValidationIssue(
                        code="SYNTAX_ERROR",
                        message="SQL 语法校验失败",
                        stage="syntax",
                    ),
                ),
            )

        syntax = validate_sql(sql, dialect=snapshot.dialect)
        if not isinstance(syntax, Mapping):
            raise TypeError("legacy validator returned a non-mapping result")
        if syntax.get("valid") is not True:
            return self._from_legacy_validation(syntax)

        scope = validate_query_scope(
            sql,
            # Per the public plan contract, () means "not supplied": tables
            # therefore default to the pinned snapshot rather than deny-all.
            allowed_tables=list(allowed_tables)
            or [entity.name for entity in snapshot.entities],
            # Passing an explicit empty list means qualified schemas are denied;
            # callers must name every schema they intend to permit.
            allowed_schemas=list(allowed_schemas),
            blocked_tables=list(self.blocked_tables),
            dialect=snapshot.dialect,
        )
        if not isinstance(scope, Mapping):
            raise TypeError("scope validator returned a non-mapping result")
        if scope.get("valid") is not True:
            return self._from_scope_validation(scope)

        semantic = self.semantic_validator.validate(sql, question, snapshot)
        if not isinstance(semantic, ValidationReport):
            raise TypeError("semantic validator returned an invalid result")
        if not semantic.valid:
            return self._ensure_failed_report(
                semantic,
                code="SEMANTIC_VALIDATION_FAILED",
                message="SQL 未通过语义校验",
                stage="semantic",
            )

        if run_preflight and self.preflight is not None:
            preflight = self.preflight.check(sql)
            if not isinstance(preflight, PreflightResult):
                raise TypeError("preflight returned an invalid result")
            if not preflight.valid:
                return self._from_preflight(preflight)
            if preflight.issues:
                return ValidationReport(valid=False, issues=preflight.issues)
        return semantic

    @staticmethod
    def _from_legacy_validation(result: Mapping[str, Any]) -> ValidationReport:
        if result.get("syntax_valid") is False:
            issue = ValidationIssue(
                code="SYNTAX_ERROR",
                message="SQL 语法校验失败",
                stage="syntax",
            )
        elif result.get("execution_valid") is False:
            issue = ValidationIssue(
                code="UNSAFE_SQL",
                message="SQL 未通过只读安全校验",
                stage="safety",
            )
        else:
            issue = ValidationIssue(
                code="SQL_VALIDATION_FAILED",
                message="SQL 未通过基础校验",
                stage="safety",
            )
        return ValidationReport(valid=False, issues=(issue,))

    @staticmethod
    def _from_scope_validation(result: Mapping[str, Any]) -> ValidationReport:
        tables = result.get("tables")
        objects = (
            tuple(sorted({item for item in tables if isinstance(item, str)}))
            if isinstance(tables, (list, tuple, set, frozenset))
            else ()
        )
        return ValidationReport(
            valid=False,
            issues=(
                ValidationIssue(
                    code="QUERY_SCOPE_DENIED",
                    message="SQL 查询范围不被允许",
                    objects=objects,
                    stage="safety",
                ),
            ),
        )

    @staticmethod
    def _from_preflight(result: PreflightResult) -> ValidationReport:
        if result.issues:
            return ValidationReport(valid=False, issues=result.issues)
        return ValidationReport(
            valid=False,
            issues=(
                ValidationIssue(
                    code="EXPLAIN_FAILED",
                    message="数据库预检失败",
                    stage="preflight",
                ),
            ),
        )

    @staticmethod
    def _ensure_failed_report(
        report: ValidationReport,
        *,
        code: str,
        message: str,
        stage: Literal["syntax", "safety", "semantic", "preflight"],
    ) -> ValidationReport:
        if report.issues:
            return report
        return ValidationReport(
            valid=False,
            issues=(ValidationIssue(code=code, message=message, stage=stage),),
        )

    @staticmethod
    def _public_issues(validation: ValidationReport) -> list[dict]:
        return [issue.model_dump(mode="json") for issue in validation.issues]

    @staticmethod
    def _has_issue(validation: ValidationReport, code: str) -> bool:
        return any(issue.code == code for issue in validation.issues)

    @staticmethod
    def _repaired_sql(result: Any) -> str | None:
        if not isinstance(result, Mapping):
            return None
        try:
            status = result.get("status")
            sql = result.get("sql")
        except Exception:
            return None
        if type(status) is not str or status != "success":
            return None
        if type(sql) is not str or not sql.strip():
            return None
        return sql.strip()

    @staticmethod
    def _result(
        sql: str,
        snapshot: SemanticSnapshot,
        validation: ValidationReport,
        repair_attempts: Literal[0, 1],
    ) -> CorrectnessResult:
        executable = validation.valid
        return CorrectnessResult(
            status="success" if executable else "rejected",
            sql=sql,
            semantic_version=snapshot.model_version,
            content_sha256=snapshot.content_sha256,
            validation=validation,
            repair_attempts=repair_attempts,
            executable=executable,
        )

    @staticmethod
    def _error_result(
        sql: str,
        semantic_version: str | None,
        repair_attempts: Literal[0, 1],
        content_sha256: str | None = None,
    ) -> CorrectnessResult:
        return CorrectnessResult(
            status="error",
            sql=sql,
            semantic_version=semantic_version,
            content_sha256=content_sha256,
            validation=ValidationReport(
                valid=False,
                issues=(_CORRECTNESS_UNAVAILABLE,),
            ),
            repair_attempts=repair_attempts,
            executable=False,
        )

    @staticmethod
    def _repair_error(
        sql: str,
        snapshot: SemanticSnapshot,
        validation: ValidationReport,
        issue: ValidationIssue,
        repair_attempts: Literal[0, 1] = 1,
    ) -> CorrectnessResult:
        return CorrectnessResult(
            status="error",
            sql=sql,
            semantic_version=snapshot.model_version,
            content_sha256=snapshot.content_sha256,
            validation=ValidationReport(
                valid=False,
                issues=(*validation.issues, issue),
            ),
            repair_attempts=repair_attempts,
            executable=False,
        )

    @staticmethod
    def _log_internal_failure(stage: str, exc: Exception) -> None:
        log_safe_exception(stage, exc)


__all__ = ["CorrectnessPipeline"]
