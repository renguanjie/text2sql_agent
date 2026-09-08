"""Privacy-safe, offline structural evaluation for Chinese Text2SQL results."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from core.semantic import FrozenModel, SemanticSnapshot
from core.semantic.validator import (
    SQLStructure,
    SemanticSQLValidator,
    analyze_sql_structure,
    normalize_filter_expression,
)


_SEMVER = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")
_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,127}")
_KNOWN_RESULT_ERROR_CODES = frozenset(
    {
        "AMBIGUOUS_COLUMN",
        "CORRECTNESS_UNAVAILABLE",
        "DIMENSION_NOT_ALLOWED",
        "EXPLAIN_FAILED",
        "INVALID_JOIN",
        "METRIC_MISMATCH",
        "POLICY_DENIED",
        "QUERY_SCOPE_DENIED",
        "REPAIR_FAILED",
        "SEMANTIC_VALIDATION_FAILED",
        "SEMANTIC_VERSION_MISMATCH",
        "SQL_VALIDATION_FAILED",
        "SYNTAX_ERROR",
        "TIME_DIMENSION_REQUIRED",
        "UNKNOWN_COLUMN",
        "UNKNOWN_TABLE",
        "UNSAFE_SQL",
        "UNSUPPORTED_SQL",
    }
)
_MODEL_PARAMETER_ALLOWLIST = frozenset(
    {
        "context_length",
        "max_tokens",
        "model_revision",
        "quantization",
        "repeat_penalty",
        "seed",
        "temperature",
        "top_k",
        "top_p",
    }
)
_Scalar = StrictBool | StrictInt | StrictFloat | StrictStr


class GoldenConfigurationError(ValueError):
    """Stable, content-free error raised for invalid evaluation inputs."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"Golden Set 配置无效（{code}）")


class GoldenCase(FrozenModel):
    id: StrictStr
    question: StrictStr
    datasource: StrictStr
    dialect: Literal["mysql"]
    expected_tables: tuple[StrictStr, ...] = ()
    forbidden_tables: tuple[StrictStr, ...] = ()
    expected_fields: tuple[StrictStr, ...] = ()
    forbidden_fields: tuple[StrictStr, ...] = ()
    expected_metrics: tuple[StrictStr, ...] = ()
    forbidden_metrics: tuple[StrictStr, ...] = ()
    expected_relationships: tuple[StrictStr, ...] = ()
    expected_filters: tuple[StrictStr, ...] = ()
    expected_error_codes: tuple[StrictStr, ...] = ()
    expected_time_grain: Literal["day", "week", "month", "quarter", "year"] | None = None
    should_execute: StrictBool

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _CASE_ID.fullmatch(value):
            raise ValueError("invalid case id")
        return value

    @field_validator("question", "datasource")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blank required text")
        return value

    @field_validator(
        "expected_tables",
        "forbidden_tables",
        "expected_fields",
        "forbidden_fields",
        "expected_metrics",
        "forbidden_metrics",
        "expected_relationships",
        "expected_filters",
        "expected_error_codes",
    )
    @classmethod
    def validate_unique_nonempty_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("expectations must be unique and non-empty")
        return values

    @field_validator("expected_error_codes")
    @classmethod
    def validate_error_codes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            not _ERROR_CODE.fullmatch(value) or value not in _KNOWN_RESULT_ERROR_CODES
            for value in values
        ):
            raise ValueError("invalid error code")
        return values


class GoldenSuite(FrozenModel):
    schema_version: Literal[1]
    datasource: StrictStr
    dialect: Literal["mysql"]
    semantic_version: StrictStr
    cases: tuple[GoldenCase, ...]

    @field_validator("semantic_version")
    @classmethod
    def validate_semantic_version(cls, value: str) -> str:
        if not _SEMVER.fullmatch(value):
            raise ValueError("invalid semantic version")
        return value

    @model_validator(mode="after")
    def validate_cases(self) -> "GoldenSuite":
        ids = [case.id for case in self.cases]
        if not self.cases or len(ids) != len(set(ids)):
            raise ValueError("cases must be non-empty and unique")
        if any(
            case.datasource != self.datasource or case.dialect != self.dialect
            for case in self.cases
        ):
            raise ValueError("case metadata mismatch")
        return self


class GeneratedIssue(FrozenModel):
    code: StrictStr
    message: StrictStr | None = None
    objects: tuple[StrictStr, ...] = ()
    stage: Literal["syntax", "safety", "semantic", "preflight"] | None = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if not _ERROR_CODE.fullmatch(value) or value not in _KNOWN_RESULT_ERROR_CODES:
            raise ValueError("invalid error code")
        return value


class GeneratedValidation(FrozenModel):
    valid: StrictBool | None = None
    issues: tuple[GeneratedIssue, ...] = ()


class GoldenGeneratedResult(FrozenModel):
    id: StrictStr
    sql: StrictStr | None
    status: Literal["success", "rejected", "error"]
    executable: StrictBool
    validation: GeneratedValidation

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _CASE_ID.fullmatch(value):
            raise ValueError("invalid result id")
        return value

    @model_validator(mode="after")
    def validate_consistent_outcome(self) -> "GoldenGeneratedResult":
        if self.sql is not None and not self.sql.strip():
            raise ValueError("blank sql")
        if self.status == "success":
            if not self.executable or self.sql is None or self.validation.valid is False:
                raise ValueError("inconsistent successful result")
        elif self.executable or self.validation.valid is True:
            raise ValueError("inconsistent non-successful result")
        return self


class GoldenResults(FrozenModel):
    schema_version: Literal[1]
    datasource: StrictStr
    dialect: Literal["mysql"]
    semantic_version: StrictStr
    results: tuple[GoldenGeneratedResult, ...]

    @field_validator("semantic_version")
    @classmethod
    def validate_semantic_version(cls, value: str) -> str:
        if not _SEMVER.fullmatch(value):
            raise ValueError("invalid semantic version")
        return value

    @model_validator(mode="after")
    def validate_result_ids(self) -> "GoldenResults":
        ids = [result.id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("result ids must be unique")
        return self


class ModelParameter(FrozenModel):
    name: StrictStr
    value: _Scalar

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value not in _MODEL_PARAMETER_ALLOWLIST:
            raise ValueError("model parameter is not report-safe")
        return value

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: _Scalar) -> _Scalar:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("model parameter value is not finite")
        if isinstance(value, str) and (not value or len(value) > 128 or "\n" in value):
            raise ValueError("model parameter value is not report-safe")
        return value


class RuleRate(FrozenModel):
    assertion: StrictStr
    total: int
    passed: int
    rate: float


class CaseResult(FrozenModel):
    id: StrictStr
    passed: bool
    failed_assertions: tuple[StrictStr, ...]
    error_codes: tuple[StrictStr, ...]
    question_fingerprint: StrictStr
    sql_fingerprint: StrictStr | None
    elapsed_ms: int
    question: StrictStr | None = None
    sql: StrictStr | None = None


class EvaluationReport(FrozenModel):
    schema_version: Literal[1] = 1
    model: StrictStr
    model_parameters: tuple[ModelParameter, ...] = ()
    datasource: StrictStr
    dialect: Literal["mysql"]
    semantic_version: StrictStr
    git_commit: StrictStr
    total: int
    passed: int
    failed: int
    pass_rate: float
    per_rule: tuple[RuleRate, ...]
    cases: tuple[CaseResult, ...]
    elapsed_ms: int
    content_included: bool = False

    @field_serializer("model_parameters")
    def serialize_model_parameters(
        self, parameters: tuple[ModelParameter, ...]
    ) -> dict[str, _Scalar]:
        return {parameter.name: parameter.value for parameter in parameters}

    def deterministic_dump(self) -> dict:
        """Return comparable report content without intentionally variable timings."""
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("elapsed_ms", None)
        for case in payload["cases"]:
            case.pop("elapsed_ms", None)
        return payload


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _fingerprint(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _read_payload(path: Path, error_code: str) -> object:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.casefold() == ".json":
            return json.loads(text)
        if path.suffix.casefold() in {".yaml", ".yml"}:
            return yaml.safe_load(text)
    except Exception as exc:
        raise GoldenConfigurationError(error_code) from exc
    raise GoldenConfigurationError("UNSUPPORTED_INPUT_FORMAT")


def _find_duplicate_id(payload: object, collection: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    items = payload.get(collection)
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return False
    ids = [item.get("id") for item in items if isinstance(item, Mapping)]
    if any(not isinstance(item_id, str) for item_id in ids):
        return False
    return len(ids) != len(set(ids))


def load_golden_suite(path: Path) -> GoldenSuite:
    payload = _read_payload(path, "INVALID_CASE_FILE")
    if _find_duplicate_id(payload, "cases"):
        raise GoldenConfigurationError("DUPLICATE_CASE_ID")
    try:
        return GoldenSuite.model_validate(payload)
    except Exception as exc:
        raise GoldenConfigurationError("INVALID_CASE_FILE") from exc


def load_golden_results(path: Path) -> GoldenResults:
    payload = _read_payload(path, "INVALID_RESULTS_FILE")
    if _find_duplicate_id(payload, "results"):
        raise GoldenConfigurationError("DUPLICATE_RESULT_ID")
    try:
        return GoldenResults.model_validate(payload)
    except Exception as exc:
        raise GoldenConfigurationError("INVALID_RESULTS_FILE") from exc


def parse_model_parameters(values: Sequence[str]) -> tuple[ModelParameter, ...]:
    parsed: dict[str, _Scalar] = {}
    for raw in values:
        if "=" not in raw:
            raise GoldenConfigurationError("INVALID_MODEL_PARAMETER")
        name, raw_value = raw.split("=", 1)
        name = name.strip()
        if name not in _MODEL_PARAMETER_ALLOWLIST:
            raise GoldenConfigurationError("MODEL_PARAMETER_DENIED")
        if name in parsed:
            raise GoldenConfigurationError("DUPLICATE_MODEL_PARAMETER")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        if value is None or isinstance(value, (list, dict)):
            raise GoldenConfigurationError("INVALID_MODEL_PARAMETER")
        if isinstance(value, float) and not math.isfinite(value):
            raise GoldenConfigurationError("INVALID_MODEL_PARAMETER")
        if isinstance(value, str) and (not value or len(value) > 128 or "\n" in value):
            raise GoldenConfigurationError("INVALID_MODEL_PARAMETER")
        parsed[name] = value
    return tuple(
        ModelParameter(name=name, value=value) for name, value in sorted(parsed.items())
    )


def get_git_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = completed.stdout.strip().casefold()
        return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else "unknown"
    except Exception:
        return "unknown"


class GoldenEvaluator:
    def __init__(
        self,
        snapshot: SemanticSnapshot,
        *,
        model: str = "unspecified",
        model_parameters: tuple[ModelParameter, ...] = (),
        git_commit: str = "unknown",
        include_content: bool = False,
    ):
        if not model.strip():
            raise GoldenConfigurationError("INVALID_MODEL_ID")
        self.snapshot = snapshot
        self.model = model
        self.model_parameters = tuple(sorted(model_parameters, key=lambda item: item.name))
        self.git_commit = git_commit
        self.include_content = include_content
        self._field_indexes = {
            _normalize(entity.name): tuple(_normalize(field.name) for field in entity.fields)
            for entity in snapshot.entities
        }
        self._metric_names = frozenset(_normalize(metric.name) for metric in snapshot.metrics)
        self._semantic_validator = SemanticSQLValidator()

    def validate_inputs(self, suite: GoldenSuite, results: GoldenResults) -> None:
        expected_identity = (
            self.snapshot.datasource,
            self.snapshot.dialect,
            self.snapshot.model_version,
        )
        if (
            (suite.datasource, suite.dialect, suite.semantic_version) != expected_identity
            or (results.datasource, results.dialect, results.semantic_version)
            != expected_identity
        ):
            raise GoldenConfigurationError("SNAPSHOT_MISMATCH")
        for case in suite.cases:
            self._validate_case_identity(case)
        case_ids = {case.id for case in suite.cases}
        result_ids = {result.id for result in results.results}
        if case_ids != result_ids:
            raise GoldenConfigurationError("RESULT_ID_MISMATCH")

    def evaluate_case(self, case: GoldenCase, result: Mapping) -> CaseResult:
        scored, _ = self._evaluate_case(case, result)
        return scored

    def _evaluate_case(
        self, case: GoldenCase, result: Mapping
    ) -> tuple[CaseResult, frozenset[str]]:
        started = time.perf_counter()
        self._validate_case_identity(case)
        generated = self._parse_generated_result(case, result)
        attempted: set[str] = {"SHOULD_EXECUTE"}
        failed: set[str] = set()
        if generated.executable != case.should_execute:
            failed.add("SHOULD_EXECUTE")

        error_codes = tuple(sorted({issue.code for issue in generated.validation.issues}))
        if case.expected_error_codes:
            attempted.add("EXPECTED_ERROR_CODES")
            if not set(case.expected_error_codes).issubset(error_codes):
                failed.add("EXPECTED_ERROR_CODES")

        structural_expectations = {
            "EXPECTED_TABLES": bool(case.expected_tables),
            "FORBIDDEN_TABLES": bool(case.forbidden_tables),
            "EXPECTED_FIELDS": bool(case.expected_fields),
            "FORBIDDEN_FIELDS": bool(case.forbidden_fields),
            "EXPECTED_RELATIONSHIPS": bool(case.expected_relationships),
            "EXPECTED_FILTERS": bool(case.expected_filters),
            "EXPECTED_TIME_GRAIN": case.expected_time_grain is not None,
        }
        attempted.update(code for code, enabled in structural_expectations.items() if enabled)
        structure: SQLStructure | None = None
        requires_structure = any(structural_expectations.values()) or bool(
            case.expected_metrics or case.forbidden_metrics
        )
        if requires_structure:
            if generated.sql is None:
                failed.update(code for code, enabled in structural_expectations.items() if enabled)
                error_codes = tuple(sorted((*error_codes, "EVALUATION_SQL_INVALID")))
            else:
                try:
                    structure = analyze_sql_structure(
                        generated.sql,
                        dialect=case.dialect,
                        semantic_indexes=self._field_indexes,
                    )
                except Exception:
                    failed.update(
                        code for code, enabled in structural_expectations.items() if enabled
                    )
                    error_codes = tuple(sorted((*error_codes, "EVALUATION_SQL_INVALID")))

        if structure is not None:
            expected_tables = {_normalize_table(value) for value in case.expected_tables}
            forbidden_tables = {_normalize_table(value) for value in case.forbidden_tables}
            if expected_tables and not expected_tables.issubset(structure.tables):
                failed.add("EXPECTED_TABLES")
            if forbidden_tables.intersection(structure.tables):
                failed.add("FORBIDDEN_TABLES")

            expected_fields = {self._normalize_field(value) for value in case.expected_fields}
            forbidden_fields = {self._normalize_field(value) for value in case.forbidden_fields}
            if expected_fields and not expected_fields.issubset(structure.fields):
                failed.add("EXPECTED_FIELDS")
            if forbidden_fields.intersection(structure.fields):
                failed.add("FORBIDDEN_FIELDS")

            expected_relationships = {
                self._normalize_relationship(value) for value in case.expected_relationships
            }
            if expected_relationships and not expected_relationships.issubset(
                set(structure.relationships)
            ):
                failed.add("EXPECTED_RELATIONSHIPS")

            if case.expected_filters:
                try:
                    expected_filters = {
                        signature
                        for expression in case.expected_filters
                        for signature in normalize_filter_expression(
                            expression,
                            dialect=case.dialect,
                            semantic_indexes=self._field_indexes,
                        )
                    }
                except Exception as exc:
                    raise GoldenConfigurationError("INVALID_FILTER_EXPECTATION") from exc
                if not expected_filters.issubset(set(structure.filters)):
                    failed.add("EXPECTED_FILTERS")

            if case.expected_time_grain is not None:
                time_fields = self._governed_time_fields()
                if not any(
                    grain == case.expected_time_grain and field in time_fields
                    for field, grain in structure.time_grains
                ):
                    failed.add("EXPECTED_TIME_GRAIN")

        for assertion, metrics, expect_present in (
            ("EXPECTED_METRICS", case.expected_metrics, True),
            ("FORBIDDEN_METRICS", case.forbidden_metrics, False),
        ):
            if not metrics:
                continue
            attempted.add(assertion)
            if generated.sql is None:
                if expect_present:
                    failed.add(assertion)
                continue
            if structure is None:
                failed.add(assertion)
                continue
            presences = [self._metric_is_present(name, generated.sql) for name in metrics]
            if (expect_present and not all(presences)) or (
                not expect_present and any(presences)
            ):
                failed.add(assertion)

        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
        return (
            CaseResult(
                id=case.id,
                passed=not failed,
                failed_assertions=tuple(sorted(failed)),
                error_codes=tuple(sorted(set(error_codes))),
                question_fingerprint=_fingerprint(case.question),
                sql_fingerprint=_fingerprint(generated.sql) if generated.sql is not None else None,
                elapsed_ms=elapsed_ms,
                question=case.question if self.include_content else None,
                sql=generated.sql if self.include_content else None,
            ),
            frozenset(attempted),
        )

    def evaluate_results(
        self, items: list[tuple[GoldenCase, Mapping]]
    ) -> EvaluationReport:
        started = time.perf_counter()
        ids = [case.id for case, _ in items]
        if len(ids) != len(set(ids)):
            raise GoldenConfigurationError("DUPLICATE_CASE_ID")
        scored: list[CaseResult] = []
        attempts: Counter[str] = Counter()
        successes: Counter[str] = Counter()
        for case, result in sorted(items, key=lambda item: item[0].id):
            case_result, attempted = self._evaluate_case(case, result)
            scored.append(case_result)
            for assertion in attempted:
                attempts[assertion] += 1
                if assertion not in case_result.failed_assertions:
                    successes[assertion] += 1
        passed = sum(result.passed for result in scored)
        total = len(scored)
        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
        return EvaluationReport(
            model=self.model,
            model_parameters=self.model_parameters,
            datasource=self.snapshot.datasource,
            dialect=self.snapshot.dialect,
            semantic_version=self.snapshot.model_version,
            git_commit=self.git_commit,
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total if total else 0.0,
            per_rule=tuple(
                RuleRate(
                    assertion=assertion,
                    total=count,
                    passed=successes[assertion],
                    rate=successes[assertion] / count,
                )
                for assertion, count in sorted(attempts.items())
            ),
            cases=tuple(scored),
            elapsed_ms=elapsed_ms,
            content_included=self.include_content,
        )

    def write_report(self, report: EvaluationReport, path: Path) -> Path:
        if report.content_included and not self.include_content:
            raise GoldenConfigurationError("CONTENT_OPT_IN_REQUIRED")
        has_content = any(
            case.question is not None or case.sql is not None for case in report.cases
        )
        if has_content and not report.content_included:
            raise GoldenConfigurationError("REPORT_CONTENT_MISMATCH")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = report.model_dump(mode="json", exclude_none=True)
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                os.chmod(temporary.name, 0o600)
                temporary.write(serialized)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        except Exception:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise
        return path

    def _validate_case_identity(self, case: GoldenCase) -> None:
        if case.datasource != self.snapshot.datasource or case.dialect != self.snapshot.dialect:
            raise GoldenConfigurationError("SNAPSHOT_MISMATCH")
        referenced_tables = {
            _normalize_table(value)
            for value in (*case.expected_tables, *case.forbidden_tables)
        }
        if not referenced_tables.issubset(self._field_indexes):
            raise GoldenConfigurationError("UNKNOWN_TABLE_EXPECTATION")
        for value in (*case.expected_fields, *case.forbidden_fields):
            self._normalize_field(value)
        referenced_metrics = {
            _normalize(value)
            for value in (*case.expected_metrics, *case.forbidden_metrics)
        }
        if not referenced_metrics.issubset(self._metric_names):
            raise GoldenConfigurationError("UNKNOWN_METRIC_EXPECTATION")
        for value in case.expected_relationships:
            self._normalize_relationship(value)
        try:
            for expression in case.expected_filters:
                normalize_filter_expression(
                    expression,
                    dialect=case.dialect,
                    semantic_indexes=self._field_indexes,
                )
        except Exception as exc:
            raise GoldenConfigurationError("INVALID_FILTER_EXPECTATION") from exc

    @staticmethod
    def _parse_generated_result(
        case: GoldenCase, result: Mapping
    ) -> GoldenGeneratedResult:
        if not isinstance(result, Mapping):
            raise GoldenConfigurationError("INVALID_RESULT")
        payload = dict(result)
        supplied_id = payload.setdefault("id", case.id)
        if supplied_id != case.id:
            raise GoldenConfigurationError("RESULT_ID_MISMATCH")
        try:
            return GoldenGeneratedResult.model_validate(payload)
        except Exception as exc:
            raise GoldenConfigurationError("INVALID_RESULT") from exc

    def _normalize_field(self, value: str) -> str:
        normalized = _normalize(value)
        if "." in normalized:
            table, field = normalized.rsplit(".", 1)
            table = _normalize_table(table)
            if table not in self._field_indexes or field not in self._field_indexes[table]:
                raise GoldenConfigurationError("UNKNOWN_FIELD_EXPECTATION")
            return f"{table}.{field}"
        candidates = [
            table for table, fields in self._field_indexes.items() if normalized in fields
        ]
        if not candidates:
            raise GoldenConfigurationError("UNKNOWN_FIELD_EXPECTATION")
        if len(candidates) != 1:
            raise GoldenConfigurationError("AMBIGUOUS_FIELD_EXPECTATION")
        return f"{candidates[0]}.{normalized}"

    def _normalize_relationship(self, value: str) -> tuple[str, str]:
        normalized = _normalize(value)
        by_name = {
            _normalize(relation.name): tuple(
                sorted(
                    (
                        f"{_normalize(relation.from_entity)}.{_normalize(relation.from_field)}",
                        f"{_normalize(relation.to_entity)}.{_normalize(relation.to_field)}",
                    )
                )
            )
            for relation in self.snapshot.relationships
        }
        if normalized in by_name:
            return by_name[normalized]
        parts = re.split(r"\s*=\s*", normalized)
        if len(parts) != 2:
            raise GoldenConfigurationError("INVALID_RELATIONSHIP_EXPECTATION")
        endpoints = tuple(
            sorted((self._normalize_field(parts[0]), self._normalize_field(parts[1])))
        )
        if endpoints not in set(by_name.values()):
            raise GoldenConfigurationError("UNKNOWN_RELATIONSHIP_EXPECTATION")
        return endpoints

    def _metric_is_present(self, name: str, sql: str) -> bool:
        normalized = _normalize(name)
        metric = next(
            (item for item in self.snapshot.metrics if _normalize(item.name) == normalized),
            None,
        )
        if metric is None:
            raise GoldenConfigurationError("UNKNOWN_METRIC_EXPECTATION")
        report = self._semantic_validator.validate(sql, metric.name, self.snapshot)
        return not any(
            issue.code == "METRIC_MISMATCH" and normalized in issue.objects
            for issue in report.issues
        )

    def _governed_time_fields(self) -> frozenset[str]:
        return frozenset(
            f"{_normalize(entity.name)}.{_normalize(field.name)}"
            for entity in self.snapshot.entities
            for field in entity.fields
            if field.role == "time"
        )


def _normalize_table(value: str) -> str:
    return _normalize(value).rsplit(".", 1)[-1]


__all__ = [
    "CaseResult",
    "EvaluationReport",
    "GoldenCase",
    "GoldenConfigurationError",
    "GoldenEvaluator",
    "GoldenGeneratedResult",
    "GoldenResults",
    "GoldenSuite",
    "ModelParameter",
    "RuleRate",
    "get_git_commit",
    "load_golden_results",
    "load_golden_suite",
    "parse_model_parameters",
]
