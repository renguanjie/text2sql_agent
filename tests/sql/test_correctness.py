from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import pytest

from core.semantic.compiler import SemanticCompiler
from core.semantic.context import SemanticContextBuilder
from core.semantic.models import (
    PreflightResult,
    SemanticEntity,
    SemanticField,
    SemanticMetric,
    SemanticSnapshot,
    ValidationIssue,
)
from core.semantic.registry import SemanticRegistry
from core.semantic.validator import SemanticSQLValidator


def _snapshot(version: str = "1.0.0") -> SemanticSnapshot:
    return SemanticSnapshot(
        schema_version=1,
        model_version=version,
        datasource="sales",
        dialect="mysql",
        entities=(
            SemanticEntity(
                name="orders",
                name_cn="订单",
                fields=(
                    SemanticField(
                        name="id",
                        name_cn="订单号",
                        data_type="bigint",
                        role="identifier",
                    ),
                    SemanticField(
                        name="amount",
                        name_cn="金额",
                        data_type="decimal",
                        role="measure",
                    ),
                    SemanticField(
                        name="created_at",
                        name_cn="下单时间",
                        data_type="datetime",
                        role="time",
                    ),
                ),
            ),
        ),
        metrics=(
            SemanticMetric(
                name="order_amount",
                name_cn="订单金额",
                expression="SUM(orders.amount)",
                allowed_dimensions=("orders.created_at",),
                default_time_field="orders.created_at",
            ),
        ),
        source_sha256="0" * 64,
        content_sha256="1" * 64,
        compiler_version="1",
        compiled_at="2026-09-09T00:00:00Z",
    )


class MutableRegistry:
    def __init__(self, snapshot: SemanticSnapshot):
        self.snapshot = snapshot
        self.calls = 0

    def get_snapshot(self) -> SemanticSnapshot:
        self.calls += 1
        return self.snapshot


class ChangingRegistry:
    def __init__(self, first: SemanticSnapshot, later: SemanticSnapshot):
        self._first = first
        self._later = later
        self.calls = 0

    def get_snapshot(self) -> SemanticSnapshot:
        self.calls += 1
        return self._first if self.calls == 1 else self._later


class SequencedPreflight:
    def __init__(self):
        self.sql: list[str] = []

    def check(self, sql: str) -> PreflightResult:
        self.sql.append(sql)
        if len(self.sql) == 1:
            return PreflightResult(
                valid=False,
                issues=(
                    ValidationIssue(
                        code="EXPLAIN_FAILED",
                        message="数据库预检失败",
                        stage="preflight",
                    ),
                ),
            )
        return PreflightResult(valid=True, estimated_rows=1)


@pytest.fixture
def registry() -> MutableRegistry:
    return MutableRegistry(_snapshot())


@pytest.fixture
def pipeline(registry: MutableRegistry):
    from core.sql.correctness import CorrectnessPipeline

    return CorrectnessPipeline(
        registry=registry,
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=SemanticSQLValidator(),
    )


def _make_pipeline():
    from core.sql.correctness import CorrectnessPipeline

    return CorrectnessPipeline(
        registry=MutableRegistry(_snapshot()),
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=SemanticSQLValidator(),
    )


def test_repairs_once_then_rejects(pipeline):
    """Catches recursive repair or accepting a repaired SQL that still fails validation."""
    calls: list[tuple[str, list[dict], str]] = []

    def repair(sql: str, issues: list[dict], context: str) -> dict:
        calls.append((sql, issues, context))
        return {"status": "success", "sql": "SELECT missing_again FROM orders"}

    result = pipeline.process(
        "订单金额", "SELECT missing FROM orders", repair=repair
    )

    assert result.status == "rejected"
    assert result.repair_attempts == 1
    assert result.executable is False
    assert result.sql == "SELECT missing_again FROM orders"
    assert len(calls) == 1


def test_successful_repair_runs_full_validation_again(pipeline):
    """Catches marking a repair executable without re-running deterministic validation."""

    def repair(sql: str, issues: list[dict], context: str) -> dict:
        return {
            "status": "success",
            "sql": "SELECT SUM(o.amount) FROM orders o",
            "error": None,
        }

    result = pipeline.process(
        "订单金额", "SELECT COUNT(*) FROM orders", repair=repair
    )

    assert result.status == "success"
    assert result.repair_attempts == 1
    assert result.executable is True
    assert result.validation.valid


def test_process_pins_one_snapshot_across_context_and_repair_validation():
    """Catches a request mixing semantic versions by reading the registry twice."""
    from core.sql.correctness import CorrectnessPipeline

    registry = ChangingRegistry(_snapshot("1.0.0"), _snapshot("1.1.0"))
    contexts: list[str] = []

    def repair(sql: str, issues: list[dict], context: str) -> dict:
        contexts.append(context)
        return {"status": "success", "sql": "SELECT SUM(amount) FROM orders"}

    result = CorrectnessPipeline(
        registry=registry,
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=SemanticSQLValidator(),
    ).process("订单金额", "SELECT COUNT(*) FROM orders", repair=repair)

    assert result.status == "success"
    assert result.semantic_version == "1.0.0"
    assert contexts and "版本: sales@1.0.0" in contexts[0]
    assert registry.calls == 1


def test_prepared_request_is_an_empty_identity_capability():
    """The token itself must expose no semantic payload that can be forged."""
    pipeline = _make_pipeline()
    request = pipeline.prepare_generation("订单金额")
    assert not hasattr(request, "_question")
    assert not hasattr(request, "_snapshot")
    assert not hasattr(request, "semantic_context")
    assert not hasattr(request, "semantic_version")
    with pytest.raises(TypeError):
        replace(request)

    semantic_context, semantic_version, content_sha256 = (
        pipeline.describe_prepared(request)
    )
    assert "版本: sales@1.0.0" in semantic_context
    assert semantic_version == "1.0.0"
    assert content_sha256 == "1" * 64


def test_prepared_request_rejects_lookalike_with_forged_fields():
    """A self-constructed payload with valid-looking fields is not a capability."""
    pipeline = _make_pipeline()
    request = pipeline.prepare_generation("订单金额")

    @dataclass(frozen=True)
    class ForgedRequest:
        semantic_context: str
        semantic_version: str
        _question: str
        _snapshot: SemanticSnapshot

    forged = ForgedRequest(
        semantic_context="版本: sales@9.9.9\n实体: archived_orders",
        semantic_version="9.9.9",
        _question="查询归档订单",
        _snapshot=_snapshot("9.9.9"),
    )

    forged_result = pipeline.process_prepared(
        forged, "SELECT SUM(amount) FROM orders"
    )
    original_result = pipeline.process_prepared(
        request, "SELECT SUM(amount) FROM orders"
    )

    assert forged_result.status == "error"
    assert forged_result.executable is False
    assert original_result.status == "success"


def test_prepared_request_is_bound_to_issuing_pipeline():
    """A request issued by one pipeline is invalid in another pipeline."""
    first = _make_pipeline()
    second = _make_pipeline()
    request = first.prepare_generation("订单金额")

    foreign = second.process_prepared(request, "SELECT SUM(amount) FROM orders")
    original = first.process_prepared(request, "SELECT SUM(amount) FROM orders")

    assert foreign.status == "error"
    assert foreign.executable is False
    assert original.status == "success"


def test_prepared_request_is_consumed_once_and_cannot_be_replayed():
    """A successful sync processing call must consume its capability."""
    pipeline = _make_pipeline()
    request = pipeline.prepare_generation("订单金额")

    first = pipeline.process_prepared(request, "SELECT SUM(amount) FROM orders")
    replay = pipeline.process_prepared(request, "SELECT SUM(amount) FROM orders")

    assert first.status == "success"
    assert replay.status == "error"
    assert replay.executable is False


@pytest.mark.asyncio
async def test_async_prepared_request_is_consumed_once_and_cannot_be_replayed():
    """Async processing must use the same atomic one-shot capability rule."""
    pipeline = _make_pipeline()
    request = pipeline.prepare_generation("订单金额")

    first = await pipeline.process_async_prepared(
        request, "SELECT SUM(amount) FROM orders"
    )
    replay = await pipeline.process_async_prepared(
        request, "SELECT SUM(amount) FROM orders"
    )

    assert first.status == "success"
    assert replay.status == "error"
    assert replay.executable is False


def test_discarded_prepared_request_cannot_be_used():
    """Discard must remove an abandoned generation capability idempotently."""
    pipeline = _make_pipeline()
    request = pipeline.prepare_generation("订单金额")

    assert pipeline.discard_prepared(request) is True
    assert pipeline.discard_prepared(request) is False
    result = pipeline.process_prepared(request, "SELECT SUM(amount) FROM orders")

    assert result.status == "error"
    assert result.executable is False


def test_original_and_repaired_candidates_both_run_preflight(registry: MutableRegistry):
    """Catches bypassing EXPLAIN for either the original or repaired candidate."""
    from core.sql.correctness import CorrectnessPipeline

    preflight = SequencedPreflight()

    def repair(sql: str, issues: list[dict], context: str) -> dict:
        assert [issue["code"] for issue in issues] == ["EXPLAIN_FAILED"]
        return {"status": "success", "sql": "SELECT SUM(o.amount) FROM orders o"}

    result = CorrectnessPipeline(
        registry=registry,
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=SemanticSQLValidator(),
        preflight=preflight,
    ).process("订单金额", "SELECT SUM(amount) FROM orders", repair=repair)

    assert result.status == "success"
    assert preflight.sql == [
        "SELECT SUM(amount) FROM orders",
        "SELECT SUM(o.amount) FROM orders o",
    ]


@pytest.mark.parametrize(
    "repair_result",
    [
        None,
        [],
        {},
        {"status": True, "sql": "SELECT SUM(amount) FROM orders"},
        {"status": "SUCCESS", "sql": "SELECT SUM(amount) FROM orders"},
        {"status": "success", "sql": 7},
        {"status": "success", "sql": "   "},
    ],
    ids=[
        "none",
        "list",
        "missing_fields",
        "truthy_status",
        "wrong_status",
        "non_string_sql",
        "blank_sql",
    ],
)
def test_repair_response_mapping_is_strict_and_fail_closed(pipeline, repair_result):
    """Catches truthy or malformed repair payloads reaching validation or execution."""

    def repair(sql: str, issues: list[dict], context: str):
        return repair_result

    result = pipeline.process(
        "订单金额", "SELECT missing FROM orders", repair=repair
    )

    assert result.status == "error"
    assert result.repair_attempts == 1
    assert result.executable is False
    assert result.sql == "SELECT missing FROM orders"
    assert result.validation.issues[-1].code == "REPAIR_FAILED"
    assert result.validation.issues[-1].message == "SQL 修复未返回有效候选"


def test_repair_exception_is_masked_in_stable_error_result(pipeline):
    """Catches callback exceptions leaking database or SQL details to public issues."""

    def repair(sql: str, issues: list[dict], context: str) -> dict:
        raise RuntimeError("password=secret database=private SELECT hidden")

    result = pipeline.process(
        "订单金额", "SELECT missing FROM orders", repair=repair
    )

    assert result.status == "error"
    assert result.repair_attempts == 1
    assert result.executable is False
    assert result.validation.issues[-1].code == "REPAIR_FAILED"
    assert result.validation.issues[-1].message == "SQL 修复失败"
    public_messages = " ".join(issue.message for issue in result.validation.issues)
    assert "secret" not in public_messages
    assert "private" not in public_messages
    assert "hidden" not in public_messages


def test_unexpected_validation_exception_is_masked_and_never_repaired(
    registry: MutableRegistry,
):
    """Catches infrastructure failures being exposed or sent to the repair model."""
    from core.sql.correctness import CorrectnessPipeline

    class ExplodingValidator:
        def validate(self, sql: str, question: str, snapshot: SemanticSnapshot):
            raise RuntimeError("password=secret database=private")

    repairs: list[str] = []

    def repair(sql: str, issues: list[dict], context: str) -> dict:
        repairs.append(sql)
        return {"status": "success", "sql": "SELECT id FROM orders"}

    result = CorrectnessPipeline(
        registry=registry,
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=ExplodingValidator(),
    ).process("查询订单", "SELECT id FROM orders", repair=repair)

    assert result.status == "error"
    assert result.semantic_version == "1.0.0"
    assert result.repair_attempts == 0
    assert result.executable is False
    assert result.validation.issues[0].model_dump() == {
        "code": "CORRECTNESS_UNAVAILABLE",
        "message": "SQL 正确性校验暂时不可用",
        "objects": [],
        "stage": "semantic",
    }
    assert repairs == []


def test_scope_failure_is_stable_and_does_not_echo_legacy_exception_text(pipeline):
    """Catches raw legacy scope diagnostics becoming public pipeline messages."""
    result = pipeline.process(
        "查询订单",
        "SELECT id FROM orders",
        repair=None,
        allowed_tables=("customers",),
    )

    assert result.status == "rejected"
    assert result.executable is False
    assert result.validation.issues[0].code == "QUERY_SCOPE_DENIED"
    assert result.validation.issues[0].message == "SQL 查询范围不被允许"


@pytest.mark.parametrize("value", [-1, 2, True, "1"])
def test_constructor_rejects_non_literal_repair_limits(registry: MutableRegistry, value):
    """Catches configuration values that could allow an undefined repair policy."""
    from core.sql.correctness import CorrectnessPipeline

    with pytest.raises(ValueError, match="max_repair_attempts"):
        CorrectnessPipeline(
            registry=registry,
            context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
            semantic_validator=SemanticSQLValidator(),
            max_repair_attempts=value,
        )


def _compiled_registry(tmp_path: Path) -> SemanticRegistry:
    source = tmp_path / "sales.yaml"
    output = tmp_path / ".compiled"
    compiler = SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z")

    def compile_version(version: str) -> None:
        source.write_text(
            f"""
schema_version: 1
model_version: {version}
datasource: sales
dialect: mysql
entities:
  - name: orders
    name_cn: 订单
    fields:
      - {{name: id, name_cn: 订单号, data_type: bigint, role: identifier}}
      - {{name: amount, name_cn: 金额, data_type: decimal, role: measure}}
metrics:
  - name: order_amount
    name_cn: 订单金额
    expression: SUM(orders.amount)
""".lstrip(),
            encoding="utf-8",
        )
        compiler.compile_file(source, output)

    compile_version("1.0.0")
    compile_version("1.1.0")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["active"]["sales"] = "1.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry = SemanticRegistry(output, "sales")
    registry.load_active()
    return registry


def test_version_change_rejects_execution_without_validation_or_repair(tmp_path: Path):
    """Catches execution authorization reusing a stale generation snapshot."""
    from core.sql.correctness import CorrectnessPipeline

    registry = _compiled_registry(tmp_path)
    preflight = SequencedPreflight()
    pipeline = CorrectnessPipeline(
        registry=registry,
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=SemanticSQLValidator(),
        preflight=preflight,
    )
    generated = pipeline.process("订单金额", "SELECT SUM(amount) FROM orders")
    assert generated.semantic_version == "1.0.0"

    manifest_path = registry.compiled_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["active"]["sales"] = "1.1.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry.load_active()
    preflight.sql.clear()

    authorized = pipeline.authorize_execution(
        question="订单金额",
        sql=generated.sql or "",
        expected_semantic_version=generated.semantic_version or "",
    )

    assert authorized.status == "rejected"
    assert authorized.semantic_version == "1.1.0"
    assert authorized.repair_attempts == 0
    assert authorized.executable is False
    assert authorized.validation.issues[0].code == "SEMANTIC_VERSION_MISMATCH"
    assert preflight.sql == []


def test_authorize_execution_reloads_current_snapshot_and_runs_preflight(
    registry: MutableRegistry,
):
    """Catches authorization trusting the earlier process result without a fresh lookup."""
    from core.sql.correctness import CorrectnessPipeline

    preflight = SequencedPreflight()
    preflight.check("warmup")
    pipeline = CorrectnessPipeline(
        registry=registry,
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=SemanticSQLValidator(),
        preflight=preflight,
    )

    result = pipeline.authorize_execution(
        question="订单金额",
        sql="SELECT SUM(amount) FROM orders",
        expected_semantic_version="1.0.0",
    )

    assert result.status == "success"
    assert result.executable is True
    assert result.repair_attempts == 0
    assert registry.calls == 1
    assert preflight.sql[-1] == "SELECT SUM(amount) FROM orders"


@pytest.mark.asyncio
async def test_async_pipeline_matches_sync_result(pipeline):
    """Catches sync and async entry points applying different validation rules."""
    sync_result = pipeline.process("订单金额", "SELECT SUM(amount) FROM orders")
    async_result = await pipeline.process_async(
        "订单金额", "SELECT SUM(amount) FROM orders"
    )

    assert async_result.model_dump() == sync_result.model_dump()


@pytest.mark.asyncio
async def test_async_repair_awaits_once_and_revalidates(pipeline):
    """Catches async repair skipping await, repeating, or bypassing revalidation."""
    calls: list[str] = []

    async def repair(sql: str, issues: list[dict], context: str) -> dict:
        calls.append(sql)
        return {"status": "success", "sql": "SELECT SUM(amount) FROM orders"}

    result = await pipeline.process_async(
        "订单金额", "SELECT COUNT(*) FROM orders", repair=repair
    )

    assert result.status == "success"
    assert result.repair_attempts == 1
    assert result.executable is True
    assert calls == ["SELECT COUNT(*) FROM orders"]


def test_process_empty_scope_tuples_use_snapshot_tables_but_no_qualified_schema(
    pipeline,
):
    """Catches empty public defaults becoming deny-all tables or allow-all schemas."""
    unqualified = pipeline.process("查询订单", "SELECT id FROM orders")
    qualified = pipeline.process("查询订单", "SELECT id FROM private.orders")

    assert unqualified.status == "success"
    assert qualified.status == "rejected"
    assert qualified.validation.issues[0].code == "QUERY_SCOPE_DENIED"


@pytest.mark.asyncio
async def test_async_custom_blocklist_does_not_disable_builtin_blocked_tables():
    """Catches custom blocked tables replacing the immutable internal blacklist."""
    from core.sql.correctness import CorrectnessPipeline

    internal = SemanticEntity(
        name="sql_generation_history",
        name_cn="内部历史",
        fields=(
            SemanticField(
                name="id",
                name_cn="编号",
                data_type="bigint",
                role="identifier",
            ),
        ),
    )
    base_snapshot = _snapshot()
    snapshot = base_snapshot.model_copy(
        update={"entities": (*base_snapshot.entities, internal)}
    )
    pipeline = CorrectnessPipeline(
        registry=MutableRegistry(snapshot),
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=SemanticSQLValidator(),
        blocked_tables=("audit_log",),
    )

    result = await pipeline.process_async(
        "查询内部历史", "SELECT id FROM sql_generation_history"
    )

    assert result.status == "rejected"
    assert result.validation.issues[0].code == "QUERY_SCOPE_DENIED"


def test_authorize_execution_requires_explicit_permission_for_qualified_schema(
    registry: MutableRegistry,
):
    """Catches execution authorization accepting arbitrary qualified schemas."""
    from core.sql.correctness import CorrectnessPipeline

    pipeline = CorrectnessPipeline(
        registry=registry,
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=SemanticSQLValidator(),
    )

    denied = pipeline.authorize_execution(
        question="查询订单",
        sql="SELECT id FROM private.orders",
        expected_semantic_version="1.0.0",
    )
    allowed = pipeline.authorize_execution(
        question="查询订单",
        sql="SELECT id FROM private.orders",
        expected_semantic_version="1.0.0",
        allowed_schemas=("private",),
    )

    assert denied.status == "rejected"
    assert denied.validation.issues[0].code == "QUERY_SCOPE_DENIED"
    assert allowed.status == "success"
