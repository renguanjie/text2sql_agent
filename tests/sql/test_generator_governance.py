from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger

from core.semantic.context import SemanticContextBuilder
from core.semantic.models import (
    CorrectnessResult,
    SemanticEntity,
    SemanticField,
    SemanticMetric,
    SemanticSnapshot,
    ValidationIssue,
    ValidationReport,
)
from core.semantic.validator import SemanticSQLValidator
from core.sql.correctness import CorrectnessPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StaticRegistry:
    def __init__(self, snapshot: SemanticSnapshot):
        self.snapshot = snapshot

    def get_snapshot(self) -> SemanticSnapshot:
        return self.snapshot


class ChangingRegistry:
    def __init__(self, first: SemanticSnapshot, later: SemanticSnapshot):
        self.first = first
        self.later = later
        self.calls = 0

    def get_snapshot(self) -> SemanticSnapshot:
        self.calls += 1
        return self.first if self.calls == 1 else self.later


class ExplodingSemanticValidator:
    def validate(self, *_args, **_kwargs):
        raise RuntimeError("password=secret SELECT private FROM hidden")


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


def _pipeline(
    *, max_repair_attempts: int = 1, **options
) -> CorrectnessPipeline:
    return CorrectnessPipeline(
        registry=StaticRegistry(_snapshot()),
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=SemanticSQLValidator(),
        max_repair_attempts=max_repair_attempts,
        **options,
    )


def _changing_pipeline() -> tuple[CorrectnessPipeline, ChangingRegistry]:
    first = _snapshot()
    later = first.model_copy(
        update={
            "entities": (
                first.entities[0].model_copy(update={"name": "archived_orders"}),
            ),
            "content_sha256": "2" * 64,
        }
    )
    registry = ChangingRegistry(first, later)
    return (
        CorrectnessPipeline(
            registry=registry,
            context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
            semantic_validator=SemanticSQLValidator(),
        ),
        registry,
    )


def _generator(
    *,
    pipeline: CorrectnessPipeline | Mock | None,
    mysql_client=None,
    candidate_sql: str = "SELECT SUM(amount) FROM orders",
    config_overrides: dict | None = None,
):
    from core.sql.generator import SQLGenerator

    retriever = Mock()
    retriever.history_index = None
    chain = Mock()
    chain.dialect = "mysql"
    chain.retrieval_top_k = 3
    generated = {"status": "success", "sql": candidate_sql, "matched_knowledge": []}
    chain.generate.return_value = dict(generated)
    chain.generate_async = AsyncMock(return_value=dict(generated))
    chain.fix_sql_governed.return_value = {
        "status": "success",
        "sql": "SELECT SUM(amount) FROM orders",
        "error": None,
    }
    chain.fix_sql_governed_async = AsyncMock(
        return_value={
            "status": "success",
            "sql": "SELECT SUM(amount) FROM orders",
            "error": None,
        }
    )
    generator_config = {
        "sql_dialect": "mysql",
        "sql_query_max_rows": 5,
        "sql_query_timeout_ms": 2500,
    }
    generator_config.update(config_overrides or {})
    generator = SQLGenerator(
        Mock(),
        retriever,
        chain,
        mysql_client=mysql_client,
        config=generator_config,
        correctness_pipeline=pipeline,
    )
    context = {
        "final_query": "订单金额",
        "detected_dialect": "mysql",
        "detected_database_id": None,
        "knowledge_results": [],
        "few_shot_examples": "",
        "schema_info": {
            "tables": [{"table_name": "orders"}],
            "columns": [{"table_name": "orders", "column_name": "amount"}],
            "relationships": [],
        },
    }
    generator._prepare_pipeline = Mock(return_value=context)
    generator._prepare_pipeline_async = AsyncMock(return_value=context)
    return generator


@contextmanager
def capture_loguru(caplog):
    caplog.set_level(logging.DEBUG)
    sink_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    try:
        yield
    finally:
        logger.remove(sink_id)


def _write_model(path: Path, *, version: str = "1.0.0") -> None:
    path.write_text(
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
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _governance_config(model_dir: Path) -> dict:
    return {
        "semantic_governance_enabled": True,
        "semantic_model_dir": str(model_dir),
        "semantic_datasource": "sales",
        "semantic_context_max_chars": 2400,
        "semantic_context_max_items": 24,
        "sql_repair_max_attempts": 1,
        "sql_explain_enabled": False,
        "sql_explain_timeout_ms": 3000,
        "sql_explain_max_estimated_rows": 0,
    }


def _import_config(**updates: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "LLM_PROVIDER": "ollama",
            "EMBEDDING_PROVIDER": "ollama",
            **updates,
        }
    )
    return subprocess.run(
        [sys.executable, "-c", "import config"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_generator_returns_governed_result_and_injects_the_same_semantic_context():
    """Catches bypassing governance or omitting the governed generation context."""
    generator = _generator(pipeline=_pipeline())

    result = generator.generate_sql("订单金额", auto_save_history=False)

    assert result["semantic_version"] == "1.0.0"
    assert result["content_sha256"] == "1" * 64
    assert result["repair_attempts"] == 0
    assert result["executable"] is True
    assert result["validation"] == {"valid": True, "issues": []}
    assert result["user_query"] == "订单金额"
    semantic_context = generator.sql_chain.generate.call_args.kwargs["semantic_context"]
    assert "版本: sales@1.0.0" in semantic_context
    assert "指标: 订单金额 = SUM(orders.amount)" in semantic_context


@pytest.mark.asyncio
async def test_async_generator_uses_the_shared_correctness_pipeline_and_result_adapter():
    """Catches async generation skipping or reshaping the shared governed result."""
    generator = _generator(pipeline=_pipeline())

    result = await generator.generate_sql_async("订单金额", auto_save_history=False)

    assert result["semantic_version"] == "1.0.0"
    assert result["repair_attempts"] == 0
    assert result["executable"] is True
    assert result["validation"] == {"valid": True, "issues": []}
    semantic_context = generator.sql_chain.generate_async.call_args.kwargs[
        "semantic_context"
    ]
    assert "版本: sales@1.0.0" in semantic_context


def test_sync_generation_pins_one_snapshot_when_same_version_content_changes():
    """Generation context and validation must use one immutable snapshot object."""
    pipeline, registry = _changing_pipeline()
    generator = _generator(
        pipeline=pipeline, candidate_sql="SELECT COUNT(*) FROM orders"
    )

    result = generator.generate_sql("订单金额", auto_save_history=False)

    assert result["status"] == "success"
    assert result["executable"] is True
    assert result["repair_attempts"] == 1
    assert registry.calls == 1
    semantic_context = generator.sql_chain.generate.call_args.kwargs[
        "semantic_context"
    ]
    assert "orders" in semantic_context
    assert "archived_orders" not in semantic_context
    repair_context = generator.sql_chain.fix_sql_governed.call_args.args[2]
    assert repair_context == semantic_context


@pytest.mark.asyncio
async def test_async_generation_pins_one_snapshot_when_same_version_content_changes():
    """Async generation must share the exact snapshot pinned before its LLM call."""
    pipeline, registry = _changing_pipeline()
    generator = _generator(
        pipeline=pipeline, candidate_sql="SELECT COUNT(*) FROM orders"
    )

    result = await generator.generate_sql_async(
        "订单金额", auto_save_history=False
    )

    assert result["status"] == "success"
    assert result["executable"] is True
    assert result["repair_attempts"] == 1
    assert registry.calls == 1
    semantic_context = generator.sql_chain.generate_async.call_args.kwargs[
        "semantic_context"
    ]
    assert "orders" in semantic_context
    assert "archived_orders" not in semantic_context
    repair_context = generator.sql_chain.fix_sql_governed_async.call_args.args[2]
    assert repair_context == semantic_context


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_generator_correctness_error_has_stable_top_level_fallback(use_async: bool):
    """Both adapters must expose a non-null safe error for legacy UI fallbacks."""
    pipeline = CorrectnessPipeline(
        registry=StaticRegistry(_snapshot()),
        context_builder=SemanticContextBuilder(max_chars=2400, max_items=24),
        semantic_validator=ExplodingSemanticValidator(),
    )
    generator = _generator(pipeline=pipeline)

    if use_async:
        result = await generator.generate_sql_async(
            "订单金额", auto_save_history=False
        )
    else:
        result = generator.generate_sql("订单金额", auto_save_history=False)

    assert result["status"] == "error"
    assert result["error"] == "SQL 正确性校验暂时不可用"
    assert result["validation"]["issues"] == [
        {
            "code": "CORRECTNESS_UNAVAILABLE",
            "message": "SQL 正确性校验暂时不可用",
            "objects": [],
            "stage": "semantic",
        }
    ]
    assert "secret" not in str(result)


def test_generator_prepare_exception_returns_stable_error_without_leaking(caplog):
    generator = _generator(pipeline=_pipeline())
    generator._prepare_pipeline.side_effect = RuntimeError(
        "Authorization: Bearer adapter-secret SELECT hidden"
    )

    with capture_loguru(caplog):
        result = generator.generate_sql("订单金额", auto_save_history=False)

    assert result["status"] == "error"
    assert result["error"] == "SQL 生成暂时不可用，请稍后重试"
    assert result["executable"] is False
    assert "adapter-secret" not in caplog.text
    assert "Bearer" not in caplog.text
    assert "Traceback" not in caplog.text
    assert "stage=sql_generator" in caplog.text


def test_generator_prepared_capacity_exhaustion_is_stable_and_fail_closed(caplog):
    pipeline = _pipeline(max_prepared_requests=1)
    pipeline.prepare_generation("private outstanding question")
    generator = _generator(pipeline=pipeline)

    with capture_loguru(caplog):
        result = generator.generate_sql(
            "second secret question", auto_save_history=False
        )

    assert result["status"] == "error"
    assert result["executable"] is False
    assert result["validation"]["issues"][0]["code"] == "CORRECTNESS_UNAVAILABLE"
    assert "private outstanding question" not in caplog.text
    assert "second secret question" not in caplog.text


def test_sync_generator_discards_prepared_request_when_chain_raises(monkeypatch):
    """An exception before correctness processing must not leave a live token."""
    pipeline = _pipeline()
    issued = []
    prepare = pipeline.prepare_generation

    def capture(question: str):
        request = prepare(question)
        issued.append(request)
        return request

    monkeypatch.setattr(pipeline, "prepare_generation", capture)
    generator = _generator(pipeline=pipeline)
    generator.sql_chain.generate.side_effect = RuntimeError("LLM unavailable")

    result = generator.generate_sql("订单金额", auto_save_history=False)

    replay = pipeline.process_prepared(
        issued[0], "SELECT SUM(amount) FROM orders"
    )
    assert result["status"] == "error"
    assert result["error"] == "SQL 生成暂时不可用，请稍后重试"
    assert replay.status == "error"
    assert replay.executable is False


def test_generator_discards_request_when_prepared_description_fails(monkeypatch):
    """Failure between issuance and return must release the server-side state."""
    pipeline = _pipeline()
    issued = []

    def fail_description(request):
        issued.append(request)
        raise RuntimeError("description unavailable")

    monkeypatch.setattr(pipeline, "describe_prepared", fail_description)
    generator = _generator(pipeline=pipeline)

    result = generator.generate_sql("订单金额", auto_save_history=False)
    replay = pipeline.process_prepared(
        issued[0], "SELECT SUM(amount) FROM orders"
    )

    assert result["status"] == "error"
    assert result["validation"]["issues"][0]["code"] == (
        "CORRECTNESS_UNAVAILABLE"
    )
    assert replay.status == "error"
    assert replay.executable is False


def test_sync_generator_discards_request_when_generation_context_is_malformed(
    monkeypatch,
):
    """A kwargs-building failure after issuance must not leave a live token."""
    pipeline = _pipeline()
    issued = []
    prepare = pipeline.prepare_generation

    def capture(question: str):
        request = prepare(question)
        issued.append(request)
        return request

    monkeypatch.setattr(pipeline, "prepare_generation", capture)
    generator = _generator(pipeline=pipeline)
    monkeypatch.setattr(generator, "_prepare_pipeline", lambda *_args: {})

    result = generator.generate_sql("订单金额", auto_save_history=False)

    with pytest.raises(ValueError, match="invalid or expired"):
        pipeline.describe_prepared(issued[0])
    replay = pipeline.process_prepared(
        issued[0], "SELECT SUM(amount) FROM orders"
    )
    assert result["status"] == "error"
    assert result["error"] == "SQL 生成暂时不可用，请稍后重试"
    assert replay.status == "error"
    assert replay.executable is False


@pytest.mark.asyncio
async def test_async_generator_discards_prepared_request_on_generation_failure(
    monkeypatch,
):
    """A non-candidate async result must release its unused prepared token."""
    pipeline = _pipeline()
    issued = []
    prepare = pipeline.prepare_generation

    def capture(question: str):
        request = prepare(question)
        issued.append(request)
        return request

    monkeypatch.setattr(pipeline, "prepare_generation", capture)
    generator = _generator(pipeline=pipeline)
    generator.sql_chain.generate_async.return_value = {
        "status": "error",
        "sql": None,
        "error": "generation unavailable",
    }

    result = await generator.generate_sql_async(
        "订单金额", auto_save_history=False
    )

    replay = pipeline.process_prepared(
        issued[0], "SELECT SUM(amount) FROM orders"
    )
    assert result["status"] == "error"
    assert replay.status == "error"
    assert replay.executable is False


@pytest.mark.asyncio
async def test_async_generator_discards_request_when_generation_context_is_malformed(
    monkeypatch,
):
    """Async kwargs-building failure must release its issued capability."""
    pipeline = _pipeline()
    issued = []
    prepare = pipeline.prepare_generation

    def capture(question: str):
        request = prepare(question)
        issued.append(request)
        return request

    monkeypatch.setattr(pipeline, "prepare_generation", capture)
    generator = _generator(pipeline=pipeline)

    async def malformed_context(*_args):
        return {}

    monkeypatch.setattr(generator, "_prepare_pipeline_async", malformed_context)

    result = await generator.generate_sql_async(
        "订单金额", auto_save_history=False
    )

    with pytest.raises(ValueError, match="invalid or expired"):
        pipeline.describe_prepared(issued[0])
    replay = pipeline.process_prepared(
        issued[0], "SELECT SUM(amount) FROM orders"
    )
    assert result["status"] == "error"
    assert result["error"] == "SQL 生成暂时不可用，请稍后重试"
    assert replay.status == "error"
    assert replay.executable is False


def test_generator_delegates_one_governed_repair_to_task_7_api():
    """Catches generator-local repair loops or use of the legacy raw-error repair API."""
    generator = _generator(
        pipeline=_pipeline(max_repair_attempts=1),
        candidate_sql="SELECT COUNT(*) FROM orders",
    )

    result = generator.generate_sql("订单金额", auto_save_history=False)

    assert result["status"] == "success"
    assert result["repair_attempts"] == 1
    generator.sql_chain.fix_sql_governed.assert_called_once()
    generator.sql_chain.fix_sql.assert_not_called()


def test_governed_history_is_saved_after_rejection_with_only_structured_codes():
    """Catches saving pre-governance success or promoting rejected SQL to Few-Shot."""
    mysql = Mock()
    mysql.save_sql_history.return_value = 17
    generator = _generator(
        pipeline=_pipeline(max_repair_attempts=0),
        mysql_client=mysql,
        candidate_sql="SELECT COUNT(*) FROM orders",
    )

    result = generator.generate_sql("订单金额")

    assert result["status"] == "rejected"
    assert result["executable"] is False
    saved = mysql.save_sql_history.call_args.kwargs
    assert saved["generated_sql"] == "SELECT COUNT(*) FROM orders"
    assert saved["validation_status"] == "fail"
    assert saved["validation_error"] == "METRIC_MISMATCH"
    generator.knowledge_retriever.add_history_record.assert_not_called()
    assert not generator._governed_history_bindings


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_governed_history_bindings_are_digest_only_and_lru_bounded():
    mysql = Mock()
    mysql.save_sql_history.side_effect = [11, 12, 13]
    generator = _generator(
        pipeline=_pipeline(),
        mysql_client=mysql,
        config_overrides={"history_max_records": 2},
    )
    generator.sql_chain.generate.side_effect = lambda **_kwargs: {
        "status": "success",
        "sql": "SELECT SUM(amount) FROM orders",
        "matched_knowledge": [],
    }

    results = [
        generator.generate_sql(f"订单金额 private-{index}", session_id=f"s-{index}")
        for index in range(3)
    ]

    assert [result["history_id"] for result in results] == [11, 12, 13]
    assert list(generator._governed_history_bindings) == [12, 13]
    stored = repr(tuple(generator._governed_history_bindings.values()))
    assert "SELECT" not in stored
    assert "订单金额" not in stored
    assert "private-" not in stored
    assert "s-" not in stored


def test_evicted_governed_history_binding_fails_closed():
    mysql = Mock()
    mysql.save_sql_history.side_effect = [11, 12]
    mysql.get_history_detail.return_value = _successful_history(
        sql="SELECT SUM(amount) FROM orders", session_id="session-a"
    ) | {"id": 11}
    generator = _generator(
        pipeline=_pipeline(),
        mysql_client=mysql,
        config_overrides={"history_max_records": 1},
    )
    generator.sql_chain.generate.side_effect = lambda **_kwargs: {
        "status": "success",
        "sql": "SELECT SUM(amount) FROM orders",
        "matched_knowledge": [],
    }
    first = generator.generate_sql("订单金额", session_id="session-a")
    generator.generate_sql("订单金额", session_id="session-b")
    mysql.reset_mock()
    mysql.get_history_detail.return_value = _successful_history(
        sql=first["sql"], session_id="session-a"
    ) | {"id": 11}

    result = generator.execute_sql(
        first["sql"],
        history_id=first["history_id"],
        session_id="session-a",
        allowed_tables=first["allowed_tables"],
        allowed_schemas=first["allowed_schemas"],
        user_query=first["user_query"],
        semantic_version=first["semantic_version"],
        content_sha256=first["content_sha256"],
    )

    assert result["status"] == "error"
    assert result["validation"]["issues"][0]["code"] == "HISTORY_CONTEXT_MISMATCH"
    mysql.execute_query_limited.assert_not_called()


def test_expired_governed_history_binding_is_pruned_and_fails_closed():
    clock = FakeClock()
    mysql = Mock()
    mysql.save_sql_history.return_value = 17
    generator = _generator(
        pipeline=_pipeline(),
        mysql_client=mysql,
        config_overrides={"governed_history_binding_ttl_seconds": 2},
    )
    generator._clock = clock
    generated = generator.generate_sql("订单金额", session_id="session-a")
    clock.advance(2.1)
    mysql.reset_mock()
    mysql.get_history_detail.return_value = _successful_history(sql=generated["sql"])

    result = generator.execute_sql(
        generated["sql"],
        history_id=generated["history_id"],
        session_id="session-a",
        allowed_tables=generated["allowed_tables"],
        allowed_schemas=generated["allowed_schemas"],
        user_query=generated["user_query"],
        semantic_version=generated["semantic_version"],
        content_sha256=generated["content_sha256"],
    )

    assert result["status"] == "error"
    assert not generator._governed_history_bindings
    mysql.execute_query_limited.assert_not_called()


def test_concurrent_governed_history_execution_consumes_binding_once():
    mysql = Mock()
    mysql.save_sql_history.return_value = 17
    mysql.get_history_detail.return_value = _successful_history(
        sql="SELECT SUM(amount) FROM orders"
    )
    mysql.update_sql_execution.return_value = True
    mysql.execute_query_limited.return_value = ([{"total": 10}], False)
    generator = _generator(pipeline=_pipeline(), mysql_client=mysql)
    generated = generator.generate_sql("订单金额", session_id="session-a")
    mysql.reset_mock()
    mysql.get_history_detail.return_value = _successful_history(sql=generated["sql"])
    mysql.update_sql_execution.return_value = True
    mysql.execute_query_limited.return_value = ([{"total": 10}], False)

    def execute(_index: int):
        return generator.execute_sql(
            generated["sql"],
            history_id=generated["history_id"],
            session_id="session-a",
            allowed_tables=generated["allowed_tables"],
            allowed_schemas=generated["allowed_schemas"],
            user_query=generated["user_query"],
            semantic_version=generated["semantic_version"],
            content_sha256=generated["content_sha256"],
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(execute, range(16)))

    assert sum(result["status"] == "success" for result in results) == 1
    assert mysql.execute_query_limited.call_count == 1
    assert mysql.update_sql_execution.call_count == 1


def _successful_history(*, sql: str, session_id: str = "session-a") -> dict:
    return {
        "id": 17,
        "session_id": session_id,
        "user_query": "订单金额",
        "generated_sql": sql,
        "validation_status": "success",
        "execution_status": "success",
    }


@pytest.mark.parametrize(
    ("execute_overrides", "record_overrides"),
    [
        ({"session_id": "session-b"}, {}),
        ({"sql": "SELECT SUM(amount) AS total FROM orders"}, {}),
        ({"user_query": "查询订单号"}, {}),
        ({"semantic_version": "9.9.9"}, {}),
        (
            {"sql": "SELECT SUM(amount) FROM orders"},
            {
                "generated_sql": "SELECT COUNT(*) FROM orders",
                "validation_status": "fail",
                "execution_status": "pending",
            },
        ),
    ],
    ids=[
        "cross_session",
        "sql_mismatch",
        "question_mismatch",
        "semantic_version_mismatch",
        "rejected_history_with_legal_sql",
    ],
)
def test_governed_execute_rejects_unbound_history_without_mutating_it(
    execute_overrides: dict, record_overrides: dict
):
    """A global history id cannot authorize or promote another request's SQL."""
    mysql = Mock()
    mysql.save_sql_history.return_value = 17
    mysql.get_history_detail.return_value = _successful_history(
        sql="SELECT SUM(amount) FROM orders"
    ) | record_overrides
    mysql.execute_query_limited.return_value = ([{"total": 10}], False)
    generator = _generator(pipeline=_pipeline(), mysql_client=mysql)
    generated = generator.generate_sql("订单金额", session_id="session-a")
    mysql.reset_mock()
    mysql.get_history_detail.return_value = _successful_history(
        sql="SELECT SUM(amount) FROM orders"
    ) | record_overrides

    arguments = {
        "sql": generated["sql"],
        "history_id": generated["history_id"],
        "session_id": "session-a",
        "allowed_tables": generated["allowed_tables"],
        "allowed_schemas": generated["allowed_schemas"],
        "user_query": generated["user_query"],
        "semantic_version": generated["semantic_version"],
        "content_sha256": generated["content_sha256"],
    }
    arguments.update(execute_overrides)
    if execute_overrides.get("semantic_version") == "9.9.9":
        generator.correctness_pipeline.registry.snapshot = _snapshot("9.9.9")
    result = generator.execute_sql(**arguments)

    assert result["status"] == "error"
    assert result["executable"] is False
    assert result["validation"]["issues"][0]["code"] == (
        "HISTORY_CONTEXT_MISMATCH"
    )
    mysql.execute_query_limited.assert_not_called()
    mysql.update_sql_execution.assert_not_called()
    generator.knowledge_retriever.add_history_record.assert_not_called()


def test_governed_success_updates_and_promotes_only_the_bound_history():
    mysql = Mock()
    mysql.save_sql_history.return_value = 17
    mysql.get_history_detail.return_value = _successful_history(
        sql="SELECT SUM(amount) FROM orders"
    )
    mysql.execute_query_limited.return_value = ([{"total": 10}], False)
    generator = _generator(pipeline=_pipeline(), mysql_client=mysql)
    generated = generator.generate_sql("订单金额", session_id="session-a")
    mysql.reset_mock()
    mysql.get_history_detail.return_value = _successful_history(sql=generated["sql"])
    mysql.execute_query_limited.return_value = ([{"total": 10}], False)
    mysql.update_sql_execution.return_value = True

    result = generator.execute_sql(
        generated["sql"],
        history_id=generated["history_id"],
        session_id="session-a",
        allowed_tables=generated["allowed_tables"],
        allowed_schemas=generated["allowed_schemas"],
        user_query=generated["user_query"],
        semantic_version=generated["semantic_version"],
        content_sha256=generated["content_sha256"],
    )

    assert result["status"] == "success"
    update = mysql.update_sql_execution.call_args.kwargs
    assert update["session_id"] == "session-a"
    assert update["generated_sql"] == generated["sql"]
    assert update["expected_user_query"] == "订单金额"
    assert update["require_validation_success"] is True
    generator.knowledge_retriever.add_history_record.assert_called_once_with(
        mysql.get_history_detail.return_value
    )


def test_history_question_changed_after_read_cannot_be_updated_or_promoted():
    """The conditional write must fail if user_query changes after the read."""
    mysql = Mock()
    mysql.save_sql_history.return_value = 17
    mysql.get_history_detail.return_value = _successful_history(
        sql="SELECT SUM(amount) FROM orders"
    )
    generator = _generator(pipeline=_pipeline(), mysql_client=mysql)
    generated = generator.generate_sql("订单金额", session_id="session-a")
    mysql.reset_mock()
    mysql.get_history_detail.return_value = _successful_history(sql=generated["sql"])
    mysql.execute_query_limited.return_value = ([{"total": 10}], False)
    # Simulates a row whose user_query changed after get_history_detail().
    mysql.update_sql_execution.return_value = False

    result = generator.execute_sql(
        generated["sql"],
        history_id=generated["history_id"],
        session_id="session-a",
        allowed_tables=generated["allowed_tables"],
        allowed_schemas=generated["allowed_schemas"],
        user_query=generated["user_query"],
        semantic_version=generated["semantic_version"],
        content_sha256=generated["content_sha256"],
    )

    assert result == {
        "status": "error",
        "error": "执行记录状态更新失败",
        "data": None,
    }
    update = mysql.update_sql_execution.call_args.kwargs
    assert update["expected_user_query"] == "订单金额"
    generator.knowledge_retriever.add_history_record.assert_not_called()


def test_restarted_generator_fails_closed_for_old_governed_history_id():
    """A process-local semantic binding cannot be inferred from legacy columns."""
    mysql = Mock()
    mysql.get_history_detail.return_value = _successful_history(
        sql="SELECT SUM(amount) FROM orders"
    )
    restarted = _generator(pipeline=_pipeline(), mysql_client=mysql)

    result = restarted.execute_sql(
        "SELECT SUM(amount) FROM orders",
        history_id=17,
        session_id="session-a",
        allowed_tables=["orders"],
        allowed_schemas=[],
        user_query="订单金额",
        semantic_version="1.0.0",
        content_sha256="1" * 64,
    )

    assert result["status"] == "error"
    assert result["validation"]["issues"][0]["code"] == (
        "HISTORY_CONTEXT_MISMATCH"
    )
    mysql.execute_query_limited.assert_not_called()
    mysql.update_sql_execution.assert_not_called()
    restarted.knowledge_retriever.add_history_record.assert_not_called()


def test_disabled_governance_preserves_legacy_result_shape_and_chain_call():
    """Catches governance fields or new prompt arguments leaking into default-off mode."""
    generator = _generator(pipeline=None)

    result = generator.generate_sql("查询订单", auto_save_history=False)

    assert "sql" in result
    assert "semantic_version" not in result
    assert "repair_attempts" not in result
    assert "executable" not in result
    assert "user_query" not in result
    assert "semantic_context" not in generator.sql_chain.generate.call_args.kwargs


def test_governed_execute_requires_original_question_and_semantic_version():
    """Catches execution authorization without the generation-time semantic context."""
    mysql = Mock()
    generator = _generator(pipeline=_pipeline(), mysql_client=mysql)

    result = generator.execute_sql("SELECT SUM(amount) FROM orders")

    assert result["status"] == "error"
    assert "原问题和语义版本" in result["error"]
    assert result["validation"]["issues"][0]["code"] == "EXECUTION_CONTEXT_REQUIRED"
    mysql.execute_query_limited.assert_not_called()


def test_governed_execute_requires_generation_content_hash():
    """The optional compatibility argument must fail closed when governance is active."""
    mysql = Mock()
    generator = _generator(pipeline=_pipeline(), mysql_client=mysql)

    result = generator.execute_sql(
        "SELECT SUM(amount) FROM orders",
        user_query="订单金额",
        semantic_version="1.0.0",
    )

    assert result["status"] == "error"
    assert result["executable"] is False
    assert result["validation"]["issues"][0]["code"] == "EXECUTION_CONTEXT_REQUIRED"
    mysql.execute_query_limited.assert_not_called()


def test_governed_history_binding_rejects_content_hash_substitution():
    mysql = Mock()
    mysql.save_sql_history.return_value = 17
    mysql.get_history_detail.return_value = _successful_history(
        sql="SELECT SUM(amount) FROM orders"
    )
    generator = _generator(pipeline=_pipeline(), mysql_client=mysql)
    generated = generator.generate_sql("订单金额", session_id="session-a")
    mysql.reset_mock()

    result = generator.execute_sql(
        generated["sql"],
        history_id=generated["history_id"],
        session_id="session-a",
        allowed_tables=generated["allowed_tables"],
        allowed_schemas=generated["allowed_schemas"],
        user_query=generated["user_query"],
        semantic_version=generated["semantic_version"],
        content_sha256="2" * 64,
    )

    assert result["status"] == "error"
    assert result["executable"] is False
    assert result["validation"]["issues"][0]["code"] in {
        "SEMANTIC_CONTENT_MISMATCH",
        "HISTORY_CONTEXT_MISMATCH",
    }
    mysql.execute_query_limited.assert_not_called()


def test_execute_reauthorizes_before_applying_row_and_timeout_constraints(monkeypatch):
    """Catches LIMIT/timeout rewriting or database access before fresh authorization."""
    import core.sql.generator as generator_module

    events: list[str] = []
    pipeline = Mock()

    def authorize(**_kwargs):
        events.append("authorize")
        return CorrectnessResult(
            status="success",
            sql="SELECT SUM(amount) FROM orders",
            semantic_version="1.0.0",
            validation=ValidationReport(valid=True),
            repair_attempts=0,
            executable=True,
        )

    pipeline.authorize_execution.side_effect = authorize
    mysql = Mock()
    mysql.execute_query_limited.side_effect = lambda *_args, **_kwargs: (
        events.append("execute") or ([{"total": 3}], False)
    )
    generator = _generator(pipeline=pipeline, mysql_client=mysql)

    def constrain(sql: str, **_kwargs) -> str:
        events.append("constrain")
        return f"{sql} LIMIT 6"

    monkeypatch.setattr(generator_module, "constrain_read_query", constrain)

    result = generator.execute_sql(
        "SELECT SUM(amount) FROM orders",
        allowed_tables=["orders"],
        allowed_schemas=[],
        user_query="订单金额",
        semantic_version="1.0.0",
        content_sha256="1" * 64,
    )

    assert result["status"] == "success"
    assert events == ["authorize", "constrain", "execute"]
    pipeline.authorize_execution.assert_called_once_with(
        question="订单金额",
        sql="SELECT SUM(amount) FROM orders",
        expected_semantic_version="1.0.0",
        expected_content_sha256="1" * 64,
        allowed_tables=("orders",),
        allowed_schemas=(),
    )


def test_governed_execute_authorizes_before_history_read_and_database_execute(
    monkeypatch,
):
    """No history or business database access may precede authorization."""
    events: list[str] = []
    pipeline = _pipeline()
    authorize = pipeline.authorize_execution

    def observed_authorize(**kwargs):
        events.append("authorize")
        return authorize(**kwargs)

    monkeypatch.setattr(pipeline, "authorize_execution", observed_authorize)
    mysql = Mock()
    mysql.save_sql_history.return_value = 17
    generator = _generator(pipeline=pipeline, mysql_client=mysql)
    generated = generator.generate_sql("订单金额", session_id="session-a")
    mysql.reset_mock()

    def read_history(_history_id: int):
        events.append("history-read")
        return _successful_history(sql=generated["sql"])

    mysql.get_history_detail.side_effect = read_history
    mysql.update_sql_execution.return_value = True
    mysql.execute_query_limited.side_effect = lambda *_args, **_kwargs: (
        events.append("execute") or ([{"total": 10}], False)
    )

    result = generator.execute_sql(
        generated["sql"],
        history_id=generated["history_id"],
        session_id="session-a",
        allowed_tables=generated["allowed_tables"],
        allowed_schemas=generated["allowed_schemas"],
        user_query=generated["user_query"],
        semantic_version=generated["semantic_version"],
        content_sha256=generated["content_sha256"],
    )

    assert result["status"] == "success"
    assert events[:3] == ["authorize", "history-read", "execute"]


def test_failed_authorization_never_reads_or_updates_history(monkeypatch):
    """A correctness denial must return before touching the history database."""
    events: list[str] = []
    pipeline = _pipeline()
    mysql = Mock()
    mysql.save_sql_history.return_value = 17
    generator = _generator(pipeline=pipeline, mysql_client=mysql)
    generated = generator.generate_sql("订单金额", session_id="session-a")
    mysql.reset_mock()

    def deny(**_kwargs):
        events.append("authorize")
        return CorrectnessResult(
            status="rejected",
            sql=generated["sql"],
            semantic_version=generated["semantic_version"],
            validation=ValidationReport(
                valid=False,
                issues=(
                    ValidationIssue(
                        code="POLICY_DENIED",
                        message="查询不符合语义策略",
                    ),
                ),
            ),
            repair_attempts=0,
            executable=False,
        )

    monkeypatch.setattr(pipeline, "authorize_execution", deny)
    mysql.get_history_detail.side_effect = lambda _history_id: (
        events.append("history-read")
        or _successful_history(sql=generated["sql"])
    )

    result = generator.execute_sql(
        generated["sql"],
        history_id=generated["history_id"],
        session_id="session-a",
        allowed_tables=generated["allowed_tables"],
        allowed_schemas=generated["allowed_schemas"],
        user_query=generated["user_query"],
        semantic_version=generated["semantic_version"],
        content_sha256=generated["content_sha256"],
    )

    assert result["status"] == "error"
    assert events == ["authorize"]
    mysql.get_history_detail.assert_not_called()
    mysql.update_sql_execution.assert_not_called()
    mysql.execute_query_limited.assert_not_called()


def test_failed_reauthorization_returns_structured_issues_without_database_access():
    """Catches version/semantic rejection being flattened or ignored by execution."""
    pipeline = Mock()
    pipeline.authorize_execution.return_value = CorrectnessResult(
        status="rejected",
        sql="SELECT SUM(amount) FROM orders",
        semantic_version="1.1.0",
        validation=ValidationReport(
            valid=False,
            issues=(
                ValidationIssue(
                    code="SEMANTIC_VERSION_MISMATCH",
                    message="SQL 生成版本与当前语义版本不一致",
                ),
            ),
        ),
        repair_attempts=0,
        executable=False,
    )
    mysql = Mock()
    generator = _generator(pipeline=pipeline, mysql_client=mysql)

    result = generator.execute_sql(
        "SELECT SUM(amount) FROM orders",
        user_query="订单金额",
        semantic_version="1.0.0",
        content_sha256="1" * 64,
    )

    assert result["status"] == "error"
    assert result["executable"] is False
    assert result["validation"]["issues"][0]["code"] == (
        "SEMANTIC_VERSION_MISMATCH"
    )
    mysql.execute_query_limited.assert_not_called()


@pytest.mark.parametrize(
    "updates,variable",
    [
        (
            {"SEMANTIC_GOVERNANCE_ENABLED": "true", "SQL_REPAIR_MAX_ATTEMPTS": "2"},
            "SQL_REPAIR_MAX_ATTEMPTS",
        ),
        ({"SEMANTIC_CONTEXT_MAX_CHARS": "0"}, "SEMANTIC_CONTEXT_MAX_CHARS"),
        ({"SEMANTIC_CONTEXT_MAX_ITEMS": "0"}, "SEMANTIC_CONTEXT_MAX_ITEMS"),
        ({"SQL_EXPLAIN_TIMEOUT_MS": "0"}, "SQL_EXPLAIN_TIMEOUT_MS"),
        ({"SQL_EXPLAIN_MAX_ESTIMATED_ROWS": "-1"}, "SQL_EXPLAIN_MAX_ESTIMATED_ROWS"),
    ],
)
def test_invalid_governance_environment_fails_import(updates: dict, variable: str):
    """Catches invalid governance configuration being downgraded to a warning."""
    result = _import_config(**updates)

    assert result.returncode != 0
    assert variable in result.stdout + result.stderr


def test_explain_enabled_without_governance_fails_environment_validation():
    """EXPLAIN is a correctness stage and must never be silently ignored."""
    result = _import_config(
        SEMANTIC_GOVERNANCE_ENABLED="false",
        SQL_EXPLAIN_ENABLED="true",
        LLM_API_KEY="do-not-leak-this-secret",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "SQL_EXPLAIN_ENABLED" in output
    assert "SEMANTIC_GOVERNANCE_ENABLED" in output
    assert "do-not-leak-this-secret" not in output


def test_validate_config_exception_names_the_invalid_explain_dependency(monkeypatch):
    import config

    monkeypatch.setattr(config, "SQL_EXPLAIN_ENABLED", True)
    monkeypatch.setattr(config, "SEMANTIC_GOVERNANCE_ENABLED", False)

    with pytest.raises(ValueError) as error:
        config.validate_config()

    message = str(error.value)
    assert "SQL_EXPLAIN_ENABLED" in message
    assert "SEMANTIC_GOVERNANCE_ENABLED" in message


def test_governance_configuration_defaults_are_exact():
    """Catches changing a compatibility or safety default during environment parsing."""
    names = (
        "SEMANTIC_GOVERNANCE_ENABLED",
        "SEMANTIC_MODEL_DIR",
        "SEMANTIC_DATASOURCE",
        "SEMANTIC_CONTEXT_MAX_CHARS",
        "SEMANTIC_CONTEXT_MAX_ITEMS",
        "SQL_REPAIR_MAX_ATTEMPTS",
        "SQL_EXPLAIN_ENABLED",
        "SQL_EXPLAIN_TIMEOUT_MS",
        "SQL_EXPLAIN_MAX_ESTIMATED_ROWS",
    )
    env = os.environ.copy()
    for name in names:
        env.pop(name, None)
    env.update({"LLM_PROVIDER": "ollama", "EMBEDDING_PROVIDER": "ollama"})
    script = """
import json
import config
names = %r
print(json.dumps({name: getattr(config, name) for name in names}, sort_keys=True))
""" % (names,)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(result.stdout.strip()) == {
        "SEMANTIC_GOVERNANCE_ENABLED": False,
        "SEMANTIC_MODEL_DIR": "semantic_models",
        "SEMANTIC_DATASOURCE": "default",
        "SEMANTIC_CONTEXT_MAX_CHARS": 2400,
        "SEMANTIC_CONTEXT_MAX_ITEMS": 24,
        "SQL_REPAIR_MAX_ATTEMPTS": 1,
        "SQL_EXPLAIN_ENABLED": False,
        "SQL_EXPLAIN_TIMEOUT_MS": 3000,
        "SQL_EXPLAIN_MAX_ESTIMATED_ROWS": 0,
    }


def test_create_app_context_from_env_threads_all_governance_values(monkeypatch):
    """Catches parsed governance settings being dropped at the construction boundary."""
    import config
    from core.app_context import create_app_context_from_env

    expected = {
        "SEMANTIC_GOVERNANCE_ENABLED": True,
        "SEMANTIC_MODEL_DIR": "/tmp/models",
        "SEMANTIC_DATASOURCE": "sales",
        "SEMANTIC_CONTEXT_MAX_CHARS": 1700,
        "SEMANTIC_CONTEXT_MAX_ITEMS": 13,
        "SQL_REPAIR_MAX_ATTEMPTS": 0,
        "SQL_EXPLAIN_ENABLED": True,
        "SQL_EXPLAIN_TIMEOUT_MS": 900,
        "SQL_EXPLAIN_MAX_ESTIMATED_ROWS": 8000,
    }
    for name, value in expected.items():
        monkeypatch.setattr(config, name, value, raising=False)

    context = create_app_context_from_env()

    assert {
        "semantic_governance_enabled": True,
        "semantic_model_dir": "/tmp/models",
        "semantic_datasource": "sales",
        "semantic_context_max_chars": 1700,
        "semantic_context_max_items": 13,
        "sql_repair_max_attempts": 0,
        "sql_explain_enabled": True,
        "sql_explain_timeout_ms": 900,
        "sql_explain_max_estimated_rows": 8000,
    }.items() <= context.config.items()


def test_governance_reload_builds_candidate_then_atomically_replaces_pair(
    tmp_path: Path, monkeypatch
):
    """Catches in-place mutation of the live registry before candidate validation."""
    from core.app_context import ApplicationContext

    source = tmp_path / "sales.yaml"
    _write_model(source)
    context = ApplicationContext({"semantic_governance_enabled": False})
    context._llm = Mock()
    context._knowledge_retriever = Mock()
    context._sql_chain = Mock()
    context._mysql_client = Mock()
    old_registry = object()
    old_generator = object()
    context._semantic_registry = old_registry
    context._correctness_pipeline = object()
    context._sql_generator = old_generator
    candidate_generator = object()
    build_generator = Mock(return_value=candidate_generator)
    monkeypatch.setattr(
        context, "_build_sql_generator", build_generator, raising=False
    )

    context.reload_governance(**_governance_config(tmp_path))

    assert context.semantic_registry is not old_registry
    assert context.semantic_registry.get_snapshot().model_version == "1.0.0"
    assert context.correctness_pipeline.registry is context.semantic_registry
    assert context.sql_generator is candidate_generator
    assert build_generator.call_args.kwargs["correctness_pipeline"] is (
        context.correctness_pipeline
    )


def test_governance_pipeline_receives_configured_sql_max_length(
    tmp_path: Path, monkeypatch
):
    from core.app_context import ApplicationContext

    _write_model(tmp_path / "sales.yaml")
    context = ApplicationContext({"semantic_governance_enabled": False})
    context._mysql_client = Mock()
    context._llm = Mock()
    context._knowledge_retriever = Mock()
    context._sql_chain = Mock()
    monkeypatch.setattr(context, "_build_sql_generator", Mock(return_value=object()))

    context.reload_governance(
        **(_governance_config(tmp_path) | {"sql_max_length": 37})
    )

    assert context.correctness_pipeline.max_sql_length == 37


def test_failed_governance_reload_keeps_old_registry_generator_and_config(
    tmp_path: Path,
):
    """Catches a failed candidate reload corrupting the live governance pair."""
    from core.app_context import ApplicationContext, ApplicationConfigurationError

    (tmp_path / "sales.yaml").write_text(
        "password=secret invalid yaml: [", encoding="utf-8"
    )
    context = ApplicationContext({"semantic_governance_enabled": False})
    old_registry = object()
    old_pipeline = object()
    old_generator = object()
    context._semantic_registry = old_registry
    context._correctness_pipeline = old_pipeline
    context._sql_generator = old_generator
    context._mysql_client = Mock()

    with pytest.raises(ApplicationConfigurationError, match="语义治理配置无效"):
        context.reload_governance(**_governance_config(tmp_path))

    assert context.config == {"semantic_governance_enabled": False}
    assert context.semantic_registry is old_registry
    assert context.correctness_pipeline is old_pipeline
    assert context.sql_generator is old_generator


def test_reload_rejects_explain_without_governance_and_keeps_live_components():
    from core.app_context import ApplicationContext, ApplicationConfigurationError

    context = ApplicationContext({"semantic_governance_enabled": False})
    old_registry = object()
    old_pipeline = object()
    old_generator = object()
    context._semantic_registry = old_registry
    context._correctness_pipeline = old_pipeline
    context._sql_generator = old_generator

    with pytest.raises(ApplicationConfigurationError) as error:
        context.reload_governance(
            sql_explain_enabled=True,
            llm_api_key="do-not-leak-this-secret",
        )

    message = str(error.value)
    assert "SQL_EXPLAIN_ENABLED" in message
    assert "SEMANTIC_GOVERNANCE_ENABLED" in message
    assert "do-not-leak-this-secret" not in message
    assert context.config == {"semantic_governance_enabled": False}
    assert context.semantic_registry is old_registry
    assert context.correctness_pipeline is old_pipeline
    assert context.sql_generator is old_generator


def test_initial_governance_failure_exposes_only_stable_configuration_error(
    tmp_path: Path, monkeypatch, caplog
):
    """Catches startup surfacing compiler content or leaving partial components live."""
    from core.app_context import ApplicationContext

    (tmp_path / "sales.yaml").write_text(
        "password=secret invalid yaml: [", encoding="utf-8"
    )
    context = ApplicationContext(_governance_config(tmp_path))
    for method in (
        "_init_llm",
        "_init_neo4j",
        "_init_mysql",
        "_init_embedding",
        "_init_retriever",
        "_init_sql_chain",
        "_init_sql_generator",
        "_load_knowledge_base",
        "_load_few_shot_history",
    ):
        monkeypatch.setattr(context, method, Mock())

    with capture_loguru(caplog):
        initialized = context.ensure_initialized()

    assert initialized is False
    assert context.initialization_error == "语义治理配置无效"
    assert "secret" not in caplog.text
    assert "invalid yaml" not in caplog.text
    assert "Traceback" not in caplog.text


def test_ui_helpers_disable_rejected_sql_and_ignore_unstructured_error_details():
    """Catches UI execution enablement or rendering of raw database error strings."""
    from core.utils.correctness_ui import (
        execution_is_disabled,
        public_failure_messages,
        structured_validation_messages,
    )

    result = {
        "executable": False,
        "error": "password=secret database=private",
        "validation": {
            "valid": False,
            "issues": [
                {
                    "code": "POLICY_DENIED",
                    "message": "字段不允许直接查询",
                    "objects": ["customers.mobile"],
                    "stage": "semantic",
                }
            ],
        },
    }

    assert execution_is_disabled(result) is True
    assert execution_is_disabled({"sql": "SELECT 1"}) is False
    assert structured_validation_messages(result) == [
        "[POLICY_DENIED] 字段不允许直接查询"
    ]
    assert public_failure_messages(result) == [
        "[POLICY_DENIED] 字段不允许直接查询"
    ]
    assert public_failure_messages({"error": None}) == ["未知错误"]


def test_ui_failure_messages_render_correctness_unavailable_without_raw_error():
    from core.utils.correctness_ui import public_failure_messages

    result = {
        "error": "password=secret SELECT private FROM hidden",
        "validation": {
            "issues": [
                {
                    "code": "CORRECTNESS_UNAVAILABLE",
                    "message": "SQL 正确性校验暂时不可用",
                }
            ]
        },
    }

    assert public_failure_messages(result) == [
        "[CORRECTNESS_UNAVAILABLE] SQL 正确性校验暂时不可用"
    ]


def test_ui_failure_helpers_do_not_render_untrusted_bearer_text():
    from core.utils.correctness_ui import public_failure_messages

    result = {
        "error": "Authorization: Bearer ui-secret",
        "validation": {
            "issues": [
                {
                    "code": "CORRECTNESS_UNAVAILABLE",
                    "message": "Authorization: Bearer issue-secret",
                }
            ]
        },
    }

    rendered = " ".join(public_failure_messages(result))
    assert rendered == "[CORRECTNESS_UNAVAILABLE] SQL 正确性校验暂时不可用"
    assert "Bearer" not in rendered
    assert "secret" not in rendered


def test_execute_exception_log_uses_safe_metadata_without_sql_or_traceback(caplog):
    """Catches execute_sql logging traceback, exception text, SQL, or database details."""
    mysql = Mock()
    mysql.get_table_schema.return_value = [{"table_name": "private_orders"}]
    mysql.execute_query_limited.side_effect = RuntimeError(
        "password=secret database=prod_private"
    )
    generator = _generator(pipeline=None, mysql_client=mysql)
    sql = "SELECT visible_amount FROM private_orders"

    with capture_loguru(caplog):
        result = generator.execute_sql(sql)

    assert result["status"] == "fail"
    assert "secret" not in caplog.text
    assert "prod_private" not in caplog.text
    assert "visible_amount" not in caplog.text
    assert "private_orders" not in caplog.text
    assert "Traceback" not in caplog.text
    assert "stage=sql_execute" in caplog.text
    assert "exception_type=RuntimeError" in caplog.text
    assert re.search(r"error_id=[0-9a-f]{32}", caplog.text)
