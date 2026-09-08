from __future__ import annotations

import pytest

from core.semantic.models import (
    SemanticEntity,
    SemanticField,
    SemanticMetric,
    SemanticPolicies,
    SemanticRelationship,
    SemanticSnapshot,
)
from core.semantic.validator import SQLScope, SemanticSQLValidator, parse_sql_scope


@pytest.fixture
def sales_snapshot() -> SemanticSnapshot:
    return SemanticSnapshot(
        schema_version=1,
        model_version="1.0.0",
        datasource="sales",
        dialect="mysql",
        entities=(
            SemanticEntity(
                name="orders",
                name_cn="订单",
                fields=(
                    SemanticField(
                        name="id", name_cn="订单号", data_type="bigint", role="identifier"
                    ),
                    SemanticField(
                        name="amount", name_cn="金额", data_type="decimal", role="measure"
                    ),
                    SemanticField(
                        name="created_at",
                        name_cn="下单时间",
                        data_type="datetime",
                        role="time",
                        synonyms=("日期",),
                    ),
                    SemanticField(
                        name="status", name_cn="状态", data_type="varchar", role="dimension"
                    ),
                    SemanticField(
                        name="customer_id",
                        name_cn="客户编号",
                        data_type="bigint",
                        role="identifier",
                    ),
                ),
            ),
            SemanticEntity(
                name="customers",
                name_cn="客户",
                fields=(
                    SemanticField(
                        name="id", name_cn="客户编号", data_type="bigint", role="identifier"
                    ),
                    SemanticField(
                        name="name", name_cn="客户名称", data_type="varchar", role="dimension"
                    ),
                    SemanticField(
                        name="mobile",
                        name_cn="手机号",
                        data_type="varchar",
                        role="dimension",
                        sensitivity="restricted",
                        directly_queryable=False,
                    ),
                ),
            ),
        ),
        relationships=(
            SemanticRelationship(
                name="order_customer",
                from_entity="orders",
                from_field="customer_id",
                to_entity="customers",
                to_field="id",
            ),
        ),
        metrics=(
            SemanticMetric(
                name="order_amount",
                name_cn="订单金额",
                synonyms=("销售额",),
                expression="SUM(orders.amount)",
                allowed_dimensions=("orders.created_at",),
                required_filters=("orders.status",),
                default_time_field="orders.created_at",
            ),
        ),
        policies=SemanticPolicies(denied_fields=("customers.mobile",)),
        source_sha256="source",
        content_sha256="content",
        compiler_version="1",
        compiled_at="2026-09-09T00:00:00Z",
    )


@pytest.fixture
def validator() -> SemanticSQLValidator:
    return SemanticSQLValidator()


@pytest.mark.parametrize(
    "sql,code",
    [
        ("SELECT * FROM ghosts", "UNKNOWN_TABLE"),
        ("SELECT o.missing FROM orders o", "UNKNOWN_COLUMN"),
        (
            "SELECT id FROM orders o JOIN customers c ON o.customer_id = c.id",
            "AMBIGUOUS_COLUMN",
        ),
    ],
)
def test_rejects_invalid_object_references(
    validator: SemanticSQLValidator,
    sales_snapshot: SemanticSnapshot,
    sql: str,
    code: str,
):
    """Break caught: accepting an unknown or ambiguous object reference."""
    report = validator.validate(sql, "查询订单", sales_snapshot)

    assert not report.valid
    assert code in {issue.code for issue in report.issues}


@pytest.mark.parametrize(
    "question,sql,code",
    [
        (
            "订单和客户",
            "SELECT * FROM orders o JOIN customers c ON o.id = c.id",
            "INVALID_JOIN",
        ),
        ("订单金额", "SELECT COUNT(*) FROM orders", "METRIC_MISMATCH"),
        ("按月统计订单金额", "SELECT SUM(amount) FROM orders", "TIME_DIMENSION_REQUIRED"),
        ("客户手机号", "SELECT mobile FROM customers", "POLICY_DENIED"),
    ],
)
def test_rejects_semantic_rule_violations(
    validator: SemanticSQLValidator,
    sales_snapshot: SemanticSnapshot,
    question: str,
    sql: str,
    code: str,
):
    """Break caught: allowing SQL that violates a governed semantic rule."""
    report = validator.validate(sql, question, sales_snapshot)

    assert not report.valid
    assert code in {issue.code for issue in report.issues}


def test_scope_resolves_aliases_and_ctes_without_a_database(sales_snapshot: SemanticSnapshot):
    """Break caught: treating a CTE name as a physical table or losing its field provenance."""
    indexes = {
        entity.name: tuple(field.name for field in entity.fields)
        for entity in sales_snapshot.entities
    }

    scope = parse_sql_scope(
        "WITH recent AS (SELECT id, amount FROM orders) "
        "SELECT r.id, SUM(r.amount) FROM recent r GROUP BY r.id",
        dialect="mysql",
        semantic_indexes=indexes,
    )

    assert isinstance(scope, SQLScope)
    assert scope.tables == frozenset({"orders"})
    assert scope.aliases["r"] == "orders"
    assert ("orders", "id") in scope.columns
    assert ("orders", "amount") in scope.columns
    assert scope.group_by == ("orders.id",)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(ORDERS.AMOUNT) FROM ORDERS",
        "SELECT SUM(o.amount) FROM orders o",
        "WITH totals AS (SELECT SUM(amount) AS value FROM orders) SELECT value FROM totals",
    ],
)
def test_metric_matching_accepts_structurally_equivalent_expressions(
    validator: SemanticSQLValidator, sales_snapshot: SemanticSnapshot, sql: str
):
    """Break caught: comparing metric SQL by casing, qualifier spelling, or raw text."""
    report = validator.validate(sql, "订单金额", sales_snapshot)

    assert report.valid, report.issues


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT AVG(amount) FROM orders",
        "SELECT SUM(id) FROM orders",
        "SELECT SUM(c.id) FROM customers c",
    ],
)
def test_metric_matching_rejects_wrong_aggregate_or_source_field(
    validator: SemanticSQLValidator, sales_snapshot: SemanticSnapshot, sql: str
):
    """Break caught: accepting any aggregate when a governed metric term is matched."""
    report = validator.validate(sql, "订单金额", sales_snapshot)

    assert {issue.code for issue in report.issues} == {"METRIC_MISMATCH"}


def test_rejects_grouping_by_a_dimension_not_allowed_for_the_metric(
    validator: SemanticSQLValidator, sales_snapshot: SemanticSnapshot
):
    """Break caught: ignoring a matched metric's allowed-dimensions allowlist."""
    report = validator.validate(
        "SELECT status, SUM(amount) FROM orders GROUP BY status",
        "按状态统计订单金额",
        sales_snapshot,
    )

    assert "DIMENSION_NOT_ALLOWED" in {issue.code for issue in report.issues}


def test_accepts_an_allowed_dimension_inside_a_grouping_expression(
    validator: SemanticSQLValidator, sales_snapshot: SemanticSnapshot
):
    """Break caught: rejecting an allowed base field when wrapped in a time function."""
    report = validator.validate(
        "SELECT DATE_FORMAT(created_at, '%Y-%m'), SUM(amount) "
        "FROM orders GROUP BY DATE_FORMAT(created_at, '%Y-%m')",
        "按月统计订单金额",
        sales_snapshot,
    )

    assert report.valid, report.issues


@pytest.mark.parametrize("wording", ["按日", "按天", "按周", "按月", "按季度", "按年"])
def test_chinese_time_granularity_requires_the_governed_time_field(
    validator: SemanticSQLValidator,
    sales_snapshot: SemanticSnapshot,
    wording: str,
):
    """Break caught: recognizing only one Chinese time-granularity phrase."""
    missing = validator.validate(
        "SELECT SUM(amount) FROM orders", f"{wording}统计订单金额", sales_snapshot
    )
    present = validator.validate(
        "SELECT created_at, SUM(amount) FROM orders GROUP BY created_at",
        f"{wording}统计订单金额",
        sales_snapshot,
    )

    assert "TIME_DIMENSION_REQUIRED" in {issue.code for issue in missing.issues}
    assert present.valid, present.issues


@pytest.mark.parametrize("question", ["查询2026年1月订单金额", "查询订单金额日期范围", "从1月到3月的订单金额"])
def test_explicit_date_or_range_intent_requires_a_time_field(
    validator: SemanticSQLValidator,
    sales_snapshot: SemanticSnapshot,
    question: str,
):
    """Break caught: overlooking explicit date and range wording outside 按月 patterns."""
    missing = validator.validate("SELECT SUM(amount) FROM orders", question, sales_snapshot)
    present = validator.validate(
        "SELECT SUM(amount) FROM orders WHERE created_at >= '2026-01-01'",
        question,
        sales_snapshot,
    )

    assert "TIME_DIMENSION_REQUIRED" in {issue.code for issue in missing.issues}
    assert present.valid, present.issues


def test_policy_rejects_denied_entities_and_non_queryable_sensitive_fields(
    validator: SemanticSQLValidator, sales_snapshot: SemanticSnapshot
):
    """Break caught: enforcing only explicit denied-fields entries and missing other policy paths."""
    denied_entity_snapshot = sales_snapshot.model_copy(
        update={"policies": SemanticPolicies(denied_entities=("customers",))}
    )
    sensitive_only_snapshot = sales_snapshot.model_copy(update={"policies": SemanticPolicies()})

    denied_entity = validator.validate(
        "SELECT name FROM customers", "客户名称", denied_entity_snapshot
    )
    sensitive_field = validator.validate(
        "SELECT mobile FROM customers", "客户手机号", sensitive_only_snapshot
    )

    assert "POLICY_DENIED" in {issue.code for issue in denied_entity.issues}
    assert "POLICY_DENIED" in {issue.code for issue in sensitive_field.issues}


def test_cte_star_checks_only_fields_exposed_by_the_cte(
    validator: SemanticSQLValidator, sales_snapshot: SemanticSnapshot
):
    """Break caught: expanding a CTE star to restricted fields the CTE did not project."""
    report = validator.validate(
        "WITH joined AS ("
        "SELECT o.amount, c.name FROM orders o "
        "JOIN customers c ON o.customer_id = c.id"
        ") SELECT * FROM joined",
        "订单和客户",
        sales_snapshot,
    )

    assert report.valid, report.issues


def test_accepts_only_the_governed_join_key_in_either_direction(
    validator: SemanticSQLValidator, sales_snapshot: SemanticSnapshot
):
    """Break caught: checking only table pairs or only one orientation of a governed join."""
    forward = validator.validate(
        "SELECT o.amount, c.name FROM orders o "
        "JOIN customers c ON o.customer_id = c.id",
        "订单和客户",
        sales_snapshot,
    )
    reverse = validator.validate(
        "SELECT o.amount, c.name FROM customers c "
        "JOIN orders o ON c.id = o.customer_id",
        "订单和客户",
        sales_snapshot,
    )

    assert forward.valid, forward.issues
    assert reverse.valid, reverse.issues


def test_parse_errors_fail_closed_as_a_syntax_issue(
    validator: SemanticSQLValidator, sales_snapshot: SemanticSnapshot
):
    """Break caught: raising or returning a valid report when sqlglot cannot parse SQL."""
    report = validator.validate("SELECT FROM", "查询订单", sales_snapshot)

    assert not report.valid
    assert tuple((issue.stage, issue.code) for issue in report.issues) == (
        ("syntax", "SYNTAX_ERROR"),
    )


def test_issues_are_deduplicated_and_sorted_by_stage_code_and_objects(
    validator: SemanticSQLValidator, sales_snapshot: SemanticSnapshot
):
    """Break caught: leaking traversal order or duplicate references into repair diagnostics."""
    report = validator.validate(
        "SELECT id, mobile, mobile FROM customers c "
        "JOIN orders o ON c.id = o.id GROUP BY mobile",
        "按月统计订单金额",
        sales_snapshot,
    )

    keys = tuple((issue.stage, issue.code, issue.objects) for issue in report.issues)
    assert keys == tuple(sorted(set(keys)))
    assert [issue.code for issue in report.issues].count("POLICY_DENIED") == 1
