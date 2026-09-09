from pathlib import Path

import pytest

from core.semantic.compiler import SemanticCompiler
from core.semantic.context import SemanticContextBuilder, SemanticContextError
from core.semantic.models import SemanticMetric, SemanticSnapshot


@pytest.fixture
def sales_snapshot(tmp_path: Path) -> SemanticSnapshot:
    """A metric whose complete dependency bundle includes governance and a join."""
    source = tmp_path / "sales.yaml"
    source.write_text(
        """
schema_version: 1
model_version: 1.0.0
datasource: sales
dialect: mysql
entities:
  - name: orders
    name_cn: 订单
    fields:
      - {name: id, name_cn: 订单号, data_type: bigint, role: identifier}
      - {name: amount, name_cn: 金额, data_type: decimal, role: measure}
      - {name: created_at, name_cn: 下单时间, data_type: datetime, role: time, synonyms: [月份]}
      - {name: status, name_cn: 状态, data_type: varchar, role: dimension}
      - {name: customer_id, name_cn: 客户编号, data_type: bigint, role: identifier}
  - name: customers
    name_cn: 客户
    fields:
      - {name: id, name_cn: 客户编号, data_type: bigint, role: identifier}
      - {name: mobile, name_cn: 手机号, data_type: varchar, role: dimension, sensitivity: restricted, directly_queryable: false}
relationships:
  - {name: order_customer, from_entity: orders, from_field: customer_id, to_entity: customers, to_field: id}
metrics:
  - name: order_amount
    name_cn: 订单金额
    synonyms: [销售额]
    expression: SUM(orders.amount)
    allowed_dimensions: [orders.created_at]
    required_filters: [orders.status]
    default_time_field: orders.created_at
terms:
  - {term: 成交额, target: order_amount, when: "orders.status = 'paid'"}
policies:
  denied_fields: [customers.mobile]
""",
        encoding="utf-8",
    )
    return SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z").compile_file(
        source, tmp_path / ".compiled"
    )


def test_context_prioritizes_metric_dependencies(sales_snapshot: SemanticSnapshot):
    """Break caught: truncating a matched metric's fields, join, or policy before lower-priority hits."""
    context = SemanticContextBuilder(max_chars=800, max_items=8).build(
        "按月份统计订单金额", sales_snapshot
    )

    assert "订单金额" in context.rendered
    assert "orders.amount" in context.rendered
    assert "orders.created_at" in context.rendered
    assert "orders.status" in context.rendered
    assert "orders.customer_id = customers.id [LEFT JOIN]" in context.rendered
    assert "禁止字段 customers.mobile" in context.rendered
    assert len(context.rendered) <= 800


def test_context_is_deterministic(sales_snapshot: SemanticSnapshot):
    """Break caught: iterating semantic objects in a non-deterministic order."""
    builder = SemanticContextBuilder(max_chars=800, max_items=8)

    assert builder.build("订单金额", sales_snapshot) == builder.build("订单金额", sales_snapshot)


def test_context_rejects_limit_that_cannot_fit_metric_dependency_bundle(
    sales_snapshot: SemanticSnapshot,
):
    """Break caught: silently omitting a required governance dependency at the character limit."""
    with pytest.raises(SemanticContextError, match="max_chars"):
        SemanticContextBuilder(max_chars=80, max_items=8).build("订单金额", sales_snapshot)


def test_context_uses_substring_matches_for_chinese_metric_names(sales_snapshot: SemanticSnapshot):
    """Break caught: only recognizing a metric when the question exactly equals its display name."""
    context = SemanticContextBuilder(max_chars=800, max_items=8).build(
        "请帮我查询订单金额趋势", sales_snapshot
    )

    assert context.metrics == ("order_amount",)


def test_context_rejects_limit_that_cannot_fit_all_matched_metric_dependencies(
    sales_snapshot: SemanticSnapshot,
):
    """Break caught: omitting a later matched metric when only the combined bundles overflow."""
    second_metric = SemanticMetric(
        name="order_amount_summary",
        name_cn="订单金额汇总",
        expression="SUM(orders.amount)",
        allowed_dimensions=("orders.created_at",),
        required_filters=("orders.status",),
        default_time_field="orders.created_at",
    )
    two_metric_snapshot = sales_snapshot.model_copy(
        update={"metrics": (sales_snapshot.metrics[0], second_metric)}
    )

    with pytest.raises(SemanticContextError, match="max_items"):
        SemanticContextBuilder(max_chars=800, max_items=8).build(
            "订单金额汇总", two_metric_snapshot
        )


def test_context_resolves_chinese_term_to_complete_metric_bundle(
    sales_snapshot: SemanticSnapshot,
):
    """Break caught: terms being published but ignored by semantic context selection."""
    context = SemanticContextBuilder(max_chars=1000, max_items=10).build(
        "查询成交额", sales_snapshot
    )

    assert context.metrics == ("order_amount",)
    assert context.terms == ("成交额",)
    assert "术语: 成交额 -> metric:order_amount" in context.rendered
    assert "条件: orders.status = 'paid'" in context.rendered
    assert "orders.customer_id = customers.id [LEFT JOIN]" in context.rendered
    assert "禁止字段 customers.mobile" in context.rendered


def test_context_fails_closed_for_ambiguous_conditional_term(
    sales_snapshot: SemanticSnapshot,
):
    """Break caught: choosing one of several conditional term definitions arbitrarily."""
    first = sales_snapshot.terms[0]
    ambiguous = sales_snapshot.model_copy(
        update={
            "terms": (
                first,
                first.model_copy(update={"when": "orders.status = 'refunded'"}),
            )
        }
    )

    with pytest.raises(SemanticContextError, match="术语"):
        SemanticContextBuilder(max_chars=1000, max_items=12).build(
            "查询成交额", ambiguous
        )
