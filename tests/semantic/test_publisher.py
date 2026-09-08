from pathlib import Path
from unittest.mock import ANY, Mock

import pytest

from core.semantic.compiler import SemanticCompiler
from core.semantic.models import SemanticSnapshot
from core.semantic.publisher import SemanticNeo4jPublisher, SemanticPublishError


@pytest.fixture
def sales_snapshot(tmp_path: Path) -> SemanticSnapshot:
    """A compiled model with a join, metric, and colliding term display text."""
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
      - {name: customer_id, name_cn: 客户编号, data_type: bigint, role: identifier}
  - name: customers
    name_cn: 客户
    fields:
      - {name: id, name_cn: 客户编号, data_type: bigint, role: identifier}
relationships:
  - {name: order_customer, from_entity: orders, from_field: customer_id,
     to_entity: customers, to_field: id}
metrics:
  - name: order_count
    name_cn: 订单数
    expression: COUNT(orders.id)
terms:
  - {term: order_count, target: orders.id}
""",
        encoding="utf-8",
    )
    return SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z").compile_file(
        source, tmp_path / ".compiled"
    )


def test_payload_projects_complete_governed_graph_with_provenance(
    sales_snapshot: SemanticSnapshot,
):
    """Break caught: omitting derived graph nodes, edges, or semantic provenance."""
    payload = SemanticNeo4jPublisher(Mock()).to_graph_payload(sales_snapshot)

    assert payload["version"] == "2.0"
    assert payload["format"] == "text2sql_knowledge_graph"
    assert payload["semantic_version"] == "1.0.0"
    assert payload["semantic_content_sha256"] == sales_snapshot.content_sha256
    assert payload["nodes"] == [
        {
            "label": "Database",
            "properties": {
                "database_id": 3763253904,
                "name": "sales",
                "db_type": "mysql",
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
            },
        },
        {
            "label": "Table",
            "properties": {
                "database_id": 3763253904,
                "name": "orders",
                "display_label": "订单",
                "synonyms": [],
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
            },
        },
        {
            "label": "Column",
            "properties": {
                "database_id": 3763253904,
                "table_name": "orders",
                "name": "id",
                "display_label": "订单号",
                "data_type": "bigint",
                "role": "identifier",
                "synonyms": [],
                "enum_values": [],
                "sensitivity": "internal",
                "directly_queryable": True,
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
            },
        },
        {
            "label": "Column",
            "properties": {
                "database_id": 3763253904,
                "table_name": "orders",
                "name": "customer_id",
                "display_label": "客户编号",
                "data_type": "bigint",
                "role": "identifier",
                "synonyms": [],
                "enum_values": [],
                "sensitivity": "internal",
                "directly_queryable": True,
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
            },
        },
        {
            "label": "Table",
            "properties": {
                "database_id": 3763253904,
                "name": "customers",
                "display_label": "客户",
                "synonyms": [],
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
            },
        },
        {
            "label": "Column",
            "properties": {
                "database_id": 3763253904,
                "table_name": "customers",
                "name": "id",
                "display_label": "客户编号",
                "data_type": "bigint",
                "role": "identifier",
                "synonyms": [],
                "enum_values": [],
                "sensitivity": "internal",
                "directly_queryable": True,
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
            },
        },
        {
            "label": "BusinessConcept",
            "properties": {
                "name": "metric:order_count",
                "display_label": "订单数",
                "concept_type": "metric",
                "expression": "COUNT(orders.id)",
                "synonyms": [],
                "allowed_dimensions": [],
                "required_filters": [],
                "default_time_field": None,
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
            },
        },
        {
            "label": "BusinessConcept",
            "properties": {
                "name": "term:order_count",
                "display_label": "order_count",
                "concept_type": "term",
                "target": "orders.id",
                "when": None,
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
            },
        },
    ]
    assert [edge["type"] for edge in payload["edges"]] == [
        "HAS_TABLE",
        "HAS_COLUMN",
        "HAS_COLUMN",
        "HAS_TABLE",
        "HAS_COLUMN",
        "CONNECTS",
    ]
    assert payload["edges"][-1]["properties"] == {
        "name": "order_customer",
        "from_column": "customer_id",
        "to_column": "id",
        "join_type": "LEFT JOIN",
        "extra_condition": None,
        "semantic_version": "1.0.0",
        "semantic_content_sha256": sales_snapshot.content_sha256,
        "source": "git_semantic_model",
    }
    assert all(
        edge["properties"]["semantic_version"] == "1.0.0" for edge in payload["edges"]
    )


def test_publish_uses_existing_atomic_import(sales_snapshot: SemanticSnapshot):
    """Break caught: bypassing the client's atomic import boundary or disabling replacement."""
    client = Mock()
    client.import_knowledge_graph.return_value = {"nodes": 7, "relationships": 8}

    result = SemanticNeo4jPublisher(client).publish(sales_snapshot)

    assert result == {"nodes": 7, "relationships": 8}
    client.import_knowledge_graph.assert_called_once_with(
        ANY, overwrite=True
    )
    assert client.import_knowledge_graph.call_args.args[0]["format"] == "text2sql_knowledge_graph"


def test_publish_surfaces_atomic_import_error(sales_snapshot: SemanticSnapshot):
    """Break caught: treating a rejected atomic graph write as a successful publication."""
    client = Mock()
    client.import_knowledge_graph.return_value = {
        "nodes": 0,
        "relationships": 0,
        "error": "invalid graph",
    }

    with pytest.raises(SemanticPublishError, match="invalid graph"):
        SemanticNeo4jPublisher(client).publish(sales_snapshot)


def test_publisher_rejects_uncompiled_semantic_data():
    """Break caught: allowing a raw mapping to bypass registry snapshot validation."""
    with pytest.raises(TypeError, match="SemanticSnapshot"):
        SemanticNeo4jPublisher(Mock()).to_graph_payload({"datasource": "sales"})
