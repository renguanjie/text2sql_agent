from hashlib import sha256
from pathlib import Path
from unittest.mock import ANY, Mock

import pytest

from core.semantic.compiler import SemanticCompiler
from core.semantic.models import SemanticSnapshot, SemanticTerm
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
    database_id = sha256(b"sales").hexdigest()

    assert payload["version"] == "2.0"
    assert payload["format"] == "text2sql_knowledge_graph"
    assert payload["semantic_version"] == "1.0.0"
    assert payload["semantic_content_sha256"] == sales_snapshot.content_sha256
    assert payload["datasource"] == "sales"
    assert payload["nodes"] == [
        {
            "label": "Database",
            "properties": {
                "database_id": database_id,
                "name": "sales",
                "db_type": "mysql",
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
                "datasource": "sales",
            },
        },
        {
            "label": "Table",
            "properties": {
                "database_id": database_id,
                "name": "orders",
                "display_label": "订单",
                "synonyms": [],
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
                "datasource": "sales",
            },
        },
        {
            "label": "Column",
            "properties": {
                "database_id": database_id,
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
                "datasource": "sales",
            },
        },
        {
            "label": "Column",
            "properties": {
                "database_id": database_id,
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
                "datasource": "sales",
            },
        },
        {
            "label": "Table",
            "properties": {
                "database_id": database_id,
                "name": "customers",
                "display_label": "客户",
                "synonyms": [],
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
                "datasource": "sales",
            },
        },
        {
            "label": "Column",
            "properties": {
                "database_id": database_id,
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
                "datasource": "sales",
            },
        },
        {
            "label": "BusinessConcept",
            "properties": {
                "name": "metric:sales:order_count",
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
                "datasource": "sales",
            },
        },
        {
            "label": "BusinessConcept",
            "properties": {
                "name": "term:sales:order_count:406d283ddbd643b4453c9a00b5300a8fbc63730745876d25a7656fd21fb931bf",
                "display_label": "order_count",
                "concept_type": "term",
                "target": "field:orders.id",
                "when": None,
                "semantic_version": "1.0.0",
                "semantic_content_sha256": sales_snapshot.content_sha256,
                "source": "git_semantic_model",
                "datasource": "sales",
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
        "datasource": "sales",
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


def test_publish_rejects_zero_write_result_without_claiming_index_update(
    sales_snapshot: SemanticSnapshot,
):
    """Break caught: disconnected/closed imports being reported as a successful publication."""
    client = Mock()
    client.import_knowledge_graph.return_value = {"nodes": 0, "relationships": 0}

    with pytest.raises(SemanticPublishError, match="未写入语义索引"):
        SemanticNeo4jPublisher(client).publish(sales_snapshot)


def test_publisher_rejects_uncompiled_semantic_data():
    """Break caught: allowing a raw mapping to bypass registry snapshot validation."""
    with pytest.raises(TypeError, match="SemanticSnapshot"):
        SemanticNeo4jPublisher(Mock()).to_graph_payload({"datasource": "sales"})


def test_term_concept_keys_preserve_conditional_definitions(sales_snapshot: SemanticSnapshot):
    """Break caught: merging distinct conditional definitions of the same term."""
    snapshot = sales_snapshot.model_copy(
        update={
            "terms": (
                SemanticTerm(
                    term="active_customer",
                    target="customers.id",
                    when="customers.status = 'active'",
                ),
                SemanticTerm(
                    term="active_customer",
                    target="customers.id",
                    when="customers.status = 'paid'",
                ),
                SemanticTerm(
                    term="active_customer",
                    target="orders.customer_id",
                    when="customers.status = 'active'",
                ),
            )
        }
    )
    publisher = SemanticNeo4jPublisher(Mock())

    first_names = [
        node["properties"]["name"]
        for node in publisher.to_graph_payload(snapshot)["nodes"]
        if node["properties"].get("concept_type") == "term"
    ]
    second_names = [
        node["properties"]["name"]
        for node in publisher.to_graph_payload(snapshot)["nodes"]
        if node["properties"].get("concept_type") == "term"
    ]

    assert first_names == second_names
    assert len(first_names) == 3
    assert len(set(first_names)) == 3
    assert all(name.startswith("term:sales:active_customer:") for name in first_names)


def test_metric_concept_keys_are_scoped_to_their_datasource(sales_snapshot: SemanticSnapshot):
    """Break caught: merging same-named metrics from separate datasource subgraphs."""
    archive_snapshot = sales_snapshot.model_copy(update={"datasource": "archive"})

    sales_name = next(
        node["properties"]["name"]
        for node in SemanticNeo4jPublisher(Mock()).to_graph_payload(sales_snapshot)["nodes"]
        if node["properties"].get("concept_type") == "metric"
    )
    archive_name = next(
        node["properties"]["name"]
        for node in SemanticNeo4jPublisher(Mock()).to_graph_payload(archive_snapshot)["nodes"]
        if node["properties"].get("concept_type") == "metric"
    )

    assert sales_name == "metric:sales:order_count"
    assert archive_name == "metric:archive:order_count"
    assert sales_name != archive_name
    assert sales_name.startswith("metric:")
    assert archive_name.startswith("metric:")


def test_full_datasource_hash_prevents_known_truncated_hash_collision(
    sales_snapshot: SemanticSnapshot,
):
    """Break caught: two datasources sharing the old first eight hex digits merging."""
    first_name = "datasource_100865"
    second_name = "datasource_127887"
    assert sha256(first_name.encode()).hexdigest()[:8] == sha256(
        second_name.encode()
    ).hexdigest()[:8]
    first = sales_snapshot.model_copy(update={"datasource": first_name})
    second = sales_snapshot.model_copy(update={"datasource": second_name})

    first_payload = SemanticNeo4jPublisher(Mock()).to_graph_payload(first)
    second_payload = SemanticNeo4jPublisher(Mock()).to_graph_payload(second)
    first_database = first_payload["nodes"][0]["properties"]
    second_database = second_payload["nodes"][0]["properties"]

    assert first_database["database_id"] == sha256(first_name.encode()).hexdigest()
    assert second_database["database_id"] == sha256(second_name.encode()).hexdigest()
    assert first_database["database_id"] != second_database["database_id"]
    assert all(
        node["properties"]["datasource"] == first_name
        for node in first_payload["nodes"]
    )
    assert all(
        edge["properties"]["datasource"] == first_name
        for edge in first_payload["edges"]
    )
