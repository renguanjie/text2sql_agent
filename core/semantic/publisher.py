"""Projection of validated semantic snapshots into the Neo4j v2 import format."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from .models import SemanticSnapshot


class SemanticPublishError(RuntimeError):
    """Raised when Neo4j rejects a semantic graph publication."""


def build_v2_payload(snapshot: SemanticSnapshot, database_id: int) -> dict[str, Any]:
    """Build the deterministic v2 graph payload for one compiled snapshot."""
    provenance = {
        "semantic_version": snapshot.model_version,
        "semantic_content_sha256": snapshot.content_sha256,
        "source": "git_semantic_model",
    }
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def node(label: str, properties: dict[str, Any]) -> dict[str, Any]:
        graph_node = {"label": label, "properties": {**properties, **provenance}}
        nodes.append(graph_node)
        return graph_node

    def reference(graph_node: dict[str, Any]) -> dict[str, Any]:
        properties = graph_node["properties"]
        label = graph_node["label"]
        if label == "Database":
            key_properties = {
                "database_id": properties["database_id"],
                "name": properties["name"],
            }
        elif label == "Table":
            key_properties = {
                "database_id": properties["database_id"],
                "name": properties["name"],
            }
        elif label == "Column":
            key_properties = {
                "database_id": properties["database_id"],
                "table_name": properties["table_name"],
                "name": properties["name"],
            }
        else:
            key_properties = {"name": properties["name"]}
        return {"label": label, "properties": key_properties}

    def edge(
        edge_type: str,
        start: dict[str, Any],
        end: dict[str, Any],
        properties: dict[str, Any] | None = None,
    ) -> None:
        edges.append(
            {
                "type": edge_type,
                "start": reference(start),
                "end": reference(end),
                "properties": {**(properties or {}), **provenance},
            }
        )

    database = node(
        "Database",
        {
            "database_id": database_id,
            "name": snapshot.datasource,
            "db_type": snapshot.dialect,
        },
    )
    tables: dict[str, dict[str, Any]] = {}
    for entity in snapshot.entities:
        table = node(
            "Table",
            {
                "database_id": database_id,
                "name": entity.name,
                "display_label": entity.name_cn,
                "synonyms": list(entity.synonyms),
            },
        )
        tables[entity.name] = table
        edge("HAS_TABLE", database, table)
        for field in entity.fields:
            column = node(
                "Column",
                {
                    "database_id": database_id,
                    "table_name": entity.name,
                    "name": field.name,
                    "display_label": field.name_cn,
                    "data_type": field.data_type,
                    "role": field.role,
                    "synonyms": list(field.synonyms),
                    "enum_values": list(field.enum_values),
                    "sensitivity": field.sensitivity,
                    "directly_queryable": field.directly_queryable,
                },
            )
            edge("HAS_COLUMN", table, column)

    for relationship in snapshot.relationships:
        edge(
            "CONNECTS",
            tables[relationship.from_entity],
            tables[relationship.to_entity],
            {
                "name": relationship.name,
                "from_column": relationship.from_field,
                "to_column": relationship.to_field,
                "join_type": relationship.join_type,
                "extra_condition": relationship.extra_condition,
            },
        )

    for metric in snapshot.metrics:
        node(
            "BusinessConcept",
            {
                "name": f"metric:{metric.name}",
                "display_label": metric.name_cn,
                "concept_type": "metric",
                "expression": metric.expression,
                "synonyms": list(metric.synonyms),
                "allowed_dimensions": list(metric.allowed_dimensions),
                "required_filters": list(metric.required_filters),
                "default_time_field": metric.default_time_field,
            },
        )
    for term in snapshot.terms:
        node(
            "BusinessConcept",
            {
                "name": f"term:{term.term}",
                "display_label": term.term,
                "concept_type": "term",
                "target": term.target,
                "when": term.when,
            },
        )

    return {
        "version": "2.0",
        "format": "text2sql_knowledge_graph",
        "semantic_version": snapshot.model_version,
        "semantic_content_sha256": snapshot.content_sha256,
        "nodes": nodes,
        "edges": edges,
    }


class SemanticNeo4jPublisher:
    """Publish a compiled semantic snapshot through Neo4jClient's atomic importer."""

    def __init__(self, neo4j_client):
        self._client = neo4j_client

    def to_graph_payload(self, snapshot: SemanticSnapshot) -> dict:
        if not isinstance(snapshot, SemanticSnapshot):
            raise TypeError("publisher requires a compiled SemanticSnapshot")
        datasource_hash = sha256(snapshot.datasource.encode("utf-8")).hexdigest()
        database_id = max(1, int(datasource_hash[:8], 16))
        return build_v2_payload(snapshot, database_id)

    def publish(self, snapshot: SemanticSnapshot) -> dict:
        payload = self.to_graph_payload(snapshot)
        result = self._client.import_knowledge_graph(payload, overwrite=True)
        if result.get("error"):
            raise SemanticPublishError(result["error"])
        return result
