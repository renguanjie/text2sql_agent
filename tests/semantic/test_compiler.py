import json
from pathlib import Path

import pytest

from core.semantic.compiler import SemanticCompileError, SemanticCompiler


def test_compile_is_deterministic_and_updates_manifest(tmp_path: Path):
    """Break caught: omitting canonical digest or active manifest update."""
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
""",
        encoding="utf-8",
    )
    compiler = SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z")

    first = compiler.compile_file(source, tmp_path / ".compiled")
    second = compiler.compile_file(source, tmp_path / ".compiled")

    assert first.content_sha256 == second.content_sha256
    manifest = json.loads((tmp_path / ".compiled/manifest.json").read_text("utf-8"))
    assert manifest["active"] == {"sales": "1.0.0"}


def test_compile_rejects_unknown_relationship_reference(tmp_path: Path):
    """Break caught: accepting a relationship endpoint absent from its model."""
    source = tmp_path / "bad.yaml"
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
relationships:
  - {name: missing_customer, from_entity: orders, from_field: id,
     to_entity: customers, to_field: id}
""",
        encoding="utf-8",
    )

    with pytest.raises(SemanticCompileError, match="customers"):
        SemanticCompiler().compile_file(source, tmp_path / ".compiled")
