import json
from pathlib import Path

import pytest

from core.semantic.compiler import SemanticCompiler
from core.semantic.registry import SemanticRegistry, SemanticRegistryError


@pytest.fixture
def compiled_model_dir(tmp_path: Path) -> Path:
    """A compiled sales snapshot with a manifest-controlled active version."""
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
    compiled_root = tmp_path / ".compiled"
    SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z").compile_file(source, compiled_root)
    return compiled_root


def test_failed_reload_keeps_previous_snapshot(compiled_model_dir: Path):
    """Break caught: replacing an active snapshot before validation has completed."""
    registry = SemanticRegistry(compiled_model_dir, "sales")
    first = registry.load_active()
    active_file = compiled_model_dir / "sales/1.0.0.json"
    active_file.write_text("{broken", encoding="utf-8")

    with pytest.raises(SemanticRegistryError):
        registry.load_active()

    assert registry.get_snapshot() == first


def test_registry_returns_same_immutable_snapshot(compiled_model_dir: Path):
    """Break caught: returning a mutable copy rather than the pinned active snapshot."""
    registry = SemanticRegistry(compiled_model_dir, "sales")

    loaded = registry.load_active()

    assert registry.get_snapshot() is loaded


def test_registry_rejects_snapshot_with_changed_semantic_content(compiled_model_dir: Path):
    """Break caught: trusting content_sha256 without recalculating canonical model content."""
    active_file = compiled_model_dir / "sales/1.0.0.json"
    payload = json.loads(active_file.read_text(encoding="utf-8"))
    payload["entities"][0]["name_cn"] = "已篡改订单"
    active_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SemanticRegistryError, match="content_sha256"):
        SemanticRegistry(compiled_model_dir, "sales").load_active()


def test_registry_accepts_historical_snapshot_without_current_yaml(compiled_model_dir: Path):
    """Break caught: comparing a valid snapshot source digest to a mutable YAML source file."""
    (compiled_model_dir.parent / "sales.yaml").unlink()
    registry = SemanticRegistry(compiled_model_dir, "sales")

    snapshot = registry.load_active()

    assert snapshot.datasource == "sales"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda payload: payload.__setitem__("source_sha256", "not-a-sha"), "source_sha256"),
        (lambda payload: payload.__setitem__("model_version", "2.0.0"), "版本"),
        (lambda payload: payload.__setitem__("datasource", "other"), "数据源"),
        (lambda payload: payload.__setitem__("dialect", "postgres"), "快照"),
    ],
)
def test_registry_rejects_manifest_or_snapshot_contract_mismatches(
    compiled_model_dir: Path, mutation, expected: str
):
    """Break caught: accepting an invalid hash, location, datasource, version, or dialect."""
    active_file = compiled_model_dir / "sales/1.0.0.json"
    payload = json.loads(active_file.read_text(encoding="utf-8"))
    mutation(payload)
    active_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SemanticRegistryError, match=expected):
        SemanticRegistry(compiled_model_dir, "sales").load_active()


def test_registry_does_not_expose_sensitive_invalid_value_in_validation_error(
    compiled_model_dir: Path,
):
    """Break caught: relaying schema-validator diagnostics containing malformed sensitive input."""
    sensitive_value = "SENSITIVE_MOBILE_13800138000"
    active_file = compiled_model_dir / "sales/1.0.0.json"
    payload = json.loads(active_file.read_text(encoding="utf-8"))
    payload["entities"][0]["fields"][0]["role"] = sensitive_value
    active_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SemanticRegistryError, match="^快照格式无效$") as exc_info:
        SemanticRegistry(compiled_model_dir, "sales").load_active()

    assert sensitive_value not in str(exc_info.value)
