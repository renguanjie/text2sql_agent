import json
import tempfile
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


def test_registry_rejects_unsafe_datasource_before_path_resolution(tmp_path: Path):
    """Break caught: registry datasource traversal being interpreted as a filesystem path."""
    with pytest.raises(SemanticRegistryError, match="数据源"):
        SemanticRegistry(tmp_path, "../escaped")


def test_registry_rejects_symlinked_compiled_root(compiled_model_dir: Path, tmp_path: Path):
    """Break caught: trusting snapshots reached through a symlinked registry root."""
    linked_root = tmp_path / "linked-compiled"
    linked_root.symlink_to(compiled_model_dir, target_is_directory=True)

    with pytest.raises(SemanticRegistryError, match="符号链接"):
        SemanticRegistry(linked_root, "sales").load_active()


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_registry_rejects_symlinked_compiled_root_ancestor(
    compiled_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: bool,
):
    """Break caught: checking only compiled_root misses a symlinked ancestor."""
    linked_parent = compiled_model_dir.parent / "linked-parent"
    linked_parent.symlink_to(compiled_model_dir.parent, target_is_directory=True)
    compiled_root = (
        Path("linked-parent/.compiled")
        if relative
        else linked_parent / compiled_model_dir.name
    )
    if relative:
        monkeypatch.chdir(compiled_model_dir.parent)

    with pytest.raises(SemanticRegistryError, match="符号链接"):
        SemanticRegistry(compiled_root, "sales").load_active()


@pytest.mark.parametrize(
    "target_name", ["manifest.json", "sales/1.0.0.json"], ids=["manifest", "artifact"]
)
def test_registry_rejects_symlinked_storage_file(
    compiled_model_dir: Path, tmp_path: Path, target_name: str
):
    """Break caught: manifest or artifact reads following a symlink."""
    target = compiled_model_dir / target_name
    outside = tmp_path / f"outside-{target.name}"
    target.rename(outside)
    target.symlink_to(outside)

    with pytest.raises(SemanticRegistryError, match="符号链接"):
        SemanticRegistry(compiled_model_dir, "sales").load_active()


@pytest.mark.parametrize(
    ("target_name", "parse_number"),
    [("manifest.json", 1), ("sales/1.0.0.json", 2)],
    ids=["manifest", "artifact"],
)
def test_registry_rejects_file_replaced_by_symlink_during_load(
    compiled_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    parse_number: int,
):
    """Break caught: a checked file being swapped after its bytes were read."""
    target = compiled_model_dir / target_name
    outside = tmp_path / f"outside-race-{target.name}"
    outside.write_bytes(target.read_bytes())
    parked = tmp_path / f"parked-{target.name}"
    real_loads = json.loads
    parse_count = 0

    def loads_then_replace(value, *args, **kwargs):
        nonlocal parse_count
        parsed = real_loads(value, *args, **kwargs)
        parse_count += 1
        if parse_count == parse_number:
            target.rename(parked)
            target.symlink_to(outside)
        return parsed

    monkeypatch.setattr("core.semantic.registry.json.loads", loads_then_replace)

    with pytest.raises(SemanticRegistryError, match="替换|符号链接"):
        SemanticRegistry(compiled_model_dir, "sales").load_active()

    assert target.is_symlink()


@pytest.mark.parametrize(
    "directory_name", [".", "sales"], ids=["compiled-root", "datasource"]
)
def test_registry_rejects_directory_replaced_by_symlink_during_load(
    compiled_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_name: str,
):
    """Break caught: a pinned snapshot parent being replaced during parsing."""
    target = (
        compiled_model_dir
        if directory_name == "."
        else compiled_model_dir / directory_name
    )
    outside = tmp_path / f"outside-directory-{target.name}"
    outside.mkdir()
    parked = tmp_path / f"parked-directory-{target.name}"
    real_loads = json.loads
    parse_count = 0

    def loads_then_replace(value, *args, **kwargs):
        nonlocal parse_count
        parsed = real_loads(value, *args, **kwargs)
        parse_count += 1
        if parse_count == 2:
            target.rename(parked)
            target.symlink_to(outside, target_is_directory=True)
        return parsed

    monkeypatch.setattr("core.semantic.registry.json.loads", loads_then_replace)

    with pytest.raises(SemanticRegistryError, match="替换|符号链接"):
        SemanticRegistry(compiled_model_dir, "sales").load_active()

    assert target.is_symlink()


def test_registry_does_not_resolve_datasource_symlink_created_after_check(
    compiled_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Break caught: resolve following an in-root symlink created after its check."""
    datasource_dir = compiled_model_dir / "sales"
    alternate = compiled_model_dir / "alternate"
    parked = compiled_model_dir / "parked-sales"
    real_reject = SemanticRegistry._reject_symlink
    attacked = False

    def reject_then_replace(path: Path, label: str):
        nonlocal attacked
        real_reject(path, label)
        if not attacked and path == datasource_dir:
            datasource_dir.rename(parked)
            alternate.mkdir()
            (alternate / "1.0.0.json").write_bytes(
                (parked / "1.0.0.json").read_bytes()
            )
            datasource_dir.symlink_to(alternate, target_is_directory=True)
            attacked = True

    monkeypatch.setattr(
        SemanticRegistry,
        "_reject_symlink",
        staticmethod(reject_then_replace),
    )

    with pytest.raises(SemanticRegistryError, match="符号链接"):
        SemanticRegistry(compiled_model_dir, "sales").load_active()

    assert attacked is True


def test_compiler_and_registry_accept_normal_nested_absolute_root(tmp_path: Path):
    """Break caught: anchoring all public roots to the repository working tree."""
    source = tmp_path / "models/sales.yaml"
    source.parent.mkdir()
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
    compiled_root = tmp_path / "nested/deeper/.compiled"

    SemanticCompiler().compile_file(source, compiled_root)
    snapshot = SemanticRegistry(compiled_root, "sales").load_active()

    assert snapshot.datasource == "sales"


def test_compiler_and_registry_accept_platform_temp_absolute_root():
    """Break caught: macOS /var temp aliases being mistaken for user symlinks."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "sales.yaml"
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
        compiled_root = root / "nested/.compiled"

        SemanticCompiler().compile_file(source, compiled_root)
        snapshot = SemanticRegistry(compiled_root, "sales").load_active()

        assert snapshot.datasource == "sales"
