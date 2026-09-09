import json
import os
import textwrap
from pathlib import Path

import pytest

from core.semantic.compiler import SemanticCompileError, SemanticCompiler


def _compile_yaml(
    tmp_path: Path,
    body: str,
    *,
    filename: str = "sales.yaml",
    expected_datasource: str | None = None,
    output_root: Path | None = None,
    compiler: SemanticCompiler | None = None,
):
    source = tmp_path / filename
    source.write_text(textwrap.dedent(body), encoding="utf-8")
    kwargs = (
        {"expected_datasource": expected_datasource}
        if expected_datasource is not None
        else {}
    )
    return (compiler or SemanticCompiler()).compile_file(
        source,
        output_root or tmp_path / ".compiled",
        **kwargs,
    )


def _rich_model(*, metric_expression: str = "SUM(orders.amount)", extra: str = "") -> str:
    return f"""
        schema_version: 1
        model_version: 1.0.0
        datasource: sales
        dialect: mysql
        entities:
          - name: orders
            name_cn: 订单
            fields:
              - {{name: id, name_cn: 订单号, data_type: bigint, role: identifier}}
              - {{name: amount, name_cn: 金额, data_type: decimal, role: measure}}
              - {{name: status, name_cn: 状态, data_type: varchar, role: dimension}}
              - {{name: created_at, name_cn: 时间, data_type: datetime, role: time}}
          - name: customers
            name_cn: 客户
            fields:
              - {{name: id, name_cn: 客户号, data_type: bigint, role: identifier}}
              - {{name: status, name_cn: 客户状态, data_type: varchar, role: dimension}}
        metrics:
          - name: gross_revenue
            name_cn: 总收入
            expression: {metric_expression}
        {extra}
    """


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
relationships:
  - {name: missing_customer, from_entity: orders, from_field: id,
     to_entity: customers, to_field: id}
""",
        encoding="utf-8",
    )

    with pytest.raises(SemanticCompileError, match="customers"):
        SemanticCompiler().compile_file(source, tmp_path / ".compiled")


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("SUM(orders.amount", id="sqlglot-21-unclosed-expression"),
        pytest.param(
            "SUM(orders.amount); COUNT(orders.id)",
            id="sqlglot-30-multiple-expressions",
        ),
    ],
)
def test_compile_rejects_metric_that_is_not_one_complete_expression(
    tmp_path: Path, expression: str
):
    """Break caught: regex-only validation accepting malformed or appended expressions."""
    with pytest.raises(SemanticCompileError, match="指标 gross_revenue"):
        _compile_yaml(tmp_path, _rich_model(metric_expression=expression))


@pytest.mark.parametrize(
    ("extra_condition", "expected"),
    [
        ("orders.ghost = customers.id", "ghost"),
        ("orders.status", "布尔"),
        ("orders.status = 'paid'; orders.id = customers.id", "单个"),
    ],
)
def test_compile_strictly_validates_relationship_extra_condition(
    tmp_path: Path, extra_condition: str, expected: str
):
    """Break caught: unknown fields and non-boolean/multiple join predicates being ignored."""
    relationship = f"""
        relationships:
          - name: order_customer
            from_entity: orders
            from_field: id
            to_entity: customers
            to_field: id
            extra_condition: {json.dumps(extra_condition)}
    """

    with pytest.raises(SemanticCompileError, match=expected):
        _compile_yaml(tmp_path, _rich_model(extra=relationship))


@pytest.mark.parametrize(
    ("metric_lines", "expected"),
    [
        ("allowed_dimensions: [orders.amount]", "维度"),
        ("default_time_field: orders.id", "时间字段"),
    ],
)
def test_compile_rejects_metric_references_with_wrong_roles(
    tmp_path: Path, metric_lines: str, expected: str
):
    """Break caught: measure fields acting as dimensions or identifiers as time fields."""
    body = _rich_model().replace(
        "expression: SUM(orders.amount)",
        f"expression: SUM(orders.amount)\n            {metric_lines}",
    )

    with pytest.raises(SemanticCompileError, match=expected):
        _compile_yaml(tmp_path, body)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("missing_target", "target"),
        ("id", "不明确"),
    ],
)
def test_compile_rejects_unknown_or_ambiguous_term_target(
    tmp_path: Path, target: str, expected: str
):
    """Break caught: inert term definitions pointing nowhere or to multiple fields."""
    terms = f"""
        terms:
          - term: 成交额
            target: {target}
    """

    with pytest.raises(SemanticCompileError, match=expected):
        _compile_yaml(tmp_path, _rich_model(extra=terms))


def test_compile_canonicalizes_term_and_all_field_references(tmp_path: Path):
    """Break caught: compiler accepting shorthand that runtime compares only as qualified refs."""
    extra = """
        relationships:
          - name: order_customer
            from_entity: ORDERS
            from_field: ID
            to_entity: CUSTOMERS
            to_field: ID
            extra_condition: ORDERS.status = CUSTOMERS.status
        metrics:
          - name: gross_revenue
            name_cn: 总收入
            expression: SUM(ORDERS.amount)
            allowed_dimensions: [CREATED_AT]
            required_filters: [ORDERS.status]
            default_time_field: CREATED_AT
        terms:
          - {term: 成交额, target: GROSS_REVENUE, when: "ORDERS.status = 'paid'"}
        policies:
          denied_entities: [CUSTOMERS]
          denied_fields: [AMOUNT]
    """
    body = _rich_model(extra=extra)
    # Replace the base metrics block because this fixture supplies a richer one.
    body = body.replace(
        "metrics:\n          - name: gross_revenue\n            name_cn: 总收入\n            expression: SUM(orders.amount)\n",
        "",
        1,
    )

    snapshot = _compile_yaml(tmp_path, body)

    relationship = snapshot.relationships[0]
    metric = snapshot.metrics[0]
    assert relationship.from_entity == "orders"
    assert relationship.from_field == "id"
    assert relationship.extra_condition == "orders.status = customers.status"
    assert metric.expression == "SUM(orders.amount)"
    assert metric.allowed_dimensions == ("orders.created_at",)
    assert metric.required_filters == ("orders.status",)
    assert metric.default_time_field == "orders.created_at"
    assert snapshot.terms[0].target == "metric:gross_revenue"
    assert snapshot.terms[0].when == "orders.status = 'paid'"
    assert snapshot.policies.denied_entities == ("customers",)
    assert snapshot.policies.denied_fields == ("orders.amount",)


def test_compile_rejects_datasource_stem_mismatch_unless_expected_is_explicit(
    tmp_path: Path,
):
    """Break caught: a source file silently publishing under another datasource name."""
    body = _rich_model()

    with pytest.raises(SemanticCompileError, match="文件名"):
        _compile_yaml(tmp_path, body, filename="incoming.yaml")

    snapshot = _compile_yaml(
        tmp_path,
        body,
        filename="incoming.yaml",
        expected_datasource="sales",
    )
    assert snapshot.datasource == "sales"


def test_compile_rejects_symlinked_output_component_before_writing(tmp_path: Path):
    """Break caught: a datasource directory symlink escaping the compiled root."""
    output_root = tmp_path / ".compiled"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    (output_root / "sales").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SemanticCompileError, match="符号链接"):
        _compile_yaml(tmp_path, _rich_model(), output_root=output_root)

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_compile_rejects_symlinked_output_ancestor_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool
):
    """Break caught: validation starting at output_root misses an earlier symlink."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    output_root = (
        Path("link/deeper/compiled")
        if relative
        else tmp_path / "link/deeper/compiled"
    )
    if relative:
        monkeypatch.chdir(tmp_path)

    with pytest.raises(SemanticCompileError, match="符号链接"):
        _compile_yaml(tmp_path, _rich_model(), output_root=output_root)

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "target_name",
    [
        ".compiled/sales/1.0.0.json",
        ".compiled/manifest.json",
        "schema/semantic-model.schema.json",
    ],
    ids=["artifact", "manifest", "schema"],
)
def test_compile_rejects_symlinked_storage_file_before_any_write(
    tmp_path: Path, target_name: str
):
    """Break caught: any semantic output target following a pre-existing link."""
    outside = tmp_path / "outside-file"
    outside.write_text("unchanged", encoding="utf-8")
    target = tmp_path / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)

    with pytest.raises(SemanticCompileError, match="符号链接"):
        _compile_yaml(tmp_path, _rich_model())

    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_compile_rejects_changed_content_at_same_version_without_any_write(
    tmp_path: Path,
):
    """Break caught: mutating an immutable version and advancing its manifest in place."""
    compiler = SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z")
    _compile_yaml(tmp_path, _rich_model(), compiler=compiler)
    output_root = tmp_path / ".compiled"
    artifact = output_root / "sales/1.0.0.json"
    manifest = output_root / "manifest.json"
    schema = tmp_path / "schema/semantic-model.schema.json"
    before = (artifact.read_bytes(), manifest.read_bytes(), schema.read_bytes())

    changed = _rich_model().replace("name_cn: 总收入", "name_cn: 修改后的总收入")
    with pytest.raises(SemanticCompileError, match="1.0.0"):
        _compile_yaml(tmp_path, changed, compiler=compiler)

    assert (artifact.read_bytes(), manifest.read_bytes(), schema.read_bytes()) == before


def test_compile_same_content_is_byte_stable_and_returns_existing_artifact(
    tmp_path: Path,
):
    """Break caught: idempotent recompilation rewriting only compiled_at/source metadata."""
    timestamps = iter(["2026-09-09T00:00:00Z", "2026-09-10T00:00:00Z"])
    compiler = SemanticCompiler(clock=lambda: next(timestamps))
    first = _compile_yaml(tmp_path, _rich_model(), compiler=compiler)
    artifact = tmp_path / ".compiled/sales/1.0.0.json"
    before = artifact.read_bytes()

    second = _compile_yaml(tmp_path, _rich_model(), compiler=compiler)

    assert second == first
    assert artifact.read_bytes() == before


def test_compile_validates_existing_manifest_before_creating_artifact(tmp_path: Path):
    """Break caught: snapshot write occurring before a malformed manifest is discovered."""
    output_root = tmp_path / ".compiled"
    output_root.mkdir()
    (output_root / "manifest.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(SemanticCompileError, match="清单"):
        _compile_yaml(tmp_path, _rich_model(), output_root=output_root)

    assert not (output_root / "sales/1.0.0.json").exists()


def test_compile_validates_every_destination_type_before_any_write(tmp_path: Path):
    """Break caught: writing the artifact before discovering an invalid schema target."""
    output_root = tmp_path / ".compiled"
    schema_destination = tmp_path / "schema/semantic-model.schema.json"
    schema_destination.mkdir(parents=True)

    with pytest.raises(SemanticCompileError, match="输出路径"):
        _compile_yaml(tmp_path, _rich_model(), output_root=output_root)

    assert not (output_root / "sales/1.0.0.json").exists()
    assert not (output_root / "manifest.json").exists()


@pytest.mark.parametrize(
    ("attacked_name", "parent_parts"),
    [
        ("1.0.0.json", (".compiled", "sales")),
        ("semantic-model.schema.json", ("schema",)),
        ("manifest.json", (".compiled",)),
    ],
    ids=["artifact", "schema", "manifest"],
)
def test_compile_rejects_parent_replaced_by_symlink_at_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attacked_name: str,
    parent_parts: tuple[str, ...],
):
    """Break caught: a checked destination parent being swapped before replace."""
    outside = tmp_path / f"outside-{attacked_name}"
    outside.mkdir()
    parent = tmp_path.joinpath(*parent_parts)
    parked = tmp_path / f"parked-{attacked_name}"
    real_replace = os.replace
    attacked = False

    def replace_after_parent_swap(source, destination, *args, **kwargs):
        nonlocal attacked
        if not attacked and Path(destination).name == attacked_name:
            parent.rename(parked)
            parent.symlink_to(outside, target_is_directory=True)
            attacked = True
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr("core.semantic.storage.os.replace", replace_after_parent_swap)

    with pytest.raises(SemanticCompileError, match="语义存储|符号链接"):
        _compile_yaml(tmp_path, _rich_model())

    assert attacked is True
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("target_name", "parse_number"),
    [("manifest.json", 1), ("sales/1.0.0.json", 2)],
    ids=["manifest", "artifact"],
)
def test_compile_rejects_existing_file_replaced_by_symlink_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    parse_number: int,
):
    """Break caught: compile decisions using a file replaced after its read."""
    compiler = SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z")
    _compile_yaml(tmp_path, _rich_model(), compiler=compiler)
    target = tmp_path / ".compiled" / target_name
    outside = tmp_path / f"outside-read-{target.name}"
    outside.write_bytes(target.read_bytes())
    parked = tmp_path / f"parked-read-{target.name}"
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

    monkeypatch.setattr("core.semantic.compiler.json.loads", loads_then_replace)

    with pytest.raises(SemanticCompileError):
        _compile_yaml(tmp_path, _rich_model(), compiler=compiler)

    assert target.is_symlink()


def test_compile_revalidates_existing_artifact_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Break caught: an existing artifact being swapped after plan validation."""
    compiler = SemanticCompiler(clock=lambda: "2026-09-09T00:00:00Z")
    _compile_yaml(tmp_path, _rich_model(), compiler=compiler)
    artifact = tmp_path / ".compiled/sales/1.0.0.json"
    outside = tmp_path / "outside-late-artifact.json"
    outside.write_bytes(artifact.read_bytes())
    parked = tmp_path / "parked-late-artifact.json"
    real_replace = os.replace
    attacked = False

    def replace_schema_then_swap_artifact(source, destination, *args, **kwargs):
        nonlocal attacked
        result = real_replace(source, destination, *args, **kwargs)
        if not attacked and Path(destination).name == "semantic-model.schema.json":
            artifact.rename(parked)
            artifact.symlink_to(outside)
            attacked = True
        return result

    monkeypatch.setattr(
        "core.semantic.storage.os.replace",
        replace_schema_then_swap_artifact,
    )

    with pytest.raises(SemanticCompileError):
        _compile_yaml(tmp_path, _rich_model(), compiler=compiler)

    assert attacked is True
    assert artifact.is_symlink()


def test_compile_fails_closed_without_dirfd_no_follow_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Break caught: unsupported platforms falling back to path-following writes."""
    monkeypatch.setattr("core.semantic.storage.os.supports_dir_fd", set())

    with pytest.raises(SemanticCompileError, match="no-follow"):
        _compile_yaml(tmp_path, _rich_model())
