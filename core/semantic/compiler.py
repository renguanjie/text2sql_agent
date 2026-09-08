"""Compile versioned semantic-model YAML into immutable JSON snapshots."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pydantic import ValidationError

from .models import SemanticModel, SemanticSnapshot


COMPILER_VERSION = "1"
_ALLOWED_AGGREGATIONS = frozenset({"sum", "count", "avg", "min", "max"})
_SQL_KEYWORDS = frozenset(
    {
        "and",
        "as",
        "case",
        "else",
        "end",
        "false",
        "if",
        "in",
        "is",
        "not",
        "null",
        "or",
        "then",
        "true",
        "when",
    }
)
_QUALIFIED_FIELD = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_FUNCTION_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_SINGLE_QUOTED_LITERAL = re.compile(r"'(?:''|[^'])*'")


class SemanticCompileError(ValueError):
    """Raised when a semantic-model source cannot safely be compiled."""


def utc_now_iso() -> str:
    """Return a UTC timestamp in the stable ISO-8601 form used by snapshots."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SemanticCompiler:
    def __init__(self, clock=None):
        self._clock = clock or utc_now_iso

    def compile_file(self, source: Path, output_root: Path) -> SemanticSnapshot:
        raw_bytes = source.read_bytes()
        try:
            payload = yaml.safe_load(raw_bytes.decode("utf-8"))
            model = SemanticModel.model_validate(payload)
            self._validate_references(model)
        except (UnicodeDecodeError, yaml.YAMLError, ValidationError, SemanticCompileError) as exc:
            raise SemanticCompileError(str(exc)) from exc

        semantic_payload = model.model_dump(mode="json")
        canonical = json.dumps(
            semantic_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        snapshot = SemanticSnapshot(
            **semantic_payload,
            source_sha256=sha256(raw_bytes).hexdigest(),
            content_sha256=sha256(canonical).hexdigest(),
            compiler_version=COMPILER_VERSION,
            compiled_at=self._clock(),
        )
        self._write_snapshot_atomically(snapshot, output_root)
        self._write_manifest_atomically(snapshot, output_root)
        self._write_json_schema(output_root.parent / "schema/semantic-model.schema.json")
        return snapshot

    def _validate_references(self, model: SemanticModel) -> None:
        entities: dict[str, Any] = {}
        fields: dict[str, dict[str, Any]] = {}
        fields_by_name: dict[str, list[str]] = {}

        for entity in model.entities:
            entity_key = entity.name.casefold()
            self._add_unique(entities, entity_key, entity.name, "实体")
            fields[entity_key] = {}
            for field in entity.fields:
                field_key = field.name.casefold()
                self._add_unique(fields[entity_key], field_key, field.name, f"实体 {entity.name} 的字段")
                fields_by_name.setdefault(field_key, []).append(f"{entity.name}.{field.name}")
                if field.sensitivity == "restricted" and field.directly_queryable:
                    raise SemanticCompileError(
                        f"受限字段 {entity.name}.{field.name} 不可直接查询"
                    )

        metrics: dict[str, Any] = {}
        for metric in model.metrics:
            self._add_unique(metrics, metric.name.casefold(), metric.name, "指标")
            self._validate_metric(metric, entities, fields, fields_by_name)

        terms: dict[str, list[Any]] = {}
        seen_terms: set[tuple[str, str | None]] = set()
        for term in model.terms:
            key = term.term.casefold()
            signature = (key, term.when.casefold() if term.when else None)
            if signature in seen_terms:
                raise SemanticCompileError(f"重复术语: {term.term}")
            seen_terms.add(signature)
            terms.setdefault(key, []).append(term)
        for name, definitions in terms.items():
            if len(definitions) > 1 and any(term.when is None for term in definitions):
                raise SemanticCompileError(f"术语 {name} 存在无消歧条件的歧义")

        for relationship in model.relationships:
            self._require_field(
                relationship.from_entity,
                relationship.from_field,
                entities,
                fields,
                f"关系 {relationship.name} 的起点",
            )
            self._require_field(
                relationship.to_entity,
                relationship.to_field,
                entities,
                fields,
                f"关系 {relationship.name} 的终点",
            )

        for entity_name in model.policies.denied_entities:
            if entity_name.casefold() not in entities:
                raise SemanticCompileError(f"策略引用了不存在的实体: {entity_name}")
        for field_ref in model.policies.denied_fields:
            self._resolve_field_reference(
                field_ref, entities, fields, fields_by_name, "策略引用了不存在的字段"
            )

    def _validate_metric(
        self,
        metric: Any,
        entities: dict[str, Any],
        fields: dict[str, dict[str, Any]],
        fields_by_name: dict[str, list[str]],
    ) -> None:
        expression = _SINGLE_QUOTED_LITERAL.sub("", metric.expression)
        functions = {match.group(1).casefold() for match in _FUNCTION_CALL.finditer(expression)}
        unknown_functions = functions - _ALLOWED_AGGREGATIONS
        if unknown_functions:
            raise SemanticCompileError(
                f"指标 {metric.name} 使用未允许的聚合: {', '.join(sorted(unknown_functions))}"
            )
        if not functions:
            raise SemanticCompileError(f"指标 {metric.name} 必须使用允许的聚合")

        qualified_spans: list[tuple[int, int]] = []
        for match in _QUALIFIED_FIELD.finditer(expression):
            qualified_spans.append(match.span())
            self._require_field(
                match.group(1), match.group(2), entities, fields, f"指标 {metric.name}"
            )
        for match in _IDENTIFIER.finditer(expression):
            if any(start <= match.start() and match.end() <= end for start, end in qualified_spans):
                continue
            identifier = match.group(0)
            identifier_key = identifier.casefold()
            following = expression[match.end() :].lstrip()
            if identifier_key in _SQL_KEYWORDS or following.startswith("("):
                continue
            self._resolve_field_reference(
                identifier, entities, fields, fields_by_name, f"指标 {metric.name} 引用了不存在的字段"
            )

        for dimension in metric.allowed_dimensions:
            self._resolve_field_reference(
                dimension, entities, fields, fields_by_name, f"指标 {metric.name} 引用了不存在的维度"
            )
        for required_filter in metric.required_filters:
            self._resolve_field_reference(
                required_filter,
                entities,
                fields,
                fields_by_name,
                f"指标 {metric.name} 引用了不存在的过滤字段",
            )
        if metric.default_time_field is not None:
            self._resolve_field_reference(
                metric.default_time_field,
                entities,
                fields,
                fields_by_name,
                f"指标 {metric.name} 引用了不存在的时间字段",
            )

    @staticmethod
    def _add_unique(index: dict[str, Any], key: str, value: str, kind: str) -> None:
        if key in index:
            raise SemanticCompileError(f"重复{kind}: {value}")
        index[key] = value

    @staticmethod
    def _require_field(
        entity_name: str,
        field_name: str,
        entities: dict[str, Any],
        fields: dict[str, dict[str, Any]],
        context: str,
    ) -> None:
        entity_key = entity_name.casefold()
        field_key = field_name.casefold()
        if entity_key not in entities:
            raise SemanticCompileError(f"{context} 引用了不存在的实体: {entity_name}")
        if field_key not in fields[entity_key]:
            raise SemanticCompileError(
                f"{context} 引用了不存在的字段: {entity_name}.{field_name}"
            )

    def _resolve_field_reference(
        self,
        reference: str,
        entities: dict[str, Any],
        fields: dict[str, dict[str, Any]],
        fields_by_name: dict[str, list[str]],
        context: str,
    ) -> None:
        if "." in reference:
            entity_name, field_name = reference.split(".", 1)
            self._require_field(entity_name, field_name, entities, fields, context)
            return
        matches = fields_by_name.get(reference.casefold(), [])
        if not matches:
            raise SemanticCompileError(f"{context}: {reference}")
        if len(matches) > 1:
            raise SemanticCompileError(f"{context} 且字段引用不明确: {reference}")

    def _write_snapshot_atomically(self, snapshot: SemanticSnapshot, output_root: Path) -> None:
        payload = snapshot.model_dump(mode="json")
        jsonschema.validate(payload, SemanticSnapshot.model_json_schema())
        destination = output_root / snapshot.datasource / f"{snapshot.model_version}.json"
        self._write_json_atomically(payload, destination)

    def _write_manifest_atomically(self, snapshot: SemanticSnapshot, output_root: Path) -> None:
        manifest_path = output_root / "manifest.json"
        manifest: dict[str, Any] = {"active": {}}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SemanticCompileError(f"无法读取现有清单: {exc}") from exc
            if not isinstance(existing, dict) or not isinstance(existing.get("active"), dict):
                raise SemanticCompileError("现有清单格式无效")
            manifest = existing
        manifest["active"][snapshot.datasource] = snapshot.model_version
        self._write_json_atomically(manifest, manifest_path)

    @staticmethod
    def _write_json_schema(destination: Path) -> None:
        SemanticCompiler._write_json_atomically(SemanticModel.model_json_schema(), destination)

    @staticmethod
    def _write_json_atomically(payload: Any, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)
