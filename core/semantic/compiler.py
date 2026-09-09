"""Compile versioned semantic-model YAML into immutable JSON snapshots."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pydantic import ValidationError
from sqlglot import exp, parse
from sqlglot.errors import ParseError

from .models import (
    SemanticEntity,
    SemanticField,
    SemanticMetric,
    SemanticModel,
    SemanticPolicies,
    SemanticRelationship,
    SemanticSnapshot,
    SemanticTerm,
)
from .relationship_predicates import (
    RelationshipPredicateError,
    parse_relationship_predicate,
)
from .storage import (
    SemanticStorageError,
    VerifiedRead,
    atomic_write_bytes,
    read_bytes_no_follow,
    remove_file_no_follow,
    semantic_transaction_lock,
    validate_no_symlink_path,
)

COMPILER_VERSION = "2"
_ALLOWED_AGGREGATIONS = frozenset({"sum", "count", "avg", "min", "max"})
_DATASOURCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_QUERY_EXPRESSION_TYPES = tuple(
    expression_type
    for name in ("Query", "Select", "SetOperation", "Subquery")
    if isinstance((expression_type := getattr(exp, name, None)), type)
)


class SemanticCompileError(ValueError):
    """Raised when a semantic-model source cannot safely be compiled."""


def utc_now_iso() -> str:
    """Return a UTC timestamp in the stable ISO-8601 form used by snapshots."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class _Indexes:
    entities: dict[str, SemanticEntity]
    fields: dict[str, dict[str, SemanticField]]
    fields_by_name: dict[str, tuple[tuple[SemanticEntity, SemanticField], ...]]
    metrics: dict[str, SemanticMetric]
    relationships: dict[str, SemanticRelationship]


@dataclass(frozen=True)
class _WritePlan:
    artifact: Path
    manifest_path: Path
    schema_path: Path
    manifest: dict[str, Any]
    existing_snapshot: SemanticSnapshot | None
    manifest_read: VerifiedRead | None
    artifact_read: VerifiedRead | None


class SemanticCompiler:
    def __init__(self, clock=None):
        self._clock = clock or utc_now_iso

    def compile_file(
        self,
        source: Path,
        output_root: Path,
        *,
        expected_datasource: str | None = None,
    ) -> SemanticSnapshot:
        """Validate completely, then publish one immutable version and its manifest."""
        source = Path(source)
        output_root = Path(output_root)
        self._validate_source_path(source)
        try:
            source_read = read_bytes_no_follow(source)
            raw_bytes = source_read.content
            payload = yaml.safe_load(raw_bytes.decode("utf-8"))
            parsed_model = SemanticModel.model_validate(payload)
            source_read.verify()
        except (
            OSError,
            UnicodeDecodeError,
            yaml.YAMLError,
            ValidationError,
            SemanticStorageError,
        ) as exc:
            raise SemanticCompileError(str(exc)) from exc

        expected = expected_datasource if expected_datasource is not None else source.stem
        self._validate_expected_datasource(expected)
        if parsed_model.datasource != expected:
            qualifier = (
                "显式 expected_datasource" if expected_datasource is not None else "源文件名"
            )
            raise SemanticCompileError(
                f"语义模型 datasource {parsed_model.datasource} 与{qualifier} {expected} 不一致"
            )

        model = self._normalize_and_validate(parsed_model)
        semantic_payload = model.model_dump(mode="json")
        candidate = SemanticSnapshot(
            **semantic_payload,
            source_sha256=sha256(raw_bytes).hexdigest(),
            content_sha256=sha256(self._canonical_bytes(semantic_payload)).hexdigest(),
            compiler_version=COMPILER_VERSION,
            compiled_at=self._clock(),
        )

        # Reject every pre-existing unsafe destination before creating the lock root.
        self._validated_write_paths(candidate, output_root)
        try:
            with semantic_transaction_lock(output_root):
                # Hold one process/thread lock across the complete storage transaction.
                plan = self._prepare_write(candidate, output_root)
                self._verify_plan_reads(plan)
                snapshot = plan.existing_snapshot or candidate
                artifact_read = plan.artifact_read
                if plan.existing_snapshot is None:
                    self._write_json_atomically(
                        snapshot.model_dump(mode="json"), plan.artifact
                    )
                    artifact_read = self._read_verified(plan.artifact)
                self._verify_plan_reads(plan, artifact_read=artifact_read)
                self._write_json_schema(plan.schema_path)
                self._verify_plan_reads(plan, artifact_read=artifact_read)
                activation_read = self._verify_activation_artifact(
                    plan,
                    snapshot,
                    artifact_read,
                )
                try:
                    self._write_json_atomically(plan.manifest, plan.manifest_path)
                    self._verify_read(activation_read)
                except SemanticCompileError:
                    self._restore_manifest(plan)
                    raise
                return snapshot
        except SemanticStorageError as exc:
            raise SemanticCompileError(str(exc)) from None

    def _normalize_and_validate(self, model: SemanticModel) -> SemanticModel:
        indexes = self._build_indexes(model)
        relationships = tuple(
            self._normalize_relationship(item, indexes, model.dialect)
            for item in model.relationships
        )
        metrics = tuple(
            self._normalize_metric(item, indexes, model.dialect)
            for item in model.metrics
        )
        terms = tuple(
            self._normalize_term(item, indexes, model.dialect) for item in model.terms
        )
        self._validate_term_uniqueness(terms)
        denied_entities = tuple(
            self._require_entity(item, indexes, "策略引用了不存在的实体").name
            for item in model.policies.denied_entities
        )
        denied_fields = tuple(
            self._canonical_field(item, indexes, "策略引用了不存在的字段")[0]
            for item in model.policies.denied_fields
        )
        return SemanticModel(
            schema_version=model.schema_version,
            model_version=model.model_version,
            datasource=model.datasource,
            dialect=model.dialect,
            entities=model.entities,
            relationships=relationships,
            metrics=metrics,
            terms=terms,
            policies=SemanticPolicies(
                denied_entities=denied_entities,
                denied_fields=denied_fields,
            ),
        )

    def _build_indexes(self, model: SemanticModel) -> _Indexes:
        entities: dict[str, SemanticEntity] = {}
        fields: dict[str, dict[str, SemanticField]] = {}
        fields_by_name: dict[str, list[tuple[SemanticEntity, SemanticField]]] = {}
        for entity in model.entities:
            key = entity.name.casefold()
            self._add_unique(entities, key, entity, "实体")
            fields[key] = {}
            for semantic_field in entity.fields:
                field_key = semantic_field.name.casefold()
                self._add_unique(
                    fields[key],
                    field_key,
                    semantic_field,
                    f"实体 {entity.name} 的字段",
                )
                fields_by_name.setdefault(field_key, []).append((entity, semantic_field))
                if (
                    semantic_field.sensitivity == "restricted"
                    and semantic_field.directly_queryable
                ):
                    raise SemanticCompileError(
                        f"受限字段 {entity.name}.{semantic_field.name} 不可直接查询"
                    )

        metrics: dict[str, SemanticMetric] = {}
        for metric in model.metrics:
            self._add_unique(metrics, metric.name.casefold(), metric, "指标")
        relationships: dict[str, SemanticRelationship] = {}
        for relationship in model.relationships:
            self._add_unique(
                relationships, relationship.name.casefold(), relationship, "关系"
            )
        return _Indexes(
            entities=entities,
            fields=fields,
            fields_by_name={
                name: tuple(values) for name, values in fields_by_name.items()
            },
            metrics=metrics,
            relationships=relationships,
        )

    def _normalize_relationship(
        self,
        relationship: SemanticRelationship,
        indexes: _Indexes,
        dialect: str,
    ) -> SemanticRelationship:
        from_entity = self._require_entity(
            relationship.from_entity, indexes, f"关系 {relationship.name} 的起点"
        )
        to_entity = self._require_entity(
            relationship.to_entity, indexes, f"关系 {relationship.name} 的终点"
        )
        from_field = self._require_field(
            from_entity.name,
            relationship.from_field,
            indexes,
            f"关系 {relationship.name} 的起点",
        )
        to_field = self._require_field(
            to_entity.name,
            relationship.to_field,
            indexes,
            f"关系 {relationship.name} 的终点",
        )
        condition = None
        if relationship.extra_condition is not None:
            context = f"关系 {relationship.name} 的 extra_condition"
            try:
                predicate = parse_relationship_predicate(
                    relationship.extra_condition,
                    dialect,
                    context,
                )
            except RelationshipPredicateError as exc:
                raise SemanticCompileError(str(exc)) from None
            self._canonicalize_columns(
                predicate.expression,
                indexes,
                context,
                allowed_entities={
                    from_entity.name.casefold(),
                    to_entity.name.casefold(),
                },
            )
            condition = predicate.sql(dialect)
        return SemanticRelationship(
            name=relationship.name,
            from_entity=from_entity.name,
            from_field=from_field.name,
            to_entity=to_entity.name,
            to_field=to_field.name,
            join_type=relationship.join_type,
            extra_condition=condition,
        )

    def _normalize_metric(
        self,
        metric: SemanticMetric,
        indexes: _Indexes,
        dialect: str,
    ) -> SemanticMetric:
        expression = self._parse_single_expression(
            metric.expression, dialect, f"指标 {metric.name} 的表达式"
        )
        nodes = tuple(self._walk_nodes(expression))
        if isinstance(expression, (exp.Alias, *_QUERY_EXPRESSION_TYPES)) or any(
            isinstance(node, _QUERY_EXPRESSION_TYPES) for node in nodes
        ):
            raise SemanticCompileError(f"指标 {metric.name} 必须是单个聚合表达式")

        aggregate_nodes = tuple(node for node in nodes if isinstance(node, exp.AggFunc))
        if not aggregate_nodes:
            raise SemanticCompileError(f"指标 {metric.name} 必须使用允许的聚合")
        for function in (node for node in nodes if isinstance(node, exp.Func)):
            function_name = type(function).__name__.casefold()
            if function_name not in _ALLOWED_AGGREGATIONS:
                raise SemanticCompileError(
                    f"指标 {metric.name} 使用未允许的聚合: {function_name}"
                )

        self._canonicalize_columns(expression, indexes, f"指标 {metric.name}")
        for column in expression.find_all(exp.Column):
            parent = column.parent
            while parent is not None:
                if isinstance(parent, exp.AggFunc):
                    break
                parent = parent.parent
            if parent is None:
                raise SemanticCompileError(
                    f"指标 {metric.name} 的字段必须位于允许的聚合内"
                )

        allowed_dimensions: list[str] = []
        for reference in metric.allowed_dimensions:
            canonical, semantic_field = self._canonical_field(
                reference, indexes, f"指标 {metric.name} 引用了不存在的维度"
            )
            if semantic_field.role not in {"dimension", "time"}:
                raise SemanticCompileError(
                    f"指标 {metric.name} 的允许维度 {canonical} 角色不是维度或时间"
                )
            allowed_dimensions.append(canonical)

        required_filters = tuple(
            self._canonical_field(
                reference,
                indexes,
                f"指标 {metric.name} 引用了不存在的过滤字段",
            )[0]
            for reference in metric.required_filters
        )
        default_time_field = None
        if metric.default_time_field is not None:
            default_time_field, semantic_field = self._canonical_field(
                metric.default_time_field,
                indexes,
                f"指标 {metric.name} 引用了不存在的时间字段",
            )
            if semantic_field.role != "time":
                raise SemanticCompileError(
                    f"指标 {metric.name} 的默认时间字段 {default_time_field} 角色不是 time"
                )

        return SemanticMetric(
            name=metric.name,
            name_cn=metric.name_cn,
            synonyms=metric.synonyms,
            expression=expression.sql(dialect=dialect),
            allowed_dimensions=tuple(allowed_dimensions),
            required_filters=required_filters,
            default_time_field=default_time_field,
        )

    def _normalize_term(
        self,
        term: SemanticTerm,
        indexes: _Indexes,
        dialect: str,
    ) -> SemanticTerm:
        return SemanticTerm(
            term=term.term,
            target=self._canonical_target(term.target, indexes),
            when=(
                self._canonical_boolean_expression(
                    term.when,
                    indexes,
                    dialect,
                    f"术语 {term.term} 的 when",
                )
                if term.when is not None
                else None
            ),
        )

    def _canonical_target(self, target: str, indexes: _Indexes) -> str:
        prefix, separator, value = target.partition(":")
        if separator and prefix.casefold() in {
            "metric",
            "entity",
            "field",
            "relationship",
        }:
            kind = prefix.casefold()
            if kind == "metric":
                metric = indexes.metrics.get(value.casefold())
                if metric is None:
                    raise SemanticCompileError(
                        f"术语 target 引用了不存在的指标: {target}"
                    )
                return f"metric:{metric.name}"
            if kind == "entity":
                entity = self._require_entity(
                    value, indexes, "术语 target 引用了不存在的实体"
                )
                return f"entity:{entity.name}"
            if kind == "relationship":
                relationship = indexes.relationships.get(value.casefold())
                if relationship is None:
                    raise SemanticCompileError(
                        f"术语 target 引用了不存在的关系: {target}"
                    )
                return f"relationship:{relationship.name}"
            field_ref = self._canonical_field(
                value, indexes, "术语 target 引用了不存在的字段"
            )[0]
            return f"field:{field_ref}"

        candidates: list[str] = []
        if metric := indexes.metrics.get(target.casefold()):
            candidates.append(f"metric:{metric.name}")
        if entity := indexes.entities.get(target.casefold()):
            candidates.append(f"entity:{entity.name}")
        if relationship := indexes.relationships.get(target.casefold()):
            candidates.append(f"relationship:{relationship.name}")
        try:
            field_reference, _ = self._canonical_field(
                target, indexes, "术语 target 引用了不存在的字段"
            )
        except SemanticCompileError as exc:
            if "不明确" in str(exc):
                raise SemanticCompileError(f"术语 target 不明确: {target}") from None
        else:
            candidates.append(f"field:{field_reference}")

        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise SemanticCompileError(f"术语 target 引用了不存在的对象: {target}")
        if len(candidates) != 1:
            raise SemanticCompileError(f"术语 target 不明确: {target}")
        return candidates[0]

    @staticmethod
    def _validate_term_uniqueness(terms: tuple[SemanticTerm, ...]) -> None:
        definitions: dict[str, list[SemanticTerm]] = {}
        signatures: set[tuple[str, str | None]] = set()
        for term in terms:
            key = term.term.casefold()
            signature = (key, term.when.casefold() if term.when else None)
            if signature in signatures:
                raise SemanticCompileError(f"重复术语: {term.term}")
            signatures.add(signature)
            definitions.setdefault(key, []).append(term)
        for name, matches in definitions.items():
            if len(matches) > 1 and any(term.when is None for term in matches):
                raise SemanticCompileError(f"术语 {name} 存在无消歧条件的歧义")

    def _canonical_boolean_expression(
        self,
        value: str,
        indexes: _Indexes,
        dialect: str,
        context: str,
        *,
        allowed_entities: set[str] | None = None,
    ) -> str:
        expression = self._parse_single_expression(value, dialect, context)
        atoms = self._conjunctive_atoms(expression)
        if (
            atoms is None
            or not atoms
            or not all(isinstance(atom, exp.Predicate) for atom in atoms)
        ):
            raise SemanticCompileError(f"{context} 必须是单个确定性布尔条件")
        self._canonicalize_columns(
            expression,
            indexes,
            context,
            allowed_entities=allowed_entities,
        )
        return expression.sql(dialect=dialect)

    @staticmethod
    def _parse_single_expression(
        value: str, dialect: str, context: str
    ) -> exp.Expression:
        try:
            statements = [item for item in parse(value, read=dialect) if item is not None]
        except ParseError as exc:
            raise SemanticCompileError(f"{context} 无法完整解析") from exc
        if len(statements) != 1:
            raise SemanticCompileError(f"{context} 必须是单个表达式")
        return statements[0]

    def _canonicalize_columns(
        self,
        expression: exp.Expression,
        indexes: _Indexes,
        context: str,
        *,
        allowed_entities: set[str] | None = None,
    ) -> None:
        for column in expression.find_all(exp.Column):
            raw = f"{column.table}.{column.name}" if column.table else column.name
            canonical, _ = self._canonical_field(
                raw, indexes, f"{context} 引用了不存在的字段"
            )
            entity_name, field_name = canonical.split(".", 1)
            if (
                allowed_entities is not None
                and entity_name.casefold() not in allowed_entities
            ):
                raise SemanticCompileError(
                    f"{context} 引用了关系端点之外的字段: {canonical}"
                )
            column.set("table", exp.to_identifier(entity_name))
            column.set("this", exp.to_identifier(field_name))

    @classmethod
    def _conjunctive_atoms(
        cls, expression: exp.Expression | None
    ) -> tuple[exp.Expression, ...] | None:
        if expression is None or isinstance(expression, exp.Or):
            return None
        if isinstance(expression, exp.And):
            left = cls._conjunctive_atoms(expression.this)
            right = cls._conjunctive_atoms(expression.expression)
            return None if left is None or right is None else left + right
        return (expression,)

    @staticmethod
    def _walk_nodes(expression: exp.Expression):
        for item in expression.walk():
            yield item[0] if isinstance(item, tuple) else item

    @staticmethod
    def _add_unique(
        index: dict[str, Any], key: str, value: Any, kind: str
    ) -> None:
        if key in index:
            name = getattr(value, "name", str(value))
            raise SemanticCompileError(f"重复{kind}: {name}")
        index[key] = value

    @staticmethod
    def _require_entity(
        reference: str, indexes: _Indexes, context: str
    ) -> SemanticEntity:
        entity = indexes.entities.get(reference.casefold())
        if entity is None:
            raise SemanticCompileError(f"{context}: {reference}")
        return entity

    def _require_field(
        self,
        entity_name: str,
        field_name: str,
        indexes: _Indexes,
        context: str,
    ) -> SemanticField:
        entity = self._require_entity(entity_name, indexes, context)
        semantic_field = indexes.fields[entity.name.casefold()].get(field_name.casefold())
        if semantic_field is None:
            raise SemanticCompileError(
                f"{context} 引用了不存在的字段: {entity_name}.{field_name}"
            )
        return semantic_field

    def _canonical_field(
        self,
        reference: str,
        indexes: _Indexes,
        context: str,
    ) -> tuple[str, SemanticField]:
        if reference.count(".") > 1:
            raise SemanticCompileError(f"{context}: {reference}")
        if "." in reference:
            entity_name, field_name = reference.split(".", 1)
            entity = self._require_entity(entity_name, indexes, context)
            semantic_field = self._require_field(
                entity.name, field_name, indexes, context
            )
            return f"{entity.name}.{semantic_field.name}", semantic_field
        matches = indexes.fields_by_name.get(reference.casefold(), ())
        if not matches:
            raise SemanticCompileError(f"{context}: {reference}")
        if len(matches) != 1:
            raise SemanticCompileError(f"{context} 且字段引用不明确: {reference}")
        entity, semantic_field = matches[0]
        return f"{entity.name}.{semantic_field.name}", semantic_field

    def _prepare_write(
        self, snapshot: SemanticSnapshot, output_root: Path
    ) -> _WritePlan:
        artifact, manifest_path, schema_path = self._validated_write_paths(
            snapshot, output_root
        )

        manifest: dict[str, Any] = {"active": {}}
        manifest_read = None
        if manifest_path.exists():
            manifest, manifest_read = self._read_manifest(manifest_path)

        existing_snapshot = None
        artifact_read = None
        if artifact.exists():
            existing_snapshot, artifact_read = self._read_existing_snapshot(artifact)
            if existing_snapshot.source_sha256 != snapshot.source_sha256:
                raise SemanticCompileError(
                    f"语义版本 {snapshot.datasource}@{snapshot.model_version} "
                    "已由不同源字节编译；请升级 model_version"
                )
            if (
                existing_snapshot.datasource != snapshot.datasource
                or existing_snapshot.model_version != snapshot.model_version
                or existing_snapshot.content_sha256 != snapshot.content_sha256
            ):
                raise SemanticCompileError(
                    f"语义版本 {snapshot.datasource}@{snapshot.model_version} "
                    "已存在且内容不同"
                )

        next_manifest = json.loads(json.dumps(manifest))
        next_manifest["active"][snapshot.datasource] = snapshot.model_version
        payload = (existing_snapshot or snapshot).model_dump(mode="json")
        try:
            jsonschema.validate(payload, SemanticSnapshot.model_json_schema())
        except jsonschema.ValidationError as exc:
            raise SemanticCompileError("编译产物格式无效") from exc
        return _WritePlan(
            artifact=artifact,
            manifest_path=manifest_path,
            schema_path=schema_path,
            manifest=next_manifest,
            existing_snapshot=existing_snapshot,
            manifest_read=manifest_read,
            artifact_read=artifact_read,
        )

    def _validated_write_paths(
        self, snapshot: SemanticSnapshot, output_root: Path
    ) -> tuple[Path, Path, Path]:
        artifact = output_root / snapshot.datasource / f"{snapshot.model_version}.json"
        manifest_path = output_root / "manifest.json"
        schema_path = output_root.parent / "schema/semantic-model.schema.json"
        self._validate_output_path(output_root, artifact)
        self._validate_output_path(output_root, manifest_path)
        self._validate_output_path(output_root.parent, schema_path)
        return artifact, manifest_path, schema_path

    @staticmethod
    def _read_manifest(path: Path) -> tuple[dict[str, Any], VerifiedRead]:
        try:
            read = read_bytes_no_follow(path)
            manifest = json.loads(read.content.decode("utf-8"))
            read.verify()
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            SemanticStorageError,
        ) as exc:
            raise SemanticCompileError(f"无法读取现有清单: {exc}") from exc
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("active"), dict
        ):
            raise SemanticCompileError("现有清单格式无效")
        if not all(
            isinstance(name, str) and isinstance(version, str)
            for name, version in manifest["active"].items()
        ):
            raise SemanticCompileError("现有清单格式无效")
        return manifest, read

    def _read_existing_snapshot(
        self, path: Path
    ) -> tuple[SemanticSnapshot, VerifiedRead]:
        try:
            read = read_bytes_no_follow(path)
            payload = json.loads(read.content.decode("utf-8"))
            read.verify()
            jsonschema.validate(payload, SemanticSnapshot.model_json_schema())
            snapshot = SemanticSnapshot.model_validate(payload)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            jsonschema.ValidationError,
            ValidationError,
            SemanticStorageError,
        ) as exc:
            raise SemanticCompileError(
                "现有版本 artifact 格式无效，拒绝覆盖"
            ) from exc
        semantic_payload = {
            name: getattr(snapshot, name) for name in SemanticModel.model_fields
        }
        digest = sha256(
            self._canonical_bytes(
                SemanticModel.model_validate(semantic_payload).model_dump(mode="json")
            )
        ).hexdigest()
        if (
            snapshot.content_sha256 != digest
            or _SHA256_HEX.fullmatch(snapshot.source_sha256) is None
        ):
            raise SemanticCompileError("现有版本 artifact 摘要无效，拒绝覆盖")
        return snapshot, read

    def _verify_activation_artifact(
        self,
        plan: _WritePlan,
        expected_snapshot: SemanticSnapshot,
        expected_read: VerifiedRead | None,
    ) -> VerifiedRead:
        if expected_read is None:
            raise SemanticCompileError("待激活 artifact 未经安全读取")
        self._verify_read(expected_read)
        current_snapshot, current_read = self._read_existing_snapshot(plan.artifact)
        if (
            current_read.file_identity != expected_read.file_identity
            or current_read.content_sha256 != expected_read.content_sha256
            or current_read.content != expected_read.content
            or current_snapshot != expected_snapshot
        ):
            raise SemanticCompileError("待激活 artifact 身份或内容已改变")
        self._verify_read(plan.manifest_read)
        self._verify_read(current_read)
        return current_read

    @staticmethod
    def _restore_manifest(plan: _WritePlan) -> None:
        try:
            if plan.manifest_read is None:
                remove_file_no_follow(plan.manifest_path)
                return
            atomic_write_bytes(plan.manifest_path, plan.manifest_read.content)
            restored = read_bytes_no_follow(plan.manifest_path)
            if (
                restored.content != plan.manifest_read.content
                or restored.content_sha256 != plan.manifest_read.content_sha256
            ):
                raise SemanticStorageError("无法恢复原始语义清单内容")
        except SemanticStorageError as exc:
            raise SemanticCompileError(f"无法恢复原始语义清单: {exc}") from None

    @staticmethod
    def _read_verified(path: Path) -> VerifiedRead:
        try:
            return read_bytes_no_follow(path)
        except SemanticStorageError as exc:
            raise SemanticCompileError(str(exc)) from None

    @staticmethod
    def _verify_read(read: VerifiedRead | None) -> None:
        if read is None:
            return
        try:
            read.verify()
        except SemanticStorageError as exc:
            raise SemanticCompileError(str(exc)) from None

    @classmethod
    def _verify_plan_reads(
        cls,
        plan: _WritePlan,
        *,
        artifact_read: VerifiedRead | None = None,
    ) -> None:
        cls._verify_read(plan.manifest_read)
        cls._verify_read(artifact_read or plan.artifact_read)

    @staticmethod
    def _validate_source_path(source: Path) -> None:
        if source.is_symlink():
            raise SemanticCompileError("语义源文件不得是符号链接")
        if not source.is_file():
            raise SemanticCompileError(f"语义源文件不存在: {source}")

    @staticmethod
    def _validate_expected_datasource(value: str) -> None:
        if not isinstance(value, str) or _DATASOURCE.fullmatch(value) is None:
            raise SemanticCompileError("expected datasource 不是安全的数据源标识符")

    @staticmethod
    def _validate_output_path(boundary: Path, destination: Path) -> None:
        boundary_absolute = boundary.absolute()
        try:
            validate_no_symlink_path(boundary_absolute)
        except SemanticStorageError as exc:
            raise SemanticCompileError(str(exc)) from None
        if boundary_absolute.is_symlink():
            raise SemanticCompileError(f"编译输出路径不得包含符号链接: {boundary}")
        if boundary_absolute.exists() and not boundary_absolute.is_dir():
            raise SemanticCompileError(f"编译输出路径的父级不是目录: {boundary}")
        try:
            relative = destination.absolute().relative_to(boundary_absolute)
        except ValueError:
            raise SemanticCompileError("编译输出路径越界") from None
        current = boundary_absolute
        for index, part in enumerate(relative.parts):
            current = current / part
            if current.is_symlink():
                raise SemanticCompileError(f"编译输出路径不得包含符号链接: {current}")
            if not current.exists():
                continue
            is_destination = index == len(relative.parts) - 1
            if is_destination and not current.is_file():
                raise SemanticCompileError(f"编译输出路径不是普通文件: {current}")
            if not is_destination and not current.is_dir():
                raise SemanticCompileError(f"编译输出路径的父级不是目录: {current}")
        try:
            destination.resolve(strict=False).relative_to(
                boundary_absolute.resolve(strict=False)
            )
        except ValueError:
            raise SemanticCompileError("编译输出解析后越界") from None

    @staticmethod
    def _canonical_bytes(payload: Any) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _write_json_schema(destination: Path) -> None:
        SemanticCompiler._write_json_atomically(
            SemanticModel.model_json_schema(), destination
        )

    @staticmethod
    def _write_json_atomically(payload: Any, destination: Path) -> None:
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode(
                "utf-8"
            )
            + b"\n"
        )
        try:
            atomic_write_bytes(destination, encoded)
        except SemanticStorageError as exc:
            raise SemanticCompileError(str(exc)) from None
