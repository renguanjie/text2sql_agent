"""Deterministic, bounded semantic context for Chinese natural-language questions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .models import (
    SemanticContext,
    SemanticEntity,
    SemanticField,
    SemanticMetric,
    SemanticRelationship,
    SemanticSnapshot,
)
from .terms import (
    ResolvedSemanticTerm,
    SemanticTermResolutionError,
    SemanticTermResolver,
)


_QUALIFIED_FIELD = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


class SemanticContextError(ValueError):
    """Raised when a complete mandatory semantic dependency bundle cannot fit."""


class SemanticPromptBundle(str):
    """String-compatible carrier marking a complete structured semantic context."""

    def __new__(cls, context: SemanticContext):
        if not isinstance(context, SemanticContext):
            raise TypeError("SemanticPromptBundle requires SemanticContext")
        instance = super().__new__(cls, context.rendered)
        instance.context = context
        return instance


@dataclass(frozen=True)
class _ContextItem:
    kind: str
    identifier: str
    line: str


class SemanticContextBuilder:
    def __init__(self, max_chars: int, max_items: int):
        if max_chars <= 0 or max_items <= 0:
            raise ValueError("max_chars 和 max_items 必须为正数")
        self._max_chars = max_chars
        self._max_items = max_items
        self._term_resolver = SemanticTermResolver()

    def build(self, question: str, snapshot: SemanticSnapshot) -> SemanticContext:
        header = f"版本: {snapshot.datasource}@{snapshot.model_version}"
        if len(header) > self._max_chars:
            raise SemanticContextError("max_chars 无法容纳版本信息")

        try:
            resolved_terms = self._term_resolver.resolve(question, snapshot)
        except SemanticTermResolutionError as exc:
            raise SemanticContextError(str(exc)) from None
        matched_metrics = [
            metric
            for metric in snapshot.metrics
            if self._matches(question, self._metric_terms(metric))
            or metric.name.casefold() in resolved_terms.metric_names
        ]
        mandatory_items = self._deduplicated(
            [],
            tuple(
                item
                for metric in matched_metrics
                for item in self._metric_dependency_bundle(metric, snapshot)
            )
            + tuple(
                item
                for term in resolved_terms.matches
                for item in self._term_dependency_bundle(term, snapshot)
            ),
        )
        self._validate_mandatory_items(header, mandatory_items)
        selected = mandatory_items

        for item in self._lower_priority_hits(question, snapshot, selected):
            self._append_if_fits(header, selected, item)

        rendered = "\n".join([header, *(item.line for item in selected)])
        return SemanticContext(
            datasource=snapshot.datasource,
            semantic_version=snapshot.model_version,
            entities=tuple(item.identifier for item in selected if item.kind == "entity"),
            fields=tuple(item.identifier for item in selected if item.kind == "field"),
            relationships=tuple(item.identifier for item in selected if item.kind == "relationship"),
            metrics=tuple(item.identifier for item in selected if item.kind == "metric"),
            terms=tuple(item.identifier for item in selected if item.kind == "term"),
            policies=tuple(item.identifier for item in selected if item.kind == "policy"),
            rendered=rendered,
        )

    def _validate_mandatory_items(self, header: str, items: list[_ContextItem]) -> None:
        if len(items) > self._max_items:
            raise SemanticContextError("max_items 无法容纳完整指标依赖组")
        rendered = "\n".join([header, *(item.line for item in items)])
        if len(rendered) > self._max_chars:
            raise SemanticContextError("max_chars 无法容纳完整指标依赖组")

    def _append_if_fits(
        self, header: str, selected: list[_ContextItem], item: _ContextItem
    ) -> None:
        candidate = self._deduplicated(selected, (item,))
        if self._fits(header, candidate):
            selected[:] = candidate

    def _fits(self, header: str, items: list[_ContextItem]) -> bool:
        if len(items) > self._max_items:
            return False
        rendered = "\n".join([header, *(item.line for item in items)])
        return len(rendered) <= self._max_chars

    @staticmethod
    def _deduplicated(
        selected: list[_ContextItem], additions: tuple[_ContextItem, ...]
    ) -> list[_ContextItem]:
        existing = {(item.kind, item.identifier) for item in selected}
        result = list(selected)
        for item in additions:
            key = (item.kind, item.identifier)
            if key not in existing:
                result.append(item)
                existing.add(key)
        return result

    def _metric_dependency_bundle(
        self, metric: SemanticMetric, snapshot: SemanticSnapshot
    ) -> tuple[_ContextItem, ...]:
        fields_by_reference = self._fields_by_reference(snapshot)
        required_fields = self._metric_fields(metric, fields_by_reference)
        related_relationships = tuple(
            relationship
            for relationship in snapshot.relationships
            if relationship.from_entity.casefold()
            in {entity.name.casefold() for entity, _ in required_fields}
            or relationship.to_entity.casefold()
            in {entity.name.casefold() for entity, _ in required_fields}
        )
        policy_fields = [
            self._resolve_field(reference, fields_by_reference)
            for reference in snapshot.policies.denied_fields
        ]
        policy_fields = [field for field in policy_fields if field is not None]

        entity_names = {
            entity.name.casefold() for entity, _ in required_fields
        } | {
            relationship.from_entity.casefold() for relationship in related_relationships
        } | {
            relationship.to_entity.casefold() for relationship in related_relationships
        } | {
            entity.name.casefold() for entity, _ in policy_fields
        } | {name.casefold() for name in snapshot.policies.denied_entities}
        entities = tuple(
            entity for entity in snapshot.entities if entity.name.casefold() in entity_names
        )

        items: list[_ContextItem] = [
            _ContextItem("metric", metric.name, f"指标: {metric.name_cn} = {metric.expression}")
        ]
        items.extend(self._entity_item(entity) for entity in entities)
        items.extend(self._field_item(entity, field) for entity, field in required_fields)
        items.extend(self._relationship_item(relationship) for relationship in related_relationships)
        items.extend(
            _ContextItem("policy", f"entity:{name}", f"策略: 禁止实体 {name}")
            for name in snapshot.policies.denied_entities
        )
        items.extend(
            _ContextItem("policy", f"field:{entity.name}.{field.name}", f"策略: 禁止字段 {entity.name}.{field.name}")
            for entity, field in policy_fields
        )
        return tuple(items)

    def _term_dependency_bundle(
        self, term: ResolvedSemanticTerm, snapshot: SemanticSnapshot
    ) -> tuple[_ContextItem, ...]:
        suffix = f"; 条件: {term.when}" if term.when is not None else ""
        items: list[_ContextItem] = [
            _ContextItem(
                "term",
                term.term,
                f"术语: {term.term} -> {term.target}{suffix}",
            )
        ]
        if term.target_kind == "metric":
            metric = next(
                item
                for item in snapshot.metrics
                if item.name.casefold() == term.target_name.casefold()
            )
            items.extend(self._metric_dependency_bundle(metric, snapshot))
        elif term.target_kind == "entity":
            entity = next(
                item
                for item in snapshot.entities
                if item.name.casefold() == term.target_name.casefold()
            )
            items.append(self._entity_item(entity))
        elif term.target_kind == "field":
            resolved = self._resolve_field(
                term.target_name, self._fields_by_reference(snapshot)
            )
            if resolved is not None:
                items.extend(
                    (self._entity_item(resolved[0]), self._field_item(*resolved))
                )
        elif term.target_kind == "relationship":
            relationship = next(
                item
                for item in snapshot.relationships
                if item.name.casefold() == term.target_name.casefold()
            )
            items.append(self._relationship_item(relationship))
            fields = self._fields_by_reference(snapshot)
            for reference in (
                f"{relationship.from_entity}.{relationship.from_field}",
                f"{relationship.to_entity}.{relationship.to_field}",
            ):
                resolved = self._resolve_field(reference, fields)
                if resolved is not None:
                    items.extend(
                        (self._entity_item(resolved[0]), self._field_item(*resolved))
                    )

        fields = self._fields_by_reference(snapshot)
        if term.when is not None:
            for match in _QUALIFIED_FIELD.finditer(term.when):
                resolved = self._resolve_field(
                    f"{match.group(1)}.{match.group(2)}", fields
                )
                if resolved is not None:
                    items.extend(
                        (self._entity_item(resolved[0]), self._field_item(*resolved))
                    )
        target_entities = {
            item.identifier.casefold() for item in items if item.kind == "entity"
        }
        target_fields = {
            item.identifier.casefold() for item in items if item.kind == "field"
        }
        items.extend(
            _ContextItem("policy", f"entity:{name}", f"策略: 禁止实体 {name}")
            for name in snapshot.policies.denied_entities
            if name.casefold() in target_entities
        )
        items.extend(
            _ContextItem(
                "policy", f"field:{reference}", f"策略: 禁止字段 {reference}"
            )
            for reference in snapshot.policies.denied_fields
            if reference.casefold() in target_fields
        )
        return tuple(items)

    def _lower_priority_hits(
        self,
        question: str,
        snapshot: SemanticSnapshot,
        selected: list[_ContextItem],
    ) -> tuple[_ContextItem, ...]:
        selected_keys = {(item.kind, item.identifier) for item in selected}
        items: list[_ContextItem] = []
        for entity in snapshot.entities:
            if self._matches(question, self._entity_terms(entity)):
                item = self._entity_item(entity)
                if (item.kind, item.identifier) not in selected_keys:
                    items.append(item)
        for entity in snapshot.entities:
            for field in entity.fields:
                if self._matches(question, self._field_terms(field)):
                    item = self._field_item(entity, field)
                    if (item.kind, item.identifier) not in selected_keys:
                        items.append(item)
        for relationship in snapshot.relationships:
            if self._matches(question, (relationship.name,)):
                item = self._relationship_item(relationship)
                if (item.kind, item.identifier) not in selected_keys:
                    items.append(item)
        return tuple(items)

    @staticmethod
    def _fields_by_reference(
        snapshot: SemanticSnapshot,
    ) -> dict[str, tuple[SemanticEntity, SemanticField]]:
        return {
            f"{entity.name}.{field.name}".casefold(): (entity, field)
            for entity in snapshot.entities
            for field in entity.fields
        }

    def _metric_fields(
        self,
        metric: SemanticMetric,
        fields_by_reference: dict[str, tuple[SemanticEntity, SemanticField]],
    ) -> tuple[tuple[SemanticEntity, SemanticField], ...]:
        references = [
            f"{match.group(1)}.{match.group(2)}" for match in _QUALIFIED_FIELD.finditer(metric.expression)
        ]
        if metric.default_time_field is not None:
            references.append(metric.default_time_field)
        references.extend(metric.allowed_dimensions)
        references.extend(metric.required_filters)

        fields: list[tuple[SemanticEntity, SemanticField]] = []
        seen: set[str] = set()
        for reference in references:
            resolved = self._resolve_field(reference, fields_by_reference)
            if resolved is None:
                continue
            key = f"{resolved[0].name}.{resolved[1].name}".casefold()
            if key not in seen:
                fields.append(resolved)
                seen.add(key)
        return tuple(fields)

    @staticmethod
    def _resolve_field(
        reference: str,
        fields_by_reference: dict[str, tuple[SemanticEntity, SemanticField]],
    ) -> tuple[SemanticEntity, SemanticField] | None:
        if "." in reference:
            return fields_by_reference.get(reference.casefold())
        matches = [
            value
            for key, value in fields_by_reference.items()
            if key.rsplit(".", 1)[1] == reference.casefold()
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _entity_item(entity: SemanticEntity) -> _ContextItem:
        return _ContextItem("entity", entity.name, f"实体: {entity.name} ({entity.name_cn})")

    @staticmethod
    def _field_item(entity: SemanticEntity, field: SemanticField) -> _ContextItem:
        identifier = f"{entity.name}.{field.name}"
        return _ContextItem(
            "field", identifier, f"字段: {identifier} ({field.name_cn}, {field.role})"
        )

    @staticmethod
    def _relationship_item(relationship: SemanticRelationship) -> _ContextItem:
        identifier = relationship.name
        return _ContextItem(
            "relationship",
            identifier,
            "关系: "
            f"{relationship.from_entity}.{relationship.from_field} = "
            f"{relationship.to_entity}.{relationship.to_field} [{relationship.join_type}]",
        )

    @classmethod
    def _matches(cls, question: str, terms: tuple[str, ...]) -> bool:
        normalized_question = cls._normalize(question)
        return any(
            normalized_term and normalized_term in normalized_question
            for term in terms
            if (normalized_term := cls._normalize(term))
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    @staticmethod
    def _metric_terms(metric: SemanticMetric) -> tuple[str, ...]:
        return (metric.name_cn, metric.name, *metric.synonyms)

    @staticmethod
    def _entity_terms(entity: SemanticEntity) -> tuple[str, ...]:
        return (entity.name_cn, entity.name, *entity.synonyms)

    @staticmethod
    def _field_terms(field: SemanticField) -> tuple[str, ...]:
        return (field.name_cn, field.name, *field.synonyms)
