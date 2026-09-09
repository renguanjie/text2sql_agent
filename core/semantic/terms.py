"""One deterministic, fail-closed resolver for compiled semantic terms."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from .models import SemanticSnapshot


class SemanticTermResolutionError(ValueError):
    """Raised when a matched term cannot resolve to one compiled definition."""


@dataclass(frozen=True)
class ResolvedSemanticTerm:
    term: str
    target_kind: str
    target_name: str
    target: str
    when: str | None


@dataclass(frozen=True)
class ResolvedSemanticTerms:
    matches: tuple[ResolvedSemanticTerm, ...] = ()

    @property
    def metric_names(self) -> frozenset[str]:
        return frozenset(
            item.target_name.casefold()
            for item in self.matches
            if item.target_kind == "metric"
        )


class SemanticTermResolver:
    """Resolve each matched spelling exactly once against a compiled snapshot."""

    def resolve(
        self, question: str, snapshot: SemanticSnapshot
    ) -> ResolvedSemanticTerms:
        normalized_question = self.normalize(question)
        grouped: dict[str, list] = {}
        order: list[str] = []
        for term in snapshot.terms:
            normalized_term = self.normalize(term.term)
            if not normalized_term or normalized_term not in normalized_question:
                continue
            if normalized_term not in grouped:
                grouped[normalized_term] = []
                order.append(normalized_term)
            grouped[normalized_term].append(term)

        selected = [
            key
            for key in order
            if not any(key != other and key in other for other in order)
        ]
        matches: list[ResolvedSemanticTerm] = []
        for key in selected:
            definitions = grouped[key]
            if len(definitions) != 1:
                raise SemanticTermResolutionError(
                    f"术语 {definitions[0].term} 无法唯一解析"
                )
            definition = definitions[0]
            kind, name = self._resolve_target(definition.target, snapshot)
            if definition.when is not None:
                self._validate_condition(definition.when, snapshot)
            matches.append(
                ResolvedSemanticTerm(
                    term=definition.term,
                    target_kind=kind,
                    target_name=name,
                    target=f"{kind}:{name}",
                    when=definition.when,
                )
            )
        return ResolvedSemanticTerms(matches=tuple(matches))

    @staticmethod
    def normalize(value: str) -> str:
        if not isinstance(value, str):
            raise SemanticTermResolutionError("术语查询必须是字符串")
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    @staticmethod
    def _resolve_target(target: str, snapshot: SemanticSnapshot) -> tuple[str, str]:
        kind, separator, raw_name = target.partition(":")
        kind = kind.casefold()
        if not separator or kind not in {"metric", "entity", "field", "relationship"}:
            raise SemanticTermResolutionError("术语 target 不是规范化引用")

        if kind == "metric":
            matches = [
                item.name
                for item in snapshot.metrics
                if item.name.casefold() == raw_name.casefold()
            ]
        elif kind == "entity":
            matches = [
                item.name
                for item in snapshot.entities
                if item.name.casefold() == raw_name.casefold()
            ]
        elif kind == "relationship":
            matches = [
                item.name
                for item in snapshot.relationships
                if item.name.casefold() == raw_name.casefold()
            ]
        else:
            if raw_name.count(".") != 1:
                raise SemanticTermResolutionError("术语字段 target 不是限定引用")
            entity_name, field_name = raw_name.split(".", 1)
            matches = [
                f"{entity.name}.{semantic_field.name}"
                for entity in snapshot.entities
                for semantic_field in entity.fields
                if entity.name.casefold() == entity_name.casefold()
                and semantic_field.name.casefold() == field_name.casefold()
            ]
        if len(matches) != 1:
            raise SemanticTermResolutionError("术语 target 无法唯一解析")
        return kind, matches[0]

    @classmethod
    def _validate_condition(cls, condition: str, snapshot: SemanticSnapshot) -> None:
        try:
            statements = [
                item
                for item in parse(condition, read=snapshot.dialect)
                if item is not None
            ]
        except ParseError as exc:
            raise SemanticTermResolutionError("术语 when 无法解析") from exc
        if len(statements) != 1:
            raise SemanticTermResolutionError("术语 when 必须是单个布尔条件")
        atoms = cls._conjunctive_atoms(statements[0])
        if (
            atoms is None
            or not atoms
            or not all(isinstance(atom, exp.Predicate) for atom in atoms)
        ):
            raise SemanticTermResolutionError("术语 when 必须是确定性布尔条件")

        fields = {
            (entity.name.casefold(), semantic_field.name.casefold())
            for entity in snapshot.entities
            for semantic_field in entity.fields
        }
        for column in statements[0].find_all(exp.Column):
            if not column.table or (
                column.table.casefold(), column.name.casefold()
            ) not in fields:
                raise SemanticTermResolutionError("术语 when 包含非规范化字段")

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


__all__ = [
    "ResolvedSemanticTerm",
    "ResolvedSemanticTerms",
    "SemanticTermResolutionError",
    "SemanticTermResolver",
]
