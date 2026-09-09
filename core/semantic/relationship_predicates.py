"""One fail-closed grammar for governed relationship predicates."""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp, parse
from sqlglot.errors import ParseError


class RelationshipPredicateError(ValueError):
    """Raised when a relationship predicate is outside the canonical grammar."""


@dataclass(frozen=True)
class CanonicalRelationshipPredicate:
    """A parsed predicate whose atoms are safe to canonicalize and compare."""

    expression: exp.Expression
    atoms: tuple[exp.Expression, ...]

    def sql(self, dialect: str) -> str:
        return " AND ".join(_canonical_atom_sql(atom, dialect) for atom in self.atoms)


def parse_relationship_predicate(
    value: str,
    dialect: str,
    context: str,
) -> CanonicalRelationshipPredicate:
    """Parse exactly one relationship predicate and enforce its atom grammar."""
    try:
        expressions = [item for item in parse(value, read=dialect) if item is not None]
    except ParseError as exc:
        raise RelationshipPredicateError(f"{context} 无法完整解析") from exc
    if len(expressions) != 1:
        raise RelationshipPredicateError(f"{context} 必须是单个表达式")
    expression = expressions[0]
    return CanonicalRelationshipPredicate(
        expression=expression,
        atoms=relationship_predicate_atoms(expression, context),
    )


def relationship_predicate_atoms(
    expression: exp.Expression,
    context: str = "关系条件",
) -> tuple[exp.Expression, ...]:
    """Return canonical AND atoms or reject every unsupported AST shape."""
    atoms = _flatten_conjunction(expression, context)
    if not atoms:
        raise _grammar_error(context)
    return atoms


def _flatten_conjunction(
    expression: exp.Expression,
    context: str,
) -> tuple[exp.Expression, ...]:
    expression = _unwrap_parentheses(expression)
    if isinstance(expression, exp.And):
        return _flatten_conjunction(expression.this, context) + _flatten_conjunction(
            expression.expression, context
        )
    _validate_atom(expression, context)
    return (expression,)


def _validate_atom(expression: exp.Expression, context: str) -> None:
    if isinstance(expression, exp.EQ):
        operands = (expression.this, expression.expression)
        if not all(_is_equality_operand(operand) for operand in operands) or not any(
            isinstance(operand, exp.Column) for operand in operands
        ):
            raise _grammar_error(context)
        return
    if _is_null_atom(expression):
        return
    if isinstance(expression, exp.Not) and _is_null_atom(
        _unwrap_parentheses(expression.this)
    ):
        return
    raise _grammar_error(context)


def _is_equality_operand(expression: exp.Expression) -> bool:
    return isinstance(expression, (exp.Column, exp.Literal, exp.Boolean))


def _is_null_atom(expression: exp.Expression) -> bool:
    expression = _unwrap_parentheses(expression)
    return (
        isinstance(expression, exp.Is)
        and isinstance(expression.this, exp.Column)
        and isinstance(expression.expression, exp.Null)
    )


def _canonical_atom_sql(expression: exp.Expression, dialect: str) -> str:
    if isinstance(expression, exp.Not):
        inner = _unwrap_parentheses(expression.this)
        if _is_null_atom(inner):
            return f"{inner.this.sql(dialect=dialect)} IS NOT NULL"
    if _is_null_atom(expression):
        return f"{expression.this.sql(dialect=dialect)} IS NULL"
    return expression.sql(dialect=dialect)


def _unwrap_parentheses(expression: exp.Expression) -> exp.Expression:
    while isinstance(expression, exp.Paren):
        expression = expression.this
    return expression


def _grammar_error(context: str) -> RelationshipPredicateError:
    return RelationshipPredicateError(
        f"{context} 必须是确定性布尔条件，仅支持 =、IS NULL、IS NOT NULL 的 AND 合取"
    )


__all__ = [
    "CanonicalRelationshipPredicate",
    "RelationshipPredicateError",
    "parse_relationship_predicate",
    "relationship_predicate_atoms",
]
