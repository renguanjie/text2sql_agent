"""Deterministic, database-independent validation of SQL against semantic snapshots."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from sqlglot import exp, parse, parse_one
from sqlglot.optimizer.scope import Scope, traverse_scope

from .models import SemanticMetric, SemanticSnapshot, ValidationIssue, ValidationReport

_FieldReference = tuple[str, str]


@dataclass(frozen=True)
class SQLScope:
    """Stable, serializable facts extracted from one SQL statement."""

    tables: frozenset[str]
    aliases: Mapping[str, str]
    columns: tuple[tuple[str | None, str], ...]
    joins: tuple[tuple[str, str, str, str], ...]
    group_by: tuple[str, ...]
    filters: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))


@dataclass(frozen=True)
class SQLStructure:
    """Public structural facts used by offline evaluators and other read-only tools."""

    scope: SQLScope
    tables: frozenset[str]
    fields: frozenset[str]
    relationships: tuple[tuple[str, str], ...]
    filters: tuple[str, ...]
    time_grains: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _Resolution:
    references: frozenset[_FieldReference] = frozenset()
    known: bool = False


@dataclass
class _ScopeInfo:
    scope: Scope
    sources: dict[str, exp.Table | Scope]
    physical_tables: frozenset[str]
    projections: dict[str, _Resolution] = field(default_factory=dict)
    projection_expressions: dict[str, exp.Expression] = field(default_factory=dict)


@dataclass(frozen=True)
class _JoinFact:
    info: _ScopeInfo
    left_tables: frozenset[str]
    right_tables: frozenset[str]
    pairs: tuple[tuple[str, str, str, str], ...]
    atom_signatures: frozenset[Any]
    supported: bool


@dataclass(frozen=True)
class _Lineage:
    fields: frozenset[str]
    metric_candidates: tuple[tuple[Any, frozenset[str], frozenset[str]], ...]
    dimensions: frozenset[str]
    filter_fields: frozenset[str]


@dataclass(frozen=True)
class _SQLAnalysis:
    tree: exp.Expression
    scope: SQLScope
    scope_infos: tuple[_ScopeInfo, ...]
    active_scope_infos: tuple[_ScopeInfo, ...]
    root_output_infos: tuple[_ScopeInfo, ...]
    join_facts: tuple[_JoinFact, ...]
    unresolved_column_indexes: frozenset[int]


class _UnsupportedSQL(ValueError):
    """Raised when semantic validation cannot prove a supported query shape."""


def _casefold(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _semantic_field_index(
    semantic_indexes: Mapping[str, Iterable[str] | Mapping[str, Any]] | None,
) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for table, columns in (semantic_indexes or {}).items():
        names = columns.keys() if isinstance(columns, Mapping) else columns
        result[_casefold(str(table)).rsplit(".", 1)[-1]] = frozenset(
            _casefold(str(column)) for column in names
        )
    return result


def _table_name(table: exp.Table) -> str:
    return _casefold(table.name).rsplit(".", 1)[-1]


def _unique_sources(info: _ScopeInfo) -> tuple[exp.Table | Scope, ...]:
    seen: set[int] = set()
    sources: list[exp.Table | Scope] = []
    for source in info.sources.values():
        marker = id(source)
        if marker not in seen:
            sources.append(source)
            seen.add(marker)
    return tuple(sources)


def _source_tables(source: exp.Table | Scope, infos: Mapping[int, _ScopeInfo]) -> frozenset[str]:
    if isinstance(source, exp.Table):
        return frozenset({_table_name(source)})
    return infos[id(source)].physical_tables


def _resolve_column(
    column: exp.Column,
    info: _ScopeInfo,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
    *,
    include_projection_aliases: bool = True,
) -> _Resolution:
    name = _casefold(column.name)
    qualifier = _casefold(column.table) if column.table else ""

    if qualifier:
        source = info.sources.get(qualifier)
        if source is None:
            return _Resolution(frozenset({(qualifier, name)}), False)
        if isinstance(source, exp.Table):
            return _Resolution(frozenset({(_table_name(source), name)}), True)
        source_projection = infos[id(source)].projections
        if name in source_projection:
            return source_projection[name]
        # A derived relation is a projection boundary: its physical inputs must
        # never make an unexposed output appear valid.
        return _Resolution(known=False)

    candidates: list[_Resolution] = []
    physical_sources: list[str] = []
    for source in _unique_sources(info):
        if isinstance(source, exp.Table):
            table = _table_name(source)
            physical_sources.append(table)
            if not semantic_indexes or name in semantic_indexes.get(table, frozenset()):
                candidates.append(_Resolution(frozenset({(table, name)}), True))
        else:
            projection = infos[id(source)].projections
            if name in projection:
                candidates.append(projection[name])

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return _Resolution(known=False)
    if include_projection_aliases and name in info.projections:
        return info.projections[name]
    if len(set(physical_sources)) == 1:
        return _Resolution(frozenset({(physical_sources[0], name)}), False)
    return _Resolution(known=False)


def _immediate_columns(expression: exp.Expression, select: exp.Select) -> tuple[exp.Column, ...]:
    return tuple(
        column
        for column in expression.find_all(exp.Column)
        if column.find_ancestor(exp.Select) is select
    )


def _projection_resolution(
    expression: exp.Expression,
    info: _ScopeInfo,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
) -> _Resolution:
    references: set[_FieldReference] = set()
    known = True
    for column in _immediate_columns(expression, info.scope.expression):
        resolved = _resolve_column(
            column,
            info,
            infos,
            semantic_indexes,
            include_projection_aliases=False,
        )
        references.update(resolved.references)
        known = known and resolved.known
    return _Resolution(frozenset(references), known)


def _expand_star(
    expression: exp.Expression,
    info: _ScopeInfo,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
) -> dict[str, _Resolution]:
    qualifier = ""
    if isinstance(expression, exp.Column) and expression.name == "*":
        qualifier = _casefold(expression.table)

    sources = (info.sources.get(qualifier),) if qualifier else _unique_sources(info)
    projections: dict[str, _Resolution] = {}
    for source in sources:
        if source is None:
            continue
        if isinstance(source, exp.Table):
            table = _table_name(source)
            for name in sorted(semantic_indexes.get(table, frozenset())):
                projections.setdefault(
                    name, _Resolution(frozenset({(table, name)}), True)
                )
        else:
            for name, resolution in infos[id(source)].projections.items():
                projections.setdefault(name, resolution)
    return projections


def _reference_strings(
    expression: exp.Expression,
    info: _ScopeInfo,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
) -> tuple[str, ...]:
    values: list[str] = []
    for column in _immediate_columns(expression, info.scope.expression):
        resolved = _resolve_column(column, info, infos, semantic_indexes)
        if resolved.references:
            values.extend(f"{table}.{name}" for table, name in sorted(resolved.references))
        elif not resolved.known:
            qualifier = _casefold(column.table) if column.table else ""
            values.append(f"{qualifier}.{_casefold(column.name)}" if qualifier else _casefold(column.name))
    if not values:
        values.append(_casefold(expression.sql()))
    return tuple(values)


def _group_reference_strings(
    expression: exp.Expression,
    info: _ScopeInfo,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
) -> tuple[str, ...]:
    supported_time_groups = {
        "datetrunc",
        "day",
        "month",
        "quarter",
        "timetostr",
        "tsordstodate",
        "week",
        "year",
    }
    if not isinstance(expression, exp.Column) and expression.key not in supported_time_groups:
        return ("unsupported_group_expression",)
    values: list[str] = []
    unsupported = False
    columns = _immediate_columns(expression, info.scope.expression)
    for column in columns:
        resolved = _resolve_column(column, info, infos, semantic_indexes)
        if resolved.references and resolved.known:
            values.extend(f"{table}.{name}" for table, name in sorted(resolved.references))
        else:
            unsupported = True
    if not columns or unsupported:
        values.append("unsupported_group_expression")
    return tuple(dict.fromkeys(values))


def _source_for_node(node: exp.Expression, info: _ScopeInfo) -> exp.Table | Scope | None:
    name = _casefold(node.alias_or_name)
    return info.sources.get(name)


def _conjunctive_atoms(predicate: exp.Expression | None) -> tuple[exp.Expression, ...] | None:
    if predicate is None or isinstance(predicate, exp.Or):
        return None
    if isinstance(predicate, exp.And):
        left = _conjunctive_atoms(predicate.this)
        right = _conjunctive_atoms(predicate.expression)
        if left is None or right is None:
            return None
        return left + right
    return (predicate,)


def _normalized_column_references(
    expression: exp.Expression,
    info: _ScopeInfo,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
    aliases: Mapping[str, str],
) -> tuple[_FieldReference, ...] | None:
    references: list[_FieldReference] = []
    for column in _immediate_columns(expression, info.scope.expression):
        resolved = _resolve_column(column, info, infos, semantic_indexes)
        if len(resolved.references) != 1:
            return None
        table, name = next(iter(resolved.references))
        table = aliases.get(table, table)
        if table not in semantic_indexes or name not in semantic_indexes[table]:
            return None
        references.append((table, name))
    return tuple(references)


def _join_fact_for_predicate(
    predicate: exp.Expression | None,
    info: _ScopeInfo,
    left_tables: set[str],
    right_tables: set[str],
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
    aliases: Mapping[str, str],
) -> _JoinFact:
    atoms = _conjunctive_atoms(predicate)
    pairs: list[tuple[str, str, str, str]] = []
    signatures: set[Any] = set()
    supported = atoms is not None
    for atom in atoms or ():
        if not isinstance(atom, exp.EQ):
            supported = False
            continue
        references = _normalized_column_references(
            atom, info, infos, semantic_indexes, aliases
        )
        if not references:
            supported = False
            continue
        signatures.add(_expression_signature(atom, info, infos, semantic_indexes))
        distinct = tuple(dict.fromkeys(references))
        if len(distinct) > 2:
            supported = False
            continue
        if len(distinct) == 2:
            left_ref, right_ref = distinct
            if left_ref[0] in right_tables and right_ref[0] in left_tables:
                left_ref, right_ref = right_ref, left_ref
            if left_ref[0] in left_tables and right_ref[0] in right_tables:
                pairs.append((left_ref[0], left_ref[1], right_ref[0], right_ref[1]))
            elif left_ref[0] != right_ref[0]:
                supported = False
    if not pairs:
        supported = False
    return _JoinFact(
        info=info,
        left_tables=frozenset(left_tables),
        right_tables=frozenset(right_tables),
        pairs=tuple(pairs),
        atom_signatures=frozenset(signatures),
        supported=supported,
    )


def _extract_join_facts(
    info: _ScopeInfo,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
    aliases: Mapping[str, str],
) -> tuple[_JoinFact, ...]:
    select = info.scope.expression
    if not isinstance(select, exp.Select):
        return ()
    from_clause = select.args.get("from_") or select.args.get("from")
    if not isinstance(from_clause, exp.From) or from_clause.this is None:
        return ()
    left_source = _source_for_node(from_clause.this, info)
    left_tables = set(_source_tables(left_source, infos)) if left_source is not None else set()
    facts: list[_JoinFact] = []
    for join in select.args.get("joins") or ():
        right_source = _source_for_node(join.this, info)
        right_tables = (
            set(_source_tables(right_source, infos)) if right_source is not None else set()
        )
        if left_tables and right_tables:
            facts.append(
                _join_fact_for_predicate(
                    join.args.get("on"),
                    info,
                    left_tables,
                    right_tables,
                    infos,
                    semantic_indexes,
                    aliases,
                )
            )
        left_tables.update(right_tables)
    return tuple(facts)


def is_single_readonly_query(sql: str, dialect: str = "mysql") -> bool:
    """Return whether *sql* is exactly one query AST with no write operation."""
    try:
        statements = [statement for statement in parse(sql, read=dialect) if statement is not None]
    except Exception:
        return False
    query_types = tuple(
        expression_type
        for name in ("Select", "Union", "Intersect", "Except")
        if (expression_type := getattr(exp, name, None)) is not None
    )
    if len(statements) != 1 or not isinstance(statements[0], query_types):
        return False
    forbidden_nodes = {
        "Alter",
        "Command",
        "Commit",
        "Create",
        "Delete",
        "Drop",
        "Grant",
        "Insert",
        "Into",
        "LoadData",
        "Lock",
        "Merge",
        "Rollback",
        "Set",
        "Transaction",
        "TruncateTable",
        "Update",
        "Use",
    }
    forbidden_functions = {
        "BENCHMARK",
        "GET_LOCK",
        "LOAD_FILE",
        "PG_SLEEP",
        "RELEASE_LOCK",
        "SLEEP",
    }
    for item in statements[0].walk():
        node = item[0] if isinstance(item, tuple) else item
        if type(node).__name__ in forbidden_nodes or (
            isinstance(node, exp.Anonymous) and node.name.upper() in forbidden_functions
        ):
            return False
    return True


def _analyze_sql(
    sql: str,
    dialect: str,
    semantic_indexes: Mapping[str, Iterable[str] | Mapping[str, Any]] | None,
) -> _SQLAnalysis:
    statements = [statement for statement in parse(sql, read=dialect) if statement is not None]
    if len(statements) != 1:
        raise ValueError("exactly one SQL statement is required")
    tree = statements[0]
    set_operations = (exp.Union, exp.Intersect, exp.Except)
    if not isinstance(tree, exp.Select) or any(
        isinstance(item[0] if isinstance(item, tuple) else item, set_operations)
        for item in tree.walk()
    ):
        raise _UnsupportedSQL("unsupported root statement")
    forbidden_nodes = {
        "Alter",
        "Command",
        "Commit",
        "Create",
        "Delete",
        "Drop",
        "Grant",
        "Insert",
        "Into",
        "LoadData",
        "Lock",
        "Merge",
        "Rollback",
        "Set",
        "Transaction",
        "TruncateTable",
        "Update",
        "Use",
    }
    forbidden_functions = {
        "BENCHMARK",
        "GET_LOCK",
        "LOAD_FILE",
        "PG_SLEEP",
        "RELEASE_LOCK",
        "SLEEP",
    }
    for item in tree.walk():
        node = item[0] if isinstance(item, tuple) else item
        if type(node).__name__ in forbidden_nodes or (
            isinstance(node, exp.Anonymous) and node.name.upper() in forbidden_functions
        ):
            raise _UnsupportedSQL("query contains an unsupported operation")
    indexes = _semantic_field_index(semantic_indexes)
    infos_by_scope: dict[int, _ScopeInfo] = {}
    aliases: dict[str, str] = {}

    for sql_scope in traverse_scope(tree):
        selected_sources = getattr(sql_scope, "selected_sources", {})
        sources = {
            _casefold(name): selected[1]
            for name, selected in selected_sources.items()
        }
        physical_tables: set[str] = set()
        for source_name, source in sources.items():
            source_tables = _source_tables(source, infos_by_scope)
            physical_tables.update(source_tables)
            if len(source_tables) == 1:
                aliases[source_name] = next(iter(source_tables))
            if isinstance(source, exp.Table):
                table = _table_name(source)
                aliases[table] = table

        info = _ScopeInfo(
            scope=sql_scope,
            sources=sources,
            physical_tables=frozenset(physical_tables),
        )
        infos_by_scope[id(sql_scope)] = info

        select = sql_scope.expression
        if not isinstance(select, exp.Select):
            continue
        for expression in select.expressions:
            unaliased = expression.this if isinstance(expression, exp.Alias) else expression
            if isinstance(unaliased, exp.Star) or (
                isinstance(unaliased, exp.Column) and unaliased.name == "*"
            ):
                info.projections.update(
                    _expand_star(unaliased, info, infos_by_scope, indexes)
                )
                continue
            output_name = _casefold(expression.alias_or_name)
            if output_name:
                info.projection_expressions[output_name] = unaliased
                info.projections[output_name] = _projection_resolution(
                    unaliased, info, infos_by_scope, indexes
                )

    root_info = next(
        (info for info in infos_by_scope.values() if info.scope.is_root), None
    )
    if root_info is None:
        raise _UnsupportedSQL("query scope is unavailable")

    active_ids: set[int] = set()

    def activate(info: _ScopeInfo) -> None:
        marker = id(info.scope)
        if marker in active_ids:
            return
        active_ids.add(marker)
        for source in _unique_sources(info):
            if isinstance(source, Scope):
                activate(infos_by_scope[id(source)])
        for child in (*info.scope.subquery_scopes, *info.scope.union_scopes):
            activate(infos_by_scope[id(child)])

    activate(root_info)
    active_infos = tuple(
        info for info in infos_by_scope.values() if id(info.scope) in active_ids
    )
    def output_infos(info: _ScopeInfo) -> tuple[_ScopeInfo, ...]:
        if isinstance(info.scope.expression, exp.Select):
            return (info,)
        return tuple(
            leaf
            for child in info.scope.union_scopes
            for leaf in output_infos(infos_by_scope[id(child)])
        )

    root_output_infos = output_infos(root_info)
    active_tables = frozenset(
        table for info in active_infos for table in info.physical_tables
    )
    if (
        not active_tables
        or not root_output_infos
        or any(not info.physical_tables for info in root_output_infos)
    ):
        raise _UnsupportedSQL("query has no resolvable source")
    columns: list[tuple[str | None, str]] = []
    unresolved_column_indexes: set[int] = set()
    group_by: list[str] = []
    filters: list[str] = []
    for info in infos_by_scope.values():
        select = info.scope.expression
        if not isinstance(select, exp.Select):
            continue
        for column in info.scope.columns:
            resolved = _resolve_column(column, info, infos_by_scope, indexes)
            if resolved.references:
                columns.extend(
                    sorted((aliases.get(table, table), name) for table, name in resolved.references)
                )
            elif not resolved.known:
                qualifier = _casefold(column.table) if column.table else None
                source = info.sources.get(qualifier) if qualifier else None
                if isinstance(source, Scope):
                    pass
                elif qualifier is not None:
                    qualifier = aliases.get(qualifier, qualifier)
                elif len(_unique_sources(info)) == 1 and isinstance(
                    _unique_sources(info)[0], Scope
                ):
                    qualifier = next(iter(info.sources))
                elif (
                    qualifier is None
                    and len(_unique_sources(info)) >= 2
                    and all(isinstance(item, Scope) for item in _unique_sources(info))
                    and all(
                        _casefold(column.name)
                        not in infos_by_scope[id(item)].projections
                        for item in _unique_sources(info)
                    )
                ):
                    unresolved_column_indexes.add(len(columns))
                columns.append((qualifier, _casefold(column.name)))

        for expression in select.expressions:
            unaliased = expression.this if isinstance(expression, exp.Alias) else expression
            if isinstance(unaliased, exp.Star) or (
                isinstance(unaliased, exp.Column) and unaliased.name == "*"
            ):
                expanded = _expand_star(unaliased, info, infos_by_scope, indexes)
                if expanded:
                    for resolution in expanded.values():
                        columns.extend(sorted(resolution.references))
                else:
                    for source in _unique_sources(info):
                        for table in sorted(_source_tables(source, infos_by_scope)):
                            columns.append((table, "*"))

        group = select.args.get("group")
        for expression in group.expressions if isinstance(group, exp.Group) else ():
            group_by.extend(
                _group_reference_strings(expression, info, infos_by_scope, indexes)
            )
        for clause_name in ("where", "having"):
            clause = select.args.get(clause_name)
            if isinstance(clause, exp.Expression) and clause.this is not None:
                filters.extend(
                    _reference_strings(clause.this, info, infos_by_scope, indexes)
                )

    join_facts: list[_JoinFact] = []
    for info in infos_by_scope.values():
        join_facts.extend(_extract_join_facts(info, infos_by_scope, indexes, aliases))
    for info in active_infos:
        if not info.scope.is_subquery or info.scope.parent is None:
            continue
        parent = infos_by_scope[id(info.scope.parent)]
        left_tables = set(parent.physical_tables) - set(info.physical_tables)
        right_tables = set(info.physical_tables)
        if not left_tables or not right_tables or left_tables == right_tables:
            continue
        select = info.scope.expression
        where = select.args.get("where") if isinstance(select, exp.Select) else None
        predicate = where.this if isinstance(where, exp.Where) else None
        join_facts.append(
            _join_fact_for_predicate(
                predicate,
                info,
                left_tables,
                right_tables,
                infos_by_scope,
                indexes,
                aliases,
            )
        )

    joins: list[tuple[str, str, str, str]] = []
    for fact in join_facts:
        if fact.pairs:
            joins.extend(fact.pairs)
        elif fact.left_tables and fact.right_tables:
            joins.append(
                (sorted(fact.left_tables)[0], "", sorted(fact.right_tables)[0], "")
            )

    scope = SQLScope(
        tables=frozenset(table for info in infos_by_scope.values() for table in info.physical_tables),
        aliases=aliases,
        columns=tuple(columns),
        joins=tuple(joins),
        group_by=tuple(dict.fromkeys(group_by)),
        filters=tuple(dict.fromkeys(filters)),
    )
    return _SQLAnalysis(
        tree=tree,
        scope=scope,
        scope_infos=tuple(infos_by_scope.values()),
        active_scope_infos=active_infos,
        root_output_infos=root_output_infos,
        join_facts=tuple(join_facts),
        unresolved_column_indexes=frozenset(unresolved_column_indexes),
    )


def parse_sql_scope(
    sql: str,
    dialect: str = "mysql",
    semantic_indexes: Mapping[str, Iterable[str] | Mapping[str, Any]] | None = None,
) -> SQLScope:
    """Parse one statement into deterministic scope facts without database access."""
    return _analyze_sql(sql, dialect, semantic_indexes).scope


def _expression_signature(
    expression: Any,
    info: _ScopeInfo | None,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
    source_tables: frozenset[str] | None = None,
) -> Any:
    if isinstance(expression, (exp.Alias, exp.Paren)):
        return _expression_signature(
            expression.this, info, infos, semantic_indexes, source_tables
        )
    if isinstance(expression, exp.Column):
        if info is not None:
            resolved = _resolve_column(expression, info, infos, semantic_indexes)
            if len(resolved.references) == 1:
                table, name = next(iter(resolved.references))
                return ("column", table, name)
        name = _casefold(expression.name)
        qualifier = _casefold(expression.table) if expression.table else ""
        if qualifier:
            return ("column", qualifier, name)
        candidates = [
            table
            for table, fields in semantic_indexes.items()
            if name in fields and (source_tables is None or table in source_tables)
        ]
        if len(candidates) == 1:
            return ("column", candidates[0], name)
        return ("column", None, name)
    if isinstance(expression, exp.Identifier):
        return ("identifier", _casefold(expression.name))
    if isinstance(expression, exp.Literal):
        return ("literal", bool(expression.is_string), expression.this)
    if isinstance(expression, exp.Expression):
        return (
            expression.key,
            tuple(
                (
                    key,
                    _expression_signature(
                        value, info, infos, semantic_indexes, source_tables
                    ),
                )
                for key, value in sorted(expression.args.items())
                if value is not None
            ),
        )
    if isinstance(expression, list):
        return tuple(
            _expression_signature(value, info, infos, semantic_indexes, source_tables)
            for value in expression
        )
    return expression


def _derived_definition(
    column: exp.Column,
    info: _ScopeInfo,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
) -> tuple[_ScopeInfo, exp.Expression] | None:
    name = _casefold(column.name)
    qualifier = _casefold(column.table) if column.table else ""
    if qualifier:
        source = info.sources.get(qualifier)
        if isinstance(source, Scope):
            source_info = infos[id(source)]
            expression = source_info.projection_expressions.get(name)
            return (source_info, expression) if expression is not None else None
        return None

    candidates: list[tuple[_ScopeInfo, exp.Expression] | None] = []
    for source in _unique_sources(info):
        if isinstance(source, exp.Table):
            table = _table_name(source)
            if name in semantic_indexes.get(table, frozenset()):
                candidates.append(None)
        else:
            source_info = infos[id(source)]
            expression = source_info.projection_expressions.get(name)
            if expression is not None:
                candidates.append((source_info, expression))
    return candidates[0] if len(candidates) == 1 else None


def _expression_children(expression: exp.Expression) -> Iterable[exp.Expression]:
    for item in expression.iter_expressions():
        child = item[1] if isinstance(item, tuple) else item
        if isinstance(child, exp.Expression):
            yield child


def _iter_expression_nodes(expression: exp.Expression) -> Iterable[exp.Expression]:
    yield expression
    for child in _expression_children(expression):
        if isinstance(child, exp.Subquery):
            yield child
            continue
        yield from _iter_expression_nodes(child)


def _lineage_for_analysis(
    analysis: _SQLAnalysis,
    semantic_indexes: Mapping[str, frozenset[str]],
) -> _Lineage:
    infos = {id(info.scope): info for info in analysis.scope_infos}
    aliases = analysis.scope.aliases
    fields: set[str] = set()
    metric_candidates: list[tuple[Any, frozenset[str], frozenset[str]]] = []
    dimensions: set[str] = set()
    filter_fields: set[str] = set()
    seen_metric_definitions: set[tuple[int, int]] = set()

    def normalized_references(expression: exp.Expression, info: _ScopeInfo) -> set[str]:
        references: set[str] = set()
        is_star = isinstance(expression, exp.Star) or (
            isinstance(expression, exp.Column) and expression.name == "*"
        )
        if is_star:
            for resolution in _expand_star(
                expression, info, infos, semantic_indexes
            ).values():
                references.update(
                    f"{aliases.get(table, table)}.{name}"
                    for table, name in resolution.references
                    if aliases.get(table, table) in semantic_indexes
                )
            return references
        for column in _immediate_columns(expression, info.scope.expression):
            resolved = _resolve_column(column, info, infos, semantic_indexes)
            references.update(
                f"{aliases.get(table, table)}.{name}"
                for table, name in resolved.references
                if aliases.get(table, table) in semantic_indexes
            )
        return references

    supported_filter_types = (
        exp.EQ,
        exp.NEQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.Like,
        exp.ILike,
        exp.In,
        exp.Between,
        exp.Is,
    )

    def scope_filter_fields(info: _ScopeInfo) -> set[str]:
        proven: set[str] = set()
        select = info.scope.expression
        if not isinstance(select, exp.Select):
            return proven
        for clause_name in ("where", "having"):
            clause = select.args.get(clause_name)
            if not isinstance(clause, exp.Expression) or clause.this is None:
                continue
            atoms = _conjunctive_atoms(clause.this)
            for atom in atoms or ():
                if not isinstance(atom, supported_filter_types):
                    continue
                columns = _immediate_columns(atom, select)
                if not columns or (
                    len(columns) > 1
                    and all(
                        isinstance(side, exp.Column)
                        for side in (atom.this, atom.expression)
                    )
                ):
                    continue
                resolved = _normalized_column_references(
                    atom, info, infos, semantic_indexes, aliases
                )
                if resolved is not None:
                    proven.update(f"{table}.{name}" for table, name in resolved)
        return proven

    def collect_metric_expression(
        expression: exp.Expression,
        info: _ScopeInfo,
        applicable_filters: frozenset[str],
    ) -> None:
        marker = (id(info.scope), id(expression))
        if marker in seen_metric_definitions:
            return
        seen_metric_definitions.add(marker)
        metric_candidates.append(
            (
                _expression_signature(expression, info, infos, semantic_indexes),
                info.physical_tables,
                applicable_filters,
            )
        )
        for column in _immediate_columns(expression, info.scope.expression):
            derived = _derived_definition(column, info, infos, semantic_indexes)
            if derived is not None:
                child_filters = applicable_filters | frozenset(
                    scope_filter_fields(derived[0])
                )
                collect_metric_expression(derived[1], derived[0], child_filters)
        for node in _iter_expression_nodes(expression):
            if not isinstance(node, exp.Subquery):
                continue
            child = next(
                (
                    candidate
                    for candidate in info.scope.subquery_scopes
                    if candidate.expression is node.this
                ),
                None,
            )
            if child is not None:
                child_info = infos[id(child)]
                for output in child_info.scope.expression.expressions:
                    collect_metric_expression(
                        output,
                        child_info,
                        frozenset(scope_filter_fields(child_info)),
                    )

    def collect_nonaggregate(
        expression: exp.Expression,
        info: _ScopeInfo,
        inside_aggregate: bool = False,
    ) -> None:
        if isinstance(expression, exp.Star) or (
            isinstance(expression, exp.Column) and expression.name == "*"
        ):
            if not inside_aggregate:
                dimensions.update(normalized_references(expression, info))
            return
        if isinstance(expression, exp.Subquery):
            child = next(
                (
                    candidate
                    for candidate in info.scope.subquery_scopes
                    if candidate.expression is expression.this
                ),
                None,
            )
            if child is not None:
                child_info = infos[id(child)]
                for output in child_info.scope.expression.expressions:
                    collect_nonaggregate(output, child_info, inside_aggregate)
            return
        if isinstance(expression, exp.Column):
            derived = _derived_definition(expression, info, infos, semantic_indexes)
            if derived is not None:
                collect_nonaggregate(derived[1], derived[0], inside_aggregate)
            elif not inside_aggregate:
                dimensions.update(normalized_references(expression, info))
            return
        nested_aggregate = inside_aggregate or isinstance(expression, exp.AggFunc)
        for child in _expression_children(expression):
            collect_nonaggregate(child, info, nested_aggregate)

    for info in analysis.root_output_infos:
        for expression in info.scope.expression.expressions:
            fields.update(normalized_references(expression, info))
            collect_metric_expression(
                expression, info, frozenset(scope_filter_fields(info))
            )
            collect_nonaggregate(expression, info)

    lineage_scope_ids: set[int] = set()

    def include_source_lineage(info: _ScopeInfo) -> None:
        marker = id(info.scope)
        if marker in lineage_scope_ids:
            return
        lineage_scope_ids.add(marker)
        for source in _unique_sources(info):
            if isinstance(source, Scope):
                include_source_lineage(infos[id(source)])

    for info in analysis.root_output_infos:
        include_source_lineage(info)

    for info in analysis.scope_infos:
        if id(info.scope) not in lineage_scope_ids:
            continue
        select = info.scope.expression
        if not isinstance(select, exp.Select):
            continue
        group = select.args.get("group")
        for expression in group.expressions if isinstance(group, exp.Group) else ():
            group_refs = _group_reference_strings(
                expression, info, infos, semantic_indexes
            )
            dimensions.update(group_refs)
            fields.update(ref for ref in group_refs if "." in ref)
        for clause_name in ("where", "having"):
            clause = select.args.get(clause_name)
            if not isinstance(clause, exp.Expression) or clause.this is None:
                continue
            fields.update(normalized_references(clause.this, info))
            filter_fields.update(scope_filter_fields(info))

    return _Lineage(
        fields=frozenset(fields),
        metric_candidates=tuple(metric_candidates),
        dimensions=frozenset(dimensions),
        filter_fields=frozenset(filter_fields),
    )


def _signature_text(signature: Any) -> str:
    return json.dumps(signature, ensure_ascii=False, separators=(",", ":"))


def _filter_signature_text(
    expression: exp.Expression,
    info: _ScopeInfo | None,
    infos: Mapping[int, _ScopeInfo],
    semantic_indexes: Mapping[str, frozenset[str]],
    source_tables: frozenset[str] | None = None,
) -> str:
    if isinstance(expression, exp.EQ):
        operands = (
            _expression_signature(
                expression.this, info, infos, semantic_indexes, source_tables
            ),
            _expression_signature(
                expression.expression, info, infos, semantic_indexes, source_tables
            ),
        )
        return _signature_text(
            ("eq", tuple(sorted(operands, key=_signature_text)))
        )
    return _signature_text(
        _expression_signature(
            expression, info, infos, semantic_indexes, source_tables
        )
    )


def normalize_filter_expression(
    expression: str,
    dialect: str = "mysql",
    semantic_indexes: Mapping[str, Iterable[str] | Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Return stable structural signatures for a required conjunctive predicate."""
    tree = parse_one(expression, read=dialect)
    atoms = _conjunctive_atoms(tree)
    if atoms is None:
        raise ValueError("filter must be a deterministic conjunction")
    indexes = _semantic_field_index(semantic_indexes)
    for column in tree.find_all(exp.Column):
        name = _casefold(column.name)
        qualifier = _casefold(column.table) if column.table else ""
        if qualifier:
            if qualifier not in indexes or name not in indexes[qualifier]:
                raise ValueError("filter references an unknown field")
            continue
        candidates = [table for table, fields in indexes.items() if name in fields]
        if len(candidates) != 1:
            raise ValueError("filter field is unknown or ambiguous")
    source_tables = frozenset(indexes)
    return tuple(
        sorted(
            _filter_signature_text(
                atom,
                None,
                {},
                indexes,
                source_tables,
            )
            for atom in atoms
        )
    )


def _date_format_time_facts(format_text: str) -> frozenset[str]:
    tokens: set[str] = set()
    index = 0
    while index < len(format_text):
        if format_text[index] != "%":
            index += 1
            continue
        if index + 1 >= len(format_text):
            return frozenset()
        token = format_text[index + 1]
        if token != "%":
            if not token.isascii() or not token.isalpha():
                return frozenset()
            tokens.add(token)
        index += 2

    # These are MySQL DATE_FORMAT tokens, not sqlglot's normalized Python
    # strftime tokens. In particular, raw %B/%G/%W must not be confused with
    # normalized forms of MySQL's legal %M/%x/%u tokens.
    year_tokens = {"Y", "y", "x", "X"}
    month_tokens = {"m", "c", "b", "M"}
    day_tokens = {"d", "e", "D", "j"}
    week_tokens = {"U", "u", "V", "v"}
    allowed = year_tokens | month_tokens | day_tokens | week_tokens
    if not tokens or not tokens <= allowed:
        return frozenset()
    has_year = bool(tokens & year_tokens)
    has_month = bool(tokens & month_tokens)
    has_day = bool(tokens & day_tokens)
    has_week = bool(tokens & week_tokens)
    if has_year and has_week and not (has_month or has_day):
        return frozenset({"exact:week"})
    if has_year and has_day and not has_week:
        if "j" in tokens or has_month:
            return frozenset({"exact:day"})
        return frozenset()
    if has_year and has_month and not (has_day or has_week):
        return frozenset({"exact:month"})
    if has_year and not (has_month or has_day or has_week):
        return frozenset({"exact:year", "part:year"})
    return frozenset()


def _raw_mysql_date_formats(
    sql: str, normalized_tree: exp.Expression
) -> Mapping[int, str]:
    """Bind raw MySQL DATE_FORMAT literals to normalized AST nodes by order."""
    try:
        neutral_statements = [item for item in parse(sql) if item is not None]
    except Exception:
        return MappingProxyType({})
    if len(neutral_statements) != 1:
        return MappingProxyType({})

    normalized_nodes = [
        node
        for item in normalized_tree.walk()
        if (node := item[0] if isinstance(item, tuple) else item).key.casefold()
        == "timetostr"
    ]
    raw_literals: list[str] = []
    for item in neutral_statements[0].walk():
        node = item[0] if isinstance(item, tuple) else item
        if not isinstance(node, exp.Anonymous) or node.name.casefold() != "date_format":
            continue
        arguments = node.expressions
        if len(arguments) != 2 or not isinstance(arguments[1], exp.Literal) or not arguments[1].is_string:
            return MappingProxyType({})
        raw_literals.append(str(arguments[1].this))
    if len(normalized_nodes) != len(raw_literals):
        return MappingProxyType({})
    return MappingProxyType(
        {id(node): raw for node, raw in zip(normalized_nodes, raw_literals, strict=True)}
    )


def _time_facts_for_node(
    node: exp.Expression, raw_date_formats: Mapping[int, str]
) -> tuple[frozenset[str], bool]:
    key = node.key.casefold()
    if key == "datetrunc":
        unit = node.args.get("unit")
        value = str(getattr(unit, "this", unit) or "").casefold()
        facts = (
            frozenset({f"exact:{value}"})
            if value in {"day", "week", "month", "quarter", "year"}
            else frozenset()
        )
        return facts, True
    if key == "timetostr":
        format_text = raw_date_formats.get(id(node))
        return (
            _date_format_time_facts(format_text)
            if format_text is not None
            else frozenset(),
            True,
        )
    if key == "year":
        return frozenset({"exact:year", "part:year"}), True
    if key in {"day", "month", "quarter", "week"}:
        return frozenset({f"part:{key}"}), True
    if key == "dayofyear":
        return frozenset({"part:dayofyear"}), True
    if key == "tsordstodate":
        return frozenset({"exact:day"}), True
    if key in {
        "hour",
        "minute",
        "second",
        "time",
        "timetrunc",
        "unixseconds",
        "unixmillis",
        "unixmicros",
    }:
        return frozenset(), True
    if isinstance(node, exp.Anonymous):
        name = node.name.casefold()
        if name == "yearweek":
            return frozenset({"exact:week"}), True
        if name == "date":
            return frozenset({"exact:day"}), True
        if name == "year":
            return frozenset({"exact:year", "part:year"}), True
        if name in {"day", "month", "quarter", "week"}:
            return frozenset({f"part:{name}"}), True
        if name == "dayofyear":
            return frozenset({"part:dayofyear"}), True
        if name == "date_trunc":
            expressions = node.expressions
            if expressions:
                value = str(getattr(expressions[0], "this", expressions[0])).casefold()
                if value in {"day", "week", "month", "quarter", "year"}:
                    return frozenset({f"exact:{value}"}), True
            return frozenset(), True
        if name in {
            "date_format",
            "dayname",
            "hour",
            "minute",
            "second",
            "time_format",
        }:
            return frozenset(), True
    return frozenset(), False


def analyze_sql_structure(
    sql: str,
    dialect: str = "mysql",
    semantic_indexes: Mapping[str, Iterable[str] | Mapping[str, Any]] | None = None,
) -> SQLStructure:
    """Extract active, alias-aware AST facts without executing SQL or opening a database."""
    analysis = _analyze_sql(sql, dialect, semantic_indexes)
    indexes = _semantic_field_index(semantic_indexes)
    infos = {id(info.scope): info for info in analysis.scope_infos}
    source_infos = analysis.active_scope_infos
    active_scope_ids = {id(info.scope) for info in source_infos}

    fields: set[str] = set()
    for info in source_infos:
        for column in info.scope.columns:
            resolved = _resolve_column(column, info, infos, indexes)
            if resolved.known:
                fields.update(f"{table}.{name}" for table, name in resolved.references)
        select = info.scope.expression
        if not isinstance(select, exp.Select):
            continue
        for expression in select.expressions:
            unaliased = expression.this if isinstance(expression, exp.Alias) else expression
            if isinstance(unaliased, exp.Star) or (
                isinstance(unaliased, exp.Column) and unaliased.name == "*"
            ):
                for resolution in _expand_star(unaliased, info, infos, indexes).values():
                    fields.update(
                        f"{table}.{name}" for table, name in resolution.references
                    )

    relationships: set[tuple[str, str]] = set()
    for fact in analysis.join_facts:
        if id(fact.info.scope) not in active_scope_ids:
            continue
        for left, left_field, right, right_field in fact.pairs:
            endpoints = tuple(sorted((f"{left}.{left_field}", f"{right}.{right_field}")))
            relationships.add(endpoints)

    filter_signatures: set[str] = set()
    for info in source_infos:
        select = info.scope.expression
        if not isinstance(select, exp.Select):
            continue
        for clause_name in ("where", "having"):
            clause = select.args.get(clause_name)
            if not isinstance(clause, exp.Expression) or clause.this is None:
                continue
            atoms = _conjunctive_atoms(clause.this)
            if atoms is None:
                continue
            for atom in atoms:
                filter_signatures.add(
                    _filter_signature_text(atom, info, infos, indexes)
                )

    raw_date_formats = _raw_mysql_date_formats(sql, analysis.tree)
    # Evidence is isolated by final output scope, underlying source occurrence,
    # and governed field. This prevents unrelated scalar/self-join scopes from
    # manufacturing a complete calendar key for the final result.
    time_evidence: dict[tuple[int, int, str], set[str]] = {}
    visited_definitions: set[tuple[int, int, int]] = set()

    def unwrap_key(node: exp.Expression) -> exp.Expression:
        while isinstance(node, (exp.Alias, exp.Paren)):
            node = node.this
        return node

    def column_origin(
        column: exp.Column,
        info: _ScopeInfo,
        visited: frozenset[tuple[int, str]],
    ) -> tuple[int, str] | None:
        marker = (id(info.scope), _casefold(column.sql()))
        if marker in visited:
            return None
        visited = visited | {marker}
        name = _casefold(column.name)
        qualifier = _casefold(column.table) if column.table else ""

        def from_source(source: exp.Table | Scope | None) -> tuple[int, str] | None:
            if isinstance(source, exp.Table):
                table = _table_name(source)
                if indexes and name not in indexes.get(table, frozenset()):
                    return None
                return id(source), f"{table}.{name}"
            if isinstance(source, Scope):
                source_info = infos[id(source)]
                definition = source_info.projection_expressions.get(name)
                definition = unwrap_key(definition) if definition is not None else None
                if isinstance(definition, exp.Column):
                    return column_origin(definition, source_info, visited)
            return None

        if qualifier:
            return from_source(info.sources.get(qualifier))
        candidates = tuple(
            candidate
            for source in _unique_sources(info)
            if (candidate := from_source(source)) is not None
        )
        return candidates[0] if len(candidates) == 1 else None

    def is_direct_time_function(node: exp.Expression) -> bool:
        for item in node.walk():
            descendant = item[0] if isinstance(item, tuple) else item
            if descendant is node:
                continue
            if isinstance(
                descendant,
                (exp.AggFunc, exp.Case, exp.Predicate, exp.Select, exp.Subquery),
            ):
                return False
            if isinstance(descendant, exp.Func) and type(descendant).__name__ not in {
                "TsOrDsToDate",
                "TsOrDsToTimestamp",
            }:
                return False
        return True

    def collect_time_key(
        node: exp.Expression,
        info: _ScopeInfo,
        output_scope_id: int,
        *,
        expand_projection_star: bool = False,
    ) -> None:
        node = unwrap_key(node)
        facts, terminal = _time_facts_for_node(node, raw_date_formats)
        if terminal:
            if not facts or not is_direct_time_function(node):
                return
            columns = _immediate_columns(node, info.scope.expression)
            if len(columns) != 1:
                return
            origin = column_origin(columns[0], info, frozenset())
            if origin is None:
                return
            source_id, reference = origin
            time_evidence.setdefault(
                (output_scope_id, source_id, reference), set()
            ).update(facts)
            return

        if isinstance(node, exp.Star) or (
            isinstance(node, exp.Column) and node.name == "*"
        ):
            if not expand_projection_star:
                return
            qualifier = _casefold(node.table) if isinstance(node, exp.Column) else ""
            sources = (info.sources.get(qualifier),) if qualifier else _unique_sources(info)
            for source in sources:
                if isinstance(source, Scope):
                    source_info = infos[id(source)]
                    for expression in source_info.projection_expressions.values():
                        collect_time_key(expression, source_info, output_scope_id)
            return

        if not isinstance(node, exp.Column):
            return
        derived = _derived_definition(node, info, infos, indexes)
        if derived is None and not node.table:
            expression = info.projection_expressions.get(_casefold(node.name))
            derived = (info, expression) if expression is not None else None
        if derived is None:
            return
        definition_info, expression = derived
        definition_marker = (output_scope_id, id(definition_info.scope), id(expression))
        if definition_marker in visited_definitions:
            return
        visited_definitions.add(definition_marker)
        collect_time_key(expression, definition_info, output_scope_id)

    for info in analysis.root_output_infos:
        select = info.scope.expression
        if not isinstance(select, exp.Select):
            continue
        expressions = list(select.expressions)
        group = select.args.get("group")
        if isinstance(group, exp.Group):
            expressions.extend(group.expressions)
        for expression in expressions:
            unaliased = expression.this if isinstance(expression, exp.Alias) else expression
            is_projection_star = isinstance(unaliased, exp.Star) or (
                isinstance(unaliased, exp.Column) and unaliased.name == "*"
            )
            collect_time_key(
                expression,
                info,
                id(info.scope),
                expand_projection_star=is_projection_star,
            )

    time_grains: set[tuple[str, str]] = set()
    grain_precision = {"year": 0, "quarter": 1, "month": 2, "week": 3, "day": 4}
    for (_, _, reference), facts in time_evidence.items():
        candidates = {
            grain
            for grain in grain_precision
            if f"exact:{grain}" in facts
        }
        if {"part:year", "part:month"} <= facts:
            candidates.add("month")
        if {"part:year", "part:quarter"} <= facts:
            candidates.add("quarter")
        if {"part:year", "part:week"} <= facts:
            candidates.add("week")
        if {"part:year", "part:month", "part:day"} <= facts or {
            "part:year",
            "part:dayofyear",
        } <= facts:
            candidates.add("day")
        if candidates:
            time_grains.add(
                (reference, max(candidates, key=grain_precision.__getitem__))
            )

    return SQLStructure(
        scope=analysis.scope,
        tables=frozenset(
            table for info in source_infos for table in info.physical_tables
        ),
        fields=frozenset(fields),
        relationships=tuple(sorted(relationships)),
        filters=tuple(sorted(filter_signatures)),
        time_grains=tuple(sorted(time_grains)),
    )


def _metric_terms(metric: SemanticMetric) -> tuple[str, ...]:
    return (metric.name, metric.name_cn, *metric.synonyms)


def _question_matches(question: str, terms: Iterable[str]) -> bool:
    normalized_question = " ".join(_casefold(question).split())
    return any(
        normalized_term and normalized_term in normalized_question
        for term in terms
        if (normalized_term := " ".join(_casefold(term).split()))
    )


_TIME_GRAIN = re.compile(r"按(?:日|天|周|月|季度|季|年)")
_DATE_LITERAL = re.compile(r"(?:\d{4}\s*年|\d{1,2}\s*月|\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?)")
_RANGE_WORDING = re.compile(r"(?:日期|时间)(?:范围|区间)?|期间|最近|过去|本月|上月|今天|昨天|今年|去年|从.+?(?:到|至)")


def _has_time_intent(question: str) -> bool:
    normalized = _casefold(question)
    return bool(
        _TIME_GRAIN.search(normalized)
        or _DATE_LITERAL.search(normalized)
        or _RANGE_WORDING.search(normalized)
    )


class SemanticSQLValidator:
    """Evaluate governed semantic rules without constructing database clients."""

    def validate(
        self,
        sql: str,
        question: str,
        snapshot: SemanticSnapshot,
    ) -> ValidationReport:
        field_indexes = {
            _casefold(entity.name): frozenset(_casefold(field.name) for field in entity.fields)
            for entity in snapshot.entities
        }
        try:
            analysis = _analyze_sql(sql, snapshot.dialect, field_indexes)
        except _UnsupportedSQL:
            return ValidationReport(
                valid=False,
                issues=(
                    ValidationIssue(
                        code="UNSUPPORTED_SQL",
                        message="SQL 不是可确定验证且具有治理数据源的只读查询",
                        stage="safety",
                    ),
                ),
            )
        except Exception:
            return ValidationReport(
                valid=False,
                issues=(
                    ValidationIssue(
                        code="SYNTAX_ERROR",
                        message="SQL 无法解析为单条语句",
                        stage="syntax",
                    ),
                ),
            )

        scope = analysis.scope
        issues: list[ValidationIssue] = []
        entities = {_casefold(entity.name): entity for entity in snapshot.entities}

        def add(code: str, message: str, *objects: str) -> None:
            issues.append(
                ValidationIssue(
                    code=code,
                    message=message,
                    objects=tuple(sorted({_casefold(item) for item in objects if item})),
                )
            )

        # Object existence.
        for table in sorted(scope.tables):
            if table not in entities:
                add("UNKNOWN_TABLE", "SQL 引用了语义模型中不存在的表", table)
        for index, (table, name) in enumerate(scope.columns):
            if name == "*":
                continue
            if index in analysis.unresolved_column_indexes:
                add("UNKNOWN_COLUMN", "SQL 引用了派生输入未暴露的字段", name)
                continue
            if table is not None:
                if table not in field_indexes or name not in field_indexes[table]:
                    add("UNKNOWN_COLUMN", "SQL 引用了语义模型中不存在的字段", f"{table}.{name}")
                continue
            matches = sorted(
                candidate
                for candidate in scope.tables
                if name in field_indexes.get(candidate, frozenset())
            )
            if not matches:
                add("UNKNOWN_COLUMN", "SQL 引用了语义模型中不存在的字段", name)

        # Ambiguity.
        for index, (table, name) in enumerate(scope.columns):
            if (
                index in analysis.unresolved_column_indexes
                or table is not None
                or name == "*"
            ):
                continue
            matches = sorted(
                candidate
                for candidate in scope.tables
                if name in field_indexes.get(candidate, frozenset())
            )
            if len(matches) > 1:
                add("AMBIGUOUS_COLUMN", "无表前缀字段可匹配多个输入表", name, *matches)

        # Policies.
        denied_entities = {_casefold(name) for name in snapshot.policies.denied_entities}
        denied_fields = {_casefold(name) for name in snapshot.policies.denied_fields}
        for table in sorted(scope.tables & denied_entities):
            add("POLICY_DENIED", "治理策略禁止访问该表", table)
        seen_columns = set(scope.columns)
        for table, name in sorted(seen_columns, key=lambda item: (item[0] or "", item[1])):
            if table not in entities:
                continue
            candidate_fields = entities[table].fields if name == "*" else (
                field for field in entities[table].fields if _casefold(field.name) == name
            )
            for semantic_field in candidate_fields:
                reference = f"{table}.{_casefold(semantic_field.name)}"
                if (
                    reference in denied_fields
                    or not semantic_field.directly_queryable
                    or semantic_field.sensitivity == "restricted"
                ):
                    add("POLICY_DENIED", "治理策略禁止直接查询该字段", reference)

        # Governed joins. Predicate structure is retained so OR, unresolved
        # atoms, correlated scopes, and relationship conditions cannot vanish.
        infos_by_scope = {id(info.scope): info for info in analysis.scope_infos}
        for fact in analysis.join_facts:
            candidate_relations = [
                relation
                for relation in snapshot.relationships
                if {
                    _casefold(relation.from_entity),
                    _casefold(relation.to_entity),
                }
                <= (set(fact.left_tables) | set(fact.right_tables))
            ]
            proven = False
            if fact.supported:
                for relation in candidate_relations:
                    endpoint = frozenset(
                        {
                            f"{_casefold(relation.from_entity)}.{_casefold(relation.from_field)}",
                            f"{_casefold(relation.to_entity)}.{_casefold(relation.to_field)}",
                        }
                    )
                    pair_sets = {
                        frozenset(
                            {f"{left}.{left_field}", f"{right}.{right_field}"}
                        )
                        for left, left_field, right, right_field in fact.pairs
                    }
                    if endpoint not in pair_sets:
                        continue
                    extra_signatures: set[Any] = set()
                    extra_cross_pairs: set[frozenset[str]] = set()
                    if relation.extra_condition:
                        try:
                            extra_expression = parse_one(
                                relation.extra_condition, read=snapshot.dialect
                            )
                            atoms = _conjunctive_atoms(extra_expression)
                            if atoms is None or not all(
                                isinstance(atom, exp.EQ) for atom in atoms
                            ):
                                continue
                            source_tables = frozenset(
                                {
                                    _casefold(relation.from_entity),
                                    _casefold(relation.to_entity),
                                }
                            )
                            extra_signatures = {
                                _expression_signature(
                                    atom,
                                    None,
                                    infos_by_scope,
                                    field_indexes,
                                    source_tables,
                                )
                                for atom in atoms
                            }
                            for atom in atoms:
                                references: set[str] = set()
                                for column in atom.find_all(exp.Column):
                                    signature = _expression_signature(
                                        column,
                                        None,
                                        infos_by_scope,
                                        field_indexes,
                                        source_tables,
                                    )
                                    if (
                                        isinstance(signature, tuple)
                                        and len(signature) == 3
                                        and signature[0] == "column"
                                        and signature[1] in source_tables
                                    ):
                                        references.add(f"{signature[1]}.{signature[2]}")
                                reference_tables = {
                                    reference.rsplit(".", 1)[0]
                                    for reference in references
                                }
                                if len(references) == 2 and len(reference_tables) == 2:
                                    extra_cross_pairs.add(frozenset(references))
                        except Exception:
                            continue
                        if not extra_signatures.issubset(fact.atom_signatures):
                            continue
                    if pair_sets - {endpoint} != extra_cross_pairs:
                        continue
                    proven = True
                    break
            if not proven:
                add(
                    "INVALID_JOIN",
                    "JOIN 条件无法证明匹配已治理关系",
                    *sorted(fact.left_tables | fact.right_tables),
                )

        matched_metrics = tuple(
            metric
            for metric in snapshot.metrics
            if _question_matches(question, _metric_terms(metric))
        )
        lineage = _lineage_for_analysis(analysis, field_indexes)

        # Matched metrics.
        for metric in matched_metrics:
            required_filters = {
                self._resolve_snapshot_reference(reference, field_indexes)
                for reference in metric.required_filters
            }
            matched_expression_and_filters = False
            try:
                metric_expression = parse_one(metric.expression, read=snapshot.dialect)
                matched_expression_and_filters = any(
                    _expression_signature(
                        metric_expression,
                        None,
                        infos_by_scope,
                        field_indexes,
                        source_tables,
                    )
                    == candidate
                    and required_filters.issubset(candidate_filters)
                    for candidate, source_tables, candidate_filters in lineage.metric_candidates
                )
            except Exception:
                matched_expression_and_filters = False
            if not matched_expression_and_filters:
                add("METRIC_MISMATCH", "SQL 聚合表达式与命中的治理指标不等价", metric.name)

        # Dimensions.
        for metric in matched_metrics:
            allowed = {
                self._resolve_snapshot_reference(reference, field_indexes)
                for reference in metric.allowed_dimensions
            }
            for dimension in sorted(lineage.dimensions):
                if dimension not in allowed:
                    add(
                        "DIMENSION_NOT_ALLOWED",
                        "指标使用了未允许的分组维度",
                        metric.name,
                        dimension,
                    )

        # Time intent.
        if _has_time_intent(question):
            time_fields: set[str] = set()
            for metric in matched_metrics:
                if metric.default_time_field:
                    time_fields.add(
                        self._resolve_snapshot_reference(metric.default_time_field, field_indexes)
                    )
                time_fields.update(
                    self._resolve_snapshot_reference(reference, field_indexes)
                    for reference in metric.allowed_dimensions
                    if self._is_time_reference(reference, snapshot)
                )
            if not time_fields:
                time_fields = {
                    f"{_casefold(entity.name)}.{_casefold(semantic_field.name)}"
                    for entity in snapshot.entities
                    for semantic_field in entity.fields
                    if semantic_field.role == "time"
                }
            used_fields = set(lineage.fields)
            if not used_fields.intersection(time_fields):
                add(
                    "TIME_DIMENSION_REQUIRED",
                    "问题包含时间意图，但 SQL 未使用受治理时间字段",
                    *sorted(time_fields),
                )

        unique: dict[tuple[str, tuple[str, ...]], ValidationIssue] = {}
        for issue in issues:
            unique.setdefault((issue.code, issue.objects), issue)
        ordered = tuple(
            sorted(unique.values(), key=lambda issue: (issue.stage, issue.code, issue.objects))
        )
        return ValidationReport(valid=not ordered, issues=ordered)

    @staticmethod
    def _resolve_snapshot_reference(
        reference: str,
        field_indexes: Mapping[str, frozenset[str]],
    ) -> str:
        normalized = _casefold(reference)
        if "." in normalized:
            table, name = normalized.rsplit(".", 1)
            return f"{table}.{name}"
        candidates = sorted(
            table for table, fields in field_indexes.items() if normalized in fields
        )
        return f"{candidates[0]}.{normalized}" if len(candidates) == 1 else normalized

    @classmethod
    def _is_time_reference(cls, reference: str, snapshot: SemanticSnapshot) -> bool:
        normalized = cls._resolve_snapshot_reference(
            reference,
            {
                _casefold(entity.name): frozenset(
                    _casefold(semantic_field.name) for semantic_field in entity.fields
                )
                for entity in snapshot.entities
            },
        )
        return any(
            normalized == f"{_casefold(entity.name)}.{_casefold(semantic_field.name)}"
            and semantic_field.role == "time"
            for entity in snapshot.entities
            for semantic_field in entity.fields
        )


__all__ = [
    "SQLScope",
    "SQLStructure",
    "SemanticSQLValidator",
    "analyze_sql_structure",
    "is_single_readonly_query",
    "normalize_filter_expression",
    "parse_sql_scope",
]
