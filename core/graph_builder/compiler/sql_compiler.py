"""
SQL 预编译器
"""
from typing import List
from loguru import logger

from ..models import RelationshipIR, SchemaExtractResult


class SQLCompiler:
    def __init__(self, dialect: str = "mysql"):
        self.dialect = dialect

    def compile(self, relationships: List[RelationshipIR]) -> List[RelationshipIR]:
        for rel in relationships:
            rel.join_sql = self._build_join_sql(rel)
        logger.info(f"预编译了 {len(relationships)} 个 JOIN SQL")
        return relationships

    def _build_join_sql(self, rel: RelationshipIR) -> str:
        quote = "`" if self.dialect == "mysql" else '"'
        from_tbl = f"{quote}{rel.from_database}{quote}.{quote}{rel.from_table}{quote}"
        to_tbl = f"{quote}{rel.to_database}{quote}.{quote}{rel.to_table}{quote}"
        from_col = f"{quote}{rel.from_table}{quote}.{quote}{rel.from_column}{quote}"
        to_col = f"{quote}{rel.to_table}{quote}.{quote}{rel.to_column}{quote}"

        join_sql = f"{from_tbl} {rel.join_type} {to_tbl} ON {from_col} = {to_col}"
        if rel.extra_condition:
            join_sql += f" AND {rel.extra_condition}"
        return join_sql


def compile_join_sqls(result: SchemaExtractResult, dialect: str = "mysql") -> SchemaExtractResult:
    compiler = SQLCompiler(dialect=dialect)
    result.relationships = compiler.compile(result.relationships)
    return result
