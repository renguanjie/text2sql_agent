"""
Extractor 基类
"""
from abc import ABC, abstractmethod
from typing import List
from loguru import logger

from ..models import SchemaExtractResult, RelationshipIR


class BaseExtractor(ABC):
    """数据抽取器抽象基类"""

    def __init__(self, **kwargs):
        self.config = kwargs
        logger.info(f"{self.__class__.__name__} 初始化完成")

    @abstractmethod
    def extract(self) -> SchemaExtractResult:
        """执行数据抽取"""
        pass

    def _infer_relationships_from_schema(self, result: SchemaExtractResult) -> List[RelationshipIR]:
        """从 Schema 中推断基础关系"""
        relationships = []
        db_tables = {}

        for table in result.tables:
            if table.database not in db_tables:
                db_tables[table.database] = []
            db_tables[table.database].append(table)

        pk_columns = {}
        for table in result.tables:
            for col in table.columns:
                if col.is_primary_key:
                    if table.name not in pk_columns:
                        pk_columns[table.name] = []
                    pk_columns[table.name].append(col.name)

        for table_name, pk_cols in pk_columns.items():
            table = next((t for t in result.tables if t.name == table_name), None)
            if table:
                for col in table.columns:
                    if col.name.endswith('_id') and not col.is_primary_key:
                        potential_ref = col.name[:-3]
                        for other_table in db_tables.get(table.database, []):
                            if potential_ref in other_table.name.lower():
                                rel = RelationshipIR(
                                    from_database=table.database,
                                    from_table=table_name,
                                    from_column=col.name,
                                    to_database=table.database,
                                    to_table=other_table.name,
                                    to_column='id',
                                    relationship_type="foreign_key",
                                    join_type="LEFT JOIN"
                                )
                                relationships.append(rel)
                                break
        return relationships
