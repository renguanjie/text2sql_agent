"""
离线 JSON 抽取器
从 JSON 文件抽取元数据
"""
import json
from pathlib import Path
from typing import Optional
from loguru import logger

from .base import BaseExtractor
from ..models import (
    SchemaExtractResult, DatabaseIR, TableIR, ColumnIR,
    RelationshipIR, BusinessConceptIR
)


class OfflineJSONExtractor(BaseExtractor):
    """离线 JSON 抽取器"""

    def __init__(self, json_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.json_path = json_path

    def extract(self) -> SchemaExtractResult:
        result = SchemaExtractResult()
        result.metadata["source_type"] = "offline_json"

        if not self.json_path:
            logger.error("JSON 文件路径未指定")
            return result

        path = Path(self.json_path)
        if not path.exists():
            logger.error(f"JSON 文件不存在：{self.json_path}")
            return result

        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败：{e}")
            return result

        # 解析元数据
        if "metadata" in data:
            result.metadata.update(data["metadata"])

        # 解析数据库
        db_id_counter = 0
        for db_data in data.get("databases", []):
            db_id_counter += 1
            db = DatabaseIR(
                id=db_id_counter,
                name=db_data.get("name", "default"),
                description=db_data.get("description"),
                db_type=db_data.get("db_type", "mysql"),
                db_language=db_data.get("db_language", "SQL"),
                create_statement=db_data.get("create_statement")
            )
            result.databases.append(db)

        # 如果没有数据库定义，创建默认数据库
        if not result.databases:
            db_id_counter = 1
            default_db = DatabaseIR(
                id=db_id_counter,
                name="default",
                db_type="mysql",
                db_language="SQL"
            )
            result.databases.append(default_db)

        # 解析表
        for table_data in data.get("tables", []):
            table = self._parse_table(table_data, result.databases[0].id if result.databases else 1)
            result.tables.append(table)

        # 解析关系
        for rel_data in data.get("relationships", []):
            rel = self._parse_relationship(rel_data)
            result.relationships.append(rel)

        # 解析业务概念
        for concept_data in data.get("concepts", []):
            concept = self._parse_concept(concept_data)
            result.concepts.append(concept)

        logger.info(
            f"JSON 抽取完成 - 数据库：{len(result.databases)}, "
            f"表：{len(result.tables)}, 关系：{len(result.relationships)}, "
            f"概念：{len(result.concepts)}"
        )
        return result

    def _parse_table(self, table_data: dict, database_id: int) -> TableIR:
        """解析表数据"""
        columns = []
        for col_data in table_data.get("columns", []):
            col = ColumnIR(
                name=col_data.get("name"),
                name_cn=col_data.get("name_cn"),
                data_type=col_data.get("data_type", "VARCHAR"),
                description=col_data.get("description"),
                is_primary_key=col_data.get("is_primary_key", False),
                is_nullable=col_data.get("is_nullable", True),
                default_value=col_data.get("default_value")
            )
            columns.append(col)

        return TableIR(
            name=table_data.get("name"),
            name_cn=table_data.get("name_cn"),
            database=table_data.get("database", "default"),
            database_id=database_id,
            description=table_data.get("description"),
            columns=columns,
            is_view=table_data.get("is_view", False),
            create_statement=table_data.get("create_statement")
        )

    def _parse_relationship(self, rel_data: dict) -> RelationshipIR:
        """解析关系数据"""
        return RelationshipIR(
            from_database=rel_data.get("from_database", "default"),
            from_database_id=rel_data.get("from_database_id"),
            from_table=rel_data.get("from_table"),
            from_column=rel_data.get("from_column"),
            to_database=rel_data.get("to_database", "default"),
            to_database_id=rel_data.get("to_database_id"),
            to_table=rel_data.get("to_table"),
            to_column=rel_data.get("to_column"),
            relationship_type=rel_data.get("relationship_type", "foreign_key"),
            join_type=rel_data.get("join_type", "LEFT JOIN"),
            extra_condition=rel_data.get("extra_condition"),
            join_sql=rel_data.get("join_sql")
        )

    def _parse_concept(self, concept_data: dict) -> BusinessConceptIR:
        """解析业务概念数据"""
        return BusinessConceptIR(
            name=concept_data.get("name"),
            description=concept_data.get("description"),
            mapped_tables=concept_data.get("mapped_tables", []),
            tags=concept_data.get("tags", [])
        )
