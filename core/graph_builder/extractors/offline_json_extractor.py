"""
离线 JSON 抽取器
从 JSON 文件抽取元数据
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
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

        # 解析数据库，保留 JSON 中的 database_id/id，支持多数据库。
        db_id_counter = 0
        database_id_by_name: Dict[str, int] = {}
        database_name_by_id: Dict[int, str] = {}
        for db_data in data.get("databases", []):
            db_id_counter += 1
            db_id = db_data.get("database_id") or db_data.get("id") or db_id_counter
            db_name = db_data.get("name", f"database_{db_id}")
            db = DatabaseIR(
                id=db_id,
                database_id=db_id,
                name=db_name,
                description=db_data.get("description"),
                db_type=db_data.get("db_type", "mysql"),
                db_language=db_data.get("db_language", "SQL"),
                create_statement=db_data.get("create_statement"),
                properties=db_data.get("properties", {})
            )
            result.databases.append(db)
            database_id_by_name[db_name] = db_id
            database_name_by_id[db_id] = db_name

        # 如果没有数据库定义，创建默认数据库
        if not result.databases:
            db_id_counter = 1
            default_db = DatabaseIR(
                id=db_id_counter,
                database_id=db_id_counter,
                name="default",
                db_type="mysql",
                db_language="SQL"
            )
            result.databases.append(default_db)
            database_id_by_name["default"] = db_id_counter
            database_name_by_id[db_id_counter] = "default"

        # 解析表
        for table_data in data.get("tables", []):
            table = self._parse_table(table_data, database_id_by_name, database_name_by_id)
            result.tables.append(table)

        # 解析关系
        tables_by_name: Dict[str, List[TableIR]] = {}
        for table in result.tables:
            tables_by_name.setdefault(table.name, []).append(table)
        for rel_data in data.get("relationships", []):
            rel = self._parse_relationship(
                rel_data, tables_by_name, database_id_by_name, database_name_by_id
            )
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

    def _parse_table(self, table_data: dict,
                     database_id_by_name: Dict[str, int],
                     database_name_by_id: Dict[int, str]) -> TableIR:
        """解析表数据"""
        database_id = table_data.get("database_id")
        database_name = table_data.get("database") or table_data.get("database_name")
        if database_id is None and database_name:
            database_id = database_id_by_name.get(database_name)
        if database_id is None:
            database_id = next(iter(database_name_by_id), 1)
        if not database_name:
            database_name = database_name_by_id.get(database_id, "default")

        columns = []
        for col_data in table_data.get("columns", []):
            col = ColumnIR(
                name=col_data.get("name"),
                name_cn=col_data.get("name_cn"),
                data_type=col_data.get("data_type") or col_data.get("type") or "VARCHAR",
                description=col_data.get("description"),
                is_primary_key=col_data.get("is_primary_key", False),
                is_nullable=col_data.get("is_nullable", True),
                is_partition=col_data.get("is_partition", False),
                default_value=col_data.get("default_value"),
                properties=col_data.get("properties", {})
            )
            columns.append(col)

        return TableIR(
            name=table_data.get("name"),
            name_cn=table_data.get("name_cn"),
            database=database_name,
            database_id=database_id,
            description=table_data.get("description"),
            columns=columns,
            is_view=table_data.get("is_view", False),
            create_statement=table_data.get("create_statement"),
            properties=table_data.get("properties", {})
        )

    @staticmethod
    def _resolve_relationship_database(
        table_name: str,
        database_name: Optional[str],
        database_id: Optional[int],
        tables_by_name: Dict[str, List[TableIR]],
        database_id_by_name: Dict[str, int],
        database_name_by_id: Dict[int, str],
    ) -> tuple[str, Optional[int]]:
        """补全关系端点的数据库信息，避免多数据库导入后关系无法匹配表。"""
        if database_name:
            return database_name, database_id or database_id_by_name.get(database_name)
        if database_id is not None:
            return database_name_by_id.get(database_id, "default"), database_id

        matches = tables_by_name.get(table_name, [])
        if len(matches) == 1:
            return matches[0].database, matches[0].database_id
        if len(matches) > 1:
            raise ValueError(f"关系中的表名 {table_name!r} 不唯一，必须指定数据库")
        return "default", None

    def _parse_relationship(
        self,
        rel_data: dict,
        tables_by_name: Dict[str, List[TableIR]],
        database_id_by_name: Dict[str, int],
        database_name_by_id: Dict[int, str],
    ) -> RelationshipIR:
        """解析关系数据"""
        from_database, from_database_id = self._resolve_relationship_database(
            rel_data.get("from_table"),
            rel_data.get("from_database"),
            rel_data.get("from_database_id"),
            tables_by_name,
            database_id_by_name,
            database_name_by_id,
        )
        to_database, to_database_id = self._resolve_relationship_database(
            rel_data.get("to_table"),
            rel_data.get("to_database"),
            rel_data.get("to_database_id"),
            tables_by_name,
            database_id_by_name,
            database_name_by_id,
        )
        return RelationshipIR(
            from_database=from_database,
            from_database_id=from_database_id,
            from_table=rel_data.get("from_table"),
            from_column=rel_data.get("from_column"),
            to_database=to_database,
            to_database_id=to_database_id,
            to_table=rel_data.get("to_table"),
            to_column=rel_data.get("to_column"),
            relationship_type=rel_data.get("relationship_type", "foreign_key"),
            join_type=rel_data.get("join_type", "LEFT JOIN"),
            extra_condition=rel_data.get("extra_condition"),
            join_sql=rel_data.get("join_sql"),
            weight=rel_data.get("weight", 1.0),
            occurrence_count=rel_data.get("occurrence_count", 1),
            source=rel_data.get("source", "manual"),
            properties=rel_data.get("properties", {})
        )

    def _parse_concept(self, concept_data: dict) -> BusinessConceptIR:
        """解析业务概念数据"""
        return BusinessConceptIR(
            name=concept_data.get("name"),
            description=concept_data.get("description"),
            mapped_tables=concept_data.get("mapped_tables") or concept_data.get("related_tables", []),
            tags=concept_data.get("tags", []),
            properties=concept_data.get("properties", {})
        )
