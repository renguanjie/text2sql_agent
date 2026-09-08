"""
活体数据库抽取器
"""
from typing import Optional
from sqlalchemy import create_engine, text
from loguru import logger

from .base import BaseExtractor
from ..models import SchemaExtractResult, DatabaseIR, TableIR, ColumnIR


class LiveDBExtractor(BaseExtractor):
    """活体数据库抽取器"""

    def __init__(self, db_uri: str, db_type: str = "mysql", schema: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.db_uri = db_uri
        self.db_type = db_type
        self.schema = schema
        self.engine = None

    def connect(self) -> bool:
        try:
            self.engine = create_engine(self.db_uri, pool_pre_ping=True)
            logger.info(f"成功连接到 {self.db_type} 数据库")
            return True
        except Exception as e:
            if self.engine is not None:
                self.engine.dispose()
                self.engine = None
            logger.error(f"数据库连接失败：{e}")
            return False

    def close(self):
        if self.engine:
            self.engine.dispose()
            self.engine = None

    def extract(self) -> SchemaExtractResult:
        if not self.engine and not self.connect():
            raise ConnectionError(f"无法连接到 {self.db_type} 数据库")

        if self.db_type != "mysql":
            raise ValueError(f"活体数据库抽取暂不支持 {self.db_type}，当前仅支持 mysql")

        result = SchemaExtractResult()
        result.metadata["source_type"] = "live_db"

        db = DatabaseIR(name=self.schema or "default", db_type=self.db_type)
        result.databases.append(db)

        self._extract_mysql(result)

        relationships = self._infer_relationships_from_schema(result)
        result.relationships = relationships
        return result

    def _extract_mysql(self, result: SchemaExtractResult):
        with self.engine.connect() as conn:
            schema_name = self.schema
            if not schema_name:
                schema_name = conn.execute(text("SELECT DATABASE()")).scalar()
            if not schema_name:
                raise ValueError("未指定 MySQL schema，且连接未选择默认数据库")

            tables = conn.execute(text("""
                SELECT TABLE_NAME, TABLE_COMMENT, TABLE_SCHEMA
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = :schema_name
            """), {"schema_name": schema_name}).fetchall()

            for row in tables:
                table = TableIR(name=row[0], database=row[2], description=row[1])
                result.tables.append(table)

            columns = conn.execute(text("""
                SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_COMMENT, IS_NULLABLE, COLUMN_KEY, TABLE_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema_name
                ORDER BY TABLE_NAME, ORDINAL_POSITION
            """), {"schema_name": schema_name}).fetchall()

            for row in columns:
                table = next((t for t in result.tables if t.name == row[5]), None)
                if table:
                    column = ColumnIR(
                        name=row[0], data_type=row[1].split('(')[0],
                        description=row[2], is_nullable=row[3] == 'YES',
                        is_primary_key=row[4] == 'PRI'
                    )
                    table.columns.append(column)
