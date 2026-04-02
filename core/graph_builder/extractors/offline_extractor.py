"""
离线文件抽取器
从 DDL 文件抽取元数据，支持 CREATE DATABASE 和 CREATE TABLE 语句
"""
import re
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

from .base import BaseExtractor
from ..models import SchemaExtractResult, DatabaseIR, TableIR, ColumnIR, RelationshipIR


class OfflineExtractor(BaseExtractor):
    """离线 DDL 抽取器"""

    DDL_TYPE_MAP = {
        "int": "INT", "integer": "INT", "bigint": "BIGINT",
        "varchar": "VARCHAR", "char": "CHAR", "text": "TEXT",
        "datetime": "DATETIME", "timestamp": "TIMESTAMP",
        "decimal": "DECIMAL", "float": "FLOAT", "double": "DOUBLE",
        "boolean": "BOOLEAN", "bool": "BOOLEAN",
        "json": "JSON", "blob": "BLOB", "binary": "BINARY",
        "date": "DATE", "time": "TIME", "year": "YEAR",
        "tinyint": "TINYINT", "smallint": "SMALLINT", "mediumint": "MEDIUMINT",
        "real": "REAL", "double precision": "DOUBLE", "numeric": "NUMERIC",
        "character": "CHAR", "nvarchar": "NVARCHAR", "nchar": "NCHAR",
        "clob": "CLOB", "longtext": "LONGTEXT", "mediumtext": "MEDIUMTEXT",
    }

    def __init__(self, ddl_path: Optional[str] = None, database_name: str = "default",
                 db_type: str = "mysql", neo4j_client=None, **kwargs):
        super().__init__(**kwargs)
        self.ddl_path = ddl_path
        self.database_name = database_name
        self.db_type = db_type
        self.neo4j_client = neo4j_client
        self._db_id_counter = 0
        self._init_db_counter()

    def _init_db_counter(self):
        """从 Neo4j 获取当前最大的 database_id 并初始化计数器"""
        if self.neo4j_client and self.neo4j_client.driver:
            try:
                result = self.neo4j_client.execute_query(
                    "MATCH (d:Database) RETURN COALESCE(MAX(d.database_id), 0) as max_id"
                )
                if result and len(result) > 0:
                    max_id = result[0].get("max_id", 0)
                    self._db_id_counter = max_id
                    logger.info(f"从 Neo4j 加载数据库 ID 计数器：{self._db_id_counter}")
            except Exception as e:
                logger.warning(f"无法从 Neo4j 加载数据库 ID 计数器，使用默认值：{e}")
                self._db_id_counter = 0
        else:
            self._db_id_counter = 0

    def _get_next_db_id(self) -> int:
        """获取下一个数据库 ID（自增）"""
        self._db_id_counter += 1
        return self._db_id_counter

    def extract(self) -> SchemaExtractResult:
        result = SchemaExtractResult()
        result.metadata["source_type"] = "offline"
        result.metadata["db_type"] = self.db_type

        # 首先尝试从 DDL 文件中解析 CREATE DATABASE 语句
        if self.ddl_path:
            self._parse_create_database(self.ddl_path, result)

        # 如果没有解析到数据库，则使用默认配置
        if not result.databases:
            db_id = self._get_next_db_id()
            db = DatabaseIR(
                id=db_id,
                database_id=db_id,  # 同时设置 database_id
                name=self.database_name,
                db_type=self.db_type,
                db_language="SQL"
            )
            result.databases.append(db)
            result.metadata["default_database"] = True

        # 解析 DDL 文件中的 CREATE TABLE 语句
        if self.ddl_path:
            self._parse_ddl(self.ddl_path, result)

        # 为所有表设置数据库 ID
        for table in result.tables:
            db = next((d for d in result.databases if d.name == table.database), None)
            if db:
                table.database_id = db.id

        relationships = self._infer_relationships_from_schema(result)
        result.relationships = relationships

        logger.info(f"抽取完成 - 数据库：{len(result.databases)}, 表：{len(result.tables)}, 关系：{len(result.relationships)}")
        return result

    def _parse_create_database(self, ddl_path: str, result: SchemaExtractResult):
        """解析 CREATE DATABASE 语句"""
        path = Path(ddl_path)
        if not path.exists():
            logger.error(f"DDL 文件不存在：{ddl_path}")
            return

        content = path.read_text(encoding="utf-8")

        # 匹配 CREATE DATABASE 语句
        db_pattern = r'CREATE\s+(?:DATABASE|SCHEMA)\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?(\w+)[`"]?\s*(?:DEFAULT\s+)?(?:CHARACTER\s+SET\s+\w+)?(?:COLLATE\s+\w+)?\s*(?:COMMENT\s+[\'"](.+?)[\'"])?\s*;'

        matches = re.findall(db_pattern, content, re.IGNORECASE)

        for match in matches:
            db_name = match[0]
            db_comment = match[1] if len(match) > 1 and match[1] else None

            # 提取完整的 CREATE DATABASE 语句
            full_stmt_match = re.search(
                r'CREATE\s+(?:DATABASE|SCHEMA)\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?' + db_name + r'[`"]?.*?;',
                content, re.IGNORECASE | re.DOTALL
            )
            create_stmt = full_stmt_match.group(0) if full_stmt_match else None

            # 提取数据库类型（从 CHARACTER SET 或其他线索）
            detected_type = self.db_type
            if create_stmt:
                if 'CHARACTER SET utf8' in create_stmt or 'CHARSET=utf8' in create_stmt:
                    detected_type = "mysql"
                elif 'ENCODING' in create_stmt.upper():
                    detected_type = "postgresql"

            db_id = self._get_next_db_id()
            db = DatabaseIR(
                id=db_id,
                name=db_name,
                description=db_comment,
                db_type=detected_type,
                db_language="SQL",
                create_statement=create_stmt
            )
            result.databases.append(db)
            logger.info(f"解析数据库：{db_name} (ID={db_id})")

    def _parse_ddl(self, ddl_path: str, result: SchemaExtractResult):
        """解析 DDL 文件中的 CREATE TABLE 语句"""
        path = Path(ddl_path)
        if not path.exists():
            logger.error(f"DDL 文件不存在：{ddl_path}")
            return

        content = path.read_text(encoding="utf-8")

        # 使用更 robust 的方式分割 SQL 语句
        statements = self._split_sql_statements(content)

        for stmt in statements:
            if stmt.strip().upper().startswith('CREATE TABLE'):
                table = self._parse_create_table(stmt, result)
                if table:
                    result.tables.append(table)

    def _split_sql_statements(self, content: str) -> list:
        """分割 SQL 语句（处理分号和注释）"""
        # 移除单行注释
        content = re.sub(r'--.*?$', '', content, flags=re.MULTILINE)
        # 移除多行注释
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        statements = []
        current_stmt = ""
        in_string = False
        string_char = None

        for char in content:
            if char in ("'", '"') and (not string_char or char == string_char):
                in_string = not in_string
                string_char = char if not string_char else None
            elif char == ';' and not in_string:
                if current_stmt.strip():
                    statements.append(current_stmt.strip() + ';')
                current_stmt = ""
                continue
            current_stmt += char

        # 处理最后一个没有分号的语句
        if current_stmt.strip():
            statements.append(current_stmt.strip())

        return statements

    def _parse_create_table(self, stmt: str, result: SchemaExtractResult) -> Optional[TableIR]:
        """解析 CREATE TABLE 语句"""
        # 匹配表名和列定义
        match = re.search(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?(\w+)[`"]?\s*\((.*)\)\s*(?:ENGINE\s*=\s*\w+)?\s*(?:DEFAULT\s+CHARSET\s*=\s*\w+)?\s*(?:COLLATE\s*=\s*\w+)?\s*(?:COMMENT\s*=\s*[\'"](.+?)[\'"])?\s*;',
            stmt, re.IGNORECASE | re.DOTALL
        )

        if not match:
            # 尝试简化匹配
            simple_match = re.search(
                r'CREATE\s+TABLE\s+[`"]?(\w+)[`"]?\s*\((.*)\)',
                stmt, re.IGNORECASE | re.DOTALL
            )
            if not simple_match:
                return None
            table_name = simple_match.group(1)
            columns_part = simple_match.group(2)
            table_comment = None
        else:
            table_name = match.group(1)
            columns_part = match.group(2)
            table_comment = match.group(3) if match.group(3) else None

        # 确定所属数据库
        database = self.database_name
        database_id = None
        if result.databases:
            # 使用第一个数据库或匹配的数据库
            database = result.databases[0].name
            database_id = result.databases[0].id

        table = TableIR(
            name=table_name,
            database=database,
            database_id=database_id,
            description=table_comment,
            create_statement=stmt.strip()
        )

        # 解析字段
        col_defs = self._split_columns(columns_part)
        for col_def in col_defs:
            col_def = col_def.strip()
            if not col_def:
                continue
            # 跳过约束定义
            upper_def = col_def.upper()
            if upper_def.startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'INDEX', 'KEY', 'UNIQUE', 'CHECK', 'FULLTEXT')):
                # 处理 PRIMARY KEY
                if 'PRIMARY KEY' in upper_def:
                    pk_match = re.search(r'PRIMARY\s+KEY\s*\(([^)]+)\)', col_def, re.IGNORECASE)
                    if pk_match:
                        pk_cols = [c.strip().strip('`"') for c in pk_match.group(1).split(',')]
                        for col in table.columns:
                            if col.name in pk_cols:
                                col.is_primary_key = True
                continue

            column = self._parse_column(col_def)
            if column:
                table.columns.append(column)

        logger.info(f"解析表：{table_name}, 字段数：{len(table.columns)}")
        return table

    def _split_columns(self, columns_part: str) -> list:
        """分割字段定义（处理嵌套括号）"""
        columns = []
        current = ""
        paren_count = 0

        for char in columns_part:
            if char == '(':
                paren_count += 1
                current += char
            elif char == ')':
                paren_count -= 1
                current += char
            elif char == ',' and paren_count == 0:
                if current.strip():
                    columns.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            columns.append(current.strip())

        return columns

    def _parse_column(self, col_def: str) -> Optional[ColumnIR]:
        """解析单个字段定义"""
        parts = col_def.split()
        if len(parts) < 2:
            return None

        col_name = parts[0].strip('`"')

        # 解析数据类型（可能包含长度和精度）
        data_type_part = parts[1]
        # 检查是否有括号（如 VARCHAR(100), DECIMAL(18,2)）
        if '(' in data_type_part:
            # 找到完整的类型定义
            paren_end = data_type_part.find(')')
            if paren_end == -1:
                # 括号在后续部分
                type_parts = [data_type_part]
                for p in parts[2:]:
                    type_parts.append(p)
                    if ')' in p:
                        break
                data_type_raw = ' '.join(type_parts).upper()
            else:
                data_type_raw = data_type_part.upper()
        else:
            data_type_raw = data_type_part.upper()

        # 规范化数据类型
        base_type = re.sub(r'\([^)]*\)', '', data_type_raw).upper().strip()
        normalized_type = self.DDL_TYPE_MAP.get(base_type.lower(), base_type)

        # 检查约束
        is_pk = 'PRIMARY KEY' in col_def.upper()
        is_nullable = 'NOT NULL' not in col_def.upper() and 'NULL' not in col_def.upper()

        # 检查默认值
        default_match = re.search(r'DEFAULT\s+([\'"]?\w+[\'"]?|\d+|NULL|CURRENT_TIMESTAMP(?:\(\d+\))?|\([^)]+\))', col_def, re.IGNORECASE)
        default_value = default_match.group(1) if default_match else None

        # 提取注释
        comment_match = re.search(r'COMMENT\s+[\'"](.+?)[\'"]', col_def, re.IGNORECASE)
        comment = comment_match.group(1) if comment_match else None

        return ColumnIR(
            name=col_name,
            data_type=normalized_type,
            description=comment,
            is_primary_key=is_pk,
            is_nullable=is_nullable,
            default_value=default_value
        )
