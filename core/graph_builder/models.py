"""
数据模型层 - 定义统一的 Intermediate Representation (IR)
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DatabaseIR(BaseModel):
    """数据库元数据"""
    id: Optional[int] = Field(None, description="数据库 ID（自增）")
    database_id: Optional[int] = Field(None, description="数据库 ID（与 id 相同，用于兼容导出导入）")
    name: str = Field(..., description="数据库名称")
    description: Optional[str] = Field(None, description="数据库描述/注释")
    db_type: str = Field(default="mysql", description="数据库类型 (mysql/oracle/postgresql/spark)")
    db_language: str = Field(default="SQL", description="数据库语言")
    create_statement: Optional[str] = Field(None, description="数据库创建语句")
    properties: Dict[str, Any] = Field(default_factory=dict, description="扩展属性")


class ColumnIR(BaseModel):
    """字段元数据"""
    name: str = Field(..., description="字段名")
    name_cn: Optional[str] = Field(None, description="字段中文名")
    data_type: str = Field(..., description="字段类型")
    description: Optional[str] = Field(None, description="字段描述/注释")
    is_primary_key: bool = Field(default=False, description="是否主键")
    is_nullable: bool = Field(default=True, description="是否可空")
    is_partition: bool = Field(default=False, description="是否分区键 (Spark 特有)")
    default_value: Optional[str] = Field(None, description="默认值")
    properties: Dict[str, Any] = Field(default_factory=dict, description="扩展属性")


class TableIR(BaseModel):
    """表元数据"""
    name: str = Field(..., description="表名")
    name_cn: Optional[str] = Field(None, description="表中文名")
    database: str = Field(..., description="所属数据库")
    database_id: Optional[int] = Field(None, description="所属数据库 ID")
    description: Optional[str] = Field(None, description="表描述/注释")
    columns: List[ColumnIR] = Field(default_factory=list, description="字段列表")
    is_view: bool = Field(default=False, description="是否视图")
    create_statement: Optional[str] = Field(None, description="表创建语句")
    properties: Dict[str, Any] = Field(default_factory=dict, description="扩展属性")


class RelationshipIR(BaseModel):
    """表关系元数据"""
    from_database: str = Field(..., description="源数据库")
    from_database_id: Optional[int] = Field(None, description="源数据库 ID")
    from_table: str = Field(..., description="源表")
    from_column: str = Field(..., description="源字段")
    to_database: str = Field(..., description="目标数据库")
    to_database_id: Optional[int] = Field(None, description="目标数据库 ID")
    to_table: str = Field(..., description="目标表")
    to_column: str = Field(..., description="目标字段")
    relationship_type: str = Field(default="foreign_key", description="关系类型")
    join_type: str = Field(default="LEFT JOIN", description="JOIN 类型")
    extra_condition: Optional[str] = Field(None, description="额外条件")
    join_sql: Optional[str] = Field(None, description="预编译的 JOIN SQL")
    properties: Dict[str, Any] = Field(default_factory=dict, description="扩展属性")


class BusinessConceptIR(BaseModel):
    """业务概念元数据"""
    name: str = Field(..., description="概念名称")
    description: Optional[str] = Field(None, description="概念描述")
    mapped_tables: List[str] = Field(default_factory=list, description="映射的表")
    tags: List[str] = Field(default_factory=list, description="业务标签")
    properties: Dict[str, Any] = Field(default_factory=dict, description="扩展属性")


class SchemaExtractResult(BaseModel):
    """Schema 抽取结果容器"""
    databases: List[DatabaseIR] = Field(default_factory=list)
    tables: List[TableIR] = Field(default_factory=list)
    relationships: List[RelationshipIR] = Field(default_factory=list)
    concepts: List[BusinessConceptIR] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据信息")

    # 数据库 ID 计数器
    _db_id_counter: int = 0

    def get_next_db_id(self) -> int:
        """获取下一个数据库 ID"""
        self._db_id_counter += 1
        return self._db_id_counter

    def get_tables_by_database(self, db_name: str) -> List[TableIR]:
        """获取指定数据库的所有表"""
        return [t for t in self.tables if t.database == db_name]

    def get_columns_by_table(self, table_name: str) -> List[ColumnIR]:
        """获取指定表的所有字段"""
        for table in self.tables:
            if table.name == table_name:
                return table.columns
        return []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()


# ==================== 辅助枚举 ====================

class DatabaseType(str):
    """支持的数据库类型"""
    MYSQL = "mysql"
    ORACLE = "oracle"
    SPARK = "spark"
    POSTGRESQL = "postgresql"


class ExtractorType(str):
    """抽取器类型"""
    LIVE_DB = "live_db"
    OFFLINE_DDL = "offline_ddl"
    OFFLINE_JSON = "offline_json"


class JoinType(str):
    """JOIN 类型"""
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"
    FULL = "FULL OUTER JOIN"
