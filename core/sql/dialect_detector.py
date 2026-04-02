"""
SQL 方言检测器
根据用户自然语言查询判断目标数据库类型，生成对应方言的 SQL
"""
import re
from typing import Optional, Dict, List, Tuple
from loguru import logger


# 数据库类型关键词映射
DATABASE_KEYWORDS = {
    "mysql": ["mysql", "mariadb", "innodb", "myisam"],
    "oracle": ["oracle", "pl/sql", "dba_", "all_", "user_"],
    "postgresql": ["postgresql", "postgres", "psql"],
    "sparksql": ["spark", "sparksql", "hive", "databricks"],
    "sqlite": ["sqlite", "sqlite3"],
    "sqlserver": ["sqlserver", "mssql", "tsql", "sybase"],
}

# 数据库表/字段名模式
DATABASE_TABLE_PATTERNS = {
    "mysql": [r"`[^`]+`", r"\binformation_schema\b", r"\bperformance_schema\b"],
    "oracle": [r"\bdba_\w+\b", r"\ball_\w+\b", r"\buser_\w+\b", r"\b(sys|system)\.\w+\b"],
    "postgresql": [r"\bpg_\w+\b", r"\binformation_schema\b"],
    "sparksql": [r"\bsys\.\w+\b", r"\bdefault\.\w+\b", r"\bhive_\w+\b"],
}


class SQLDialectDetector:
    """SQL 方言检测器"""

    def __init__(self, neo4j_client=None):
        """
        初始化检测器

        Args:
            neo4j_client: Neo4j 客户端实例
        """
        self.neo4j_client = neo4j_client
        self.database_cache: List[Dict] = []
        self._load_database_metadata()

    def _load_database_metadata(self):
        """从 Neo4j 加载数据库元数据"""
        if not self.neo4j_client:
            return

        result = self.neo4j_client.execute_query("""
            MATCH (d:Database)
            RETURN d.database_id as database_id, d.name as name, d.db_type as db_type,
                   d.db_language as db_language, d.description as description
        """)

        self.database_cache = [
            {
                "database_id": r["database_id"],
                "name": r["name"],
                "db_type": r["db_type"],
                "db_language": r.get("db_language", "SQL"),
                "description": r.get("description", "")
            }
            for r in result
        ]
        logger.info(f"加载数据库元数据：{len(self.database_cache)} 个数据库")

    def detect_from_query(self, query: str) -> Tuple[str, Optional[int]]:
        """
        从用户查询中检测目标数据库类型

        Args:
            query: 用户自然语言查询

        Returns:
            Tuple[str, Optional[int]]: (数据库类型，数据库 ID)
        """
        query_lower = query.lower()

        # 1. 检查是否明确提到数据库名称
        for db in self.database_cache:
            db_name = db["name"].lower()
            if db_name in query_lower:
                logger.info(f"检测到数据库名称：{db['name']} -> {db['db_type']}")
                return db["db_type"], db["database_id"]

        # 2. 检查数据库类型关键词
        for db_type, keywords in DATABASE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    logger.info(f"检测到数据库关键词：{keyword} -> {db_type}")
                    # 返回第一个匹配的数据库类型
                    matching_db = next((d for d in self.database_cache if d["db_type"] == db_type), None)
                    if matching_db:
                        return db_type, matching_db["id"]
                    return db_type, None

        # 3. 检查表名模式
        for db_type, patterns in DATABASE_TABLE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"检测到表名模式：{pattern} -> {db_type}")
                    matching_db = next((d for d in self.database_cache if d["db_type"] == db_type), None)
                    if matching_db:
                        return db_type, matching_db["database_id"]
                    return db_type, None

        # 4. 默认返回第一个数据库类型
        if self.database_cache:
            default_db = self.database_cache[0]
            logger.info(f"使用默认数据库：{default_db['name']} ({default_db['db_type']})")
            return default_db["db_type"], default_db["database_id"]

        logger.warning("未检测到数据库，返回默认 mysql")
        return "mysql", None

    def get_database_info(self, database_id: Optional[int] = None) -> Optional[Dict]:
        """
        获取数据库信息

        Args:
            database_id: 数据库 ID，为 None 时返回第一个

        Returns:
            数据库信息字典
        """
        if not self.database_cache:
            return None

        if database_id is None:
            return self.database_cache[0]

        for db in self.database_cache:
            if db["database_id"] == database_id:
                return db
        return None

    def get_all_databases(self) -> List[Dict]:
        """获取所有数据库列表"""
        return self.database_cache.copy()

    def refresh_metadata(self):
        """刷新数据库元数据缓存"""
        self.database_cache = []
        self._load_database_metadata()
