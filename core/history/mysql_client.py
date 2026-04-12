"""
MySQL 数据库客户端
负责元数据存储、历史记录管理
"""
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from loguru import logger
import json
from datetime import datetime
import uuid


class MySQLClient:
    """MySQL 数据库客户端 - 使用 SQLAlchemy 连接池"""

    def __init__(self, host: str, port: int, user: str, password: str, database: str,
                 pool_size: int = 10, max_overflow: int = 20, pool_timeout: int = 30):
        """
        初始化 MySQL 客户端

        Args:
            host: 主机地址
            port: 端口
            user: 用户名
            password: 密码
            database: 数据库名
            pool_size: 连接池大小，默认 10
            max_overflow: 最大溢出连接数，默认 20
            pool_timeout: 获取连接超时时间（秒），默认 30
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.engine = None

    def connect(self) -> bool:
        """
        连接到 MySQL 数据库（使用连接池）

        Returns:
            bool: 连接是否成功
        """
        try:
            # 创建 SQLAlchemy engine，使用 QueuePool 连接池
            self.engine = create_engine(
                f"mysql+mysqlconnector://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}",
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=3600,  # 1 小时后回收连接
                pool_pre_ping=True,  # 自动检测并清理失效连接
                echo=False  # 生产环境关闭 SQL 日志
            )
            # 验证连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"MySQL 连接池初始化成功：{self.host}:{self.port}/{self.database}, "
                       f"pool_size={self.pool_size}, max_overflow={self.max_overflow}")
            return True
        except Exception as e:
            logger.error(f"MySQL 连接池初始化失败：{e}")
            return False

    def close(self):
        """关闭连接池"""
        if self.engine:
            self.engine.dispose()
            logger.info("MySQL 连接池已关闭")

    @contextmanager
    def get_connection(self):
        """
        获取数据库连接（上下文管理器）

        Yields:
            SQLAlchemy Connection
        """
        if not self.engine:
            logger.error("MySQL 未连接")
            raise RuntimeError("MySQL 未连接")

        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def execute_query(self, query: str, params: Optional[dict] = None) -> List[Dict]:
        """
        执行查询（使用连接池）

        Args:
            query: SQL 查询语句
            params: 查询参数（字典格式）

        Returns:
            List[Dict]: 查询结果
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"MySQL 查询执行失败：{e}, query={query}")
            return []

    def execute_update(self, query: str, params: Optional[dict] = None) -> int:
        """
        执行更新操作（使用连接池）

        Args:
            query: SQL 语句
            params: 参数（字典格式）

        Returns:
            int: 受影响的行数
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query), params or {})
                return result.rowcount
        except Exception as e:
            logger.error(f"MySQL 更新失败：{e}, query={query}")
            return 0

    def upsert_table_schema(self, table_name: str, table_comment: str = "",
                           properties: Optional[Dict] = None) -> bool:
        """
        插入或更新表结构信息

        Args:
            table_name: 表名
            table_comment: 表注释
            properties: 额外属性

        Returns:
            bool: 是否成功
        """
        query = """
        INSERT INTO table_schema (table_name, table_comment, properties)
        VALUES (:table_name, :table_comment, :properties)
        ON DUPLICATE KEY UPDATE
            table_comment = VALUES(table_comment),
            properties = VALUES(properties),
            updated_at = CURRENT_TIMESTAMP
        """
        params = {
            "table_name": table_name,
            "table_comment": table_comment,
            "properties": json.dumps(properties) if properties else None
        }
        result = self.execute_update(query, params)
        return result >= 0

    def upsert_column_schema(self, table_name: str, column_name: str,
                            column_type: str = "VARCHAR", column_comment: str = "",
                            is_nullable: bool = True, is_primary_key: bool = False) -> bool:
        """
        插入或更新字段信息

        Args:
            table_name: 表名
            column_name: 字段名
            column_type: 字段类型
            column_comment: 字段注释
            is_nullable: 是否可为空
            is_primary_key: 是否主键

        Returns:
            bool: 是否成功
        """
        # 先获取表 ID
        table_query = "SELECT id FROM table_schema WHERE table_name = :table_name"
        table_results = self.execute_query(table_query, {"table_name": table_name})

        if not table_results:
            logger.warning(f"表 {table_name} 不存在")
            return False

        table_id = table_results[0]['id']

        query = """
        INSERT INTO column_schema (table_id, column_name, column_type, column_comment,
                                   is_nullable, is_primary_key)
        VALUES (:table_id, :column_name, :column_type, :column_comment, :is_nullable, :is_primary_key)
        ON DUPLICATE KEY UPDATE
            column_type = VALUES(column_type),
            column_comment = VALUES(column_comment),
            is_nullable = VALUES(is_nullable),
            is_primary_key = VALUES(is_primary_key),
            updated_at = CURRENT_TIMESTAMP
        """
        params = {
            "table_id": table_id,
            "column_name": column_name,
            "column_type": column_type,
            "column_comment": column_comment,
            "is_nullable": 1 if is_nullable else 0,
            "is_primary_key": 1 if is_primary_key else 0
        }
        result = self.execute_update(query, params)
        return result >= 0

    def save_sql_history(self, user_query: str, generated_sql: str,
                        session_id: Optional[str] = None,
                        matched_knowledge: Optional[List] = None,
                        retrieval_query: Optional[str] = None,
                        validation_status: str = "pending",
                        validation_error: Optional[str] = None) -> Optional[int]:
        """
        保存 SQL 生成历史

        Args:
            user_query: 用户查询
            generated_sql: 生成的 SQL
            session_id: 会话 ID
            matched_knowledge: 匹配的知识
            retrieval_query: 检索查询
            validation_status: 验证状态
            validation_error: 验证错误

        Returns:
            Optional[int]: 插入的记录 ID
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        query = """
        INSERT INTO sql_generation_history
        (session_id, user_query, generated_sql, matched_knowledge, retrieval_query,
         validation_status, validation_error)
        VALUES (:session_id, :user_query, :generated_sql, :matched_knowledge, :retrieval_query,
         :validation_status, :validation_error)
        """
        params = {
            "session_id": session_id,
            "user_query": user_query,
            "generated_sql": generated_sql,
            "matched_knowledge": json.dumps(matched_knowledge, ensure_ascii=False) if matched_knowledge else None,
            "retrieval_query": retrieval_query,
            "validation_status": validation_status,
            "validation_error": validation_error
        }

        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query), params)
                conn.commit()
                history_id = result.lastrowid
                logger.info(f"保存 SQL 历史记录，ID={history_id}")
                return history_id
        except Exception as e:
            logger.error(f"保存 SQL 历史失败：{e}")
            return None

    def update_sql_execution(self, history_id: int, execution_status: str,
                            execution_error: Optional[str] = None,
                            execution_result: Optional[Any] = None) -> bool:
        """
        更新 SQL 执行结果

        Args:
            history_id: 历史记录 ID
            execution_status: 执行状态
            execution_error: 执行错误
            execution_result: 执行结果

        Returns:
            bool: 是否成功
        """
        query = """
        UPDATE sql_generation_history
        SET execution_status = %s, execution_error = %s, execution_result = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """
        params = (
            execution_status,
            execution_error,
            json.dumps(execution_result, ensure_ascii=False, default=str) if execution_result else None,
            history_id
        )
        result = self.execute_update(query, params)
        return result > 0

    def get_history_list(self, session_id: Optional[str] = None,
                        limit: int = 50, offset: int = 0,
                        status_filter: Optional[str] = None) -> List[Dict]:
        """
        获取历史记录列表

        Args:
            session_id: 会话 ID 过滤
            limit: 限制数量
            offset: 偏移量
            status_filter: 状态过滤 (success/fail/pending)

        Returns:
            List[Dict]: 历史记录列表
        """
        conditions = []
        params = {}

        if session_id:
            conditions.append("session_id = :session_id")
            params["session_id"] = session_id

        if status_filter:
            conditions.append("execution_status = :status_filter")
            params["status_filter"] = status_filter

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        query = f"""
        SELECT id, session_id, user_query, generated_sql, validation_status,
               execution_status, execution_error, execution_result, created_at
        FROM sql_generation_history
        {where_clause}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
        """
        params["limit"] = limit
        params["offset"] = offset

        return self.execute_query(query, params)

    def get_successful_history_for_few_shot(self, limit: int = 100) -> List[Dict]:
        """
        获取成功的历史记录用于 Few-Shot 检索

        Args:
            limit: 限制数量

        Returns:
            List[Dict]: 成功历史记录列表
        """
        query = """
        SELECT id, user_query, generated_sql, validation_status, execution_status
        FROM sql_generation_history
        WHERE execution_status = 'success' OR validation_status = 'success'
        ORDER BY created_at DESC
        LIMIT :limit
        """
        return self.execute_query(query, {"limit": limit})

    def get_history_detail(self, history_id: int) -> Optional[Dict]:
        """
        获取历史记录详情

        Args:
            history_id: 历史记录 ID

        Returns:
            Optional[Dict]: 历史记录详情
        """
        query = "SELECT * FROM sql_generation_history WHERE id = :history_id"
        results = self.execute_query(query, {"history_id": history_id})
        return results[0] if results else None

    def get_table_schema(self, table_name: Optional[str] = None) -> List[Dict]:
        """
        获取表结构信息

        Args:
            table_name: 表名过滤

        Returns:
            List[Dict]: 表结构信息
        """
        if table_name:
            query = """
            SELECT ts.*,
                   GROUP_CONCAT(cs.column_name ORDER BY cs.id) AS columns
            FROM table_schema ts
            LEFT JOIN column_schema cs ON ts.id = cs.table_id
            WHERE ts.table_name = :table_name
            GROUP BY ts.id
            """
            return self.execute_query(query, {"table_name": table_name})
        else:
            query = """
            SELECT ts.*,
                   GROUP_CONCAT(cs.column_name ORDER BY cs.id) AS columns
            FROM table_schema ts
            LEFT JOIN column_schema cs ON ts.id = cs.table_id
            GROUP BY ts.id
            """
            return self.execute_query(query)

    def get_column_schema(self, table_name: str) -> List[Dict]:
        """
        获取字段信息

        Args:
            table_name: 表名

        Returns:
            List[Dict]: 字段信息
        """
        query = """
        SELECT cs.*, ts.table_name
        FROM column_schema cs
        JOIN table_schema ts ON cs.table_id = ts.id
        WHERE ts.table_name = :table_name
        ORDER BY cs.id
        """
        return self.execute_query(query, {"table_name": table_name})

    def save_feedback(self, history_id: int, feedback_type: str,
                     feedback_score: Optional[int] = None,
                     feedback_comment: Optional[str] = None,
                     corrected_sql: Optional[str] = None) -> bool:
        """
        保存用户反馈

        Args:
            history_id: 历史记录 ID
            feedback_type: 反馈类型
            feedback_score: 评分
            feedback_comment: 反馈备注
            corrected_sql: 修正后的 SQL

        Returns:
            bool: 是否成功
        """
        # 同时更新历史记录的反馈字段
        if feedback_score:
            self.execute_update(
                "UPDATE sql_generation_history SET feedback_score = :feedback_score WHERE id = :history_id",
                {"feedback_score": feedback_score, "history_id": history_id}
            )

        query = """
        INSERT INTO user_feedback (history_id, feedback_type, feedback_score,
                                   feedback_comment, corrected_sql)
        VALUES (:history_id, :feedback_type, :feedback_score, :feedback_comment, :corrected_sql)
        """
        params = {
            "history_id": history_id,
            "feedback_type": feedback_type,
            "feedback_score": feedback_score,
            "feedback_comment": feedback_comment,
            "corrected_sql": corrected_sql
        }
        result = self.execute_update(query, params)
        return result > 0

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns:
            Dict: 统计数据
        """
        queries = {
            'total_history': "SELECT COUNT(*) as count FROM sql_generation_history",
            'success_rate': """
                SELECT
                    SUM(CASE WHEN execution_status = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as rate
                FROM sql_generation_history WHERE execution_status != 'pending'
            """,
            'avg_feedback_score': "SELECT AVG(feedback_score) as score FROM sql_generation_history WHERE feedback_score IS NOT NULL",
            'total_tables': "SELECT COUNT(*) as count FROM table_schema",
            'total_columns': "SELECT COUNT(*) as count FROM column_schema"
        }

        stats = {}
        for key, query in queries.items():
            results = self.execute_query(query)
            if results:
                stats[key] = results[0].get(list(results[0].keys())[0])

        return stats


# ==================== 向后兼容（已废弃，请使用 ApplicationContext） ====================
# 保留全局单例以兼容旧代码，新功能应使用 ApplicationContext

_mysql_client: Optional[MySQLClient] = None


def get_mysql_client(host: str, port: int, user: str, password: str,
                     database: str, pool_size: int = 10,
                     max_overflow: int = 20, pool_timeout: int = 30) -> MySQLClient:
    """
    获取或创建 MySQL 客户端单例（带连接池配置）

    .. deprecated:: 请使用 ApplicationContext 管理组件生命周期

    Args:
        host: 主机
        port: 端口
        user: 用户名
        password: 密码
        database: 数据库名
        pool_size: 连接池大小，默认 10
        max_overflow: 最大溢出连接数，默认 20
        pool_timeout: 获取连接超时时间（秒），默认 30

    Returns:
        MySQLClient: 客户端实例
    """
    global _mysql_client
    if _mysql_client is None:
        _mysql_client = MySQLClient(host, port, user, password, database,
                                    pool_size, max_overflow, pool_timeout)
        _mysql_client.connect()
    return _mysql_client
