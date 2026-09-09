"""
MySQL 数据库客户端
负责元数据存储、历史记录管理
"""
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import re

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.pool import QueuePool
from loguru import logger
from sqlglot import exp, parse_one
from ..config_validation import validate_history_max_records
from ..utils.log_sanitizer import mask_sensitive, sanitize_sql, truncate_log
from ..sql.preflight import ExplainDeadlineExceeded, run_with_wall_clock_deadline
import json
import uuid

_SEMANTIC_CONTENT_SHA256 = re.compile(r"[0-9a-f]{64}")
_HISTORY_GOVERNANCE_COLUMNS = (
    (
        "semantic_datasource",
        "VARCHAR(191) NULL COMMENT '语义治理数据源'",
    ),
    (
        "semantic_dialect",
        "VARCHAR(32) NULL COMMENT '语义治理 SQL 方言'",
    ),
    (
        "semantic_version",
        "VARCHAR(64) NULL COMMENT '语义模型版本'",
    ),
    (
        "semantic_content_sha256",
        "CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL "
        "COMMENT '语义快照内容 SHA-256'",
    ),
    (
        "governance_validated",
        "TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否通过语义治理校验'",
    ),
)
_GOVERNED_FEW_SHOT_INDEX = "idx_governed_few_shot"
_DUPLICATE_SCHEMA_CHANGE_CODES = frozenset((1060, 1061))


class MySQLClient:
    """MySQL 数据库客户端 - 使用 SQLAlchemy 连接池"""

    explain_query_enforces_wall_clock_deadline = True

    def __init__(self, host: str, port: int, user: str, password: str, database: str,
                 pool_size: int = 10, max_overflow: int = 20, pool_timeout: int = 30,
                 history_max_records: int = 1000):
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
        self.history_max_records = validate_history_max_records(
            history_max_records, key="history_max_records"
        )
        self.engine = None

    def connect(self) -> bool:
        """
        连接到 MySQL 数据库（使用连接池）

        Returns:
            bool: 连接是否成功
        """
        try:
            # 创建 SQLAlchemy engine，使用 QueuePool 连接池
            db_url = URL.create(
                "mysql+mysqlconnector",
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
            )
            self.engine = create_engine(
                db_url,
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
            self._migrate_history_schema()
            logger.info(f"MySQL 连接池初始化成功：{self.host}:{self.port}/{self.database}, "
                       f"pool_size={self.pool_size}, max_overflow={self.max_overflow}")
            return True
        except Exception as e:
            if self.engine is not None:
                self.engine.dispose()
                self.engine = None
            logger.error(f"MySQL 连接池初始化失败：{e}")
            return False

    def _migrate_history_schema(self) -> None:
        """Idempotently add governed-history identity columns to an existing table."""
        column_rows = self.execute_query(
            """
            SELECT COLUMN_NAME AS column_name
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = :database_name
              AND TABLE_NAME = 'sql_generation_history'
            """,
            {"database_name": self.database},
            raise_on_error=True,
        )
        existing_columns = {
            row.get("column_name") for row in column_rows if row.get("column_name")
        }
        for column_name, definition in _HISTORY_GOVERNANCE_COLUMNS:
            if column_name not in existing_columns:
                self._apply_history_schema_change(
                    "ALTER TABLE sql_generation_history "
                    f"ADD COLUMN {column_name} {definition}",
                )

        index_rows = self.execute_query(
            """
            SELECT DISTINCT INDEX_NAME AS index_name
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = :database_name
              AND TABLE_NAME = 'sql_generation_history'
            """,
            {"database_name": self.database},
            raise_on_error=True,
        )
        existing_indexes = {
            row.get("index_name") for row in index_rows if row.get("index_name")
        }
        if _GOVERNED_FEW_SHOT_INDEX not in existing_indexes:
            self._apply_history_schema_change(
                """
                ALTER TABLE sql_generation_history
                ADD INDEX idx_governed_few_shot
                (governance_validated, semantic_datasource, semantic_dialect,
                 semantic_version, semantic_content_sha256,
                 validation_status, execution_status, created_at)
                """,
            )

    def _apply_history_schema_change(self, query: str) -> None:
        """Apply a migration DDL statement while tolerating only a concurrent winner."""
        try:
            self.execute_update(query, raise_on_error=True)
        except Exception as exc:
            if not self._is_duplicate_schema_change(exc):
                raise
            logger.info("历史治理迁移已由并发启动完成")

    @staticmethod
    def _is_duplicate_schema_change(error: Exception) -> bool:
        """Recognize MySQL's duplicate-column and duplicate-index error codes."""
        pending = [error]
        seen = set()
        while pending:
            current = pending.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            if getattr(current, "errno", None) in _DUPLICATE_SCHEMA_CHANGE_CODES:
                return True
            if any(
                value in _DUPLICATE_SCHEMA_CHANGE_CODES
                for value in getattr(current, "args", ())
            ):
                return True
            pending.extend(
                (
                    getattr(current, "orig", None),
                    getattr(current, "__cause__", None),
                    getattr(current, "__context__", None),
                )
            )
        return False

    def close(self):
        """关闭连接池"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
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
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute_query(self, query: str, params: Optional[dict] = None,
                      raise_on_error: bool = False) -> List[Dict]:
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
            safe_error = truncate_log(mask_sensitive(str(e)), 160)
            logger.error(f"MySQL 查询执行失败：{safe_error}, query={sanitize_sql(query)}")
            if raise_on_error:
                raise
            return []

    def execute_query_limited(self, query: str, max_rows: int,
                              params: Optional[dict] = None) -> tuple[List[Dict], bool]:
        """流式获取有限结果，避免将无界结果集一次性载入内存。"""
        if max_rows <= 0:
            raise ValueError("max_rows 必须大于 0")
        with self.get_connection() as conn:
            result = conn.execution_options(stream_results=True).execute(text(query), params or {})
            rows = result.mappings().fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            return [dict(row) for row in rows[:max_rows]], truncated

    def explain_query(self, query: str, timeout_ms: int) -> List[Dict]:
        """Run EXPLAIN on a disposable checkout within a wall-clock deadline."""
        if timeout_ms <= 0:
            raise ValueError("timeout_ms 必须大于 0")

        statement = parse_one(query, read="mysql")
        if not isinstance(statement, exp.Select):
            raise ValueError("仅支持 SELECT 预检")

        statement.set(
            "hint",
            exp.Hint(expressions=[exp.Var(this=f"MAX_EXECUTION_TIME({int(timeout_ms)})")]),
        )
        explain_sql = f"EXPLAIN {statement.sql(dialect='mysql')}"
        if self.engine is None:
            raise RuntimeError("MySQL 未连接")

        def execute(cancelled) -> List[Dict]:
            conn = self.engine.connect()
            try:
                if cancelled.is_set():
                    raise ExplainDeadlineExceeded("EXPLAIN deadline exceeded")
                result = conn.execute(text(explain_sql))
                rows = [dict(row._mapping) for row in result]
                if cancelled.is_set():
                    raise ExplainDeadlineExceeded("EXPLAIN deadline exceeded")
                return rows
            finally:
                if cancelled.is_set():
                    try:
                        conn.invalidate()
                    except Exception:
                        pass
                conn.close()

        return run_with_wall_clock_deadline(execute, timeout_ms)

    def execute_update(self, query: str, params: Optional[dict] = None,
                       raise_on_error: bool = False) -> int:
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
            safe_error = truncate_log(mask_sensitive(str(e)), 160)
            logger.error(f"MySQL 更新失败：{safe_error}, query={sanitize_sql(query)}")
            if raise_on_error:
                raise
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
                        validation_error: Optional[str] = None,
                        *,
                        semantic_datasource: Optional[str] = None,
                        semantic_dialect: Optional[str] = None,
                        semantic_version: Optional[str] = None,
                        semantic_content_sha256: Optional[str] = None,
                        governance_validated: bool = False) -> Optional[int]:
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

        self._validate_governance_identity(
            semantic_datasource=semantic_datasource,
            semantic_dialect=semantic_dialect,
            semantic_version=semantic_version,
            semantic_content_sha256=semantic_content_sha256,
            governance_validated=governance_validated,
        )

        query = """
        INSERT INTO sql_generation_history
        (session_id, user_query, generated_sql, matched_knowledge, retrieval_query,
         validation_status, validation_error, semantic_datasource, semantic_dialect,
         semantic_version, semantic_content_sha256, governance_validated)
        VALUES (:session_id, :user_query, :generated_sql, :matched_knowledge, :retrieval_query,
         :validation_status, :validation_error, :semantic_datasource, :semantic_dialect,
         :semantic_version, :semantic_content_sha256, :governance_validated)
        """
        params = {
            "session_id": session_id,
            "user_query": user_query,
            "generated_sql": generated_sql,
            "matched_knowledge": json.dumps(matched_knowledge, ensure_ascii=False) if matched_knowledge else None,
            "retrieval_query": retrieval_query,
            "validation_status": validation_status,
            "validation_error": validation_error,
            "semantic_datasource": semantic_datasource,
            "semantic_dialect": semantic_dialect,
            "semantic_version": semantic_version,
            "semantic_content_sha256": semantic_content_sha256,
            "governance_validated": governance_validated,
        }

        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query), params)
                history_id = result.lastrowid
                logger.info(f"保存 SQL 历史记录，ID={history_id}")
            self._prune_history()
            return history_id
        except Exception as e:
            logger.error(f"保存 SQL 历史失败：{e}")
            return None

    @staticmethod
    def _validate_governance_identity(
        *,
        semantic_datasource: Optional[str],
        semantic_dialect: Optional[str],
        semantic_version: Optional[str],
        semantic_content_sha256: Optional[str],
        governance_validated: bool,
    ) -> None:
        identity = (
            semantic_datasource,
            semantic_dialect,
            semantic_version,
            semantic_content_sha256,
        )
        if type(governance_validated) is not bool:
            raise ValueError("governance_validated must be a boolean")
        if all(value is None for value in identity):
            if governance_validated:
                raise ValueError("governed history requires a semantic identity")
            return
        if (
            not all(isinstance(value, str) and value for value in identity)
            or len(semantic_datasource or "") > 191
            or len(semantic_dialect or "") > 32
            or len(semantic_version or "") > 64
            or _SEMANTIC_CONTENT_SHA256.fullmatch(semantic_content_sha256 or "") is None
        ):
            raise ValueError("semantic history identity is invalid")

    def _prune_history(self) -> None:
        """保留最新的有限条历史，避免数据表无限增长。"""
        self.execute_update(
            """
            DELETE h FROM sql_generation_history AS h
            LEFT JOIN (
                SELECT id FROM (
                    SELECT id FROM sql_generation_history
                    ORDER BY created_at DESC, id DESC
                    LIMIT :history_limit
                ) AS recent_ids
            ) AS keep_rows ON keep_rows.id = h.id
            WHERE keep_rows.id IS NULL
            """,
            {"history_limit": self.history_max_records},
        )

    def update_sql_execution(self, history_id: int, execution_status: str,
                            execution_error: Optional[str] = None,
                            execution_result: Optional[Any] = None,
                            session_id: Optional[str] = None,
                            generated_sql: Optional[str] = None,
                            require_validation_success: bool = False,
                            *,
                            expected_user_query: Optional[str] = None,
                            expected_semantic_datasource: Optional[str] = None,
                            expected_semantic_dialect: Optional[str] = None,
                            expected_semantic_version: Optional[str] = None,
                            expected_semantic_content_sha256: Optional[str] = None,
                            require_governance_validated: bool = False) -> bool:
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
        predicates = ["id = :history_id"]
        if session_id is not None:
            predicates.append("session_id = :session_id")
        if generated_sql is not None:
            predicates.append("generated_sql = :generated_sql")
        if expected_user_query is not None:
            predicates.append("user_query = :expected_user_query")
        if require_validation_success:
            predicates.append("validation_status = 'success'")
        expected_identity = (
            expected_semantic_datasource,
            expected_semantic_dialect,
            expected_semantic_version,
            expected_semantic_content_sha256,
        )
        if any(value is not None for value in expected_identity):
            self._validate_governance_identity(
                semantic_datasource=expected_semantic_datasource,
                semantic_dialect=expected_semantic_dialect,
                semantic_version=expected_semantic_version,
                semantic_content_sha256=expected_semantic_content_sha256,
                governance_validated=require_governance_validated,
            )
            predicates.extend(
                (
                    "semantic_datasource = :expected_semantic_datasource",
                    "semantic_dialect = :expected_semantic_dialect",
                    "semantic_version = :expected_semantic_version",
                    "semantic_content_sha256 = :expected_semantic_content_sha256",
                )
            )
        elif require_governance_validated:
            raise ValueError("governed update requires a semantic identity")
        if require_governance_validated:
            predicates.append("governance_validated = 1")
        where_clause = " AND ".join(predicates)
        query = f"""
        UPDATE sql_generation_history
        SET execution_status = :execution_status,
            execution_error = :execution_error,
            execution_result = :execution_result,
            updated_at = CURRENT_TIMESTAMP
        WHERE {where_clause}
        """
        params = {
            "execution_status": execution_status,
            "execution_error": execution_error,
            "execution_result": json.dumps(execution_result, ensure_ascii=False, default=str)
            if execution_result is not None else None,
            "history_id": history_id,
        }
        if session_id is not None:
            params["session_id"] = session_id
        if generated_sql is not None:
            params["generated_sql"] = generated_sql
        if expected_user_query is not None:
            params["expected_user_query"] = expected_user_query
        if any(value is not None for value in expected_identity):
            params.update(
                {
                    "expected_semantic_datasource": expected_semantic_datasource,
                    "expected_semantic_dialect": expected_semantic_dialect,
                    "expected_semantic_version": expected_semantic_version,
                    "expected_semantic_content_sha256": (
                        expected_semantic_content_sha256
                    ),
                }
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

    def get_successful_history_for_few_shot(
        self,
        limit: int = 100,
        *,
        semantic_datasource: Optional[str] = None,
        semantic_dialect: Optional[str] = None,
        semantic_version: Optional[str] = None,
        semantic_content_sha256: Optional[str] = None,
        governance_validated: Optional[bool] = None,
    ) -> List[Dict]:
        """
        获取成功的历史记录用于 Few-Shot 检索

        Args:
            limit: 限制数量

        Returns:
            List[Dict]: 成功历史记录列表
        """
        identity = (
            semantic_datasource,
            semantic_dialect,
            semantic_version,
            semantic_content_sha256,
        )
        governed = governance_validated is not None or any(
            value is not None for value in identity
        )
        conditions = [
            "validation_status = 'success'",
            "execution_status = 'success'",
        ]
        params = {"limit": limit}
        if governed:
            if governance_validated is not True:
                raise ValueError("governed Few-Shot retrieval requires validation")
            self._validate_governance_identity(
                semantic_datasource=semantic_datasource,
                semantic_dialect=semantic_dialect,
                semantic_version=semantic_version,
                semantic_content_sha256=semantic_content_sha256,
                governance_validated=True,
            )
            conditions.extend(
                (
                    "governance_validated = :governance_validated",
                    "semantic_datasource = :semantic_datasource",
                    "semantic_dialect = :semantic_dialect",
                    "semantic_version = :semantic_version",
                    "semantic_content_sha256 = :semantic_content_sha256",
                )
            )
            params.update(
                {
                    "semantic_datasource": semantic_datasource,
                    "semantic_dialect": semantic_dialect,
                    "semantic_version": semantic_version,
                    "semantic_content_sha256": semantic_content_sha256,
                    "governance_validated": True,
                }
            )
        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, user_query, generated_sql, validation_status, execution_status,
               semantic_datasource, semantic_dialect, semantic_version,
               semantic_content_sha256, governance_validated
        FROM sql_generation_history
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT :limit
        """
        return self.execute_query(query, params)

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
        获取统计信息（合并为单次查询，减少数据库往返）

        Returns:
            Dict: 统计数据
        """
        query = """
            SELECT
                (SELECT COUNT(*) FROM sql_generation_history) AS total_history,
                (SELECT CASE WHEN COUNT(*) = 0 THEN NULL
                        ELSE SUM(CASE WHEN execution_status = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
                   END
                 FROM sql_generation_history WHERE execution_status != 'pending') AS success_rate,
                (SELECT AVG(feedback_score) FROM sql_generation_history WHERE feedback_score IS NOT NULL) AS avg_feedback_score,
                (SELECT COUNT(*) FROM table_schema) AS total_tables,
                (SELECT COUNT(*) FROM column_schema) AS total_columns
        """
        results = self.execute_query(query)
        if results:
            row = results[0]
            return {
                'total_history': row.get('total_history', 0),
                'success_rate': row.get('success_rate'),
                'avg_feedback_score': row.get('avg_feedback_score'),
                'total_tables': row.get('total_tables', 0),
                'total_columns': row.get('total_columns', 0),
            }
        return {}


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
