"""
SQL 生成器模块
整合检索、链式处理，提供统一的 SQL 生成接口（支持异步）
支持多数据库方言动态检测
"""
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from loguru import logger

from ..retrieval.bm25_tfidf import KnowledgeRetriever
from ..chain.sql_chain import SQLGenerationChain, QueryRewriteChain
from ..history.mysql_client import MySQLClient
from .dialect_detector import SQLDialectDetector


class SQLGenerator:
    """SQL 生成器 - 核心业务逻辑封装"""

    def __init__(self, llm, knowledge_retriever: KnowledgeRetriever,
                 sql_chain: SQLGenerationChain,
                 mysql_client: Optional[MySQLClient] = None,
                 config: Optional[Dict] = None,
                 neo4j_client=None):
        """
        初始化 SQL 生成器

        Args:
            llm: LLM 实例
            knowledge_retriever: 知识检索器
            sql_chain: SQL 生成链
            mysql_client: MySQL 客户端
            config: 配置字典
            neo4j_client: Neo4j 客户端（用于方言检测）
        """
        self.llm = llm
        self.knowledge_retriever = knowledge_retriever
        self.sql_chain = sql_chain
        self.mysql_client = mysql_client
        self.config = config or {}
        self.neo4j_client = neo4j_client

        self.rewrite_chain = QueryRewriteChain(llm) if self.config.get('enable_rewrite', False) else None

        # 初始化方言检测器
        self.dialect_detector = SQLDialectDetector(neo4j_client) if neo4j_client else None
        logger.info(f"SQL 生成器初始化完成，方言检测器：{'已启用' if self.dialect_detector else '未启用'}")

    def generate_sql(self, user_query: str, session_id: Optional[str] = None,
                    auto_save_history: bool = True,
                    use_few_shot: bool = True,
                    target_dialect: Optional[str] = None,
                    target_database_id: Optional[int] = None) -> Dict[str, Any]:
        """
        生成 SQL 的主入口 - 支持动态 Few-Shot 示例检索

        Args:
            user_query: 用户自然语言查询
            session_id: 会话 ID
            auto_save_history: 是否自动保存历史
            use_few_shot: 是否使用动态 Few-Shot 示例
            target_dialect: 目标 SQL 方言（可选，优先于自动检测）
            target_database_id: 目标数据库 ID（可选，优先于自动检测）

        Returns:
            Dict: 生成结果
        """
        logger.info(f"开始生成 SQL, query={user_query}")

        # 1. 检测 SQL 方言和目标数据库（如果未指定）
        detected_dialect = target_dialect
        detected_database_id = target_database_id

        if self.dialect_detector and not target_dialect:
            detected_dialect, detected_database_id = self.dialect_detector.detect_from_query(user_query)
            logger.info(f"方言检测结果：dialect={detected_dialect}, database_id={detected_database_id}")

        # 2. 可选：改写查询
        final_query = user_query
        if self.rewrite_chain:
            final_query = self.rewrite_chain.rewrite(user_query)
            logger.info(f"查询改写：{user_query} -> {final_query}")

        # 3. 检索相关知识（使用检测到的 database_id 过滤）
        knowledge_results = self.knowledge_retriever.retrieve_all(
            final_query,
            top_k=self.sql_chain.retrieval_top_k,
            database_id=detected_database_id
        )
        logger.info(f"检索到 {len(knowledge_results)} 个相关知识节点")

        # 4. 检索动态 Few-Shot 示例
        few_shot_examples = ""
        if use_few_shot and self.knowledge_retriever.history_index is not None:
            few_shot_examples = self.knowledge_retriever.get_few_shot_examples_formatted(
                final_query,
                top_k=self.config.get('few_shot_top_k', 3)
            )
            logger.info(f"检索到 Few-Shot 示例")

        # 5. 获取 Schema 信息（使用检测到的 database_id 过滤）
        schema_info = self._get_schema_info(database_id=detected_database_id)

        # 6. 如果方言改变，重新创建 SQLChain
        if detected_dialect and detected_dialect != self.sql_chain.dialect:
            logger.info(f"切换方言：{self.sql_chain.dialect} -> {detected_dialect}")
            from ..chain.sql_chain import SQLGenerationChain
            self.sql_chain = SQLGenerationChain(
                llm=self.llm,
                dialect=detected_dialect,
                retrieval_top_k=self.config.get('retrieval_top_k', 5)
            )

        # 7. 生成 SQL
        result = self.sql_chain.generate(
            user_query=final_query,
            schema_info=schema_info,
            knowledge_results=knowledge_results,
            session_id=session_id,
            few_shot_examples=few_shot_examples
        )

        # 6. 保存历史（同时添加到 Few-Shot 索引）
        if auto_save_history and self.mysql_client and result.get('sql'):
            history_id = self.mysql_client.save_sql_history(
                user_query=user_query,
                generated_sql=result['sql'],
                session_id=session_id,
                matched_knowledge=result.get('matched_knowledge'),
                retrieval_query=final_query,
                validation_status=result.get('status', 'pending'),
                validation_error=result.get('error')
            )

            # 添加到 Few-Shot 索引（如果执行成功）
            if history_id and result.get('status') == 'success':
                history_record = {
                    'id': history_id,
                    'user_query': user_query,
                    'generated_sql': result['sql'],
                    'execution_status': 'success',
                    'validation_status': 'success'
                }
                self.knowledge_retriever.add_history_record(history_record)

        return result

    def _get_schema_info(self, database_id: Optional[int] = None) -> Dict:
        """获取 Schema 信息"""
        if not self.mysql_client:
            return {'tables': [], 'columns': [], 'relationships': []}

        tables = self.mysql_client.get_table_schema()

        # 如果指定了 database_id，需要过滤表
        # 这里假设表信息中包含 database_id 字段
        if database_id is not None:
            # 从 Neo4j 获取指定数据库的表
            if self.neo4j_client:
                try:
                    result = self.neo4j_client.execute_query("""
                        MATCH (d:Database {database_id: $database_id})-[:HAS_TABLE]->(t:Table)
                        RETURN t.name as table_name, t.description as table_comment
                    """, {"database_id": database_id})
                    table_names = [r["table_name"] for r in result]
                    tables = [t for t in tables if t['table_name'] in table_names]
                except Exception as e:
                    logger.warning(f"从 Neo4j 过滤表失败：{e}")

        all_columns = []
        for table in tables:
            columns = self.mysql_client.get_column_schema(table['table_name'])
            all_columns.extend(columns)

        return {
            'tables': tables,
            'columns': all_columns,
            'relationships': []  # 可以从 Neo4j 获取
        }

    def execute_sql(self, sql: str, history_id: Optional[int] = None) -> Dict[str, Any]:
        """
        执行 SQL

        Args:
            sql: SQL 语句
            history_id: 历史记录 ID

        Returns:
            Dict: 执行结果
        """
        if not self.mysql_client:
            return {
                'status': 'error',
                'error': 'MySQL 客户端未连接',
                'data': None
            }

        logger.info(f"执行 SQL: {sql[:100]}...")

        try:
            # 只允许 SELECT 查询
            if not self._is_safe_query(sql):
                return {
                    'status': 'error',
                    'error': '只允许执行 SELECT 查询',
                    'data': None
                }

            results = self.mysql_client.execute_query(sql)

            # 更新历史记录
            if history_id and self.mysql_client:
                self.mysql_client.update_sql_execution(
                    history_id=history_id,
                    execution_status='success',
                    execution_result={'row_count': len(results), 'data_preview': results[:5]}
                )

            return {
                'status': 'success',
                'error': None,
                'data': results,
                'row_count': len(results)
            }

        except Exception as e:
            logger.error(f"SQL 执行失败：{e}")

            # 更新历史记录
            if history_id and self.mysql_client:
                self.mysql_client.update_sql_execution(
                    history_id=history_id,
                    execution_status='fail',
                    execution_error=str(e)
                )

            return {
                'status': 'fail',
                'error': str(e),
                'data': None
            }

    def _is_safe_query(self, sql: str) -> bool:
        """检查 SQL 是否安全（只允许 SELECT）"""
        sql_upper = sql.strip().upper()
        unsafe_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']

        for keyword in unsafe_keywords:
            if sql_upper.startswith(keyword) or f" {keyword} " in sql_upper:
                return False

        return True

    def fix_and_retry(self, history_id: int, error: str) -> Dict[str, Any]:
        """
        修复 SQL 并重试

        Args:
            history_id: 历史记录 ID
            error: 错误信息

        Returns:
            Dict: 结果
        """
        # 获取原始记录
        history = self.mysql_client.get_history_detail(history_id)
        if not history:
            return {'status': 'error', 'error': '历史记录不存在'}

        original_sql = history.get('generated_sql', '')
        schema_info = self._get_schema_info()

        # 修复 SQL
        fix_result = self.sql_chain.fix_sql(
            sql=original_sql,
            error=error,
            schema_info=schema_info
        )

        if fix_result['status'] != 'success':
            return fix_result

        # 更新历史记录
        self.mysql_client.execute_update(
            "UPDATE sql_generation_history SET generated_sql = %s, validation_status = 'pending' WHERE id = %s",
            (fix_result['sql'], history_id)
        )

        # 重新执行
        return self.execute_sql(fix_result['sql'], history_id)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'chain_stats': self.sql_chain.get_stats(),
            'retriever_stats': {
                'document_count': self.knowledge_retriever.retriever.get_document_count()
            }
        }

        if self.mysql_client:
            stats['history_stats'] = self.mysql_client.get_statistics()

        return stats

    def detect_dialect(self, user_query: str) -> Tuple[str, Optional[int]]:
        """
        检测用户查询的目标数据库方言

        Args:
            user_query: 用户自然语言查询

        Returns:
            Tuple[str, Optional[int]]: (方言类型，数据库 ID)
        """
        if self.dialect_detector:
            return self.dialect_detector.detect_from_query(user_query)
        return self.config.get('sql_dialect', 'mysql'), None

    async def generate_sql_async(self, user_query: str, session_id: Optional[str] = None,
                                 auto_save_history: bool = True,
                                 use_few_shot: bool = True,
                                 target_dialect: Optional[str] = None,
                                 target_database_id: Optional[int] = None) -> Dict[str, Any]:
        """
        生成 SQL 的异步入口

        Args:
            user_query: 用户自然语言查询
            session_id: 会话 ID
            auto_save_history: 是否自动保存历史
            use_few_shot: 是否使用动态 Few-Shot 示例
            target_dialect: 目标 SQL 方言（可选，优先于自动检测）
            target_database_id: 目标数据库 ID（可选，优先于自动检测）

        Returns:
            Dict: 生成结果
        """
        logger.info(f"开始异步生成 SQL, query={user_query}")

        # 1. 检测 SQL 方言和目标数据库（如果未指定）
        detected_dialect = target_dialect
        detected_database_id = target_database_id

        if self.dialect_detector and not target_dialect:
            detected_dialect, detected_database_id = self.dialect_detector.detect_from_query(user_query)
            logger.info(f"方言检测结果：dialect={detected_dialect}, database_id={detected_database_id}")

        # 2. 可选：改写查询（异步）
        final_query = user_query
        if self.rewrite_chain:
            final_query = await self.rewrite_chain.rewrite_async(user_query)
            logger.info(f"查询改写：{user_query} -> {final_query}")

        # 3. 检索相关知识（使用检测到的 database_id 过滤）
        knowledge_results = self.knowledge_retriever.retrieve_all(
            final_query,
            top_k=self.sql_chain.retrieval_top_k,
            database_id=detected_database_id
        )
        logger.info(f"检索到 {len(knowledge_results)} 个相关知识节点")

        # 4. 检索动态 Few-Shot 示例
        few_shot_examples = ""
        if use_few_shot and self.knowledge_retriever.history_index is not None:
            few_shot_examples = self.knowledge_retriever.get_few_shot_examples_formatted(
                final_query,
                top_k=self.config.get('few_shot_top_k', 3)
            )
            logger.info(f"检索到 Few-Shot 示例")

        # 5. 获取 Schema 信息（使用检测到的 database_id 过滤）
        schema_info = self._get_schema_info(database_id=detected_database_id)

        # 6. 如果方言改变，重新创建 SQLChain
        if detected_dialect and detected_dialect != self.sql_chain.dialect:
            logger.info(f"切换方言：{self.sql_chain.dialect} -> {detected_dialect}")
            from ..chain.sql_chain import SQLGenerationChain
            self.sql_chain = SQLGenerationChain(
                llm=self.llm,
                dialect=detected_dialect,
                retrieval_top_k=self.config.get('retrieval_top_k', 5)
            )

        # 7. 异步生成 SQL
        result = await self.sql_chain.generate_async(
            user_query=final_query,
            schema_info=schema_info,
            knowledge_results=knowledge_results,
            session_id=session_id,
            few_shot_examples=few_shot_examples
        )

        # 6. 保存历史
        if auto_save_history and self.mysql_client and result.get('sql'):
            history_id = self.mysql_client.save_sql_history(
                user_query=user_query,
                generated_sql=result['sql'],
                session_id=session_id,
                matched_knowledge=result.get('matched_knowledge'),
                retrieval_query=final_query,
                validation_status=result.get('status', 'pending'),
                validation_error=result.get('error')
            )

            # 添加到 Few-Shot 索引
            if history_id and result.get('status') == 'success':
                history_record = {
                    'id': history_id,
                    'user_query': user_query,
                    'generated_sql': result['sql'],
                    'execution_status': 'success',
                    'validation_status': 'success'
                }
                self.knowledge_retriever.add_history_record(history_record)

        return result


def create_sql_generator(llm, neo4j_client, mysql_client, config: Optional[Dict] = None,
                         embedding_factory: Optional = None) -> SQLGenerator:
    """
    创建 SQL 生成器工厂函数

    Args:
        llm: LLM 实例
        neo4j_client: Neo4j 客户端
        mysql_client: MySQL 客户端
        config: 配置
        embedding_factory: Embedding 工厂实例

    Returns:
        SQLGenerator: SQL 生成器实例
    """
    config = config or {}

    # 1. 从 Neo4j 同步知识到检索器
    knowledge_nodes = neo4j_client.get_schema_metadata() if neo4j_client else {'tables': [], 'columns': []}

    # 2. 构建检索器（支持真实 Embedding）
    knowledge_retriever = KnowledgeRetriever(
        config=config,
        embedding_factory=embedding_factory
    )

    # 将知识节点转换为检索器可用的格式
    documents = []
    metadata = []

    for table in knowledge_nodes.get('tables', []):
        doc = f"{table.get('name', '')} {table.get('description', '')}"
        if doc.strip():
            documents.append(doc)
            metadata.append({
                'node_type': 'table',
                'name': table.get('name', ''),
                'description': table.get('description', '')
            })

    # 3. 创建 SQL 生成链
    sql_chain = SQLGenerationChain(
        llm=llm,
        dialect=config.get('sql_dialect', 'mysql'),
        retrieval_top_k=config.get('retrieval_top_k', 5)
    )

    # 4. 创建 SQL 生成器
    generator = SQLGenerator(
        llm=llm,
        knowledge_retriever=knowledge_retriever,
        sql_chain=sql_chain,
        mysql_client=mysql_client,
        config=config
    )

    # 5. 构建知识库索引
    if documents:
        knowledge_retriever.retriever.index_documents(documents, metadata)

    # 6. 构建 Few-Shot 索引（从历史成功记录）
    if mysql_client and config.get('few_shot_enabled', True):
        try:
            history_records = mysql_client.get_history_list(limit=100)
            knowledge_retriever.build_few_shot_index(history_records)
            logger.info(f"Few-Shot 索引构建完成，共 {len(knowledge_retriever.history_records)} 条成功记录")
        except Exception as e:
            logger.warning(f"构建 Few-Shot 索引失败：{e}")

    return generator
