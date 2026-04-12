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
        self.llm = llm
        self.knowledge_retriever = knowledge_retriever
        self.sql_chain = sql_chain
        self.mysql_client = mysql_client
        self.config = config or {}
        self.neo4j_client = neo4j_client

        self.rewrite_chain = QueryRewriteChain(llm) if self.config.get('enable_rewrite', False) else None
        self.dialect_detector = SQLDialectDetector(neo4j_client) if neo4j_client else None
        logger.info(f"SQL 生成器初始化完成，方言检测器：{'已启用' if self.dialect_detector else '未启用'}")

    # ==================== 核心共享逻辑 ====================

    async def _prepare_query(self, user_query: str, use_async: bool = False) -> str:
        """查询改写（可选）"""
        if self.rewrite_chain:
            if use_async:
                return await self.rewrite_chain.rewrite_async(user_query)
            return self.rewrite_chain.rewrite(user_query)
        return user_query

    def _detect_dialect_and_db(self, user_query: str,
                                 target_dialect: Optional[str],
                                 target_database_id: Optional[int]) -> Tuple[str, Optional[int]]:
        """检测方言和目标数据库"""
        if target_dialect:
            return target_dialect, target_database_id
        if self.dialect_detector:
            return self.dialect_detector.detect_from_query(user_query)
        return self.config.get('sql_dialect', 'mysql'), None

    def _switch_dialect_if_needed(self, dialect: str):
        """按需切换方言（避免重建整个 SQLChain）"""
        if dialect and dialect != self.sql_chain.dialect:
            logger.info(f"切换方言：{self.sql_chain.dialect} -> {dialect}")
            self.sql_chain.dialect = dialect
            from ..chain.prompts import SQLPromptTemplates
            self.sql_chain.prompt_templates = SQLPromptTemplates(dialect=dialect)

    def _get_few_shot_examples(self, query: str) -> str:
        """获取 Few-Shot 示例"""
        if self.knowledge_retriever.history_index is None:
            return ""
        return self.knowledge_retriever.get_few_shot_examples_formatted(
            query, top_k=self.config.get('few_shot_top_k', 3)
        )

    def _retrieve_knowledge(self, query: str, database_id: Optional[int]) -> List:
        """检索知识库"""
        results = self.knowledge_retriever.retrieve_all(
            query, top_k=self.sql_chain.retrieval_top_k, database_id=database_id
        )
        logger.info(f"检索到 {len(results)} 个相关知识节点")
        return results

    def _save_history_and_few_shot(self, user_query: str, result: Dict,
                                     retrieval_query: str, session_id: Optional[str]):
        """保存历史记录并更新 Few-Shot 索引"""
        if not self.mysql_client or not result.get('sql'):
            return
        history_id = self.mysql_client.save_sql_history(
            user_query=user_query,
            generated_sql=result['sql'],
            session_id=session_id,
            matched_knowledge=result.get('matched_knowledge'),
            retrieval_query=retrieval_query,
            validation_status=result.get('status', 'pending'),
            validation_error=result.get('error')
        )
        if history_id and result.get('status') == 'success':
            self.knowledge_retriever.add_history_record({
                'id': history_id,
                'user_query': user_query,
                'generated_sql': result['sql'],
                'execution_status': 'success',
                'validation_status': 'success'
            })

    # ==================== 公共方法 ====================

    def generate_sql(self, user_query: str, session_id: Optional[str] = None,
                    auto_save_history: bool = True,
                    use_few_shot: bool = True,
                    target_dialect: Optional[str] = None,
                    target_database_id: Optional[int] = None) -> Dict[str, Any]:
        """生成 SQL 的主入口（同步）"""
        logger.info(f"开始生成 SQL, query={user_query}")

        # 1. 检测方言和目标数据库
        detected_dialect, detected_database_id = self._detect_dialect_and_db(
            user_query, target_dialect, target_database_id
        )
        logger.info(f"方言检测结果：dialect={detected_dialect}, database_id={detected_database_id}")

        # 2. 查询改写
        final_query = asyncio.run(self._prepare_query(user_query, use_async=False))
        if self.rewrite_chain:
            logger.info(f"查询改写：{user_query} -> {final_query}")

        # 3. 检索知识库
        knowledge_results = self._retrieve_knowledge(final_query, detected_database_id)

        # 4. Few-Shot 示例
        few_shot_examples = self._get_few_shot_examples(final_query) if use_few_shot else ""

        # 5. 获取 Schema 信息
        schema_info = self._get_schema_info(database_id=detected_database_id)

        # 6. 方言切换
        self._switch_dialect_if_needed(detected_dialect)

        # 7. 生成 SQL
        result = self.sql_chain.generate(
            user_query=final_query,
            schema_info=schema_info,
            knowledge_results=knowledge_results,
            session_id=session_id,
            few_shot_examples=few_shot_examples
        )

        # 8. 保存历史
        self._save_history_and_few_shot(user_query, result, final_query, session_id)

        return result

    async def generate_sql_async(self, user_query: str, session_id: Optional[str] = None,
                                 auto_save_history: bool = True,
                                 use_few_shot: bool = True,
                                 target_dialect: Optional[str] = None,
                                 target_database_id: Optional[int] = None) -> Dict[str, Any]:
        """生成 SQL 的异步入口"""
        logger.info(f"开始异步生成 SQL, query={user_query}")

        # 1. 检测方言和目标数据库
        detected_dialect, detected_database_id = self._detect_dialect_and_db(
            user_query, target_dialect, target_database_id
        )
        logger.info(f"方言检测结果：dialect={detected_dialect}, database_id={detected_database_id}")

        # 2. 查询改写（异步）
        final_query = await self._prepare_query(user_query, use_async=True)
        if self.rewrite_chain:
            logger.info(f"查询改写：{user_query} -> {final_query}")

        # 3. 检索知识库
        knowledge_results = self._retrieve_knowledge(final_query, detected_database_id)

        # 4. Few-Shot 示例
        few_shot_examples = self._get_few_shot_examples(final_query) if use_few_shot else ""

        # 5. 获取 Schema 信息
        schema_info = self._get_schema_info(database_id=detected_database_id)

        # 6. 方言切换
        self._switch_dialect_if_needed(detected_dialect)

        # 7. 异步生成 SQL
        result = await self.sql_chain.generate_async(
            user_query=final_query,
            schema_info=schema_info,
            knowledge_results=knowledge_results,
            session_id=session_id,
            few_shot_examples=few_shot_examples
        )

        # 8. 保存历史
        self._save_history_and_few_shot(user_query, result, final_query, session_id)

        return result

    def _get_schema_info(self, database_id: Optional[int] = None) -> Dict:
        """获取 Schema 信息"""
        if self.neo4j_client:
            return self._get_schema_from_neo4j(database_id)
        if self.mysql_client:
            return self._get_schema_from_mysql(database_id)
        return {'tables': [], 'columns': [], 'relationships': []}

    def _get_schema_from_neo4j(self, database_id: Optional[int] = None) -> Dict:
        """从 Neo4j 获取完整 Schema（表 + 列 + 关系）"""
        try:
            if database_id is not None:
                tables_result = self.neo4j_client.execute_query("""
                    MATCH (d:Database {database_id: $database_id})-[:HAS_TABLE]->(t:Table)
                    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                    RETURN t.name as table_name, t.description as table_comment,
                           collect({
                               column_name: c.name,
                               column_type: c.type,
                               column_comment: c.description,
                               is_primary_key: c.is_primary_key,
                               is_nullable: c.is_nullable,
                               table_name: t.name
                           }) as columns
                """, {"database_id": database_id})

                tables = []
                all_columns = []
                for row in tables_result:
                    tables.append({
                        'table_name': row['table_name'],
                        'table_comment': row.get('table_comment', '')
                    })
                    for col in row.get('columns', []):
                        if col.get('column_name'):
                            all_columns.append(dict(col))

                # 获取关系（CONNECTS 类型，带权重信息）
                rel_result = self.neo4j_client.execute_query("""
                    MATCH (t1:Table)-[r:CONNECTS]->(t2:Table)
                    WHERE t1.database_id = $database_id OR t2.database_id = $database_id
                    RETURN t1.name as from_table, t2.name as to_table,
                           r.relationship_type as type, r.join_type as join_type,
                           r.join_sql as join_sql, r.from_column as from_column,
                           r.to_column as to_column, r.weight as weight,
                           r.extra_condition as extra_condition
                """, {"database_id": database_id})
            else:
                tables_result = self.neo4j_client.execute_query("""
                    MATCH (t:Table)
                    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                    RETURN t.name as table_name, t.description as table_comment,
                           collect({
                               column_name: c.name,
                               column_type: c.type,
                               column_comment: c.description,
                               is_primary_key: c.is_primary_key,
                               is_nullable: c.is_nullable,
                               table_name: t.name
                           }) as columns
                """)

                tables = []
                all_columns = []
                for row in tables_result:
                    tables.append({
                        'table_name': row['table_name'],
                        'table_comment': row.get('table_comment', '')
                    })
                    for col in row.get('columns', []):
                        if col.get('column_name'):
                            all_columns.append(dict(col))

                rel_result = self.neo4j_client.execute_query("""
                    MATCH (t1:Table)-[r:CONNECTS]->(t2:Table)
                    RETURN t1.name as from_table, t2.name as to_table,
                           r.relationship_type as type, r.join_type as join_type,
                           r.join_sql as join_sql, r.from_column as from_column,
                           r.to_column as to_column, r.weight as weight,
                           r.extra_condition as extra_condition
                """)

            return {
                'tables': tables,
                'columns': all_columns,
                'relationships': rel_result or []
            }
        except Exception as e:
            logger.warning(f"从 Neo4j 获取 Schema 失败：{e}，回退到 MySQL")
            return self._get_schema_from_mysql(database_id)

    def _get_schema_from_mysql(self, database_id: Optional[int] = None) -> Dict:
        """从 MySQL 获取 Schema 信息"""
        if not self.mysql_client:
            return {'tables': [], 'columns': [], 'relationships': []}

        tables = self.mysql_client.get_table_schema()
        if database_id is not None and self.neo4j_client:
            try:
                result = self.neo4j_client.execute_query("""
                    MATCH (d:Database {database_id: $database_id})-[:HAS_TABLE]->(t:Table)
                    RETURN t.name as table_name
                """, {"database_id": database_id})
                table_names = [r["table_name"] for r in result]
                tables = [t for t in tables if t['table_name'] in table_names]
            except Exception as e:
                logger.warning(f"从 Neo4j 过滤表失败：{e}")

        all_columns = []
        for table in tables:
            columns = self.mysql_client.get_column_schema(table['table_name'])
            all_columns.extend(columns)

        return {'tables': tables, 'columns': all_columns, 'relationships': []}

    # ==================== 其他公共方法 ====================

    def execute_sql(self, sql: str, history_id: Optional[int] = None) -> Dict[str, Any]:
        """执行 SQL"""
        if not self.mysql_client:
            return {'status': 'error', 'error': 'MySQL 客户端未连接', 'data': None}

        logger.info(f"执行 SQL: {sql[:100]}...")

        try:
            if not self._is_safe_query(sql):
                return {'status': 'error', 'error': '只允许执行 SELECT 查询', 'data': None}

            results = self.mysql_client.execute_query(sql)

            if history_id and self.mysql_client:
                self.mysql_client.update_sql_execution(
                    history_id=history_id,
                    execution_status='success',
                    execution_result={'row_count': len(results), 'data_preview': results[:5]}
                )

            return {
                'status': 'success', 'error': None,
                'data': results, 'row_count': len(results)
            }
        except Exception as e:
            logger.error(f"SQL 执行失败：{e}")
            if history_id and self.mysql_client:
                self.mysql_client.update_sql_execution(
                    history_id=history_id, execution_status='fail', execution_error=str(e)
                )
            return {'status': 'fail', 'error': str(e), 'data': None}

    def _is_safe_query(self, sql: str) -> bool:
        """检查 SQL 是否安全（只允许 SELECT）"""
        sql_upper = sql.strip().upper()
        unsafe_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        for keyword in unsafe_keywords:
            if sql_upper.startswith(keyword) or f" {keyword} " in sql_upper:
                return False
        return True

    def fix_and_retry(self, history_id: int, error: str) -> Dict[str, Any]:
        """修复 SQL 并重试"""
        history = self.mysql_client.get_history_detail(history_id)
        if not history:
            return {'status': 'error', 'error': '历史记录不存在'}

        original_sql = history.get('generated_sql', '')
        schema_info = self._get_schema_info()

        fix_result = self.sql_chain.fix_sql(sql=original_sql, error=error, schema_info=schema_info)
        if fix_result['status'] != 'success':
            return fix_result

        self.mysql_client.execute_update(
            "UPDATE sql_generation_history SET generated_sql = %s, validation_status = 'pending' WHERE id = %s",
            (fix_result['sql'], history_id)
        )
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
        """检测用户查询的目标数据库方言"""
        if self.dialect_detector:
            return self.dialect_detector.detect_from_query(user_query)
        return self.config.get('sql_dialect', 'mysql'), None


def create_sql_generator(llm, neo4j_client, mysql_client, config: Optional[Dict] = None,
                         embedding_factory: Optional = None) -> SQLGenerator:
    """创建 SQL 生成器工厂函数"""
    config = config or {}

    # 1. 从 Neo4j 同步知识到检索器
    knowledge_nodes = neo4j_client.get_schema_metadata() if neo4j_client else {'tables': [], 'columns': []}

    # 2. 构建检索器
    knowledge_retriever = KnowledgeRetriever(config=config, embedding_factory=embedding_factory)

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

    # 6. 构建 Few-Shot 索引
    if mysql_client and config.get('few_shot_enabled', True):
        try:
            history_records = mysql_client.get_history_list(limit=100)
            knowledge_retriever.build_few_shot_index(history_records)
            logger.info(f"Few-Shot 索引构建完成，共 {len(knowledge_retriever.history_records)} 条成功记录")
        except Exception as e:
            logger.warning(f"构建 Few-Shot 索引失败：{e}")

    return generator
