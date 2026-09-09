"""
应用上下文 - 统一管理所有组件的生命周期
替代原有的全局单例模式
"""
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from core.config_validation import RuntimeConfigurationError, validate_runtime_config
from core.utils.log_sanitizer import log_safe_exception

_CURRENT_GOVERNANCE_PIPELINE = object()


class ApplicationConfigurationError(ValueError):
    """Stable public error raised for invalid application configuration."""


class ApplicationContext:
    """应用上下文 - 统一管理 LLM、数据库、检索器等组件"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化应用上下文

        Args:
            config: 配置字典（从 config.py 读取）
        """
        self.config = config
        self._initialized = False
        self._initialization_error = None

        # 组件实例（延迟初始化）
        self._llm = None
        self._neo4j_client = None
        self._mysql_client = None
        self._embedding_factory = None
        self._knowledge_retriever = None
        self._sql_generator = None
        self._sql_chain = None
        self._semantic_registry = None
        self._correctness_pipeline = None

    @property
    def llm(self):
        return self._llm

    @property
    def neo4j_client(self):
        return self._neo4j_client

    @property
    def mysql_client(self):
        return self._mysql_client

    @property
    def embedding_factory(self):
        return self._embedding_factory

    @property
    def knowledge_retriever(self):
        return self._knowledge_retriever

    @property
    def sql_generator(self):
        return self._sql_generator

    @property
    def sql_chain(self):
        return self._sql_chain

    @property
    def semantic_registry(self):
        return self._semantic_registry

    @property
    def correctness_pipeline(self):
        return self._correctness_pipeline

    @property
    def initialization_error(self):
        return self._initialization_error

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def ensure_initialized(self) -> bool:
        """确保所有组件已初始化"""
        if self._initialized:
            return True

        logger.info("开始初始化系统组件...")

        try:
            self._validate_candidate_config(self.config)
            self._init_llm()
            self._init_neo4j()
            self._init_mysql()
            self._init_embedding()
            self._init_retriever()
            self._init_sql_chain()
            self._init_governance()
            self._init_sql_generator()
            self._load_knowledge_base()
            self._load_few_shot_history()

            self._initialized = True
            self._initialization_error = None
            logger.info("系统初始化完成")
            return True

        except ApplicationConfigurationError as exc:
            self._initialization_error = str(exc)
            logger.error("系统初始化失败：{}", self._initialization_error)
            self.shutdown()
            self._initialized = False
            return False
        except Exception as e:
            self._initialization_error = "系统初始化失败"
            log_safe_exception("application_startup", e)
            self.shutdown()
            self._initialized = False
            return False

    def _init_llm(self):
        """初始化 LLM"""
        from core.llm_factory import create_llm
        self._llm = create_llm(
            provider=self.config.get('llm_provider', 'openai'),
            model=self.config.get('llm_model', 'gpt-4'),
            api_key=self.config.get('llm_api_key', ''),
            base_url=self.config.get('llm_base_url', '')
        )
        logger.info(f"LLM 初始化成功：{self.config.get('llm_provider')}/{self.config.get('llm_model')}")

    def _init_neo4j(self):
        """初始化 Neo4j 客户端"""
        from core.knowledge.neo4j_client import Neo4jClient
        self._neo4j_client = Neo4jClient(
            uri=self.config.get('neo4j_uri', 'bolt://localhost:7687'),
            user=self.config.get('neo4j_user', 'neo4j'),
            password=self.config.get('neo4j_password', 'neo4j@123'),
            database=self.config.get('neo4j_database', 'neo4j'),
            max_connection_pool_size=self.config.get('neo4j_max_pool_size', 50),
            connection_timeout=self.config.get('neo4j_connection_timeout', 30),
            graph_max_nodes=self.config.get('graph_max_nodes', 10000),
            graph_max_edges=self.config.get('graph_max_edges', 20000),
            graph_max_depth=self.config.get('graph_max_depth', 20),
            graph_max_string_length=self.config.get('graph_max_string_length', 10000),
        )
        neo4j_enabled = self.config.get('neo4j_enabled', False)
        if neo4j_enabled:
            if not self._neo4j_client.connect():
                logger.warning("Neo4j 连接失败，将使用 MySQL 元数据模式")
        else:
            logger.info("Neo4j 未启用，使用 MySQL 元数据模式")

    def _init_mysql(self):
        """初始化 MySQL 客户端"""
        from core.history.mysql_client import MySQLClient
        self._mysql_client = MySQLClient(
            host=self.config.get('mysql_host', 'localhost'),
            port=self.config.get('mysql_port', 3306),
            user=self.config.get('mysql_user', 'root'),
            password=self.config.get('mysql_password', 'root'),
            database=self.config.get('mysql_database', 'text2sql_db'),
            pool_size=self.config.get('mysql_pool_size', 10),
            max_overflow=self.config.get('mysql_max_overflow', 20),
            pool_timeout=self.config.get('mysql_pool_timeout', 30),
            history_max_records=self.config.get('history_max_records', 1000),
        )
        if not self._mysql_client.connect():
            logger.warning("MySQL 连接失败，历史记录功能可能不可用")

    def _init_embedding(self):
        """初始化 Embedding Factory"""
        from core.embedding_factory import EmbeddingFactory
        self._embedding_factory = EmbeddingFactory(
            provider=self.config.get('embedding_provider', 'dashscope'),
            model=self.config.get('embedding_model', 'text-embedding-v1'),
            api_key=self.config.get('embedding_api_key', ''),
            api_base=self.config.get('embedding_api_base', '')
        )
        logger.info(f"Embedding 工厂初始化：{self.config.get('embedding_provider')}/{self.config.get('embedding_model')}")

    def _init_retriever(self):
        """初始化知识检索器"""
        from core.retrieval.bm25_tfidf import KnowledgeRetriever
        self._knowledge_retriever = KnowledgeRetriever(
            config=self._retriever_config(self.config),
            embedding_factory=self._embedding_factory
        )

    @staticmethod
    def _retriever_config(config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'bm25_k1': config.get('bm25_k1', 1.5),
            'bm25_b': config.get('bm25_b', 0.75),
            'bm25_weight': config.get('bm25_weight', 0.6),
            'dense_weight': config.get('dense_weight', 0.4),
            'embedding_dim': config.get('faiss_embedding_dim', 1536),
            'index_type': config.get('faiss_index_type', 'flat')
            if config.get('faiss_enabled', True) else None,
            'few_shot_enabled': config.get('few_shot_enabled', True),
            'few_shot_top_k': config.get('few_shot_top_k', 3),
            'few_shot_min_similarity': config.get('few_shot_min_similarity', 0.5),
        }

    @staticmethod
    def _generator_config(config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'enable_rewrite': False,
            'few_shot_enabled': config.get('few_shot_enabled', True),
            'few_shot_top_k': config.get('few_shot_top_k', 3),
            'sql_dialect': config.get('sql_dialect', 'mysql'),
            'sql_validation_enabled': config.get('sql_validation_enabled', True),
            'sql_max_length': config.get('sql_max_length', 2000),
            'sql_query_max_rows': config.get('sql_query_max_rows', 500),
            'sql_query_timeout_ms': config.get('sql_query_timeout_ms', 10000),
            'sql_scope_enforcement': config.get('sql_scope_enforcement', True),
            'sql_blocked_tables': config.get('sql_blocked_tables'),
            'prompt_max_length': config.get('prompt_max_length', 4000),
            'history_max_records': config.get('history_max_records', 1000),
            'governed_history_binding_ttl_seconds': config.get(
                'governed_history_binding_ttl_seconds', 900
            ),
        }

    def _init_sql_chain(self):
        """初始化 SQL 生成链"""
        from core.chain.sql_chain import SQLGenerationChain
        self._sql_chain = SQLGenerationChain(
            llm=self._llm,
            dialect=self.config.get('sql_dialect', 'mysql'),
            retrieval_top_k=self.config.get('retrieval_top_k', 5),
            prompt_max_length=self.config.get('prompt_max_length', 4000),
        )

    @staticmethod
    def _resolve_semantic_model_dir(config: Dict[str, Any]) -> Path:
        model_dir = Path(config.get('semantic_model_dir', 'semantic_models'))
        if model_dir.is_absolute():
            return model_dir
        return Path(__file__).resolve().parents[1] / model_dir

    @staticmethod
    def _validate_candidate_config(config: Dict[str, Any]) -> None:
        try:
            validate_runtime_config(config)
        except RuntimeConfigurationError as exc:
            raise ApplicationConfigurationError(str(exc)) from None

    def _build_governance_components(self, config: Dict[str, Any]):
        """Compile and validate a complete candidate pair before publication."""
        self._validate_candidate_config(config)
        if not config.get('semantic_governance_enabled', False):
            return None, None

        from core.semantic.compiler import SemanticCompiler
        from core.semantic.context import SemanticContextBuilder
        from core.semantic.registry import SemanticRegistry
        from core.semantic.validator import SemanticSQLValidator
        from core.sql.correctness import CorrectnessPipeline
        from core.sql.preflight import MySQLExplainPreflight

        try:
            model_dir = self._resolve_semantic_model_dir(config)
            datasource = config.get('semantic_datasource', 'default')
            source = model_dir / f"{datasource}.yaml"
            compiled_root = model_dir / '.compiled'
            SemanticCompiler().compile_file(source, compiled_root)
            registry = SemanticRegistry(compiled_root, datasource)
            registry.load_active()
            context_builder = SemanticContextBuilder(
                max_chars=config.get('semantic_context_max_chars', 2400),
                max_items=config.get('semantic_context_max_items', 24),
            )
            preflight = None
            if config.get('sql_explain_enabled', False):
                preflight = MySQLExplainPreflight(
                    self._mysql_client,
                    timeout_ms=config.get('sql_explain_timeout_ms', 3000),
                    max_estimated_rows=config.get(
                        'sql_explain_max_estimated_rows', 0
                    ),
                )
            pipeline = CorrectnessPipeline(
                registry=registry,
                context_builder=context_builder,
                semantic_validator=SemanticSQLValidator(),
                preflight=preflight,
                max_repair_attempts=config.get('sql_repair_max_attempts', 1),
                blocked_tables=tuple(config.get('sql_blocked_tables') or ()),
                max_sql_length=config.get('sql_max_length', 2000),
            )
            return registry, pipeline
        except Exception as exc:
            log_safe_exception("semantic_governance_config", exc)
            raise ApplicationConfigurationError("语义治理配置无效") from None

    def _init_governance(self) -> None:
        """Initialize the optional governed snapshot and pipeline."""
        registry, pipeline = self._build_governance_components(self.config)
        self._semantic_registry = registry
        self._correctness_pipeline = pipeline

    def _build_sql_generator(
        self,
        *,
        llm,
        knowledge_retriever,
        sql_chain,
        config: Dict[str, Any],
        correctness_pipeline,
    ):
        from core.sql.generator import SQLGenerator

        return SQLGenerator(
            llm=llm,
            knowledge_retriever=knowledge_retriever,
            sql_chain=sql_chain,
            mysql_client=self._mysql_client,
            config=self._generator_config(config),
            neo4j_client=(
                self._neo4j_client
                if self._neo4j_client and self._neo4j_client.driver else None
            ),
            correctness_pipeline=correctness_pipeline,
        )

    def _init_sql_generator(self):
        """初始化 SQL 生成器"""
        self._sql_generator = self._build_sql_generator(
            llm=self._llm,
            knowledge_retriever=self._knowledge_retriever,
            sql_chain=self._sql_chain,
            config=self.config,
            correctness_pipeline=self._correctness_pipeline,
        )

    def reload_llm(self, **updates: Any) -> None:
        """为当前会话事务式重载 LLM 链路。"""
        from core.chain.sql_chain import SQLGenerationChain
        from core.llm_factory import create_llm

        candidate_config = {**self.config, **updates}
        self._validate_candidate_config(candidate_config)
        candidate_llm = create_llm(
            provider=candidate_config.get('llm_provider', 'openai'),
            model=candidate_config.get('llm_model', 'gpt-4'),
            api_key=candidate_config.get('llm_api_key', ''),
            base_url=candidate_config.get('llm_base_url', ''),
        )
        candidate_chain = SQLGenerationChain(
            llm=candidate_llm,
            dialect=candidate_config.get('sql_dialect', 'mysql'),
            retrieval_top_k=candidate_config.get('retrieval_top_k', 5),
            prompt_max_length=candidate_config.get('prompt_max_length', 4000),
        )
        candidate_generator = self._build_sql_generator(
            llm=candidate_llm,
            knowledge_retriever=self._knowledge_retriever,
            sql_chain=candidate_chain,
            config=candidate_config,
            correctness_pipeline=self._correctness_pipeline,
        )
        self.config = candidate_config
        self._llm = candidate_llm
        self._sql_chain = candidate_chain
        self._sql_generator = candidate_generator
        self._initialized = True

    def reload_embedding(self, **updates: Any) -> None:
        """为当前会话构建新检索链路，成功后一次性替换旧组件。"""
        from core.embedding_factory import EmbeddingFactory
        from core.retrieval.bm25_tfidf import KnowledgeRetriever

        candidate_config = {**self.config, **updates}
        self._validate_candidate_config(candidate_config)
        candidate_embedding = EmbeddingFactory(
            provider=candidate_config.get('embedding_provider', 'dashscope'),
            model=candidate_config.get('embedding_model', 'text-embedding-v1'),
            api_key=candidate_config.get('embedding_api_key', ''),
            api_base=candidate_config.get('embedding_api_base', ''),
        )
        try:
            candidate_retriever = KnowledgeRetriever(
                config=self._retriever_config(candidate_config),
                embedding_factory=candidate_embedding,
            )
            self._populate_retriever(candidate_retriever, candidate_config)
            candidate_generator = self._build_sql_generator(
                llm=self._llm,
                knowledge_retriever=candidate_retriever,
                sql_chain=self._sql_chain,
                config=candidate_config,
                correctness_pipeline=self._correctness_pipeline,
            )
        except Exception:
            try:
                candidate_embedding.clear_cache()
            except Exception as cleanup_error:
                logger.warning(f"候选 Embedding 缓存清理失败：{cleanup_error}")
            raise

        old_embedding = self._embedding_factory
        self.config = candidate_config
        self._embedding_factory = candidate_embedding
        self._knowledge_retriever = candidate_retriever
        self._sql_generator = candidate_generator
        self._initialized = True
        if old_embedding:
            try:
                old_embedding.clear_cache()
            except Exception as cleanup_error:
                logger.warning(f"旧 Embedding 缓存清理失败：{cleanup_error}")

    def reload_governance(self, **updates: Any) -> None:
        """Candidate-build governance and atomically replace registry/generator."""
        from core.retrieval.bm25_tfidf import KnowledgeRetriever

        candidate_config = {**self.config, **updates}
        candidate_registry, candidate_pipeline = self._build_governance_components(
            candidate_config
        )
        candidate_retriever = KnowledgeRetriever(
            config=self._retriever_config(candidate_config),
            embedding_factory=self._embedding_factory,
        )
        self._populate_retriever(
            candidate_retriever,
            candidate_config,
            correctness_pipeline=candidate_pipeline,
        )
        candidate_generator = self._build_sql_generator(
            llm=self._llm,
            knowledge_retriever=candidate_retriever,
            sql_chain=self._sql_chain,
            config=candidate_config,
            correctness_pipeline=candidate_pipeline,
        )

        self.config = candidate_config
        self._semantic_registry = candidate_registry
        self._correctness_pipeline = candidate_pipeline
        self._knowledge_retriever = candidate_retriever
        self._sql_generator = candidate_generator
        self._initialized = True

    def _populate_retriever(
        self,
        retriever,
        config: Dict[str, Any],
        *,
        correctness_pipeline=_CURRENT_GOVERNANCE_PIPELINE,
    ) -> None:
        """把当前知识与执行成功历史加载到候选检索器。"""
        if correctness_pipeline is _CURRENT_GOVERNANCE_PIPELINE:
            correctness_pipeline = self._correctness_pipeline
        documents = []
        metadata_list = []
        if config.get('neo4j_enabled', False) and getattr(self._neo4j_client, 'driver', None):
            metadata = self._neo4j_client.get_schema_metadata()
            for table in metadata.get('tables', []):
                name = table.get('name', '')
                doc = f"{name} {table.get('description', '')}".strip()
                if doc:
                    documents.append(doc)
                    metadata_list.append({
                        'node_type': 'table', 'name': name,
                        'description': table.get('description', ''),
                    })
        elif self._mysql_client and self._mysql_client.engine:
            for table in self._mysql_client.get_table_schema():
                name = table.get('table_name', '')
                doc = f"{name} {table.get('table_comment', '')}".strip()
                if doc:
                    documents.append(doc)
                    metadata_list.append({
                        'node_type': 'table', 'name': name,
                        'description': table.get('table_comment', ''),
                    })
        if documents:
            retriever.retriever.index_documents(documents, metadata_list)
        if (
            config.get('few_shot_enabled', True)
            and self._mysql_client
            and self._mysql_client.engine
        ):
            identity, retrieval_fields = self._governed_history_retrieval_fields(
                correctness_pipeline
            )
            records = self._mysql_client.get_successful_history_for_few_shot(
                limit=100,
                **retrieval_fields,
            )
            if identity is None:
                retriever.build_few_shot_index(records)
            else:
                retriever.build_few_shot_index(
                    records,
                    governance_identity=identity,
                )

    @staticmethod
    def _governed_history_retrieval_fields(correctness_pipeline):
        if correctness_pipeline is None:
            return None, {}
        from core.history.identity import GovernedHistoryIdentity

        identity = GovernedHistoryIdentity.from_snapshot(
            correctness_pipeline.registry.get_snapshot()
        )
        return identity, identity.retrieval_fields()

    def _load_knowledge_base(self):
        """加载知识库到检索器；磁盘索引缓存需显式开启以避免读到旧图谱。"""
        index_cache_enabled = self.config.get('index_cache_enabled', False)
        index_dir = self.config.get('index_cache_dir', 'data/index_cache')
        if index_cache_enabled:
            try:
                if self._knowledge_retriever.load_index(index_dir):
                    logger.info(f"知识库索引从缓存加载成功：{index_dir}")
                    return
            except Exception as e:
                logger.warning(f"知识库索引缓存加载失败，将从数据源重建：{e}")

        # 缓存未命中，从数据源重建
        try:
            neo4j_enabled = self.config.get('neo4j_enabled', False)
            if neo4j_enabled and self._neo4j_client and self._neo4j_client.driver:
                self._load_knowledge_from_neo4j()
            else:
                self._load_knowledge_from_mysql()
            if index_cache_enabled:
                try:
                    self._knowledge_retriever.save_index(index_dir)
                except Exception as e:
                    logger.warning(f"知识库索引缓存保存失败：{e}")
        except Exception as e:
            logger.warning(f"知识库加载失败：{e}")

    def _load_knowledge_from_neo4j(self):
        """从 Neo4j 加载知识库"""
        metadata = self._neo4j_client.get_schema_metadata()
        documents = []
        metadata_list = []
        for table in metadata.get('tables', []):
            doc = f"{table.get('name', '')} {table.get('description', '')}"
            if doc.strip():
                documents.append(doc)
                metadata_list.append({
                    'node_type': 'table',
                    'name': table.get('name', ''),
                    'description': table.get('description', '')
                })
        if documents:
            self._knowledge_retriever.retriever.index_documents(documents, metadata_list)
            logger.info(f"知识库索引完成：{len(documents)} 个文档 (Neo4j)")

    def _load_knowledge_from_mysql(self):
        """从 MySQL 加载元数据"""
        tables = self._mysql_client.get_table_schema()
        documents = []
        metadata_list = []
        for table in tables:
            doc = f"{table.get('table_name', '')} {table.get('table_comment', '')}"
            if doc.strip():
                documents.append(doc)
                metadata_list.append({
                    'node_type': 'table',
                    'name': table.get('table_name', ''),
                    'description': table.get('table_comment', '')
                })
        if documents:
            self._knowledge_retriever.retriever.index_documents(documents, metadata_list)
            logger.info(f"知识库索引完成：{len(documents)} 个文档 (MySQL)")

    def _load_few_shot_history(self):
        """从已验证或执行成功的历史记录构建 Few-Shot 索引。"""
        if not self.config.get('few_shot_enabled', True):
            return
        if not self._mysql_client or not self._mysql_client.engine:
            return
        try:
            identity, retrieval_fields = self._governed_history_retrieval_fields(
                self._correctness_pipeline
            )
            records = self._mysql_client.get_successful_history_for_few_shot(
                limit=100,
                **retrieval_fields,
            )
            if identity is None:
                self._knowledge_retriever.build_few_shot_index(records)
            else:
                self._knowledge_retriever.build_few_shot_index(
                    records,
                    governance_identity=identity,
                )
        except Exception as e:
            logger.warning(f"Few-Shot 索引加载失败：{e}")

    def reset(self):
        """重置所有组件状态，先关闭现有连接以避免泄漏"""
        # 先关闭现有连接
        self.shutdown()
        # 再清空所有组件引用
        self._initialized = False
        self._llm = None
        self._neo4j_client = None
        self._mysql_client = None
        self._embedding_factory = None
        self._knowledge_retriever = None
        self._sql_generator = None
        self._sql_chain = None
        self._semantic_registry = None
        self._correctness_pipeline = None
        self._initialization_error = None
        logger.info("应用上下文已重置")

    def shutdown(self):
        """关闭所有连接，释放缓存资源"""
        if self._neo4j_client:
            self._neo4j_client.close()
        if self._mysql_client:
            self._mysql_client.close()
        # 释放 Embedding 缓存
        if self._embedding_factory:
            try:
                self._embedding_factory.clear_cache()
            except Exception as e:
                logger.warning(f"Embedding 缓存释放失败：{e}")
        logger.info("应用上下文已关闭")


def create_app_context_from_env() -> ApplicationContext:
    """从环境变量创建应用上下文（便捷函数）"""
    import config
    return ApplicationContext(config={
        # MySQL
        'mysql_host': config.MYSQL_HOST,
        'mysql_port': config.MYSQL_PORT,
        'mysql_user': config.MYSQL_USER,
        'mysql_password': config.MYSQL_PASSWORD,
        'mysql_database': config.MYSQL_DATABASE,
        'mysql_pool_size': config.MYSQL_POOL_SIZE,
        'mysql_max_overflow': config.MYSQL_MAX_OVERFLOW,
        'mysql_pool_timeout': config.MYSQL_POOL_TIMEOUT,
        'history_max_records': config.HISTORY_MAX_RECORDS,
        # Neo4j
        'neo4j_enabled': config.NEO4J_ENABLED,
        'neo4j_uri': config.NEO4J_URI,
        'neo4j_user': config.NEO4J_USER,
        'neo4j_password': config.NEO4J_PASSWORD,
        'neo4j_database': config.NEO4J_DATABASE,
        'neo4j_max_pool_size': config.NEO4J_MAX_POOL_SIZE,
        'neo4j_connection_timeout': config.NEO4J_CONNECTION_TIMEOUT,
        'graph_max_nodes': config.GRAPH_MAX_NODES,
        'graph_max_edges': config.GRAPH_MAX_EDGES,
        'graph_max_depth': config.GRAPH_MAX_DEPTH,
        'graph_max_string_length': config.GRAPH_MAX_STRING_LENGTH,
        # LLM
        'llm_provider': config.LLM_PROVIDER,
        'llm_model': config.LLM_MODEL,
        'llm_api_key': config.LLM_API_KEY,
        'llm_base_url': config.LLM_BASE_URL,
        # Retrieval
        'retrieval_top_k': config.RETRIEVAL_TOP_K,
        'prompt_max_length': config.PROMPT_MAX_LENGTH,
        'bm25_k1': config.BM25_K1,
        'bm25_b': config.BM25_B,
        'bm25_weight': config.BM25_WEIGHT,
        'dense_weight': config.DENSE_WEIGHT,
        # FAISS
        'faiss_enabled': config.FAISS_ENABLED,
        'faiss_embedding_dim': config.FAISS_EMBEDDING_DIM,
        'faiss_index_type': config.FAISS_INDEX_TYPE,
        'index_cache_enabled': config.INDEX_CACHE_ENABLED,
        'index_cache_dir': config.INDEX_CACHE_DIR,
        # Embedding
        'embedding_provider': config.EMBEDDING_PROVIDER,
        'embedding_model': config.EMBEDDING_MODEL,
        'embedding_api_key': config.EMBEDDING_API_KEY,
        'embedding_api_base': config.EMBEDDING_API_BASE,
        # Few-Shot
        'few_shot_enabled': config.FEW_SHOT_ENABLED,
        'few_shot_top_k': config.FEW_SHOT_TOP_K,
        'few_shot_min_similarity': config.FEW_SHOT_MIN_SIMILARITY,
        # SQL
        'sql_dialect': config.SQL_DIALECT,
        'sql_validation_enabled': config.SQL_VALIDATION_ENABLED,
        'sql_max_length': config.SQL_MAX_LENGTH,
        'sql_query_max_rows': config.SQL_QUERY_MAX_ROWS,
        'sql_query_timeout_ms': config.SQL_QUERY_TIMEOUT_MS,
        'sql_scope_enforcement': config.SQL_SCOPE_ENFORCEMENT,
        'sql_blocked_tables': config.SQL_BLOCKED_TABLES,
        # Semantic governance and EXPLAIN
        'semantic_governance_enabled': config.SEMANTIC_GOVERNANCE_ENABLED,
        'semantic_model_dir': config.SEMANTIC_MODEL_DIR,
        'semantic_datasource': config.SEMANTIC_DATASOURCE,
        'semantic_context_max_chars': config.SEMANTIC_CONTEXT_MAX_CHARS,
        'semantic_context_max_items': config.SEMANTIC_CONTEXT_MAX_ITEMS,
        'sql_repair_max_attempts': config.SQL_REPAIR_MAX_ATTEMPTS,
        'sql_explain_enabled': config.SQL_EXPLAIN_ENABLED,
        'sql_explain_timeout_ms': config.SQL_EXPLAIN_TIMEOUT_MS,
        'sql_explain_max_estimated_rows': config.SQL_EXPLAIN_MAX_ESTIMATED_ROWS,
    })
