"""
应用上下文 - 统一管理所有组件的生命周期
替代原有的全局单例模式
"""
from typing import Optional, Dict, Any
from loguru import logger


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

        # 组件实例（延迟初始化）
        self._llm = None
        self._neo4j_client = None
        self._mysql_client = None
        self._embedding_factory = None
        self._knowledge_retriever = None
        self._sql_generator = None
        self._sql_chain = None

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
    def is_initialized(self) -> bool:
        return self._initialized

    def ensure_initialized(self) -> bool:
        """确保所有组件已初始化"""
        if self._initialized:
            return True

        logger.info("开始初始化系统组件...")

        try:
            self._init_llm()
            self._init_neo4j()
            self._init_mysql()
            self._init_embedding()
            self._init_retriever()
            self._init_sql_chain()
            self._init_sql_generator()
            self._load_knowledge_base()

            self._initialized = True
            logger.info("系统初始化完成")
            return True

        except Exception as e:
            logger.error(f"系统初始化失败：{e}")
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
            max_connection_pool_size=50,
            connection_timeout=30
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
            pool_timeout=self.config.get('mysql_pool_timeout', 30)
        )
        if not self._mysql_client.connect():
            logger.warning("MySQL 连接失败，历史记录功能可能不可用")

    def _init_embedding(self):
        """初始化 Embedding Factory"""
        from core.embedding_factory import EmbeddingFactory
        self._embedding_factory = EmbeddingFactory(
            provider=self.config.get('embedding_provider', 'dashscope'),
            model=self.config.get('embedding_model', 'text-embedding-v1'),
            api_key=self.config.get('embedding_api_key', '')
        )
        logger.info(f"Embedding 工厂初始化：{self.config.get('embedding_provider')}/{self.config.get('embedding_model')}")

    def _init_retriever(self):
        """初始化知识检索器"""
        from core.retrieval.bm25_tfidf import KnowledgeRetriever
        self._knowledge_retriever = KnowledgeRetriever(
            config={
                'bm25_k1': self.config.get('bm25_k1', 1.5),
                'bm25_b': self.config.get('bm25_b', 0.75),
                'bm25_weight': self.config.get('bm25_weight', 0.6),
                'dense_weight': self.config.get('dense_weight', 0.4),
                'embedding_dim': self.config.get('faiss_embedding_dim', 1536),
                'index_type': self.config.get('faiss_index_type', 'flat') if self.config.get('faiss_enabled', True) else None,
                'few_shot_enabled': self.config.get('few_shot_enabled', True),
                'few_shot_top_k': self.config.get('few_shot_top_k', 3)
            },
            embedding_factory=self._embedding_factory
        )

    def _init_sql_chain(self):
        """初始化 SQL 生成链"""
        from core.chain.sql_chain import SQLGenerationChain
        self._sql_chain = SQLGenerationChain(
            llm=self._llm,
            dialect=self.config.get('sql_dialect', 'mysql'),
            retrieval_top_k=self.config.get('retrieval_top_k', 5)
        )

    def _init_sql_generator(self):
        """初始化 SQL 生成器"""
        from core.sql.generator import SQLGenerator
        self._sql_generator = SQLGenerator(
            llm=self._llm,
            knowledge_retriever=self._knowledge_retriever,
            sql_chain=self._sql_chain,
            mysql_client=self._mysql_client,
            config={
                'enable_rewrite': False,
                'few_shot_enabled': self.config.get('few_shot_enabled', True),
                'few_shot_top_k': self.config.get('few_shot_top_k', 3)
            },
            neo4j_client=self._neo4j_client
        )

    def _load_knowledge_base(self):
        """加载知识库到检索器"""
        try:
            neo4j_enabled = self.config.get('neo4j_enabled', False)
            if neo4j_enabled and self._neo4j_client and self._neo4j_client.driver:
                self._load_knowledge_from_neo4j()
            else:
                self._load_knowledge_from_mysql()
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

    def reset(self):
        """重置所有组件状态（不关闭连接）"""
        self._initialized = False
        self._llm = None
        self._neo4j_client = None
        self._mysql_client = None
        self._embedding_factory = None
        self._knowledge_retriever = None
        self._sql_generator = None
        self._sql_chain = None
        logger.info("应用上下文已重置")

    def shutdown(self):
        """关闭所有连接"""
        if self._neo4j_client:
            self._neo4j_client.close()
        if self._mysql_client:
            self._mysql_client.close()
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
        # Neo4j
        'neo4j_enabled': config.NEO4J_ENABLED,
        'neo4j_uri': config.NEO4J_URI,
        'neo4j_user': config.NEO4J_USER,
        'neo4j_password': config.NEO4J_PASSWORD,
        'neo4j_database': config.NEO4J_DATABASE,
        # LLM
        'llm_provider': config.LLM_PROVIDER,
        'llm_model': config.LLM_MODEL,
        'llm_api_key': config.LLM_API_KEY,
        'llm_base_url': config.LLM_BASE_URL,
        # Retrieval
        'retrieval_top_k': config.RETRIEVAL_TOP_K,
        'bm25_k1': config.BM25_K1,
        'bm25_b': config.BM25_B,
        'bm25_weight': config.BM25_WEIGHT,
        'dense_weight': config.DENSE_WEIGHT,
        # FAISS
        'faiss_enabled': config.FAISS_ENABLED,
        'faiss_embedding_dim': config.FAISS_EMBEDDING_DIM,
        'faiss_index_type': config.FAISS_INDEX_TYPE,
        # Embedding
        'embedding_provider': config.EMBEDDING_PROVIDER,
        'embedding_model': config.EMBEDDING_MODEL,
        'embedding_api_key': config.EMBEDDING_API_KEY,
        'embedding_api_base': config.EMBEDDING_API_BASE,
        # Few-Shot
        'few_shot_enabled': config.FEW_SHOT_ENABLED,
        'few_shot_top_k': config.FEW_SHOT_TOP_K,
        # SQL
        'sql_dialect': config.SQL_DIALECT,
    })
