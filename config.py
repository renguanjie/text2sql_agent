"""
项目配置文件
包含数据库配置、Neo4j 配置、LangChain 配置等
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from core.config_validation import (
    parse_env_bool,
    parse_env_float,
    parse_env_int,
    validate_runtime_config,
)

# 提供商目录定义在 core/provider_catalog.py，此处重新导出以保持既有导入路径
# （`from config import LLM_PROVIDERS`）；core/config_validation.py 也依赖它
# 来校验"提供商是否需要 API Key"。
from core.provider_catalog import EMBEDDING_PROVIDERS, LLM_PROVIDERS

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# ==================== MySQL 配置 ====================
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = parse_env_int("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "text2sql_db")

# MySQL 连接池配置
MYSQL_POOL_SIZE = parse_env_int("MYSQL_POOL_SIZE", 10)  # 连接池大小
MYSQL_MAX_OVERFLOW = parse_env_int("MYSQL_MAX_OVERFLOW", 20)  # 最大溢出连接数
MYSQL_POOL_TIMEOUT = parse_env_int("MYSQL_POOL_TIMEOUT", 30)  # 获取连接超时时间（秒）

# ==================== Neo4j 配置 ====================
NEO4J_ENABLED = parse_env_bool("NEO4J_ENABLED", False)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j@123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Neo4j 连接池配置
NEO4J_MAX_POOL_SIZE = parse_env_int("NEO4J_MAX_POOL_SIZE", 50)  # 最大连接池大小
NEO4J_CONNECTION_TIMEOUT = parse_env_int("NEO4J_CONNECTION_TIMEOUT", 30)  # 连接超时时间（秒）

# ==================== LangChain 配置 ====================
# LLM 模型配置 (支持 OpenAI、本地模型等)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:e4b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # 可选，用于本地模型或代理

# 提示词模板配置
PROMPT_MAX_LENGTH = parse_env_int("PROMPT_MAX_LENGTH", 4000)
RETRIEVAL_TOP_K = parse_env_int("RETRIEVAL_TOP_K", 5)
UPLOAD_MAX_BYTES = parse_env_int("UPLOAD_MAX_BYTES", 5 * 1024 * 1024)
GRAPH_MAX_NODES = parse_env_int("GRAPH_MAX_NODES", 10_000)
GRAPH_MAX_EDGES = parse_env_int("GRAPH_MAX_EDGES", 20_000)
GRAPH_MAX_DEPTH = parse_env_int("GRAPH_MAX_DEPTH", 20)
GRAPH_MAX_STRING_LENGTH = parse_env_int("GRAPH_MAX_STRING_LENGTH", 10_000)

# ==================== 检索配置 ====================
# BM25 + FAISS 两路混合检索配置
BM25_K1 = parse_env_float("BM25_K1", 1.5)
BM25_B = parse_env_float("BM25_B", 0.75)

# 两路混合检索权重配置 (总和应为 1)
BM25_WEIGHT = parse_env_float("BM25_WEIGHT", 0.6)
DENSE_WEIGHT = parse_env_float("DENSE_WEIGHT", 0.4)

# FAISS 向量检索配置
FAISS_ENABLED = parse_env_bool("FAISS_ENABLED", True)
FAISS_EMBEDDING_DIM = parse_env_int("FAISS_EMBEDDING_DIM", 1024)  # bge-m3 dense 维度
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "flat")  # flat, ivf, hnsw
INDEX_CACHE_ENABLED = parse_env_bool("INDEX_CACHE_ENABLED", False)
INDEX_CACHE_DIR = os.getenv("INDEX_CACHE_DIR", str(PROJECT_ROOT / "data" / "index_cache"))

# ==================== Embedding 配置 ====================
# 通义千问 Embedding 模型配置
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")  # 通义千问 API Key
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "")  # 可选 API Base URL

# ==================== Few-Shot 配置 ====================
# 动态 Few-Shot 示例检索配置
FEW_SHOT_ENABLED = parse_env_bool("FEW_SHOT_ENABLED", True)
FEW_SHOT_TOP_K = parse_env_int("FEW_SHOT_TOP_K", 3)  # 检索示例数量
FEW_SHOT_MIN_SIMILARITY = parse_env_float("FEW_SHOT_MIN_SIMILARITY", 0.5)  # 最小相似度阈值

# ==================== SQL 配置 ====================
# SQLFluff 配置
SQL_DIALECT = os.getenv("SQL_DIALECT", "mysql")
SQL_MAX_LENGTH = parse_env_int("SQL_MAX_LENGTH", 2000)
SQL_VALIDATION_ENABLED = parse_env_bool("SQL_VALIDATION_ENABLED", True)
SQL_QUERY_MAX_ROWS = parse_env_int("SQL_QUERY_MAX_ROWS", 500)
SQL_QUERY_TIMEOUT_MS = parse_env_int("SQL_QUERY_TIMEOUT_MS", 10_000)
SQL_SCOPE_ENFORCEMENT = parse_env_bool("SQL_SCOPE_ENFORCEMENT", True)
SQL_BLOCKED_TABLES = [
    value.strip() for value in os.getenv(
        "SQL_BLOCKED_TABLES",
        "sql_generation_history,user_feedback,table_schema,column_schema",
    ).split(',') if value.strip()
]

# 语义治理与执行预检（默认关闭以保持旧调用行为）
SEMANTIC_GOVERNANCE_ENABLED = parse_env_bool(
    "SEMANTIC_GOVERNANCE_ENABLED", False
)
SEMANTIC_MODEL_DIR = os.getenv("SEMANTIC_MODEL_DIR", "semantic_models")
SEMANTIC_DATASOURCE = os.getenv("SEMANTIC_DATASOURCE", "default")
SEMANTIC_CONTEXT_MAX_CHARS = parse_env_int("SEMANTIC_CONTEXT_MAX_CHARS", 2400)
SEMANTIC_CONTEXT_MAX_ITEMS = parse_env_int("SEMANTIC_CONTEXT_MAX_ITEMS", 24)
SQL_REPAIR_MAX_ATTEMPTS = parse_env_int("SQL_REPAIR_MAX_ATTEMPTS", 1)
SQL_EXPLAIN_ENABLED = parse_env_bool("SQL_EXPLAIN_ENABLED", False)
SQL_EXPLAIN_TIMEOUT_MS = parse_env_int("SQL_EXPLAIN_TIMEOUT_MS", 3000)
SQL_EXPLAIN_MAX_ESTIMATED_ROWS = parse_env_int(
    "SQL_EXPLAIN_MAX_ESTIMATED_ROWS", 0
)
MAX_PREPARED_REQUESTS = parse_env_int("MAX_PREPARED_REQUESTS", 256)
PREPARED_REQUEST_TTL_SECONDS = parse_env_float(
    "PREPARED_REQUEST_TTL_SECONDS", 120.0
)
GOVERNED_HISTORY_BINDING_TTL_SECONDS = parse_env_float(
    "GOVERNED_HISTORY_BINDING_TTL_SECONDS", 900.0
)

# ==================== 日志配置 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"

# ==================== 应用配置 ====================
STREAMLIT_SERVER_PORT = parse_env_int("STREAMLIT_SERVER_PORT", 8501)
STREAMLIT_HEAD = parse_env_bool("STREAMLIT_HEAD", True)

# ==================== 历史配置 ====================
# 历史记录保留条数
HISTORY_MAX_RECORDS = parse_env_int("HISTORY_MAX_RECORDS", 1000)
# 自动保存历史记录
HISTORY_AUTO_SAVE = parse_env_bool("HISTORY_AUTO_SAVE", True)


# ==================== 配置验证 ====================
def validate_config():
    """验证配置参数的有效性，在模块加载时执行"""
    validate_runtime_config(
        {
            "neo4j_enabled": NEO4J_ENABLED,
            "faiss_enabled": FAISS_ENABLED,
            "index_cache_enabled": INDEX_CACHE_ENABLED,
            "few_shot_enabled": FEW_SHOT_ENABLED,
            "sql_validation_enabled": SQL_VALIDATION_ENABLED,
            "sql_scope_enforcement": SQL_SCOPE_ENFORCEMENT,
            "semantic_governance_enabled": SEMANTIC_GOVERNANCE_ENABLED,
            "sql_explain_enabled": SQL_EXPLAIN_ENABLED,
            "streamlit_head": STREAMLIT_HEAD,
            "history_auto_save": HISTORY_AUTO_SAVE,
            "history_max_records": HISTORY_MAX_RECORDS,
            "sql_dialect": SQL_DIALECT,
            "llm_provider": LLM_PROVIDER,
            "llm_api_key": LLM_API_KEY,
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_api_key": EMBEDDING_API_KEY,
            "mysql_port": MYSQL_PORT,
            "mysql_pool_size": MYSQL_POOL_SIZE,
            "mysql_max_overflow": MYSQL_MAX_OVERFLOW,
            "mysql_pool_timeout": MYSQL_POOL_TIMEOUT,
            "neo4j_max_pool_size": NEO4J_MAX_POOL_SIZE,
            "neo4j_connection_timeout": NEO4J_CONNECTION_TIMEOUT,
            "prompt_max_length": PROMPT_MAX_LENGTH,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "upload_max_bytes": UPLOAD_MAX_BYTES,
            "graph_max_nodes": GRAPH_MAX_NODES,
            "graph_max_edges": GRAPH_MAX_EDGES,
            "graph_max_depth": GRAPH_MAX_DEPTH,
            "graph_max_string_length": GRAPH_MAX_STRING_LENGTH,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "bm25_weight": BM25_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
            "faiss_embedding_dim": FAISS_EMBEDDING_DIM,
            "few_shot_top_k": FEW_SHOT_TOP_K,
            "few_shot_min_similarity": FEW_SHOT_MIN_SIMILARITY,
            "sql_max_length": SQL_MAX_LENGTH,
            "sql_query_max_rows": SQL_QUERY_MAX_ROWS,
            "sql_query_timeout_ms": SQL_QUERY_TIMEOUT_MS,
            "semantic_context_max_chars": SEMANTIC_CONTEXT_MAX_CHARS,
            "semantic_context_max_items": SEMANTIC_CONTEXT_MAX_ITEMS,
            "sql_repair_max_attempts": SQL_REPAIR_MAX_ATTEMPTS,
            "sql_explain_timeout_ms": SQL_EXPLAIN_TIMEOUT_MS,
            "sql_explain_max_estimated_rows": SQL_EXPLAIN_MAX_ESTIMATED_ROWS,
            "max_prepared_requests": MAX_PREPARED_REQUESTS,
            "prepared_request_ttl_seconds": PREPARED_REQUEST_TTL_SECONDS,
            "governed_history_binding_ttl_seconds": (
                GOVERNED_HISTORY_BINDING_TTL_SECONDS
            ),
            "streamlit_server_port": STREAMLIT_SERVER_PORT,
        }
    )
    errors = []

    # 验证检索权重总和
    if abs(BM25_WEIGHT + DENSE_WEIGHT - 1.0) > 0.01:
        errors.append(f"检索权重总和必须为 1.0，当前：BM25_WEIGHT={BM25_WEIGHT}, DENSE_WEIGHT={DENSE_WEIGHT}")

    # 验证 LLM 提供商
    if LLM_PROVIDER not in LLM_PROVIDERS:
        errors.append(f"未知的 LLM 提供商：{LLM_PROVIDER}，支持的提供商：{list(LLM_PROVIDERS.keys())}")

    # 验证 Embedding 提供商
    if EMBEDDING_PROVIDER not in EMBEDDING_PROVIDERS:
        errors.append(f"未知的 Embedding 提供商：{EMBEDDING_PROVIDER}，支持的提供商：{list(EMBEDDING_PROVIDERS.keys())}")

    # 「提供商需要 API Key」的校验已上移到 validate_runtime_config，
    # 这样启动和 reload_llm/reload_embedding 热重载走的是同一条规则。
    # 上面两条「未知提供商」检查刻意只保留在启动路径：热重载对未知 provider
    # 是容错回退，且其原子性测试依赖该语义。

    if errors:
        from loguru import logger
        logger.error("配置验证失败：")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError(f"配置验证失败，共 {len(errors)} 个错误")


# 在模块加载时执行配置验证；无效安全配置必须阻止应用启动。
validate_config()
