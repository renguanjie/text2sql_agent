"""
项目配置文件
包含数据库配置、Neo4j 配置、LangChain 配置等
"""
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

from core.config_validation import parse_env_bool, validate_runtime_config

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# ==================== 模型目录 ====================

# 支持的 LLM 提供商及推荐模型
LLM_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-4o",
        "needs_api_key": True,
        "needs_base_url": False,
        "default_base_url": "",
    },
    "ollama": {
        "label": "Ollama (本地)",
        "default_model": "gemma3:e4b",
        "needs_api_key": False,
        "needs_base_url": True,
        "default_base_url": "http://localhost:11434",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "default_model": "claude-sonnet-4-20250514",
        "needs_api_key": True,
        "needs_base_url": False,
        "default_base_url": "",
    },
    "dashscope": {
        "label": "DashScope (通义千问)",
        "default_model": "qwen-max",
        "needs_api_key": True,
        "needs_base_url": True,
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "deepseek": {
        "label": "DeepSeek (深度求索)",
        "default_model": "deepseek-chat",
        "needs_api_key": True,
        "needs_base_url": True,
        "default_base_url": "https://api.deepseek.com/v1",
    },
    "zhipu": {
        "label": "智谱 AI",
        "default_model": "glm-4-flash",
        "needs_api_key": True,
        "needs_base_url": True,
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    "azure": {
        "label": "Azure OpenAI",
        "default_model": "gpt-4o",
        "needs_api_key": True,
        "needs_base_url": True,
        "default_base_url": "",
    },
}

# 支持的 Embedding 模型提供商
EMBEDDING_PROVIDERS = {
    "dashscope": {
        "label": "DashScope (通义千问)",
        "default_model": "text-embedding-v3",
        "dimensions": 1024,
        "needs_api_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "text-embedding-3-large",
        "dimensions": 3072,
        "needs_api_key": True,
    },
    "ollama": {
        "label": "Ollama (本地)",
        "default_model": "bge-m3:latest",
        "dimensions": 1024,
        "needs_api_key": False,
    },
}

# ==================== MySQL 配置 ====================
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "text2sql_db")

# MySQL 连接池配置
MYSQL_POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", "10"))  # 连接池大小
MYSQL_MAX_OVERFLOW = int(os.getenv("MYSQL_MAX_OVERFLOW", "20"))  # 最大溢出连接数
MYSQL_POOL_TIMEOUT = int(os.getenv("MYSQL_POOL_TIMEOUT", "30"))  # 获取连接超时时间（秒）

# MySQL 连接 URL
MYSQL_URI = (
    f"mysql+mysqlconnector://{quote_plus(MYSQL_USER)}:{quote_plus(MYSQL_PASSWORD)}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)

# ==================== Neo4j 配置 ====================
NEO4J_ENABLED = parse_env_bool("NEO4J_ENABLED", False)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j@123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Neo4j 连接池配置
NEO4J_MAX_POOL_SIZE = int(os.getenv("NEO4J_MAX_POOL_SIZE", "50"))  # 最大连接池大小
NEO4J_CONNECTION_TIMEOUT = int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30"))  # 连接超时时间（秒）

# ==================== LangChain 配置 ====================
# LLM 模型配置 (支持 OpenAI、本地模型等)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:e4b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # 可选，用于本地模型或代理

# 提示词模板配置
PROMPT_MAX_LENGTH = int(os.getenv("PROMPT_MAX_LENGTH", 4000))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 5))
UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(5 * 1024 * 1024)))
GRAPH_MAX_NODES = int(os.getenv("GRAPH_MAX_NODES", "10000"))
GRAPH_MAX_EDGES = int(os.getenv("GRAPH_MAX_EDGES", "20000"))
GRAPH_MAX_DEPTH = int(os.getenv("GRAPH_MAX_DEPTH", "20"))
GRAPH_MAX_STRING_LENGTH = int(os.getenv("GRAPH_MAX_STRING_LENGTH", "10000"))

# ==================== 检索配置 ====================
# BM25 + FAISS 两路混合检索配置
BM25_K1 = float(os.getenv("BM25_K1", 1.5))
BM25_B = float(os.getenv("BM25_B", 0.75))

# 两路混合检索权重配置 (总和应为 1)
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.6))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.4))

# FAISS 向量检索配置
FAISS_ENABLED = parse_env_bool("FAISS_ENABLED", True)
FAISS_EMBEDDING_DIM = int(os.getenv("FAISS_EMBEDDING_DIM", "1024"))  # bge-m3 dense 维度
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
FEW_SHOT_TOP_K = int(os.getenv("FEW_SHOT_TOP_K", "3"))  # 检索示例数量
FEW_SHOT_MIN_SIMILARITY = float(os.getenv("FEW_SHOT_MIN_SIMILARITY", "0.5"))  # 最小相似度阈值

# ==================== SQL 配置 ====================
# SQLFluff 配置
SQL_DIALECT = os.getenv("SQL_DIALECT", "mysql")
SQL_MAX_LENGTH = int(os.getenv("SQL_MAX_LENGTH", 2000))
SQL_VALIDATION_ENABLED = parse_env_bool("SQL_VALIDATION_ENABLED", True)
SQL_QUERY_MAX_ROWS = int(os.getenv("SQL_QUERY_MAX_ROWS", "500"))
SQL_QUERY_TIMEOUT_MS = int(os.getenv("SQL_QUERY_TIMEOUT_MS", "10000"))
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
SEMANTIC_CONTEXT_MAX_CHARS = int(os.getenv("SEMANTIC_CONTEXT_MAX_CHARS", "2400"))
SEMANTIC_CONTEXT_MAX_ITEMS = int(os.getenv("SEMANTIC_CONTEXT_MAX_ITEMS", "24"))
SQL_REPAIR_MAX_ATTEMPTS = int(os.getenv("SQL_REPAIR_MAX_ATTEMPTS", "1"))
SQL_EXPLAIN_ENABLED = parse_env_bool("SQL_EXPLAIN_ENABLED", False)
SQL_EXPLAIN_TIMEOUT_MS = int(os.getenv("SQL_EXPLAIN_TIMEOUT_MS", "3000"))
SQL_EXPLAIN_MAX_ESTIMATED_ROWS = int(
    os.getenv("SQL_EXPLAIN_MAX_ESTIMATED_ROWS", "0")
)

# ==================== 日志配置 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"

# ==================== 应用配置 ====================
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_HEAD = parse_env_bool("STREAMLIT_HEAD", True)

# ==================== 历史配置 ====================
# 历史记录保留条数
HISTORY_MAX_RECORDS = int(os.getenv("HISTORY_MAX_RECORDS", 1000))
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
            "sql_dialect": SQL_DIALECT,
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

    # 验证需要 API Key 的提供商
    if LLM_PROVIDERS.get(LLM_PROVIDER, {}).get("needs_api_key") and not LLM_API_KEY:
        errors.append(f"LLM 提供商 {LLM_PROVIDER} 需要配置 LLM_API_KEY")

    if EMBEDDING_PROVIDERS.get(EMBEDDING_PROVIDER, {}).get("needs_api_key") and not EMBEDDING_API_KEY:
        errors.append(f"Embedding 提供商 {EMBEDDING_PROVIDER} 需要配置 EMBEDDING_API_KEY")

    # 验证端口号范围
    if not (1 <= MYSQL_PORT <= 65535):
        errors.append(f"MySQL 端口号无效：{MYSQL_PORT}，应在 1-65535 范围内")

    # 验证连接池参数
    if MYSQL_POOL_SIZE <= 0:
        errors.append(f"MySQL 连接池大小必须大于 0：{MYSQL_POOL_SIZE}")
    if MYSQL_MAX_OVERFLOW < 0:
        errors.append(f"MySQL 最大溢出连接数不能为负：{MYSQL_MAX_OVERFLOW}")
    if NEO4J_MAX_POOL_SIZE <= 0:
        errors.append(f"Neo4j 连接池大小必须大于 0：{NEO4J_MAX_POOL_SIZE}")

    # 验证检索参数
    if RETRIEVAL_TOP_K <= 0:
        errors.append(f"RETRIEVAL_TOP_K 必须大于 0：{RETRIEVAL_TOP_K}")
    if FEW_SHOT_TOP_K <= 0:
        errors.append(f"FEW_SHOT_TOP_K 必须大于 0：{FEW_SHOT_TOP_K}")
    if not (0 <= FEW_SHOT_MIN_SIMILARITY <= 1):
        errors.append(f"FEW_SHOT_MIN_SIMILARITY 应在 0-1 范围内：{FEW_SHOT_MIN_SIMILARITY}")
    if SQL_QUERY_MAX_ROWS <= 0:
        errors.append(f"SQL_QUERY_MAX_ROWS 必须大于 0：{SQL_QUERY_MAX_ROWS}")
    if SQL_QUERY_TIMEOUT_MS <= 0:
        errors.append(f"SQL_QUERY_TIMEOUT_MS 必须大于 0：{SQL_QUERY_TIMEOUT_MS}")
    if SQL_MAX_LENGTH <= 0:
        errors.append(f"SQL_MAX_LENGTH 必须大于 0：{SQL_MAX_LENGTH}")
    if SQL_REPAIR_MAX_ATTEMPTS not in (0, 1):
        errors.append(
            "SQL_REPAIR_MAX_ATTEMPTS 必须是 0 或 1："
            f"{SQL_REPAIR_MAX_ATTEMPTS}"
        )
    if SEMANTIC_CONTEXT_MAX_CHARS <= 0:
        errors.append(
            "SEMANTIC_CONTEXT_MAX_CHARS 必须大于 0："
            f"{SEMANTIC_CONTEXT_MAX_CHARS}"
        )
    if SEMANTIC_CONTEXT_MAX_ITEMS <= 0:
        errors.append(
            "SEMANTIC_CONTEXT_MAX_ITEMS 必须大于 0："
            f"{SEMANTIC_CONTEXT_MAX_ITEMS}"
        )
    if SQL_EXPLAIN_TIMEOUT_MS <= 0:
        errors.append(
            f"SQL_EXPLAIN_TIMEOUT_MS 必须大于 0：{SQL_EXPLAIN_TIMEOUT_MS}"
        )
    if SQL_EXPLAIN_MAX_ESTIMATED_ROWS < 0:
        errors.append(
            "SQL_EXPLAIN_MAX_ESTIMATED_ROWS 不能为负："
            f"{SQL_EXPLAIN_MAX_ESTIMATED_ROWS}"
        )
    if PROMPT_MAX_LENGTH < 1000:
        errors.append(f"PROMPT_MAX_LENGTH 不应小于 1000：{PROMPT_MAX_LENGTH}")
    if UPLOAD_MAX_BYTES <= 0 or GRAPH_MAX_NODES <= 0 or GRAPH_MAX_EDGES <= 0:
        errors.append("上传大小、图节点数和图关系数限制必须大于 0")

    # 验证 BM25 参数
    if BM25_K1 <= 0:
        errors.append(f"BM25_K1 必须大于 0：{BM25_K1}")
    if not (0 <= BM25_B <= 1):
        errors.append(f"BM25_B 应在 0-1 范围内：{BM25_B}")

    if errors:
        from loguru import logger
        logger.error("配置验证失败：")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError(f"配置验证失败，共 {len(errors)} 个错误")


# 在模块加载时执行配置验证；无效安全配置必须阻止应用启动。
validate_config()
