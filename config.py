"""
项目配置文件
包含数据库配置、Neo4j 配置、LangChain 配置等
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

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
MYSQL_URI = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

# ==================== Neo4j 配置 ====================
NEO4J_ENABLED = os.getenv("NEO4J_ENABLED", "false").lower() == "true"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j@123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Neo4j 连接池配置
NEO4J_MAX_POOL_SIZE = int(os.getenv("NEO4J_MAX_POOL_SIZE", "50"))  # 最大连接池大小
NEO4J_CONNECTION_TIMEOUT = int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30"))  # 连接超时时间（秒）

# ==================== LangChain 配置 ====================
# LLM 模型配置 (支持 OpenAI、本地模型等)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # 可选，用于本地模型或代理

# 提示词模板配置
PROMPT_MAX_LENGTH = int(os.getenv("PROMPT_MAX_LENGTH", 4000))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 5))

# ==================== 检索配置 ====================
# BM25 + TF-IDF + FAISS 三路混合检索配置
BM25_K1 = float(os.getenv("BM25_K1", 1.5))
BM25_B = float(os.getenv("BM25_B", 0.75))
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", 5000))

# 三路混合检索权重配置 (总和应为 1)
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.4))
TFIDF_WEIGHT = float(os.getenv("TFIDF_WEIGHT", 0.3))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.3))

# FAISS 向量检索配置
FAISS_ENABLED = os.getenv("FAISS_ENABLED", "true").lower() == "true"
FAISS_EMBEDDING_DIM = int(os.getenv("FAISS_EMBEDDING_DIM", "1536"))  # 通义千问 v1 维度
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "flat")  # flat, ivf, hnsw

# ==================== Embedding 配置 ====================
# 通义千问 Embedding 模型配置
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "dashscope")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")  # 通义千问 API Key
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "")  # 可选 API Base URL

# ==================== Few-Shot 配置 ====================
# 动态 Few-Shot 示例检索配置
FEW_SHOT_ENABLED = os.getenv("FEW_SHOT_ENABLED", "true").lower() == "true"
FEW_SHOT_TOP_K = int(os.getenv("FEW_SHOT_TOP_K", "3"))  # 检索示例数量
FEW_SHOT_MIN_SIMILARITY = float(os.getenv("FEW_SHOT_MIN_SIMILARITY", "0.5"))  # 最小相似度阈值

# ==================== SQL 配置 ====================
# SQLFluff 配置
SQL_DIALECT = os.getenv("SQL_DIALECT", "mysql")
SQL_MAX_LENGTH = int(os.getenv("SQL_MAX_LENGTH", 2000))
SQL_VALIDATION_ENABLED = os.getenv("SQL_VALIDATION_ENABLED", "true").lower() == "true"

# ==================== 日志配置 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"

# ==================== 应用配置 ====================
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_HEAD = os.getenv("STREAMLIT_HEAD", "true").lower() == "true"

# ==================== 历史配置 ====================
# 历史记录保留条数
HISTORY_MAX_RECORDS = int(os.getenv("HISTORY_MAX_RECORDS", 1000))
# 自动保存历史记录
HISTORY_AUTO_SAVE = os.getenv("HISTORY_AUTO_SAVE", "true").lower() == "true"
