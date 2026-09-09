"""
Text2SQL 智能体 - Streamlit 主入口
基于本地知识库的 SQL 智能生成应用
"""
# ruff: noqa: E402
import json
import sys
from pathlib import Path

import streamlit as st

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import uuid

from loguru import logger

from config import (
    BM25_B,
    BM25_K1,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    EMBEDDING_API_BASE,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_PROVIDERS,
    FAISS_EMBEDDING_DIM,
    FAISS_ENABLED,
    FAISS_INDEX_TYPE,
    FEW_SHOT_ENABLED,
    FEW_SHOT_MIN_SIMILARITY,
    FEW_SHOT_TOP_K,
    GRAPH_MAX_DEPTH,
    GRAPH_MAX_EDGES,
    GRAPH_MAX_NODES,
    GRAPH_MAX_STRING_LENGTH,
    HISTORY_AUTO_SAVE,
    HISTORY_MAX_RECORDS,
    INDEX_CACHE_DIR,
    INDEX_CACHE_ENABLED,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_PROVIDERS,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_MAX_OVERFLOW,
    MYSQL_PASSWORD,
    MYSQL_POOL_SIZE,
    MYSQL_POOL_TIMEOUT,
    MYSQL_PORT,
    MYSQL_USER,
    NEO4J_CONNECTION_TIMEOUT,
    NEO4J_DATABASE,
    NEO4J_ENABLED,
    NEO4J_MAX_POOL_SIZE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PROMPT_MAX_LENGTH,
    RETRIEVAL_TOP_K,
    SEMANTIC_CONTEXT_MAX_CHARS,
    SEMANTIC_CONTEXT_MAX_ITEMS,
    SEMANTIC_DATASOURCE,
    SEMANTIC_GOVERNANCE_ENABLED,
    SEMANTIC_MODEL_DIR,
    SQL_BLOCKED_TABLES,
    SQL_DIALECT,
    SQL_EXPLAIN_ENABLED,
    SQL_EXPLAIN_MAX_ESTIMATED_ROWS,
    SQL_EXPLAIN_TIMEOUT_MS,
    SQL_MAX_LENGTH,
    SQL_QUERY_MAX_ROWS,
    SQL_QUERY_TIMEOUT_MS,
    SQL_REPAIR_MAX_ATTEMPTS,
    SQL_SCOPE_ENFORCEMENT,
    SQL_VALIDATION_ENABLED,
    UPLOAD_MAX_BYTES,
)
from core.app_context import ApplicationContext
from core.sql.validator import validate_sql
from core.utils.correctness_ui import (
    execution_is_disabled,
    public_failure_messages,
    structured_validation_messages,
)
from core.utils.log_sanitizer import (
    log_safe_exception,
    make_safe_streamlit_sink,
    safe_log_format,
)
from core.utils.upload_limits import UploadValidationError, read_uploaded_text


def get_knowledge_graph_template() -> dict:
    """返回知识图谱 JSON 导入模板。"""
    return {
        "version": "2.0",
        "format": "text2sql_knowledge_graph",
        "nodes": [
            {
                "label": "Database",
                "properties": {
                    "database_id": 1,
                    "name": "example_db",
                    "db_type": "mysql",
                    "db_language": "SQL",
                    "description": "示例数据库"
                }
            },
            {
                "label": "Table",
                "properties": {
                    "name": "customer",
                    "database": "example_db",
                    "database_id": 1,
                    "description": "客户表",
                    "is_view": False
                }
            },
            {
                "label": "Column",
                "properties": {
                    "name": "customer_id",
                    "table_name": "customer",
                    "database_id": 1,
                    "data_type": "BIGINT",
                    "description": "客户 ID",
                    "is_primary_key": True,
                    "is_nullable": False
                }
            }
        ],
        "edges": [
            {
                "type": "HAS_TABLE",
                "start": {
                    "label": "Database",
                    "properties": {"database_id": 1, "name": "example_db"}
                },
                "end": {
                    "label": "Table",
                    "properties": {"database_id": 1, "name": "customer"}
                },
                "properties": {}
            },
            {
                "type": "HAS_COLUMN",
                "start": {
                    "label": "Table",
                    "properties": {"database_id": 1, "name": "customer"}
                },
                "end": {
                    "label": "Column",
                    "properties": {"database_id": 1, "table_name": "customer", "name": "customer_id"}
                },
                "properties": {}
            }
        ],
        "databases": [
            {
                "database_id": 1,
                "name": "example_db",
                "db_type": "mysql",
                "db_language": "SQL",
                "description": "示例数据库"
            }
        ],
        "tables": [
            {
                "name": "customer",
                "database": "example_db",
                "database_id": 1,
                "description": "客户表",
                "is_view": False
            }
        ],
        "columns": [
            {
                "name": "customer_id",
                "table_name": "customer",
                "database_id": 1,
                "data_type": "BIGINT",
                "description": "客户 ID",
                "is_primary_key": True,
                "is_nullable": False
            }
        ],
        "relationships": [
            {
                "relationship_type": "CONNECTS",
                "from_table": "customer",
                "from_database_id": 1,
                "from_column": "customer_id",
                "to_table": "order",
                "to_database_id": 1,
                "to_column": "customer_id",
                "join_type": "LEFT JOIN",
                "weight": 1.0,
                "occurrence_count": 1,
                "source": "manual"
            }
        ],
        "concepts": [
            {
                "name": "客户域",
                "description": "客户相关数据",
                "tags": ["客户"],
                "related_tables": ["customer"]
            }
        ]
    }

# 配置日志
logger.remove()
logger.add(
    make_safe_streamlit_sink(st.write),
    level="INFO",
    format=safe_log_format,
)
logger.add(
    project_root / "logs" / "app.log",
    level="DEBUG",
    rotation="1 day",
    format=safe_log_format,
)

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="Text2SQL 智能体",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/renguanjie/text2sql-agent',
        'Report a bug': 'https://github.com/renguanjie/text2sql-agent/issues',
        'About': '# Text2SQL 智能体\n基于本地知识库的 SQL 智能生成系统\n\n核心技术：BM25+FAISS 检索 + LangChain + SQLFluff'
    }
)

# ==================== 自定义 CSS 样式 ====================
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* 副标题样式 */
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* 卡片样式 */
    .stCard {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 按钮样式优化 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* 代码块样式 */
    .stCode {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    /* 成功/错误提示优化 */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 会话状态初始化 ====================
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'app_context' not in st.session_state:
    st.session_state.app_context = None

if 'history' not in st.session_state:
    st.session_state.history = []

if 'last_sql' not in st.session_state:
    st.session_state.last_sql = None

if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# ==================== 初始化函数 ====================
def initialize_system():
    """初始化系统组件（委托给 ApplicationContext）"""
    logger.info("开始初始化系统...")

    app_ctx = ApplicationContext(config={
        'mysql_host': MYSQL_HOST, 'mysql_port': MYSQL_PORT,
        'mysql_user': MYSQL_USER, 'mysql_password': MYSQL_PASSWORD,
        'mysql_database': MYSQL_DATABASE,
        'mysql_pool_size': MYSQL_POOL_SIZE,
        'mysql_max_overflow': MYSQL_MAX_OVERFLOW,
        'mysql_pool_timeout': MYSQL_POOL_TIMEOUT,
        'history_max_records': HISTORY_MAX_RECORDS,
        'neo4j_enabled': NEO4J_ENABLED, 'neo4j_uri': NEO4J_URI,
        'neo4j_user': NEO4J_USER, 'neo4j_password': NEO4J_PASSWORD,
        'neo4j_database': NEO4J_DATABASE,
        'neo4j_max_pool_size': NEO4J_MAX_POOL_SIZE,
        'neo4j_connection_timeout': NEO4J_CONNECTION_TIMEOUT,
        'graph_max_nodes': GRAPH_MAX_NODES, 'graph_max_edges': GRAPH_MAX_EDGES,
        'graph_max_depth': GRAPH_MAX_DEPTH,
        'graph_max_string_length': GRAPH_MAX_STRING_LENGTH,
        'llm_provider': LLM_PROVIDER, 'llm_model': LLM_MODEL,
        'llm_api_key': LLM_API_KEY, 'llm_base_url': LLM_BASE_URL,
        'retrieval_top_k': RETRIEVAL_TOP_K, 'prompt_max_length': PROMPT_MAX_LENGTH,
        'sql_dialect': SQL_DIALECT,
        'sql_validation_enabled': SQL_VALIDATION_ENABLED, 'sql_max_length': SQL_MAX_LENGTH,
        'sql_query_max_rows': SQL_QUERY_MAX_ROWS,
        'sql_query_timeout_ms': SQL_QUERY_TIMEOUT_MS,
        'sql_scope_enforcement': SQL_SCOPE_ENFORCEMENT,
        'sql_blocked_tables': SQL_BLOCKED_TABLES,
        'semantic_governance_enabled': SEMANTIC_GOVERNANCE_ENABLED,
        'semantic_model_dir': SEMANTIC_MODEL_DIR,
        'semantic_datasource': SEMANTIC_DATASOURCE,
        'semantic_context_max_chars': SEMANTIC_CONTEXT_MAX_CHARS,
        'semantic_context_max_items': SEMANTIC_CONTEXT_MAX_ITEMS,
        'sql_repair_max_attempts': SQL_REPAIR_MAX_ATTEMPTS,
        'sql_explain_enabled': SQL_EXPLAIN_ENABLED,
        'sql_explain_timeout_ms': SQL_EXPLAIN_TIMEOUT_MS,
        'sql_explain_max_estimated_rows': SQL_EXPLAIN_MAX_ESTIMATED_ROWS,
        'bm25_k1': BM25_K1, 'bm25_b': BM25_B,
        'bm25_weight': BM25_WEIGHT, 'dense_weight': DENSE_WEIGHT,
        'faiss_enabled': FAISS_ENABLED, 'faiss_embedding_dim': FAISS_EMBEDDING_DIM,
        'faiss_index_type': FAISS_INDEX_TYPE,
        'index_cache_enabled': INDEX_CACHE_ENABLED, 'index_cache_dir': INDEX_CACHE_DIR,
        'embedding_provider': EMBEDDING_PROVIDER, 'embedding_model': EMBEDDING_MODEL,
        'embedding_api_key': EMBEDDING_API_KEY, 'embedding_api_base': EMBEDDING_API_BASE,
        'few_shot_enabled': FEW_SHOT_ENABLED, 'few_shot_top_k': FEW_SHOT_TOP_K,
        'few_shot_min_similarity': FEW_SHOT_MIN_SIMILARITY,
    })

    if app_ctx.ensure_initialized():
        st.session_state.app_context = app_ctx
        st.session_state.initialized = True
        logger.info("系统初始化完成")
        return True
    else:
        st.session_state.initialization_error = app_ctx.initialization_error
        logger.error("系统初始化失败")
        return None


# ==================== 模型配置辅助函数 ====================

def _apply_llm_config(provider: str, model: str, api_key: str, base_url: str) -> bool:
    """应用 LLM 配置变更，热重载 LLM 相关组件"""
    try:
        ctx = st.session_state.get("app_context")
        if not ctx:
            st.error("系统未初始化")
            return False

        ctx.reload_llm(
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key,
            llm_base_url=base_url,
        )

        logger.info(f"LLM 配置已更新：{provider}/{model}")
        return True
    except Exception as e:
        log_safe_exception("llm_config_reload", e)
        return False


def _apply_embedding_config(provider: str, model: str, api_key: str) -> bool:
    """应用 Embedding 配置变更，热重载 Embedding 和检索器"""
    try:
        ctx = st.session_state.get("app_context")
        if not ctx:
            st.error("系统未初始化")
            return False

        # 获取新 embedding 维度
        from config import EMBEDDING_PROVIDERS as _EP
        dim = _EP.get(provider, {}).get("dimensions", 1536)

        ctx.reload_embedding(
            embedding_provider=provider,
            embedding_model=model,
            embedding_api_key=api_key,
            faiss_embedding_dim=dim,
        )
        logger.info(f"Embedding 配置已更新：{provider}/{model}")
        return True
    except Exception as e:
        log_safe_exception("embedding_config_reload", e)
        return False


def _render_model_config_panel():
    """渲染模型配置 UI 面板（LLM + Embedding）"""
    st.markdown("### ️ 模型配置")
    st.caption("切换大语言模型或向量模型后点击保存，系统将自动重载相关组件。")

    # 读取当前配置（优先从 session_state 覆盖值，否则从环境变量）
    current_llm_provider = st.session_state.get("_llm_provider", LLM_PROVIDER)
    current_llm_model = st.session_state.get("_llm_model", LLM_MODEL)
    current_llm_api_key = st.session_state.get("_llm_api_key", LLM_API_KEY)
    current_llm_base_url = st.session_state.get("_llm_base_url", LLM_BASE_URL)

    current_emb_provider = st.session_state.get("_emb_provider", EMBEDDING_PROVIDER)
    current_emb_model = st.session_state.get("_emb_model", EMBEDDING_MODEL)
    current_emb_api_key = st.session_state.get("_emb_api_key", EMBEDDING_API_KEY)

    # ── LLM 配置 ──
    with st.expander("── 大语言模型 (LLM) ──", expanded=False):
        provider_labels = {k: v["label"] for k, v in LLM_PROVIDERS.items()}
        provider_keys = list(LLM_PROVIDERS.keys())
        provider_display = [provider_labels[k] for k in provider_keys]

        current_idx = provider_keys.index(current_llm_provider) if current_llm_provider in provider_keys else 0

        selected_label = st.selectbox(
            "提供商",
            options=provider_display,
            index=current_idx,
            key="cfg_llm_provider",
        )
        selected_provider = provider_keys[provider_display.index(selected_label)]
        provider_info = LLM_PROVIDERS[selected_provider]

        # 默认模型：如果用户手动填过就用用户的，否则用提供商推荐
        if selected_provider == current_llm_provider:
            default_model = current_llm_model
        else:
            default_model = provider_info["default_model"]

        st.text_input(
            "模型名称",
            value=default_model,
            key="cfg_llm_model",
            help="输入模型名称，如 gpt-4o、qwen-max、qwen2.5 等",
        )

        if provider_info["needs_api_key"]:
            st.text_input(
                "API Key",
                value=current_llm_api_key,
                type="password",
                key="cfg_llm_api_key",
                help="留空则使用 .env 中的配置",
            )
        else:
            st.caption("此提供商无需 API Key")
            st.session_state["cfg_llm_api_key"] = ""

        if provider_info["needs_base_url"]:
            st.text_input(
                "Base URL",
                value=current_llm_base_url or provider_info["default_base_url"],
                key="cfg_llm_base_url",
                help=f"默认：{provider_info['default_base_url'] or '（无）'}",
            )
        else:
            st.caption("使用官方默认 API 地址")
            st.session_state["cfg_llm_base_url"] = ""

        col_save_llm, col_reset_llm = st.columns([1, 1])
        with col_save_llm:
            if st.button(" 保存并重新加载 LLM", use_container_width=True, type="primary", key="btn_save_llm"):
                new_provider = selected_provider
                new_model = st.session_state.get("cfg_llm_model", provider_info["default_model"])
                new_api_key = st.session_state.get("cfg_llm_api_key", current_llm_api_key)
                new_base_url = st.session_state.get("cfg_llm_base_url", "")

                with st.spinner("正在重新加载 LLM..."):
                    ok = _apply_llm_config(new_provider, new_model, new_api_key, new_base_url)

                if ok:
                    # 持久化到 session_state，下次刷新时保持
                    st.session_state["_llm_provider"] = new_provider
                    st.session_state["_llm_model"] = new_model
                    st.session_state["_llm_api_key"] = new_api_key
                    st.session_state["_llm_base_url"] = new_base_url
                    st.success(f"✅ LLM 已切换为：{LLM_PROVIDERS[new_provider]['label']} / {new_model}")
                else:
                    st.error("LLM 加载失败，请检查配置")

        with col_reset_llm:
            if st.button("️ 恢复默认 LLM 配置", use_container_width=True, key="btn_reset_llm"):
                for k in ("_llm_provider", "_llm_model", "_llm_api_key", "_llm_base_url"):
                    st.session_state.pop(k, None)
                st.success("已恢复为 .env 中的默认配置，请刷新页面")

    st.markdown("")

    # ── Embedding 配置 ──
    with st.expander("── 向量模型 (Embedding) ─", expanded=False):
        emb_provider_labels = {k: v["label"] for k, v in EMBEDDING_PROVIDERS.items()}
        emb_provider_keys = list(EMBEDDING_PROVIDERS.keys())
        emb_provider_display = [emb_provider_labels[k] for k in emb_provider_keys]

        current_emb_idx = emb_provider_keys.index(current_emb_provider) if current_emb_provider in emb_provider_keys else 0

        selected_emb_label = st.selectbox(
            "提供商",
            options=emb_provider_display,
            index=current_emb_idx,
            key="cfg_emb_provider",
        )
        selected_emb_provider = emb_provider_keys[emb_provider_display.index(selected_emb_label)]
        emb_provider_info = EMBEDDING_PROVIDERS[selected_emb_provider]

        if selected_emb_provider == current_emb_provider:
            emb_default_model = current_emb_model
        else:
            emb_default_model = emb_provider_info["default_model"]

        st.text_input(
            "模型名称",
            value=emb_default_model,
            key="cfg_emb_model",
            help="输入 Embedding 模型名称，如 text-embedding-v3、nomic-embed-text 等",
        )

        if emb_provider_info["needs_api_key"]:
            st.text_input(
                "API Key",
                value=current_emb_api_key,
                type="password",
                key="cfg_emb_api_key",
                help="留空则使用 .env 中的配置",
            )
        else:
            st.caption("此提供商无需 API Key（本地模型）")
            st.session_state["cfg_emb_api_key"] = ""

        col_save_emb, col_reset_emb = st.columns([1, 1])
        with col_save_emb:
            if st.button("💾 保存并重新加载 Embedding", use_container_width=True, type="primary", key="btn_save_emb"):
                new_emb_provider = selected_emb_provider
                new_emb_model = st.session_state.get("cfg_emb_model", emb_provider_info["default_model"])
                new_emb_api_key = st.session_state.get("cfg_emb_api_key", current_emb_api_key)

                with st.spinner("正在重新加载 Embedding 和知识库索引..."):
                    ok = _apply_embedding_config(new_emb_provider, new_emb_model, new_emb_api_key)

                if ok:
                    st.session_state["_emb_provider"] = new_emb_provider
                    st.session_state["_emb_model"] = new_emb_model
                    st.session_state["_emb_api_key"] = new_emb_api_key
                    st.success(f"✅ Embedding 已切换为：{EMBEDDING_PROVIDERS[new_emb_provider]['label']} / {new_emb_model}")
                else:
                    st.error("Embedding 加载失败，请检查配置")

        with col_reset_emb:
            if st.button("↩️ 恢复默认 Embedding 配置", use_container_width=True, key="btn_reset_emb"):
                for k in ("_emb_provider", "_emb_model", "_emb_api_key"):
                    st.session_state.pop(k, None)
                st.success("已恢复为 .env 中的默认配置，请刷新页面")


# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("⚙️ 控制中心")

    # 系统状态卡片
    st.markdown("### 📊 系统状态")
    ctx = st.session_state.app_context
    if st.session_state.initialized and ctx:
        st.success("✅ 系统已初始化")

        col1, col2 = st.columns(2)
        with col1:
            if NEO4J_ENABLED:
                st.markdown("🔗 **Neo4j**")
                st.caption("已启用" if ctx.neo4j_client and ctx.neo4j_client.driver else "未连接")
            else:
                st.markdown("📄 **MySQL 模式**")
                st.caption("元数据模式")
        with col2:
            st.markdown("💾 **MySQL**")
            st.caption("已连接" if ctx.mysql_client else "未连接")

        st.markdown("🔌 **Embedding**")
        st.caption(f"{EMBEDDING_PROVIDER}/{EMBEDDING_MODEL}")

        st.markdown("🎯 **Few-Shot**")
        st.caption(f"{'已启用' if FEW_SHOT_ENABLED else '禁用'} (top {FEW_SHOT_TOP_K})")
    else:
        st.warning("⚠️ 系统未初始化")
        with st.spinner("正在初始化系统..."):
            init_result = initialize_system()
            if init_result:
                st.success("✅ 初始化成功!")
                st.rerun()
            else:
                st.error(
                    st.session_state.get("initialization_error")
                    or "初始化失败，请检查配置"
                )

    st.divider()

    # 重新初始化系统按钮
    if st.button("🔄 重新初始化系统", use_container_width=True, type="secondary"):
        if ctx:
            ctx.reset()
        for key in ['initialized', 'app_context']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("系统状态已重置，正在重新初始化...")
        st.rerun()

    st.divider()

    # 会话信息
    st.markdown("### 💬 会话信息")
    st.code(f"{st.session_state.session_id[:12]}...", language="text")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_sql = None
            st.session_state.last_result = None
            st.success("历史已清空")
    with col2:
        st.metric("会话数", len(st.session_state.get('history', [])))

    st.divider()

    # 知识库管理
    st.markdown("### 🗄️ 知识库管理")

    if NEO4J_ENABLED and ctx and ctx.neo4j_client and ctx.neo4j_client.driver:
        neo4j = ctx.neo4j_client
        try:
            stats = neo4j.execute_query("""
                MATCH (n)
                OPTIONAL MATCH ()-[r]-()
                RETURN
                    count(DISTINCT n) as nodes,
                    count(r) as relationships
            """)
            if stats:
                st.caption(f"节点：{stats[0].get('nodes', 0)} | 关系：{stats[0].get('relationships', 0)}")
        except Exception:
            st.caption("无法获取知识库统计")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 导出", use_container_width=True, type="secondary"):
                if st.session_state.initialized:
                    with st.spinner("正在导出知识库..."):
                        export_data = neo4j.export_knowledge_graph()
                        if export_data:
                            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
                            st.download_button(
                                label="📥 点击下载 JSON 文件",
                                data=json_str,
                                file_name=f"knowledge_graph_{st.session_state.session_id[:8]}.json",
                                mime="application/json",
                                use_container_width=True
                            )
        with col2:
            st.download_button(
                label="⬇️ 下载模板",
                data=json.dumps(get_knowledge_graph_template(), ensure_ascii=False, indent=2),
                file_name="knowledge_graph_template.json",
                mime="application/json",
                use_container_width=True,
                help="下载可导入的知识图谱 JSON 模板"
            )

        uploaded_file = st.file_uploader(
            "上传知识库 JSON 文件",
            type=["json"],
            accept_multiple_files=False,
            help="上传之前导出的知识库 JSON 文件进行恢复"
        )

        if uploaded_file is not None:
            try:
                content = read_uploaded_text(uploaded_file, UPLOAD_MAX_BYTES)
                import_data = json.loads(content)
            except json.JSONDecodeError as e:
                import_data = None
                st.error(f"JSON 格式异常：第 {e.lineno} 行第 {e.colno} 列，{e.msg}")
            except UploadValidationError as e:
                import_data = None
                st.error(str(e))

            if import_data is None:
                st.warning("格式异常，是否下载模板后按模板调整？")
                st.download_button(
                    label="⬇️ 下载知识图谱 JSON 模板",
                    data=json.dumps(get_knowledge_graph_template(), ensure_ascii=False, indent=2),
                    file_name="knowledge_graph_template.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                validation = neo4j.validate_knowledge_graph_json(import_data)
                if not validation["valid"]:
                    st.error("知识图谱 JSON 结构异常：")
                    for error in validation["errors"]:
                        st.write(f"- {error}")
                    st.warning("是否下载模板后按模板调整？")
                    st.download_button(
                        label="⬇️ 下载知识图谱 JSON 模板",
                        data=json.dumps(get_knowledge_graph_template(), ensure_ascii=False, indent=2),
                        file_name="knowledge_graph_template.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    database_refs = validation.get("database_refs", [])
                    existing_databases = neo4j.get_existing_import_databases(import_data)
                    if database_refs:
                        st.caption("导入文件包含数据库：" + "、".join(
                            str(db.get("name") or db.get("database_id")) for db in database_refs
                        ))

                    if existing_databases:
                        st.warning(
                            "检测到导入文件中的数据库图谱已存在："
                            + "、".join(str(db.get("name") or db.get("database_id")) for db in existing_databases)
                            + "。请确认是否覆盖这些数据库的现有图谱。"
                        )
                        confirm_import = st.checkbox(
                            "✓ 确认覆盖已存在数据库图谱",
                            key="confirm_overwrite_import_kg"
                        )
                        import_clicked = st.button(
                            "覆盖并导入",
                            type="primary",
                            use_container_width=True,
                            disabled=not confirm_import,
                            key="overwrite_import_kg_btn"
                        )
                        overwrite = True
                    else:
                        st.info("未检测到同名或同 ID 数据库图谱，将直接导入。")
                        import_clicked = st.button(
                            "直接导入",
                            type="primary",
                            use_container_width=True,
                            key="direct_import_kg_btn"
                        )
                        overwrite = False

                    if import_clicked:
                        with st.spinner("正在导入知识库..."):
                            result = neo4j.import_knowledge_graph(import_data, overwrite=overwrite)
                            if "error" in result:
                                st.error(f"导入失败：{result['error']}")
                            else:
                                st.success(
                                    f"✅ 导入完成：{result['nodes']} 个节点，"
                                    f"{result['relationships']} 个关系"
                                )
                                if ctx:
                                    ctx.reset()
                                for key in ['initialized', 'app_context']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.rerun()

        col_clear1, col_clear2 = st.columns([3, 1])
        with col_clear1:
            if "confirm_clear_kg" not in st.session_state:
                st.session_state.confirm_clear_kg = False
            clear_confirm = st.checkbox("✓ 确认清除", key="confirm_clear_kg",
                                        help="勾选后点击清除按钮将删除所有知识图谱数据")
        with col_clear2:
            clear_btn = st.button("🗑️ 清除知识库", use_container_width=True, type="secondary",
                                 disabled=not clear_confirm)

        if clear_btn:
            if st.session_state.initialized:
                with st.spinner("正在清除知识库..."):
                    result = neo4j.clear_knowledge_graph()
                    if "error" in result:
                        st.error(f"清除失败：{result['error']}")
                    else:
                        st.success(f"✅ 已删除 {result['nodes']} 个节点，{result['relationships']} 个关系")
                        if ctx:
                            ctx.reset()
                        for key in ['initialized', 'app_context']:
                            if key in st.session_state:
                                del st.session_state[key]
                        if "confirm_clear_kg" in st.session_state:
                            del st.session_state.confirm_clear_kg
                        st.success("知识库已清除，系统将重新初始化...")
                        st.rerun()
    else:
        st.caption("Neo4j 未启用")

    st.divider()

    # 关于
    st.markdown("### ℹ️ 关于")
    st.markdown("""
    **Text2SQL 智能体 v1.0**

    基于本地知识库的 SQL 智能生成系统

    **核心技术**:
    - 🔍 BM25 + FAISS 混合检索
    - 🤖 LangChain + LLM 智能生成
    - ✅ SQLFluff 语法校验
    - 💾 Neo4j/MySQL 双模式
    """)

    st.markdown("---")
    st.caption("© 2026 Text2SQL Agent")


# ==================== 主界面 ====================
# 主标题
st.markdown('<p class="main-title">📊 Text2SQL 智能体</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">基于本地知识库的 SQL 智能生成系统</p>', unsafe_allow_html=True)

# 选项卡
tab1, tab2, tab3, tab4 = st.tabs(["🔮 SQL 生成", "📜 执行历史", "🏗️ 知识图谱构建", "ℹ️ 系统信息"])

# ==================== Tab 1: SQL 生成 ====================
with tab1:
    # 输入区域 - 使用卡片样式
    st.markdown("### 📝 自然语言转 SQL")

    # 数据库选择
    db_options = []
    db_labels = []
    db_list = []
    ctx = st.session_state.app_context
    if st.session_state.initialized and ctx and ctx.neo4j_client and ctx.neo4j_client.driver:
        try:
            db_list = ctx.neo4j_client.get_database_list()
            db_options = [db["database_id"] for db in db_list]
            db_labels = [f"{db['name']} ({db['db_type']})" for db in db_list]
        except Exception as e:
            logger.warning(f"获取数据库列表失败：{e}")

    col_db, col1, col2 = st.columns([1, 2, 1])

    with col_db:
        if db_options:
            # 默认选择第一个数据库
            default_idx = 0
            selected_db_label = st.selectbox(
                "🗄️ 选择数据库",
                options=db_labels,
                index=default_idx if len(db_labels) > 0 else 0,
                key="selected_database",
                help="选择要查询的数据库，系统将生成对应语法的 SQL"
            )
            # 获取选中的数据库 ID
            selected_idx = db_labels.index(selected_db_label)
            selected_db_id = db_options[selected_idx]
            selected_db_info = next((db for db in db_list if db["database_id"] == selected_db_id), None)
            selected_db_type = selected_db_info["db_type"] if selected_db_info else "mysql"
        else:
            st.selectbox("🗄️ 选择数据库", options=["暂无数据库"], disabled=True)
            selected_db_id = None
            selected_db_type = "mysql"

    with col1:
        user_query = st.text_area(
            "💭 请输入您的查询需求",
            placeholder="例如：查询所有订单金额大于 100 的用户",
            height=120,
            key="user_query",
            help="输入自然语言描述，系统将自动生成对应的 SQL 语句"
        )

    with col2:
        st.markdown("""
        <div class="stCard">
        <h4>💡 示例查询</h4>
        <ul style="padding-left: 20px; line-height: 2;">
        <li>查询所有用户</li>
        <li>查询订单数量</li>
        <li>查询金额最大的订单</li>
        <li>统计每个用户订单数</li>
        <li>查询创建时间最近的订单</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # 生成按钮
    st.markdown("")  # 间距
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        generate_btn = st.button("🚀 生成 SQL", type="primary", use_container_width=True)
    with col2:
        execute_btn = st.button(
            "▶️ 执行 SQL",
            use_container_width=True,
            disabled=execution_is_disabled(st.session_state.last_result),
        )
    with col3:
        copy_btn = st.button("📋 复制 SQL", use_container_width=True)

    # 生成 SQL
    if generate_btn and user_query:
        if not st.session_state.initialized or not ctx or not ctx.sql_generator:
            st.error("请先初始化系统")
        else:
            # 优先使用用户选择的数据库
            if selected_db_id is not None:
                detected_dialect = selected_db_type
                detected_db_id = selected_db_id
                db_name = selected_db_label.split(' ')[0]
            else:
                # 自动检测
                detected_dialect, detected_db_id = ctx.sql_generator.detect_dialect(user_query)
                db_info = ctx.sql_generator.dialect_detector.get_database_info(detected_db_id) if detected_db_id else None
                db_name = db_info.get('name', '未知') if db_info else '默认数据库'

            st.info(f"🎯 检测到目标数据库：**{db_name}** ({detected_dialect.upper()})")

            with st.spinner("正在生成 SQL..."):
                result = ctx.sql_generator.generate_sql(
                    user_query=user_query,
                    session_id=st.session_state.session_id,
                    target_dialect=detected_dialect,
                    target_database_id=detected_db_id,
                    auto_save_history=HISTORY_AUTO_SAVE,
                )

            # 检查是否检索到知识
            if result.get('matched_knowledge') is not None and len(result.get('matched_knowledge', [])) == 0:
                st.warning("""
                ⚠️ **未检索到相关知识库内容**

                可能的原因：
                1. 知识库为空或未初始化 - 请先在"知识图谱构建"页面导入 DDL 文件
                2. 知识库索引未刷新 - 请点击侧边栏的 **"🔄 重新初始化系统"** 按钮
                3. 查询问题与知识库内容不匹配 - 尝试调整查询措辞
                """)

            if result.get('status') in {'success', 'rejected'} and result.get('sql'):
                st.session_state.last_sql = result['sql']
                st.session_state.last_result = result

                # 显示思考过程（CoT）
                if result.get('thinking'):
                    with st.expander("🧠 思考过程（CoT）", expanded=False):
                        st.markdown(result['thinking'])

                # 显示生成的 SQL
                st.code(result['sql'], language='sql', line_numbers=True)

                # 显示匹配的知识
                if result.get('matched_knowledge'):
                    with st.expander("📚 匹配的知识"):
                        for knowledge in result['matched_knowledge']:
                            st.markdown(f"- **{knowledge['name']}** (类型：{knowledge['node_type']}, 相似度：{knowledge['score']:.4f})")

                # 显示校验结果
                validation = result.get('validation') or validate_sql(
                    result['sql'], dialect=detected_dialect
                )
                if validation['valid']:
                    st.success("✅ SQL 校验通过")
                else:
                    structured = structured_validation_messages(result)
                    if structured:
                        for message in structured:
                            st.warning(f"⚠️ SQL 校验问题：{message}")
                    else:
                        st.warning(
                            f"⚠️ SQL 校验问题：{', '.join(validation.get('errors', []))}"
                        )

                # 添加到历史
                st.session_state.history.append({
                    'query': user_query,
                    'user_query': result.get('user_query', user_query),
                    'sql': result['sql'],
                    'status': result.get('status'),
                    'semantic_version': result.get('semantic_version'),
                    'content_sha256': result.get('content_sha256'),
                    'validation': result.get('validation'),
                    'executable': result.get('executable'),
                    'allowed_tables': result.get('allowed_tables'),
                    'allowed_schemas': result.get('allowed_schemas'),
                    'timestamp': '刚刚'
                })

                st.rerun()
            else:
                structured = structured_validation_messages(result)
                error_msg = public_failure_messages(result)[0]
                # 检查是否是连接错误
                if not structured and (
                    'Connection' in error_msg or 'connection' in error_msg.lower()
                ):
                    st.error("""
                    ⚠️ **LLM 服务连接失败**

                    可能的原因：
                    1. 网络连接问题 - 请检查网络连接
                    2. API Key 无效或过期 - 请检查 `.env` 文件中的 `LLM_API_KEY` 配置
                    3. 服务不可用 - 阿里云 DashScope 服务可能暂时不可用

                    建议操作：
                    - 检查 `.env` 文件中的 API Key 配置是否正确
                    - 稍后重试
                    """)
                elif not structured:
                    st.error(f"生成失败：{error_msg}")
                else:
                    for message in structured:
                        st.error(f"生成失败：{message}")

    # 显示当前 SQL
    if st.session_state.last_sql:
        st.markdown("### 📄 生成的 SQL")
        sql_container = st.container()
        with sql_container:
            st.code(st.session_state.last_sql, language='sql', line_numbers=False)

            persisted_validation = structured_validation_messages(
                st.session_state.last_result
            )
            for message in persisted_validation:
                st.warning(f"⚠️ SQL 校验问题：{message}")

            if copy_btn:
                st.info("💡 点击代码块右上角的复制按钮即可复制 SQL")

    # 执行 SQL
    if execute_btn and st.session_state.last_sql and ctx and ctx.sql_generator:
        with st.spinner("⏳ 正在执行 SQL..."):
            exec_result = ctx.sql_generator.execute_sql(
                sql=st.session_state.last_sql,
                history_id=(st.session_state.last_result or {}).get('history_id'),
                allowed_tables=(st.session_state.last_result or {}).get('allowed_tables'),
                allowed_schemas=(st.session_state.last_result or {}).get('allowed_schemas'),
                user_query=(st.session_state.last_result or {}).get('user_query'),
                semantic_version=(st.session_state.last_result or {}).get(
                    'semantic_version'
                ),
                content_sha256=(st.session_state.last_result or {}).get(
                    'content_sha256'
                ),
                session_id=st.session_state.session_id,
            )

        if exec_result.get('status') == 'success':
            st.success(f"✅ 执行成功，返回 {exec_result.get('row_count', 0)} 条记录")
            if exec_result.get('truncated'):
                st.warning(f"结果超过安全上限，仅显示前 {SQL_QUERY_MAX_ROWS} 条")

            # 显示结果
            data = exec_result.get('data', [])
            if data:
                st.dataframe(data, use_container_width=True, height=300)
        else:
            for message in public_failure_messages(exec_result):
                st.error(f"❌ 执行失败：{message}")

    # 无输入提示
    if generate_btn and not user_query:
        st.warning("⚠️ 请输入查询内容")


# ==================== Tab 2: 执行历史 ====================
with tab2:
    st.markdown("### 📜 执行历史")
    
    if st.session_state.history:
        # 统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总记录数", len(st.session_state.history))
        with col2:
            success_count = len([h for h in st.session_state.history if h.get('status') == 'success'])
            st.metric("成功", success_count)
        with col3:
            st.metric("待执行", len(st.session_state.history) - success_count)
        
        st.divider()
        
        # 历史列表
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"**{i+1}.** {item['query'][:60]}{'...' if len(item['query']) > 60 else ''}", expanded=(i==0)):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**查询:** `{item['query']}`")
                with col2:
                    status_icon = "✅" if item.get('status') == 'success' else "⏳"
                    st.markdown(f"**状态:** {status_icon}")
                
                st.code(item['sql'], language='sql', line_numbers=False)
                
                # 操作按钮
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("📥 使用此 SQL", key=f"use_{i}", use_container_width=True):
                        st.session_state.last_sql = item['sql']
                        st.session_state.last_result = item
                        st.rerun()
                with col2:
                    if st.button("📋 复制", key=f"copy_{i}", use_container_width=True):
                        st.success("已复制到剪贴板")
                with col3:
                    if st.button("🗑️ 删除", key=f"delete_{i}", use_container_width=True):
                        st.session_state.history.pop(len(st.session_state.history) - 1 - i)
                        st.rerun()
                with col4:
                    if st.button("提取&增强", key=f"extract_{i}", use_container_width=True):
                        # 提取 JOIN 关系
                        from core.sql.join_extractor import extract_joins_from_sql
                        joins = extract_joins_from_sql(item['sql'])
                        # 转为字典列表存储（避免 Streamlit session_state 序列化 dataclass 的问题）
                        if joins:
                            st.session_state[f"extracted_joins_{i}"] = [
                                {
                                    "left_table": jn.left_table,
                                    "right_table": jn.right_table,
                                    "join_type": jn.join_type,
                                    "join_condition": jn.join_condition,
                                    "from_column": jn.from_column,
                                    "to_column": jn.to_column,
                                    "extra_condition": jn.extra_condition
                                }
                                for jn in joins
                            ]
                        else:
                            st.session_state[f"extracted_joins_{i}"] = []

                # 显示提取的 JOIN 关系
                extracted = st.session_state.get(f"extracted_joins_{i}", [])
                if extracted:
                    st.markdown(f"**提取到 {len(extracted)} 个 JOIN 关系：**")
                    for j, join in enumerate(extracted):
                        st.markdown(
                            f"  **{j+1}.** `{join['left_table']}` → `{join['right_table']}` "
                            f"({join['join_type']}) — ON `{join['join_condition']}`"
                        )

                    if st.button("确认增强到知识图谱", key=f"confirm_{i}", type="primary", use_container_width=True):
                        # 更新 Neo4j 权重
                        if ctx and ctx.neo4j_client and ctx.neo4j_client.driver:
                            from core.graph_builder.loader.weight_initializer import (
                                rebalance_with_history,
                            )
                            stats = rebalance_with_history(extracted, ctx.neo4j_client)
                            st.success(f"增强完成：更新 {stats['updated']} 条，新建 {stats['created']} 条关系")
                        else:
                            st.warning("Neo4j 未连接，无法增强")
    else:
        st.info("💡 暂无历史记录，快去生成第一条 SQL 吧！")


# ==================== Tab 3: 知识图谱构建 ====================
with tab3:
    from app.pages.kg_builder import show_kg_builder
    show_kg_builder()

with tab4:
    st.markdown("### ℹ️ 系统信息")

    # 读取当前生效的配置（UI 覆盖值优先于 .env）
    _current_llm_provider = st.session_state.get("_llm_provider", LLM_PROVIDER)
    _current_llm_model = st.session_state.get("_llm_model", LLM_MODEL)
    _current_llm_base_url = st.session_state.get("_llm_base_url", LLM_BASE_URL)
    _current_emb_provider = st.session_state.get("_emb_provider", EMBEDDING_PROVIDER)
    _current_emb_model = st.session_state.get("_emb_model", EMBEDDING_MODEL)

    # 系统状态卡片
    st.markdown("#### 🔧 配置信息")
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        # LLM 配置展示（显示当前生效值 + 标签）
        llm_label = LLM_PROVIDERS.get(_current_llm_provider, {}).get("label", _current_llm_provider)
        st.markdown(f"""
        **LLM 配置**
        - 提供商：{llm_label}
        - 模型：{_current_llm_model}
        - Base URL: {_current_llm_base_url or "默认"}
        """)

        emb_label = EMBEDDING_PROVIDERS.get(_current_emb_provider, {}).get("label", _current_emb_provider)
        st.markdown(f"""
        **Embedding 配置**
        - 提供商：{emb_label}
        - 模型：{_current_emb_model}
        """)

    with config_col2:
        # 脱敏展示：隐藏主机地址和端口详情，仅显示连接状态
        mysql_status = "已连接" if (ctx and ctx.mysql_client and ctx.mysql_client.engine) else "未连接"
        neo4j_status = "已连接" if NEO4J_ENABLED and (ctx and ctx.neo4j_client and ctx.neo4j_client.driver) else "未连接/未启用"
        st.markdown("""
        **数据库配置**
        - MySQL: {mysql_status}
        - Neo4j: {neo4j_status}
        - SQL 方言：{dialect}
        """.format(
            mysql_status=mysql_status,
            neo4j_status=neo4j_status,
            dialect=SQL_DIALECT
        ))

        st.markdown("""
        **检索配置**
        - BM25+FAISS 混合检索
        - Few-Shot: {few_shot} (top {top_k})
        """.format(
            few_shot="已启用" if FEW_SHOT_ENABLED else "禁用",
            top_k=FEW_SHOT_TOP_K
        ))

    st.divider()

    # 模型配置 UI 面板
    _render_model_config_panel()

    st.divider()

    # 统计信息
    st.markdown("#### 📊 统计信息")
    if ctx and ctx.sql_generator:
        stats = ctx.sql_generator.get_statistics()
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("总请求数", stats.get('total_requests', 0))
        with stat_col2:
            st.metric("成功生成", stats.get('successful_generations', 0))
        with stat_col3:
            st.metric("失败", stats.get('failed_generations', 0))
    
    st.divider()
    
    # 日志
    st.markdown("#### 📝 最近日志")
    try:
        log_file = project_root / "logs" / "app.log"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = f.readlines()[-20:]
                st.text("".join(logs))
        else:
            st.info("日志文件不存在")
    except Exception as e:
        st.write(f"无法读取日志：{e}")


# ==================== Footer ====================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999; padding: 20px 0;'>"
    "Text2SQL 智能体 v1.0 | 基于本地知识库的 SQL 智能生成系统 | © 2026"
    "</div>",
    unsafe_allow_html=True
)
