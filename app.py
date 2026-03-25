"""
Text2SQL 智能体 - Streamlit 主入口
基于本地知识库的 SQL 智能生成应用
"""
import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    NEO4J_ENABLED, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    RETRIEVAL_TOP_K, SQL_DIALECT,
    BM25_K1, BM25_B, TFIDF_MAX_FEATURES,
    BM25_WEIGHT, TFIDF_WEIGHT, DENSE_WEIGHT,
    FAISS_ENABLED, FAISS_EMBEDDING_DIM, FAISS_INDEX_TYPE,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_KEY,
    FEW_SHOT_ENABLED, FEW_SHOT_TOP_K,
    MYSQL_POOL_SIZE, MYSQL_MAX_OVERFLOW, MYSQL_POOL_TIMEOUT
)
from core.llm_factory import create_llm
from core.knowledge.neo4j_client import Neo4jClient
from core.history.mysql_client import MySQLClient
from core.retrieval.bm25_tfidf import KnowledgeRetriever
from core.chain.sql_chain import SQLGenerationChain
from core.sql.generator import SQLGenerator
from core.sql.validator import validate_sql
from core.embedding_factory import EmbeddingFactory
from loguru import logger
import uuid

# 配置日志
logger.remove()
logger.add(lambda msg: st.write(msg), level="INFO")
logger.add(project_root / "logs" / "app.log", level="DEBUG", rotation="1 day")

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="Text2SQL 智能体",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/renguanjie/text2sql-agent',
        'Report a bug': 'https://github.com/renguanjie/text2sql-agent/issues',
        'About': '# Text2SQL 智能体\n基于本地知识库的 SQL 智能生成系统\n\n核心技术：BM25+TF-IDF 检索 + LangChain + SQLFluff'
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

if 'sql_generator' not in st.session_state:
    st.session_state.sql_generator = None

if 'history' not in st.session_state:
    st.session_state.history = []

if 'last_sql' not in st.session_state:
    st.session_state.last_sql = None

if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# ==================== 初始化函数 ====================
@st.cache_resource
def initialize_system():
    """初始化系统组件"""
    logger.info("开始初始化系统...")

    # 1. 创建 LLM
    try:
        llm = create_llm(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )
        logger.info(f"LLM 初始化成功：{LLM_PROVIDER}/{LLM_MODEL}")
    except Exception as e:
        logger.error(f"LLM 初始化失败：{e}")
        st.error(f"LLM 初始化失败：{e}")
        return None

    # 2. 创建 Neo4j 客户端 (可选)
    neo4j_client = Neo4jClient(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        max_connection_pool_size=50,
        connection_timeout=30
    )
    if NEO4J_ENABLED:
        if not neo4j_client.connect():
            logger.warning("Neo4j 连接失败，将使用 MySQL 元数据模式")
            st.warning("⚠️ Neo4j 未连接，使用 MySQL 元数据模式")
    else:
        logger.info("Neo4j 未启用，使用 MySQL 元数据模式")

    # 3. 创建 MySQL 客户端（使用连接池）
    mysql_client = MySQLClient(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        pool_size=MYSQL_POOL_SIZE,
        max_overflow=MYSQL_MAX_OVERFLOW,
        pool_timeout=MYSQL_POOL_TIMEOUT
    )
    if not mysql_client.connect():
        logger.warning("MySQL 连接失败，历史记录功能可能不可用")

    # 4. 创建 Embedding Factory
    embedding_factory = EmbeddingFactory(
        provider=EMBEDDING_PROVIDER,
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY
    )
    logger.info(f"Embedding 工厂初始化：{EMBEDDING_PROVIDER}/{EMBEDDING_MODEL}")

    # 5. 创建知识检索器 - 三路混合检索 (BM25 + TF-IDF + FAISS)
    knowledge_retriever = KnowledgeRetriever(
        config={
            'bm25_k1': BM25_K1,
            'bm25_b': BM25_B,
            'tfidf_max_features': TFIDF_MAX_FEATURES,
            'bm25_weight': BM25_WEIGHT,
            'tfidf_weight': TFIDF_WEIGHT,
            'dense_weight': DENSE_WEIGHT,
            'embedding_dim': FAISS_EMBEDDING_DIM,
            'index_type': FAISS_INDEX_TYPE if FAISS_ENABLED else None,
            'few_shot_enabled': FEW_SHOT_ENABLED,
            'few_shot_top_k': FEW_SHOT_TOP_K
        },
        embedding_factory=embedding_factory
    )

    # 5. 加载知识库 (Neo4j 或 MySQL)
    try:
        if NEO4J_ENABLED and neo4j_client.driver:
            # 从 Neo4j 加载
            metadata = neo4j_client.get_schema_metadata()
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
                knowledge_retriever.retriever.index_documents(documents, metadata_list)
                logger.info(f"知识库索引完成：{len(documents)} 个文档 (Neo4j)")
        else:
            # 从 MySQL 加载元数据
            tables = mysql_client.get_table_schema()
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
                knowledge_retriever.retriever.index_documents(documents, metadata_list)
                logger.info(f"知识库索引完成：{len(documents)} 个文档 (MySQL)")
    except Exception as e:
        logger.warning(f"知识库加载失败：{e}")

    # 6. 创建 SQL 生成链
    sql_chain = SQLGenerationChain(
        llm=llm,
        dialect=SQL_DIALECT,
        retrieval_top_k=RETRIEVAL_TOP_K
    )

    # 7. 创建 SQL 生成器
    generator = SQLGenerator(
        llm=llm,
        knowledge_retriever=knowledge_retriever,
        sql_chain=sql_chain,
        mysql_client=mysql_client,
        config={
            'enable_rewrite': False,
            'few_shot_enabled': FEW_SHOT_ENABLED,
            'few_shot_top_k': FEW_SHOT_TOP_K
        }
    )

    st.session_state.sql_generator = generator
    st.session_state.initialized = True
    st.session_state.neo4j_client = neo4j_client
    st.session_state.mysql_client = mysql_client

    logger.info("系统初始化完成")
    return True


# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("⚙️ 控制中心")
    
    # 系统状态卡片
    st.markdown("### 📊 系统状态")
    if st.session_state.initialized:
        st.success("✅ 系统已初始化")

        # 显示连接状态
        col1, col2 = st.columns(2)
        with col1:
            if NEO4J_ENABLED:
                st.markdown("🔗 **Neo4j**")
                st.caption("已启用" if hasattr(st.session_state, 'neo4j_client') and st.session_state.neo4j_client.driver else "未连接")
            else:
                st.markdown("📄 **MySQL 模式**")
                st.caption("元数据模式")
        with col2:
            st.markdown("💾 **MySQL**")
            st.caption("已连接" if hasattr(st.session_state, 'mysql_client') else "未连接")

        # 显示 Embedding 和 Few-Shot 状态
        st.markdown("🔌 **Embedding**")
        st.caption(f"{EMBEDDING_PROVIDER}/{EMBEDDING_MODEL}")

        st.markdown("🎯 **Few-Shot**")
        st.caption(f"{'已启用' if FEW_SHOT_ENABLED else '禁用'} (top {FEW_SHOT_TOP_K})")
    else:
        st.warning("⚠️ 系统未初始化")
        if st.button("🚀 初始化系统", use_container_width=True, type="primary"):
            with st.spinner("初始化中..."):
                initialize_system()
                if st.session_state.initialized:
                    st.success("✅ 初始化成功!")
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
    
    # 关于
    st.markdown("### ℹ️ 关于")
    st.markdown("""
    **Text2SQL 智能体 v1.0**
    
    基于本地知识库的 SQL 智能生成系统
    
    **核心技术**:
    - 🔍 BM25 + TF-IDF 混合检索
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
tab1, tab2, tab3 = st.tabs(["🔮 SQL 生成", "📜 执行历史", "ℹ️ 系统信息"])

# ==================== Tab 1: SQL 生成 ====================
with tab1:
    # 输入区域 - 使用卡片样式
    st.markdown("### 📝 自然语言转 SQL")
    
    col1, col2 = st.columns([3, 1])

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
        execute_btn = st.button("▶️ 执行 SQL", use_container_width=True)
    with col3:
        copy_btn = st.button("📋 复制 SQL", use_container_width=True)

    # 生成 SQL
    if generate_btn and user_query:
        if not st.session_state.initialized:
            st.error("请先初始化系统")
        else:
            with st.spinner("正在生成 SQL..."):
                result = st.session_state.sql_generator.generate_sql(
                    user_query=user_query,
                    session_id=st.session_state.session_id
                )

            if result.get('status') == 'success' and result.get('sql'):
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
                validation = validate_sql(result['sql'])
                if validation['valid']:
                    st.success("✅ SQL 校验通过")
                else:
                    st.warning(f"⚠️ SQL 校验问题：{', '.join(validation['errors'])}")

                # 添加到历史
                st.session_state.history.append({
                    'query': user_query,
                    'sql': result['sql'],
                    'status': result.get('status'),
                    'timestamp': '刚刚'
                })

                st.rerun()
            else:
                st.error(f"生成失败：{result.get('error', '未知错误')}")

    # 显示当前 SQL
    if st.session_state.last_sql:
        st.markdown("### 📄 生成的 SQL")
        sql_container = st.container()
        with sql_container:
            st.code(st.session_state.last_sql, language='sql', line_numbers=False)

            if copy_btn:
                st.info("💡 点击代码块右上角的复制按钮即可复制 SQL")

    # 执行 SQL
    if execute_btn and st.session_state.last_sql:
        with st.spinner("⏳ 正在执行 SQL..."):
            # 这里需要获取 history_id，简化处理
            exec_result = st.session_state.sql_generator.execute_sql(
                sql=st.session_state.last_sql
            )

        if exec_result.get('status') == 'success':
            st.success(f"✅ 执行成功，返回 {exec_result.get('row_count', 0)} 条记录")

            # 显示结果
            data = exec_result.get('data', [])
            if data:
                st.dataframe(data, use_container_width=True, height=300)
        else:
            st.error(f"❌ 执行失败：{exec_result.get('error', '未知错误')}")

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
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📥 使用此 SQL", key=f"use_{i}", use_container_width=True):
                        st.session_state.last_sql = item['sql']
                        st.rerun()
                with col2:
                    if st.button("📋 复制", key=f"copy_{i}", use_container_width=True):
                        st.success("已复制到剪贴板")
                with col3:
                    if st.button("🗑️ 删除", key=f"delete_{i}", use_container_width=True):
                        st.session_state.history.pop(len(st.session_state.history) - 1 - i)
                        st.rerun()
    else:
        st.info("💡 暂无历史记录，快去生成第一条 SQL 吧！")


# ==================== Tab 3: 系统信息 ====================
with tab3:
    st.markdown("### ℹ️ 系统信息")
    
    # 系统状态卡片
    st.markdown("#### 🔧 配置信息")
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        st.markdown("""
        **LLM 配置**
        - 提供商：{provider}
        - 模型：{model}
        - Base URL: {base_url}
        """.format(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            base_url=LLM_BASE_URL or "OpenAI 默认"
        ))

        st.markdown("""
        **Embedding 配置**
        - 提供商：{emb_provider}
        - 模型：{emb_model}
        """.format(
            emb_provider=EMBEDDING_PROVIDER,
            emb_model=EMBEDDING_MODEL
        ))

    with config_col2:
        st.markdown("""
        **数据库配置**
        - MySQL: {mysql_host}:{mysql_port}
        - Neo4j: {neo4j_uri}
        - SQL 方言：{dialect}
        """.format(
            mysql_host=MYSQL_HOST,
            mysql_port=MYSQL_PORT,
            neo4j_uri=NEO4J_URI,
            dialect=SQL_DIALECT
        ))

        st.markdown("""
        **检索配置**
        - BM25+TF-IDF+FAISS 混合检索
        - Few-Shot: {few_shot} (top {top_k})
        """.format(
            few_shot="已启用" if FEW_SHOT_ENABLED else "禁用",
            top_k=FEW_SHOT_TOP_K
        ))
    
    st.divider()
    
    # 统计信息
    st.markdown("#### 📊 统计信息")
    if st.session_state.sql_generator:
        stats = st.session_state.sql_generator.get_statistics()
        
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
