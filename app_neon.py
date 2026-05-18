"""
Text2SQL 智能体 - Neon PostgreSQL 版本
基于阿里云 Qwen-Max + Neo4j 知识库的 SQL 智能生成
"""
import streamlit as st
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    LLM_API_KEY, LLM_MODEL, LLM_BASE_URL,
    SQL_DIALECT, RETRIEVAL_TOP_K
)
from core.knowledge.neon_knowledge import Neo4jKnowledgeRetriever, build_dynamic_prompt
from core.llm.qwen_generator import QwenMaxLLM, generate_sql_with_qwen
from core.sql.validator import SQLValidator
from sqlalchemy import create_engine, text
import json

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="Text2SQL 智能体 - Neon 版",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义 CSS ====================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State 初始化 ====================
if 'neo4j_retriever' not in st.session_state:
    st.session_state.neo4j_retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'neon_engine' not in st.session_state:
    st.session_state.neon_engine = None

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("⚙️ 配置")

    # 数据库配置
    st.subheader("🗄️ 数据库配置")
    neon_url = st.text_input(
        "Neon PostgreSQL URL",
        value=os.getenv("NEON_DB_URL", ""),
        type="password",
        help="Neon PostgreSQL 连接字符串"
    )

    # LLM 配置
    st.subheader("🤖 LLM 配置")
    api_key = st.text_input(
        "阿里云 API Key",
        value=LLM_API_KEY,
        type="password",
        help="阿里云 DashScope API Key"
    )
    model_name = st.text_input(
        "模型名称",
        value=LLM_MODEL or "qwen-max",
        help="阿里云模型名称"
    )

    # 检索配置
    st.subheader("🔍 检索配置")
    top_k = st.slider("检索返回数量", 1, 10, RETRIEVAL_TOP_K or 5)
    dialect = st.selectbox("SQL 方言", ["postgresql", "mysql"], index=0)

    # 连接按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 连接数据库", use_container_width=True):
            try:
                # 初始化 Neo4j 检索器
                st.session_state.neo4j_retriever = Neo4jKnowledgeRetriever(
                    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
                )
                st.success("✅ Neo4j 连接成功")

                # 初始化 LLM
                if api_key:
                    st.session_state.llm = QwenMaxLLM(
                        api_key=api_key,
                        model=model_name,
                        base_url=LLM_BASE_URL
                    )
                    st.success("✅ LLM 初始化成功")

                # 初始化 Neon 连接
                st.session_state.neon_engine = create_engine(neon_url)
                st.success("✅ Neon PostgreSQL 连接成功")

            except Exception as e:
                st.error(f"❌ 连接失败：{str(e)}")

    with col2:
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.query_history = []
            st.rerun()

    # 显示统计信息
    if st.session_state.neo4j_retriever:
        st.divider()
        st.subheader("📊 知识库统计")
        try:
            tables = st.session_state.neo4j_retriever.get_all_tables()
            st.metric("表数量", len(tables))

            total_columns = sum(t['column_count'] for t in tables)
            st.metric("字段总数", total_columns)
        except Exception as e:
            st.warning(f"无法获取统计：{str(e)}")

    # 显示历史记录
    if st.session_state.query_history:
        st.divider()
        st.subheader("📝 查询历史")
        for item in reversed(st.session_state.query_history[-5:]):
            with st.expander(f"🔍 {item['query'][:30]}..."):
                st.code(item['sql'], language="sql")
                if item.get('result'):
                    st.info(f"结果：{item['result']}")

# ==================== 主页面 ====================
st.markdown('<h1 class="main-title">📊 Text2SQL 智能体 - Neon 版</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">基于阿里云 Qwen-Max + Neo4j 知识库的 SQL 智能生成系统</p>', unsafe_allow_html=True)

# ==================== 功能选择 ====================
tab1, tab2, tab3 = st.tabs(["🔍 SQL 生成", "📋 表浏览器", "ℹ️ 关于"])

with tab1:
    st.header("自然语言生成 SQL")

    # 查询输入
    query = st.text_area(
        "请输入您的查询",
        placeholder="例如：查询月收入 10000 元以上的客户及其账户信息",
        height=100,
        key="user_query"
    )

    # 生成按钮
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        generate_btn = st.button("🚀 生成 SQL", type="primary", use_container_width=True)

    if generate_btn and query:
        if not st.session_state.neo4j_retriever or not st.session_state.llm:
            st.error("❌ 请先在侧边栏连接数据库和配置 LLM")
        else:
            with st.spinner("🤖 正在生成 SQL..."):
                try:
                    # 1. 构建动态提示词
                    prompt, context = build_dynamic_prompt(
                        st.session_state.neo4j_retriever,
                        query,
                        dialect=dialect,
                        top_k=top_k
                    )

                    # 显示上下文
                    with st.expander("📚 检索到的知识", expanded=False):
                        st.write(f"**相关表**: {', '.join(context['table_names'])}")
                        st.text(context['schema'][:1000] + "...")

                    # 2. 调用 LLM 生成
                    result = generate_sql_with_qwen(
                        user_query=query,
                        schema_prompt=context['schema'],
                        api_key=api_key,
                        model=model_name
                    )

                    if result['success']:
                        sql = result.get('sql', '')

                        if sql == "INSUFFICIENT_INFO":
                            st.warning("⚠️ 信息不足以生成 SQL，请提供更具体的查询")
                        else:
                            # 显示生成的 SQL
                            st.success("✅ SQL 生成成功")
                            st.code(sql, language="sql")

                            # 3. SQL 验证
                            st.subheader("🔍 SQL 验证")
                            is_valid = sql.strip().upper().startswith('SELECT')
                            if is_valid:
                                st.success("✅ SQL 格式有效")
                            else:
                                st.warning("⚠️ SQL 格式可能有问题")

                            # 4. 执行 SQL（如果是 SELECT）
                            if is_valid and st.session_state.neon_engine:
                                st.subheader("▶️ 执行结果")
                                try:
                                    with st.session_state.neon_engine.connect() as conn:
                                        query_result = conn.execute(text(sql))
                                        rows = query_result.fetchall()

                                        if rows:
                                            # 显示列名
                                            columns = [key for key in rows[0]._mapping.keys()]
                                            st.write(f"**返回 {len(rows)} 条记录**")

                                            # 显示前 10 条
                                            st.dataframe(
                                                [dict(row._mapping) for row in rows[:10]],
                                                use_container_width=True
                                            )

                                            # 保存到历史
                                            st.session_state.query_history.append({
                                                'query': query,
                                                'sql': sql,
                                                'result': f"{len(rows)} 条记录"
                                            })
                                        else:
                                            st.info("ℹ️ 查询结果为空")

                                except Exception as e:
                                    st.error(f"❌ 执行失败：{str(e)}")

                            # 5. 显示使用量
                            if result.get('usage'):
                                st.caption(f"Tokens 使用：{result['usage']}")
                    else:
                        st.error(f"❌ 生成失败：{result.get('error', '未知错误')}")

                except Exception as e:
                    st.error(f"❌ 处理失败：{str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    elif generate_btn and not query:
        st.warning("⚠️ 请输入查询内容")

with tab2:
    st.header("📋 表浏览器")

    if st.session_state.neo4j_retriever:
        # 选择表
        try:
            tables = st.session_state.neo4j_retriever.get_all_tables()
            table_names = [f"{t['name']} ({t['name_cn']})" for t in tables]

            selected = st.selectbox("选择表", table_names)

            if selected:
                table_name = selected.split(' ')[0]
                schema = st.session_state.neo4j_retriever.get_table_schema(table_name)

                if schema:
                    st.subheader(f"{schema['name_cn']} ({schema['name']})")
                    st.write(schema.get('comment', ''))

                    # 字段列表
                    st.write(f"**字段 ({len(schema['columns'])} 个)**:")
                    st.dataframe(
                        {
                            '字段名': [c['name'] for c in schema['columns']],
                            '中文名': [c['name_cn'] for c in schema['columns']],
                            '类型': [c['type'] for c in schema['columns']],
                            '可空': ['是' if c['nullable'] else '否' for c in schema['columns']]
                        },
                        use_container_width=True
                    )

                    # 关联关系
                    if schema['relations']:
                        st.write(f"**关联关系 ({len(schema['relations'])} 个)**:")
                        for rel in schema['relations']:
                            st.write(f"- → {rel['target_table_cn']} ({rel['target_table']}) [{rel['relationship']}]")

        except Exception as e:
            st.error(f"❌ 加载失败：{str(e)}")
    else:
        st.info("ℹ️ 请先在侧边栏连接数据库")

with tab3:
    st.header("ℹ️ 关于")

    st.markdown("""
    ### Text2SQL 智能体 - Neon 版

    这是一个基于阿里云 Qwen-Max 大模型和 Neo4j 知识库的 SQL 智能生成系统。

    #### 核心技术

    - **LLM**: 阿里云 Qwen-Max
    - **知识库**: Neo4j 图数据库（存储表结构和关联关系）
    - **数据库**: Neon PostgreSQL（云原生数据库）
    - **动态提示词**: 基于 Neo4j 知识库动态构建提示词

    #### 功能特点

    1. **自然语言查询**: 用中文直接提问，自动生成 SQL
    2. **智能检索**: 基于关键词自动匹配相关表和字段
    3. **动态提示词**: 根据查询动态构建最优提示词
    4. **SQL 验证**: 自动验证生成的 SQL 格式
    5. **即时执行**: 直接在数据库中执行并查看结果
    6. **查询历史**: 自动保存查询历史

    #### 使用示例

    - "查询所有客户的基本信息"
    - "查找月收入在 10000 元以上的客户"
    - "统计每个城市的客户数量"
    - "查询客户及其账户信息"
    - "查找有贷款记录的客户"
    - "统计每个风险等级客户的平均贷款余额"

    #### 技术架构

    ```
    用户查询 → Neo4j 知识检索 → 动态提示词构建 → Qwen-Max → SQL 生成 → 验证 → 执行
    ```

    #### 版本信息

    - 版本：1.0.0
    - 开发日期：2026-03-22
    - 开发者：特能干 💪
    """)

# ==================== Footer ====================
st.divider()
st.caption("Text2SQL 智能体 - Neon 版 | 基于阿里云 Qwen-Max + Neo4j 知识库")
