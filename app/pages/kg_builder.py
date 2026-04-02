"""
知识图谱构建页面
支持从 DDL 文件或活体数据库构建知识图谱
"""
import streamlit as st
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.graph_builder import GraphBuilderPipeline
from core.knowledge.neo4j_client import get_neo4j_client
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
)


def render_graph_builder():
    """渲染知识图谱构建页面"""

    st.markdown("""
    <div class="hero-wrapper">
        <p class="main-title">🏗️ 知识图谱构建</p>
        <p class="subtitle">企业级知识图谱自动构建工具 - 支持多源数据融合 + AI 语义增强</p>
        <div class="badge-row">
            <span class="badge">📁 DDL 文件</span>
            <span class="badge">💾 活体数据库</span>
            <span class="badge">🤖 AI 增强</span>
            <span class="badge">🕒 UNWIND+MERGE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 初始化 Neo4j 客户端
    if 'kg_neo4j_client' not in st.session_state:
        try:
            st.session_state.kg_neo4j_client = get_neo4j_client(
                NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
            )
            st.session_state.kg_connected = True
        except Exception as e:
            st.session_state.kg_neo4j_client = None
            st.session_state.kg_connected = False
            st.error(f"Neo4j 连接失败：{e}")

    # 选项卡
    tab1, tab2, tab3 = st.tabs(["📁 DDL/JSON 文件构建", "💾 活体数据库构建", "📊 图谱状态"])

    # ==================== Tab 1: DDL/JSON 文件构建 ====================
    with tab1:
        st.markdown("### 📁 从 DDL/JSON 文件构建知识图谱")

        st.info("""
        💡 **支持格式**：
        - **DDL 文件** (.sql, .txt)：标准的 MySQL CREATE TABLE 语句
        - **JSON 文件** (.json)：结构化元数据定义

        **DDL 格式说明**：
        - 可包含 CREATE DATABASE 语句
        - 可包含 COMMENT 注释
        - 可包含 PRIMARY KEY 定义
        - 多个表语句可在同一文件中

        **JSON 格式说明**：
        - databases: 数据库定义数组
        - tables: 表定义数组（含 columns 字段数组）
        - relationships: 表关系数组（可选）
        - concepts: 业务概念数组（可选）
        """)

        # 文件上传
        uploaded_file = st.file_uploader(
            "上传 DDL 或 JSON 文件",
            type=["sql", "txt", "json"],
            help="选择包含 CREATE TABLE 语句的 SQL 文件或 JSON 元数据文件"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            file_type = st.selectbox("文件类型", ["自动检测", "DDL (.sql)", "JSON (.json)"])
        with col2:
            db_name = st.text_input("数据库名称", value="default_db")
        with col3:
            db_type = st.selectbox("数据库类型", ["mysql", "oracle", "hive", "postgresql", "sparksql"], index=0)

        enable_ai = st.checkbox("启用 AI 增强", value=True,
                               help="使用 LLM 自动推断表关系和生成业务标签")

        if st.button("🚀 开始构建", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.warning("请先上传 DDL 或 JSON 文件")
            else:
                # 检测文件类型
                file_ext = uploaded_file.name.lower().split('.')[-1]
                if file_type == "JSON (.json)" or file_ext == "json":
                    is_json = True
                else:
                    is_json = False

                # 保存上传的文件
                if is_json:
                    temp_path = Path("/tmp/kg_upload.json")
                    temp_path.write_text(uploaded_file.getvalue().decode('utf-8'))
                else:
                    temp_path = Path("/tmp/kg_upload.sql")
                    temp_path.write_text(uploaded_file.getvalue().decode('utf-8'))

                try:
                    with st.spinner("正在构建知识图谱..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # 创建流水线
                        pipeline = GraphBuilderPipeline(
                            neo4j_client=st.session_state.kg_neo4j_client,
                            llm_provider=LLM_PROVIDER,
                            llm_model=LLM_MODEL,
                            llm_config={"api_key": LLM_API_KEY, "base_url": LLM_BASE_URL} if LLM_API_KEY else {},
                            dialect=db_type
                        )

                        status_text.text("阶段 1/4: 抽取元数据...")
                        progress_bar.progress(25)
                        if is_json:
                            result = pipeline.extract_from_json(json_path=str(temp_path))
                        else:
                            result = pipeline.extract_from_files(
                                ddl_path=str(temp_path),
                                database_name=db_name,
                                db_type=db_type
                            )

                        status_text.text(f"阶段 2/4: AI 语义增强...")
                        progress_bar.progress(50)
                        if enable_ai and LLM_API_KEY:
                            pipeline.enrich(enable_concepts=True, enable_relations=True)
                        else:
                            st.info("跳过 AI 增强（未启用或缺少 API Key）")

                        status_text.text("阶段 3/4: 预编译 JOIN SQL...")
                        progress_bar.progress(75)
                        pipeline.compile()

                        status_text.text("阶段 4/4: 装载到 Neo4j...")
                        progress_bar.progress(90)
                        stats = pipeline.load(st.session_state.kg_neo4j_client)
                        progress_bar.progress(100)

                        # 显示结果
                        st.success("✅ 知识图谱构建完成！")

                        # 统计信息
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("数据库", len(pipeline.result.databases) if pipeline.result else 0)
                        with col2:
                            st.metric("表", len(pipeline.result.tables) if pipeline.result else 0)
                        with col3:
                            st.metric("字段", sum(len(t.columns) for t in pipeline.result.tables) if pipeline.result else 0)
                        with col4:
                            st.metric("关系", len(pipeline.result.relationships) if pipeline.result else 0)

                        # 详细信息
                        with st.expander("📋 查看详细统计", expanded=False):
                            st.json({
                                "extract": {
                                    "databases": len(pipeline.result.databases) if pipeline.result else 0,
                                    "tables": len(pipeline.result.tables) if pipeline.result else 0,
                                    "columns": sum(len(t.columns) for t in pipeline.result.tables) if pipeline.result else 0
                                },
                                "load": stats
                            })

                        # 图谱统计
                        st.markdown("#### 🕒 Neo4j 图谱当前状态")
                        st.write(f"- 表节点：{stats.get('Table', 0)}")
                        st.write(f"- 字段节点：{stats.get('Column', 0)}")
                        st.write(f"- 关系边：{stats.get('Relationships', 0)}")

                        # 提示用户重新初始化主应用
                        st.success("""
                        ✅ **构建成功！**

                        请返回主页面，点击侧边栏的 **"🔄 重新初始化系统"** 按钮来刷新检索索引，
                        然后才能使用新的知识库进行 SQL 查询。
                        """)

                except Exception as e:
                    st.error(f"构建失败：{e}")
                finally:
                    if temp_path.exists():
                        temp_path.unlink()

    # ==================== Tab 2: 活体数据库构建 ====================
    with tab2:
        st.markdown("### 💾 从活体数据库构建知识图谱")

        st.warning("""
        ⚠️ **注意事项**
        - 需要提供数据库连接 URI
        - 将读取 INFORMATION_SCHEMA 获取元数据
        - 支持 MySQL、Oracle、SparkSQL
        """)

        col1, col2 = st.columns(2)
        with col1:
            db_type = st.selectbox("数据库类型", ["mysql", "postgresql", "oracle"], index=0)
            schema_name = st.text_input("Schema/数据库名", value="")
        with col2:
            db_host = st.text_input("主机", value="localhost")
            db_port = st.text_input("端口", value="3306")

        db_user = st.text_input("用户名", value="root")
        db_password = st.text_input("密码", type="password", value="")

        enable_ai_db = st.checkbox("启用 AI 增强", value=True, key="ai_db")

        if st.button("🚀 连接并构建", type="primary", use_container_width=True, key="build_db"):
            if not db_host or not db_user:
                st.warning("请填写完整的数据库连接信息")
            else:
                # 构建连接 URI
                if db_type == "mysql":
                    db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{schema_name}"
                else:
                    db_uri = f"{db_type}://{db_user}:{db_password}@{db_host}:{db_port}/{schema_name}"

                try:
                    with st.spinner("正在连接数据库并构建知识图谱..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        pipeline = GraphBuilderPipeline(
                            neo4j_client=st.session_state.kg_neo4j_client,
                            llm_provider=LLM_PROVIDER,
                            llm_model=LLM_MODEL,
                            llm_config={"api_key": LLM_API_KEY, "base_url": LLM_BASE_URL} if LLM_API_KEY else {},
                            dialect=db_type
                        )

                        status_text.text("阶段 1/4: 连接数据库...")
                        progress_bar.progress(25)
                        result = pipeline.extract_from_db(
                            db_uri=db_uri,
                            db_type=db_type,
                            schema=schema_name
                        )

                        status_text.text("阶段 2/4: AI 语义增强...")
                        progress_bar.progress(50)
                        if enable_ai_db and LLM_API_KEY:
                            pipeline.enrich(enable_concepts=True, enable_relations=True)

                        status_text.text("阶段 3/4: 预编译 JOIN SQL...")
                        progress_bar.progress(75)
                        pipeline.compile()

                        status_text.text("阶段 4/4: 装载到 Neo4j...")
                        progress_bar.progress(90)
                        stats = pipeline.load(st.session_state.kg_neo4j_client)
                        progress_bar.progress(100)

                        st.success("✅ 知识图谱构建完成！")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("数据库", stats.get('extract', {}).get('databases', 0))
                        with col2:
                            st.metric("表", stats.get('extract', {}).get('tables', 0))
                        with col3:
                            st.metric("字段", stats.get('extract', {}).get('columns', 0))
                        with col4:
                            st.metric("关系", stats.get('enrichment', {}).get('relationships', 0))

                except Exception as e:
                    st.error(f"构建失败：{e}")

    # ==================== Tab 3: 图谱状态 ====================
    with tab3:
        st.markdown("### 📊 Neo4j 图谱状态")

        if st.session_state.kg_connected and st.session_state.kg_neo4j_client:
            # 刷新按钮
            if st.button("🔄 刷新统计", use_container_width=True):
                st.rerun()

            # 获取图谱统计
            try:
                stats = st.session_state.kg_neo4j_client.execute_query("""
                    MATCH (d:Database)
                    OPTIONAL MATCH (t:Table)
                    OPTIONAL MATCH (c:Column)
                    OPTIONAL MATCH (bc:BusinessConcept)
                    OPTIONAL MATCH ()-[r:CONNECTS]->()
                    RETURN
                      count(distinct d) as databases,
                      count(distinct t) as tables,
                      count(distinct c) as columns,
                      count(distinct bc) as concepts,
                      count(distinct r) as relationships
                """)

                if stats:
                    stat = stats[0]
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("数据库", stat.get('databases', 0))
                    with col2:
                        st.metric("表", stat.get('tables', 0))
                    with col3:
                        st.metric("字段", stat.get('columns', 0))
                    with col4:
                        st.metric("业务概念", stat.get('concepts', 0))
                    with col5:
                        st.metric("关系边", stat.get('relationships', 0))

                    st.divider()

                    # 按标签统计
                    st.markdown("#### 节点分布")
                    labels_query = """
                        MATCH (n)
                        RETURN labels(n)[0] as label, count(n) as count
                        ORDER BY count DESC
                    """
                    labels_result = st.session_state.kg_neo4j_client.execute_query(labels_query)
                    if labels_result:
                        for row in labels_result:
                            st.write(f"- **{row['label']}**: {row['count']}")

                    # 最近更新的表
                    st.divider()
                    st.markdown("#### 最近的表")
                    recent_tables = st.session_state.kg_neo4j_client.execute_query("""
                        MATCH (t:Table)
                        RETURN t.name as name, t.description as description, t.database as database
                        ORDER BY t.updated_at DESC
                        LIMIT 10
                    """)
                    if recent_tables:
                        for row in recent_tables:
                            st.write(f"- `{row['database']}.{row['name']}` - {row['description'] or '无描述'}")

            except Exception as e:
                st.error(f"查询失败：{e}")
        else:
            st.warning("⚠️ Neo4j 未连接，请先在侧边栏配置连接")

    # ==================== 帮助信息 ====================
    with st.expander("❓ 使用帮助"):
        st.markdown("""
        ### 知识图谱构建模块说明

        #### 数据来源
        1. **DDL 文件**: 上传包含 CREATE TABLE 语句的 SQL 文件
        2. **活体数据库**: 直接连接 MySQL/Oracle/Spark 数据库

        #### AI 增强功能
        - **业务标签**: LLM 自动分析表结构，生成 3-5 个业务标签
        - **关系推断**: LLM 根据字段命名推断潜在的表关联关系
        - **JOIN 预编译**: 自动生成标准化的 JOIN SQL 片段

        #### Neo4j 图谱结构
        ```
        (Database)-[:HAS_TABLE]->(Table)-[:HAS_COLUMN]->(Column)
        (BusinessConcept)-[:MAPPED_TO]->(Table)
        (Table)-[:CONNECTS]->(Table)
        ```

        #### 幂等性保证
        使用 `UNWIND + MERGE` 模式，确保重复执行不会创建重复节点。
        """)


def show_kg_builder():
    """显示知识图谱构建界面（兼容旧调用）"""
    render_graph_builder()


if __name__ == "__main__":
    render_graph_builder()
