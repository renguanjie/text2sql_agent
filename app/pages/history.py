"""
执行历史页面
查看和管理 SQL 生成历史
"""
import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from core.history.mysql_client import MySQLClient

# 页面配置
st.set_page_config(
    page_title="执行历史 - Text2SQL",
    page_icon="📜",
    layout="wide"
)

st.title("📜 SQL 生成历史")

# ==================== 数据库连接 ====================
@st.cache_resource
def get_mysql_client():
    """获取 MySQL 客户端"""
    client = MySQLClient(MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE)
    client.connect()
    return client


# ==================== 侧边栏过滤器 ====================
with st.sidebar:
    st.header("过滤器")

    # 会话 ID 过滤
    session_filter = st.text_input("会话 ID", placeholder="输入会话 ID 过滤")

    # 状态过滤
    status_options = ["全部", "success", "fail", "pending"]
    status_filter = st.selectbox("执行状态", status_options)

    # 每页数量
    page_size = st.selectbox("每页数量", [10, 20, 50, 100], index=1)

    # 刷新按钮
    if st.button("🔄 刷新"):
        st.rerun()

# ==================== 获取数据 ====================
mysql_client = get_mysql_client()

# 构建查询条件
conditions = []
params = []

if session_filter:
    conditions.append("session_id = %s")
    params.append(session_filter)

if status_filter != "全部":
    conditions.append("execution_status = %s")
    params.append(status_filter)

where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

# 分页
page = st.number_input("页码", min_value=1, value=1)
offset = (page - 1) * page_size

# 查询历史
query = f"""
SELECT id, session_id, user_query, generated_sql, validation_status,
       execution_status, execution_error, feedback_score, created_at
FROM sql_generation_history
{where_clause}
ORDER BY created_at DESC
LIMIT %s OFFSET %s
"""
params.extend([page_size, offset])

try:
    history_list = mysql_client.execute_query(query, tuple(params))
except Exception as e:
    st.error(f"查询失败：{e}")
    history_list = []

# 获取总数
count_query = f"""
SELECT COUNT(*) as total FROM sql_generation_history {where_clause}
"""
try:
    total_result = mysql_client.execute_query(count_query, tuple(params[:len(params)-2]))
    total = total_result[0].get('total', 0) if total_result else 0
except:
    total = 0

# ==================== 统计卡片 ====================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("总记录数", total)

with col2:
    success_count = sum(1 for h in history_list if h.get('execution_status') == 'success')
    st.metric("成功", success_count)

with col3:
    fail_count = sum(1 for h in history_list if h.get('execution_status') == 'fail')
    st.metric("失败", fail_count)

with col4:
    pending_count = sum(1 for h in history_list if h.get('execution_status') == 'pending')
    st.metric("待执行", pending_count)

# ==================== 历史列表 ====================
st.divider()

if history_list:
    for record in history_list:
        # 状态颜色
        status = record.get('execution_status', 'pending')
        if status == 'success':
            status_color = "🟢"
        elif status == 'fail':
            status_color = "🔴"
        else:
            status_color = "🟡"

        with st.expander(f"{status_color} {record.get('user_query', '')[:80]}...", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**查询:** {record.get('user_query', '')}")
                st.markdown(f"**会话 ID:** `{record.get('session_id', '')[:8]}...`")
                st.markdown(f"**时间:** {record.get('created_at', '')}")

            with col2:
                st.markdown(f"**验证状态:** `{record.get('validation_status', '')}`")
                st.markdown(f"**执行状态:** `{record.get('execution_status', '')}`")
                if record.get('feedback_score'):
                    st.markdown(f"**评分:** {'⭐' * record['feedback_score']}")

            # SQL
            st.subheader("SQL")
            if record.get('generated_sql'):
                st.code(record['generated_sql'], language='sql')

            # 错误信息
            if record.get('execution_error'):
                st.error(f"错误：{record['execution_error']}")

            # 操作按钮
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📋 复制 SQL", key=f"copy_{record['id']}"):
                    st.write("请手动复制上方代码块")
            with col2:
                if st.button("🔄 重新执行", key=f"retry_{record['id']}"):
                    st.session_state.last_sql = record.get('generated_sql')
                    st.switch_page("app.py")
            with col3:
                # 反馈
                feedback_cols = st.columns(5)
                for i, col in enumerate(feedback_cols, 1):
                    with col:
                        if st.button(f"{i}⭐", key=f"rate_{record['id']}_{i}"):
                            mysql_client.save_feedback(
                                history_id=record['id'],
                                feedback_type='correct' if i >= 3 else 'incorrect',
                                feedback_score=i
                            )
                            st.success("已提交反馈")
                            st.rerun()
else:
    st.info("暂无历史记录")

# ==================== 分页控件 ====================
if total > page_size:
    total_pages = (total + page_size - 1) // page_size

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ 上一页", disabled=(page <= 1)):
            st.query_params["page"] = page - 1
            st.rerun()
    with col2:
        st.write(f"第 {page} 页 / 共 {total_pages} 页")
    with col3:
        if st.button("下一页 ➡️", disabled=(page >= total_pages)):
            st.query_params["page"] = page + 1
            st.rerun()

# ==================== 快捷操作 ====================
st.divider()
st.subheader("快捷操作")

col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ 清空所有历史", type="secondary"):
        if st.confirm("确定要清空所有历史记录吗？此操作不可恢复。"):
            mysql_client.execute_update("DELETE FROM sql_generation_history")
            st.success("已清空所有历史")
            st.rerun()

with col2:
    if st.button("🗑️ 清空失败记录", type="secondary"):
        if st.confirm("确定要清空所有失败记录吗？"):
            mysql_client.execute_update("DELETE FROM sql_generation_history WHERE execution_status = 'fail'")
            st.success("已清空失败记录")
            st.rerun()
