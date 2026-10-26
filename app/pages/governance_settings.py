"""
语义治理与执行预检设置面板
把 ApplicationContext.reload_governance 支持的 18 项治理配置接到前端，
使正确性内核可以热开关，而不必修改 .env 后重启。
"""
# ruff: noqa: E402
from app.ui_icons import icon, icon_header, icon_line, inject_icon_styles
import streamlit as st
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from config import (
    GOVERNED_HISTORY_BINDING_TTL_SECONDS,
    MAX_PREPARED_REQUESTS,
    PREPARED_REQUEST_TTL_SECONDS,
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
)
from core.app_context import ApplicationConfigurationError
from core.utils.governance_settings import (
    EDITABLE_KEYS,
    active_semantic_version,
    describe_status,
    diff_updates,
    format_blocked_tables,
    parse_blocked_tables,
    validate_updates,
)
from core.utils.log_sanitizer import log_safe_exception

# 整数控件需要显式转换的键（st.number_input 可能给出 float）
_INT_WIDGET_KEYS = (
    "semantic_context_max_chars",
    "semantic_context_max_items",
    "sql_explain_timeout_ms",
    "sql_explain_max_estimated_rows",
    "sql_query_max_rows",
    "sql_query_timeout_ms",
    "sql_max_length",
    "max_prepared_requests",
)
_FLOAT_WIDGET_KEYS = (
    "prepared_request_ttl_seconds",
    "governed_history_binding_ttl_seconds",
)

# 纠错次数上限刻意用自描述字符串而非 format_func：Streamlit 的 AppTest 会以
# str(value) 反查 options，格式化标签与原始值不一致时无法渲染测试。
_REPAIR_ATTEMPT_CHOICES = ("0 — 校验失败即拒绝", "1 — 允许一次纠错")


def repair_choice_index(attempts: int) -> int:
    """纠错次数上限 → 选项下标。"""
    return 0 if int(attempts) == 0 else 1


def repair_choice_value(choice: str) -> int:
    """选项文本 → 纠错次数上限。"""
    return int(str(choice).split(" ", 1)[0])


def governance_defaults() -> dict:
    """`.env` 中的治理配置，作为面板初值与「恢复默认」的目标值。"""
    return {
        'semantic_governance_enabled': SEMANTIC_GOVERNANCE_ENABLED,
        'semantic_model_dir': SEMANTIC_MODEL_DIR,
        'semantic_datasource': SEMANTIC_DATASOURCE,
        'semantic_context_max_chars': SEMANTIC_CONTEXT_MAX_CHARS,
        'semantic_context_max_items': SEMANTIC_CONTEXT_MAX_ITEMS,
        'sql_repair_max_attempts': SQL_REPAIR_MAX_ATTEMPTS,
        'sql_explain_enabled': SQL_EXPLAIN_ENABLED,
        'sql_explain_timeout_ms': SQL_EXPLAIN_TIMEOUT_MS,
        'sql_explain_max_estimated_rows': SQL_EXPLAIN_MAX_ESTIMATED_ROWS,
        'sql_dialect': SQL_DIALECT,
        'sql_max_length': SQL_MAX_LENGTH,
        'sql_query_max_rows': SQL_QUERY_MAX_ROWS,
        'sql_query_timeout_ms': SQL_QUERY_TIMEOUT_MS,
        'sql_scope_enforcement': SQL_SCOPE_ENFORCEMENT,
        'sql_blocked_tables': list(SQL_BLOCKED_TABLES),
        'max_prepared_requests': MAX_PREPARED_REQUESTS,
        'prepared_request_ttl_seconds': PREPARED_REQUEST_TTL_SECONDS,
        'governed_history_binding_ttl_seconds': GOVERNED_HISTORY_BINDING_TTL_SECONDS,
    }


def current_governance_config(ctx=None) -> dict:
    """当前生效的治理配置；`ctx.config` 是原子发布后的只读快照。"""
    if ctx is None:
        ctx = st.session_state.get("app_context")
    merged = governance_defaults()
    if ctx is not None:
        try:
            live = ctx.config
        except Exception as e:
            log_safe_exception("governance_config_read", e)
            live = {}
        merged.update({k: v for k, v in live.items() if k in merged})
    return merged


def apply_governance_config(updates: dict) -> tuple[bool, str]:
    """把治理配置变更交给 `reload_governance` 事务式重载。"""
    ctx = st.session_state.get("app_context")
    if not ctx:
        return False, "系统未初始化，无法应用治理配置"
    if not updates:
        return True, "配置未发生变化"
    try:
        with st.spinner("正在重新加载语义治理配置..."):
            ctx.reload_governance(**updates)
    except ApplicationConfigurationError as e:
        # 该异常的消息按设计只含配置键名、不含取值，可安全展示
        log_safe_exception("governance_config_reload", e)
        return False, f"配置未生效：{e}"
    except Exception as e:
        log_safe_exception("governance_config_reload", e)
        return False, "配置未生效，详情请查看 logs/app.log 中的错误编号"
    logger.info(f"语义治理配置已更新：{sorted(updates)}")
    return True, f"已应用 {len(updates)} 项治理配置变更"


def _collect_desired() -> dict:
    """从控件状态收集期望配置，只保留白名单内的键。"""
    desired = {
        key[len("cfg_gov_"):]: value
        for key, value in st.session_state.items()
        if key.startswith("cfg_gov_") and key[len("cfg_gov_"):] in EDITABLE_KEYS
    }
    if "sql_blocked_tables" in desired:
        desired["sql_blocked_tables"] = parse_blocked_tables(
            desired["sql_blocked_tables"]
        )
    if "sql_repair_max_attempts" in desired:
        desired["sql_repair_max_attempts"] = repair_choice_value(
            desired["sql_repair_max_attempts"]
        )
    for key in _INT_WIDGET_KEYS:
        if key in desired:
            desired[key] = int(desired[key])
    for key in _FLOAT_WIDGET_KEYS:
        if key in desired:
            desired[key] = float(desired[key])
    if "semantic_datasource" in desired:
        desired["semantic_datasource"] = str(desired["semantic_datasource"]).strip()
    return desired


def _submit(current: dict, desired: dict, ctx=None) -> None:
    """前置校验 → 只提交变化项 → 就地反馈结果。

    刻意不调用 `st.rerun()`：重跑会让控件状态与结果反馈分离，且 Streamlit 的
    AppTest 会因按钮状态被重复投递而无限重跑，使这条链路无法被测试。
    """
    errors = validate_updates(current, desired)
    if errors:
        st.error("；".join(errors))
        return
    ok, message = apply_governance_config(diff_updates(current, desired))
    (st.success if ok else st.error)(message)
    if ok:
        # 控件在下一轮渲染前清掉，使其从新的 ctx.config 重新播种
        st.session_state["_gov_reseed"] = True
        refreshed = current_governance_config(ctx)
        st.caption(
            "当前生效："
            + describe_status(
                refreshed,
                active_semantic_version(getattr(ctx, "semantic_registry", None)),
            )
        )


def show_governance_settings() -> None:
    """渲染语义治理与执行预检设置面板。"""
    st.markdown(icon_header("语义治理与执行预检", "shield-check", level=3), unsafe_allow_html=True)

    ctx = st.session_state.get("app_context")
    current = current_governance_config(ctx)

    # 必须在实例化控件之前清理，否则会触发 Streamlit 的控件状态保护
    if st.session_state.pop("_gov_reseed", False):
        for key in list(st.session_state.keys()):
            if key.startswith("cfg_gov_"):
                st.session_state.pop(key, None)

    semantic_version = active_semantic_version(
        getattr(ctx, "semantic_registry", None)
    )
    st.caption(describe_status(current, semantic_version))
    st.caption(
        "开关变更会事务式重建治理组件；语义快照由 "
        f"`{current['semantic_model_dir']}/{current['semantic_datasource']}.yaml` "
        "自动编译，校验失败时保持原有配置不变。"
    )

    # ── 语义治理 ──
    with st.expander("── 语义治理 ──", expanded=False):
        gov_enabled = st.toggle(
            "启用语义治理",
            value=bool(current['semantic_governance_enabled']),
            key="cfg_gov_semantic_governance_enabled",
            help="开启后执行前做表/字段/关联/指标/时间维度/策略的确定性校验",
        )
        st.text_input(
            "数据源",
            value=str(current['semantic_datasource']),
            key="cfg_gov_semantic_datasource",
            help="对应 semantic_models/<数据源>.yaml；仅允许字母、数字、下划线和连字符",
        )
        col_chars, col_items = st.columns(2)
        with col_chars:
            st.number_input(
                "语义上下文最大字符",
                min_value=1, max_value=100_000,
                value=int(current['semantic_context_max_chars']),
                key="cfg_gov_semantic_context_max_chars",
            )
        with col_items:
            st.number_input(
                "语义上下文最大条目",
                min_value=1, max_value=10_000,
                value=int(current['semantic_context_max_items']),
                key="cfg_gov_semantic_context_max_items",
            )
        st.radio(
            "模型纠错次数上限",
            options=list(_REPAIR_ATTEMPT_CHOICES),
            index=repair_choice_index(int(current['sql_repair_max_attempts'])),
            key="cfg_gov_sql_repair_max_attempts",
            horizontal=True,
            help="设计上限恒为 1；纠错输入只含结构化错误码与最小语义切片",
        )
        if gov_enabled:
            if semantic_version:
                st.info(f"激活语义版本：{semantic_version}")
            else:
                st.info("语义快照尚未加载，保存时将编译并激活")

    # ── MySQL EXPLAIN 预检 ──
    with st.expander("── MySQL EXPLAIN 预检 ──", expanded=False):
        st.toggle(
            "启用 EXPLAIN 预检",
            value=bool(current['sql_explain_enabled']),
            key="cfg_gov_sql_explain_enabled",
            disabled=not gov_enabled,
            help="需先启用语义治理；预检与正式查询共用只读连接与超时上限",
        )
        if not gov_enabled:
            st.caption("语义治理关闭时不可启用预检")
        col_explain_timeout, col_explain_rows = st.columns(2)
        with col_explain_timeout:
            st.number_input(
                "预检超时 (毫秒)",
                min_value=1, max_value=3_600_000,
                value=int(current['sql_explain_timeout_ms']),
                key="cfg_gov_sql_explain_timeout_ms",
            )
        with col_explain_rows:
            st.number_input(
                "估算扫描行数上限",
                min_value=0, max_value=1_000_000_000,
                value=int(current['sql_explain_max_estimated_rows']),
                key="cfg_gov_sql_explain_max_estimated_rows",
                help="0 表示不按估算行数拒绝",
            )

    # ── 执行范围与安全 ──
    with st.expander("── 执行范围与安全 ──", expanded=False):
        st.caption(
            f"当前 SQL 方言：`{current['sql_dialect']}`"
            "（语义治理仅支持 mysql，如需修改请编辑 .env 后重启）"
        )
        col_rows, col_timeout = st.columns(2)
        with col_rows:
            st.number_input(
                "最大返回行数",
                min_value=1, max_value=100_000,
                value=int(current['sql_query_max_rows']),
                key="cfg_gov_sql_query_max_rows",
            )
        with col_timeout:
            st.number_input(
                "查询超时 (毫秒)",
                min_value=1, max_value=3_600_000,
                value=int(current['sql_query_timeout_ms']),
                key="cfg_gov_sql_query_timeout_ms",
            )
        st.number_input(
            "SQL 最大长度",
            min_value=1, max_value=100_000,
            value=int(current['sql_max_length']),
            key="cfg_gov_sql_max_length",
        )
        st.toggle(
            "强制查询范围校验",
            value=bool(current['sql_scope_enforcement']),
            key="cfg_gov_sql_scope_enforcement",
            help="关闭后不再拦截越权表访问，仅建议在隔离环境调试时使用",
        )
        st.text_area(
            "禁用表（逗号或换行分隔）",
            value=format_blocked_tables(current['sql_blocked_tables']),
            key="cfg_gov_sql_blocked_tables",
            height=88,
            help="这些表即使用户提问命中也会被 POLICY_DENIED 拒绝",
        )

    # ── 高级：进程内缓存与 TTL ──
    with st.expander("── 高级（进程内缓存与 TTL）──", expanded=False):
        st.number_input(
            "已预备请求上限",
            min_value=1, max_value=4_096,
            value=int(current['max_prepared_requests']),
            key="cfg_gov_max_prepared_requests",
        )
        col_prep, col_hist = st.columns(2)
        with col_prep:
            st.number_input(
                "已预备请求存活时间 (秒)",
                min_value=0.0, max_value=3_600.0, step=1.0,
                value=float(current['prepared_request_ttl_seconds']),
                key="cfg_gov_prepared_request_ttl_seconds",
            )
        with col_hist:
            st.number_input(
                "治理历史绑定存活时间 (秒)",
                min_value=0.0, max_value=3_600.0, step=1.0,
                value=float(current['governed_history_binding_ttl_seconds']),
                key="cfg_gov_governed_history_binding_ttl_seconds",
            )

    col_save, col_reset = st.columns([1, 1])
    with col_save:
        if st.button("保存并重新加载治理配置", use_container_width=True,
                     type="primary", key="btn_save_gov"):
            _submit(current, _collect_desired(), ctx)

    with col_reset:
        if st.button("↩恢复 .env 默认治理配置", use_container_width=True,
                     key="btn_reset_gov"):
            _submit(current, governance_defaults(), ctx)
