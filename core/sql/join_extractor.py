"""
JOIN 提取引擎
使用 sqlglot AST 解析从 SQL 中提取表关联关系
"""
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import sqlglot
    from sqlglot import exp, parse, ParseError
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    logger.warning("sqlglot 未安装，JOIN 提取功能将不可用。安装：pip install sqlglot")


# JOIN 类型映射：sqlglot side -> SQL JOIN 字符串
JOIN_SIDE_MAP = {
    "LEFT": "LEFT JOIN",
    "RIGHT": "RIGHT JOIN",
    "FULL": "FULL OUTER JOIN",
    "INNER": "INNER JOIN",
    "CROSS": "CROSS JOIN",
    None: "INNER JOIN",  # 默认无 side 为 INNER JOIN
}


@dataclass
class JoinIR:
    """JOIN 关系中间表示"""
    left_table: str
    right_table: str
    join_condition: str       # e.g. "orders.customer_id = customers.id"
    join_type: str            # e.g. "LEFT JOIN"
    from_column: Optional[str] = None   # e.g. "customer_id"
    to_column: Optional[str] = None     # e.g. "id"
    extra_condition: Optional[str] = None  # AND 后面的额外条件


def extract_joins_from_sql(sql: str, dialect: str = "mysql") -> List[JoinIR]:
    """
    从 SQL 中提取 JOIN 关系（使用 sqlglot AST 解析）

    Args:
        sql: SQL 语句
        dialect: SQL 方言 (mysql, postgresql, oracle, sparksql 等)

    Returns:
        List[JoinIR]: 提取的 JOIN 关系列表
    """
    if not SQLGLOT_AVAILABLE:
        logger.warning("sqlglot 不可用，无法提取 JOIN 关系")
        return []

    results = []

    try:
        parsed = parse(sql, read=dialect)

        for statement in parsed:
            if statement is None:
                continue

            # 获取 FROM 子句中的基表
            from_clause = statement.find(exp.From)
            if not from_clause:
                continue

            # 收集所有表别名映射 (alias -> table_name)
            table_aliases = {}
            base_table_name = _get_table_name_from_node(from_clause.this)
            if base_table_name:
                base_alias = _get_alias(from_clause.this)
                if base_alias:
                    table_aliases[base_alias.lower()] = base_table_name

            # 提取 JOIN 节点
            for join_node in statement.find_all(exp.Join):
                join_info = _extract_single_join(join_node, base_table_name, table_aliases)
                if join_info:
                    results.append(join_info)
                    # 更新别名映射
                    table_aliases[join_info.right_table.lower()] = join_info.right_table
                    join_alias = _get_alias(join_node.this)
                    if join_alias:
                        table_aliases[join_alias.lower()] = join_info.right_table

    except ParseError as e:
        logger.warning(f"SQL 解析失败，无法提取 JOIN: {e}")
    except Exception as e:
        logger.error(f"JOIN 提取失败: {e}")

    logger.info(f"从 SQL 中提取到 {len(results)} 个 JOIN 关系")
    return results


def _extract_single_join(join_node: exp.Join, base_table: Optional[str],
                         aliases: dict) -> Optional[JoinIR]:
    """
    从单个 Join 节点提取关系信息

    Args:
        join_node: sqlglot Join 节点
        base_table: 基表名称
        aliases: 表别名映射

    Returns:
        JoinIR 或 None
    """
    # 获取右表（JOIN 的表）
    right_table_name = _get_table_name_from_node(join_node.this)
    if not right_table_name:
        return None

    # 获取 JOIN 类型
    join_type = JOIN_SIDE_MAP.get(join_node.side, "INNER JOIN")

    # 获取 ON 条件
    on_condition = join_node.args.get("on")
    if not on_condition:
        # 没有 ON 条件（CROSS JOIN 或自然连接）
        return JoinIR(
            left_table=base_table or "unknown",
            right_table=right_table_name,
            join_condition="",
            join_type=join_type
        )

    # 解析 ON 条件
    condition_str = on_condition.sql(dialect="mysql")
    from_column, to_column, extra_condition = _parse_on_condition(on_condition)

    # 解析左表和右表的字段
    # ON 条件通常是 left_table.col = right_table.col 形式
    left_table = base_table or "unknown"

    # 尝试从字段前缀推断实际表名
    if from_column:
        # 字段可能带表前缀或别名
        parts = from_column.split(".")
        if len(parts) == 2:
            prefix = parts[0].lower()
            from_column = parts[1]
            # 前缀可能是别名
            left_table = aliases.get(prefix, prefix)

    if to_column:
        parts = to_column.split(".")
        if len(parts) == 2:
            prefix = parts[0].lower()
            to_column = parts[1]
            right_table_name = aliases.get(prefix, right_table_name)

    return JoinIR(
        left_table=left_table,
        right_table=right_table_name,
        join_condition=condition_str,
        join_type=join_type,
        from_column=from_column,
        to_column=to_column,
        extra_condition=extra_condition
    )


def _parse_on_condition(on_expr) -> tuple:
    """
    解析 ON 条件表达式，提取关联字段和额外条件

    返回: (left_column, right_column, extra_condition)
    例: ON a.id = b.a_id AND b.status = 'active'
        -> ("a.id", "b.a_id", "b.status = 'active'")
    """
    if not on_expr:
        return (None, None, None)

    left_col = None
    right_col = None
    extra_parts = []

    # 处理 AND 连接的多条件
    conditions = _split_and_conditions(on_expr)

    for cond in conditions:
        # 查找等式条件 (col = col)
        if isinstance(cond, exp.EQ):
            left = cond.this
            right = cond.expression

            left_str = _extract_column_ref(left)
            right_str = _extract_column_ref(right)

            if left_str and right_str:
                # 第一个等式作为主连接条件
                if left_col is None:
                    left_col = left_str
                    right_col = right_str
                else:
                    extra_parts.append(cond.sql(dialect="mysql"))
            else:
                # 等式但一边不是列引用（如 col = 'value'），作为额外条件
                extra_parts.append(cond.sql(dialect="mysql"))
        else:
            # 非等式条件作为额外条件
            extra_parts.append(cond.sql(dialect="mysql"))

    extra_condition = " AND ".join(extra_parts) if extra_parts else None
    return (left_col, right_col, extra_condition)


def _split_and_conditions(expr) -> list:
    """
    将 AND 连接的表达式拆分为条件列表
    """
    if isinstance(expr, exp.And):
        result = []
        # 递归拆分 AND
        _flatten_and(expr, result)
        return result
    return [expr]


def _flatten_and(node, result: list):
    """递归展开 AND 树"""
    if isinstance(node, exp.And):
        _flatten_and(node.this, result)
        _flatten_and(node.expression, result)
    else:
        result.append(node)


def _extract_column_ref(node) -> Optional[str]:
    """
    从表达式节点提取列引用字符串
    """
    if isinstance(node, exp.Column):
        parts = []
        if node.table:
            parts.append(node.table)
        if node.name:
            parts.append(node.name)
        return ".".join(parts) if parts else None
    elif isinstance(node, exp.Identifier):
        return node.this
    else:
        return None


def _get_table_name_from_node(node) -> Optional[str]:
    """
    从 sqlglot 表节点提取表名
    """
    if node is None:
        return None

    if isinstance(node, exp.Table):
        return node.name

    # 处理带别名的情况
    if isinstance(node, exp.Alias):
        return _get_table_name_from_node(node.this)

    return None


def _get_alias(node) -> Optional[str]:
    """
    从表节点获取别名
    """
    if node is None:
        return None

    if isinstance(node, exp.Alias):
        return node.alias

    if isinstance(node, exp.Table):
        return node.alias

    return None
