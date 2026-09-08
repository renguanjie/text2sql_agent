"""
SQL 校验器模块
使用 SQLFluff 进行 SQL 语法校验和格式化
"""
from typing import Dict, List, Optional, Any
import sqlfluff
from sqlfluff.core import Linter
from sqlfluff.core.errors import SQLParseError
from loguru import logger


SYSTEM_SCHEMAS = {"information_schema", "mysql", "performance_schema", "sys", "pg_catalog"}
DEFAULT_BLOCKED_TABLES = {
    "sql_generation_history", "user_feedback", "table_schema", "column_schema"
}


def validate_query_scope(sql: str, allowed_tables: Optional[List[str]] = None,
                         allowed_schemas: Optional[List[str]] = None,
                         blocked_tables: Optional[List[str]] = None,
                         dialect: str = "mysql") -> Dict[str, Any]:
    """使用 AST 校验查询引用的库表，正确忽略 CTE 别名。"""
    try:
        import sqlglot
        from sqlglot import exp

        statement = sqlglot.parse_one(sql, read=dialect)
    except Exception as exc:
        return {'valid': False, 'errors': [f'SQL 无法解析：{exc}'], 'tables': []}

    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    allowed_table_set = {
        str(name).lower().split('.')[-1] for name in (allowed_tables or []) if name
    }
    allowed_schema_set = {str(name).lower() for name in (allowed_schemas or []) if name}
    blocked_table_set = {
        str(name).lower() for name in (
            DEFAULT_BLOCKED_TABLES if blocked_tables is None else blocked_tables
        ) if name
    }
    errors = []
    tables = []

    for table in statement.find_all(exp.Table):
        table_name = table.name.lower()
        schema_name = (table.db or table.catalog or "").lower()
        if not table_name or (not schema_name and table_name in cte_names):
            continue
        qualified_name = f"{schema_name}.{table_name}" if schema_name else table_name
        tables.append(qualified_name)
        if table_name in blocked_table_set:
            errors.append(f"禁止访问内部表：{qualified_name}")
        if schema_name in SYSTEM_SCHEMAS:
            errors.append(f"禁止访问系统 Schema：{schema_name}")
        elif schema_name and allowed_schema_set and schema_name not in allowed_schema_set:
            errors.append(f"Schema 不在允许范围内：{schema_name}")
        if allowed_tables is not None and table_name not in allowed_table_set:
            errors.append(f"表不在允许范围内：{qualified_name}")

    return {'valid': not errors, 'errors': list(dict.fromkeys(errors)), 'tables': tables}


def constrain_read_query(sql: str, max_rows: int, timeout_ms: int = 0,
                         dialect: str = "mysql") -> str:
    """通过 AST 强制外层行数限制，并为 MySQL 添加执行超时提示。"""
    import sqlglot
    from sqlglot import exp

    statement = sqlglot.parse_one(sql, read=dialect)
    if max_rows > 0:
        statement = statement.limit(max_rows + 1)
    if dialect == "mysql" and timeout_ms > 0 and isinstance(statement, exp.Select):
        statement.set(
            "hint",
            exp.Hint(expressions=[exp.Var(this=f"MAX_EXECUTION_TIME({int(timeout_ms)})")]),
        )
    return statement.sql(dialect=dialect)


def is_safe_read_query(sql: str, dialect: str = "mysql") -> bool:
    """仅接受一条无锁、无副作用的 SELECT/集合查询。

    解析器缺失或 SQL 无法解析时采用 fail-closed 策略，拒绝执行。
    """
    if not isinstance(sql, str) or not sql.strip():
        return False

    try:
        import sqlglot

        statements = [statement for statement in sqlglot.parse(sql, read=dialect) if statement is not None]
    except Exception as exc:
        logger.warning(f"SQL 安全解析失败，拒绝执行：{exc}")
        return False

    if len(statements) != 1:
        return False

    statement = statements[0]
    if type(statement).__name__ not in {"Select", "Union", "Intersect", "Except"}:
        return False

    forbidden_nodes = {
        "Alter", "Command", "Commit", "Create", "Delete", "Drop", "Grant",
        "Insert", "Into", "LoadData", "Lock", "Merge", "PropertyEQ", "Rollback", "Set",
        "Transaction", "TruncateTable", "Update", "Use",
    }
    forbidden_functions = {
        "BENCHMARK", "GET_LOCK", "LOAD_FILE", "PG_SLEEP", "RELEASE_LOCK", "SLEEP"
    }
    for item in statement.walk():
        # sqlglot 旧版本返回 (node, parent, key)，新版本直接返回 node。
        node = item[0] if isinstance(item, tuple) else item
        if type(node).__name__ in forbidden_nodes:
            return False
        if type(node).__name__ == "Anonymous" and node.name.upper() in forbidden_functions:
            return False
    return True


class SQLValidator:
    """SQL 校验器 - 使用 SQLFluff"""

    def __init__(self, dialect: str = "mysql", rules: Optional[List[str]] = None):
        """
        初始化 SQL 校验器

        Args:
            dialect: SQL 方言
            rules: 启用的规则列表
        """
        self.dialect = dialect
        self.rules = rules
        self.linter = Linter(dialect=dialect, rules=rules)

    def validate(self, sql: str) -> Dict[str, Any]:
        """
        校验 SQL

        Args:
            sql: SQL 语句

        Returns:
            Dict: 校验结果 {valid: bool, errors: List, warnings: List}
        """
        if not sql or not sql.strip():
            return {
                'valid': False,
                'errors': ['SQL 语句为空'],
                'warnings': []
            }

        try:
            # 使用 sqlfluff lint
            result = sqlfluff.lint(sql, dialect=self.dialect)

            errors = []
            warnings = []

            for violation in result:
                code = violation.get('code', '')
                description = violation.get('description', '')
                line = violation.get('line_no', '?')
                pos = violation.get('line_pos', '?')

                message = f"L{line}:P{pos} [{code}] {description}"

                # 区分错误和警告
                if code.startswith(('PRS', 'LXR')):
                    # 解析错误、词法错误 - 严重错误
                    errors.append(message)
                elif code.startswith(('LT', 'CV')):
                    # 布局问题、约定问题 - 警告
                    warnings.append(message)
                else:
                    # 其他问题
                    warnings.append(message)

            is_valid = len(errors) == 0

            logger.info(f"SQL 校验完成：valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}")

            return {
                'valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'raw_violations': result
            }

        except SQLParseError as e:
            logger.error(f"SQL 解析错误：{e}")
            return {
                'valid': False,
                'errors': [f'解析错误：{str(e)}'],
                'warnings': []
            }
        except Exception as e:
            logger.error(f"SQL 校验失败：{e}")
            return {
                'valid': False,
                'errors': [f'校验失败：{str(e)}'],
                'warnings': []
            }

    def format(self, sql: str) -> Dict[str, Any]:
        """
        格式化 SQL

        Args:
            sql: SQL 语句

        Returns:
            Dict: 格式化结果 {formatted_sql: str, success: bool, error: str}
        """
        if not sql or not sql.strip():
            return {
                'formatted_sql': '',
                'success': False,
                'error': 'SQL 语句为空'
            }

        try:
            formatted = sqlfluff.format(sql, dialect=self.dialect)
            return {
                'formatted_sql': formatted,
                'success': True,
                'error': None
            }
        except Exception as e:
            logger.error(f"SQL 格式化失败：{e}")
            return {
                'formatted_sql': sql,  # 返回原 SQL
                'success': False,
                'error': str(e)
            }

    def fix(self, sql: str) -> Dict[str, Any]:
        """
        自动修复 SQL（修复可自动修复的问题）

        Args:
            sql: SQL 语句

        Returns:
            Dict: 修复结果 {fixed_sql: str, success: bool, errors: List}
        """
        if not sql or not sql.strip():
            return {
                'fixed_sql': '',
                'success': False,
                'errors': ['SQL 语句为空']
            }

        try:
            # 使用 sqlfluff fix
            fixed = sqlfluff.fix(sql, dialect=self.dialect, fix_even_unparsable=False)
            return {
                'fixed_sql': fixed,
                'success': True,
                'errors': []
            }
        except Exception as e:
            logger.error(f"SQL 自动修复失败：{e}")
            return {
                'fixed_sql': sql,  # 返回原 SQL
                'success': False,
                'errors': [str(e)]
            }

    def parse(self, sql: str) -> Dict[str, Any]:
        """
        解析 SQL 结构

        Args:
            sql: SQL 语句

        Returns:
            Dict: 解析结果
        """
        if not sql or not sql.strip():
            return {
                'success': False,
                'error': 'SQL 语句为空',
                'tree': None
            }

        try:
            parsed = sqlfluff.parse(sql, dialect=self.dialect)

            if parsed:
                return {
                    'success': True,
                    'error': None,
                    'tree': self._tree_to_dict(parsed),
                    'raw_tree': parsed
                }
            else:
                return {
                    'success': False,
                    'error': '解析结果为空',
                    'tree': None
                }

        except SQLParseError as e:
            return {
                'success': False,
                'error': f'解析错误：{str(e)}',
                'tree': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'解析失败：{str(e)}',
                'tree': None
            }

    def _tree_to_dict(self, tree) -> Dict:
        """将解析树转换为字典（简化版）"""
        if tree is None:
            return None

        result = {
            'type': type(tree).__name__,
            'raw': getattr(tree, 'raw', None),
            'children': []
        }

        if hasattr(tree, 'segments'):
            for child in tree.segments:
                child_dict = self._tree_to_dict(child)
                if child_dict:
                    result['children'].append(child_dict)

        return result

    def get_violation_types(self, sql: str) -> Dict[str, int]:
        """
        获取违规类型统计

        Args:
            sql: SQL 语句

        Returns:
            Dict: 类型计数字典
        """
        result = sqlfluff.lint(sql, dialect=self.dialect)

        type_counts = {}
        for violation in result:
            code = violation.get('code', 'UNKNOWN')
            type_counts[code] = type_counts.get(code, 0) + 1

        return type_counts


class SQLExecutionValidator:
    """SQL 执行前校验器 - 业务规则校验"""

    def __init__(self, allowed_tables: Optional[List[str]] = None,
                 allow_select_only: bool = True, dialect: str = "mysql"):
        """
        初始化执行校验器

        Args:
            allowed_tables: 允许的表列表
            allow_select_only: 是否只允许 SELECT
        """
        self.allowed_tables = allowed_tables
        self.allow_select_only = allow_select_only
        self.dialect = dialect

    def validate(self, sql: str) -> Dict[str, Any]:
        """
        执行前校验

        Args:
            sql: SQL 语句

        Returns:
            Dict: 校验结果
        """
        if not isinstance(sql, str) or not sql.strip():
            return {'valid': False, 'errors': ['SQL 语句为空'], 'warnings': []}

        errors = []
        warnings = []

        sql_upper = sql.strip().upper()

        # 检查是否只允许 SELECT
        if self.allow_select_only:
            if not is_safe_read_query(sql, dialect=self.dialect):
                errors.append("只允许执行单条、无副作用的 SELECT 查询")
        else:
            dangerous_patterns = (
                'DROP', 'TRUNCATE', 'DELETE FROM', 'INSERT INTO',
                'UPDATE ', 'ALTER ', 'CREATE ',
            )
            if any(pattern in sql_upper for pattern in dangerous_patterns):
                errors.append("危险 SQL 操作被禁止")

        # 检查允许的表
        if self.allowed_tables:
            found_tables = validate_query_scope(sql, dialect=self.dialect)['tables']
            for table in found_tables:
                if table.upper() not in [t.upper() for t in self.allowed_tables]:
                    errors.append(f"表 {table} 不在允许列表中")

        # 检查 LIMIT（防止全表扫描）
        if 'SELECT' in sql_upper and 'LIMIT' not in sql_upper and 'WHERE' not in sql_upper:
            warnings.append("查询没有 WHERE 或 LIMIT 条件，可能导致全表扫描")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def _extract_tables(self, sql: str) -> List[str]:
        """使用 AST 提取真实表引用。"""
        return validate_query_scope(sql, dialect=self.dialect)['tables']


# 模块级 Linter 缓存，避免重复创建（按方言缓存）
_validator_cache: Dict[str, 'SQLValidator'] = {}


def _get_cached_validator(dialect: str) -> 'SQLValidator':
    """获取或创建指定方言的缓存 SQLValidator 实例"""
    if dialect not in _validator_cache:
        _validator_cache[dialect] = SQLValidator(dialect=dialect)
    return _validator_cache[dialect]


def validate_sql(sql: str, dialect: str = "mysql",
                execution_check: bool = True) -> Dict[str, Any]:
    """
    便捷函数：完整校验 SQL

    Args:
        sql: SQL 语句
        dialect: SQL 方言
        execution_check: 是否执行前校验

    Returns:
        Dict: 校验结果
    """
    # SQLFluff 校验（使用缓存的 validator）
    validator = _get_cached_validator(dialect)
    syntax_result = validator.validate(sql)

    # 执行前校验
    exec_result = {'valid': True, 'errors': [], 'warnings': []}
    if execution_check:
        exec_validator = SQLExecutionValidator(allow_select_only=True, dialect=dialect)
        exec_result = exec_validator.validate(sql)

    # 合并结果
    all_errors = syntax_result['errors'] + exec_result['errors']
    all_warnings = syntax_result['warnings'] + exec_result['warnings']

    return {
        'valid': len(all_errors) == 0,
        'syntax_valid': syntax_result['valid'],
        'execution_valid': exec_result['valid'],
        'errors': all_errors,
        'warnings': all_warnings,
        'formatted_sql': validator.format(sql).get('formatted_sql', sql)
    }
