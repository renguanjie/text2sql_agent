"""
SQL 校验器模块
使用 SQLFluff 进行 SQL 语法校验和格式化
"""
from typing import Dict, List, Optional, Any, Tuple
import sqlfluff
from sqlfluff.core import Linter
from sqlfluff.core.errors import SQLLintError, SQLParseError
from loguru import logger


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
                 allow_select_only: bool = True):
        """
        初始化执行校验器

        Args:
            allowed_tables: 允许的表列表
            allow_select_only: 是否只允许 SELECT
        """
        self.allowed_tables = allowed_tables
        self.allow_select_only = allow_select_only

    def validate(self, sql: str) -> Dict[str, Any]:
        """
        执行前校验

        Args:
            sql: SQL 语句

        Returns:
            Dict: 校验结果
        """
        errors = []
        warnings = []

        sql_upper = sql.strip().upper()

        # 检查是否只允许 SELECT
        if self.allow_select_only:
            if not sql_upper.startswith('SELECT') and not sql_upper.startswith('WITH'):
                errors.append("只允许执行 SELECT 查询")

        # 检查危险操作
        dangerous_patterns = [
            ('DROP', 'DROP 操作被禁止'),
            ('TRUNCATE', 'TRUNCATE 操作被禁止'),
            ('DELETE FROM', 'DELETE 操作被禁止'),
            ('INSERT INTO', 'INSERT 操作被禁止'),
            ('UPDATE ', 'UPDATE 操作被禁止'),
            ('ALTER ', 'ALTER 操作被禁止'),
            ('CREATE ', 'CREATE 操作被禁止'),
        ]

        for pattern, message in dangerous_patterns:
            if pattern in sql_upper:
                errors.append(message)

        # 检查允许的表
        if self.allowed_tables:
            found_tables = self._extract_tables(sql)
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
        """简单提取 SQL 中的表名"""
        import re

        # 简单的 FROM 子句提取
        from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables = re.findall(from_pattern, sql, re.IGNORECASE)

        # JOIN 子句提取
        join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables.extend(re.findall(join_pattern, sql, re.IGNORECASE))

        return tables


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
    # SQLFluff 校验
    validator = SQLValidator(dialect=dialect)
    syntax_result = validator.validate(sql)

    # 执行前校验
    exec_result = {'valid': True, 'errors': [], 'warnings': []}
    if execution_check:
        exec_validator = SQLExecutionValidator(allow_select_only=True)
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
