"""
SQL 字段验证器
验证生成的 SQL 中使用的表名和字段名是否存在于数据库中
使用 sqlglot 进行 AST 解析，精准提取表名和字段名
"""
from sqlalchemy import create_engine, inspect, text
from typing import Dict, List, Set, Tuple, Optional, Any
from loguru import logger

try:
    import sqlglot
    from sqlglot import exp, parse, ParseError
    from sqlglot.expressions import (
        Select, Table, Column, Identifier,
        Join, From, CTE, Subquery, Alias
    )
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    logger.warning("sqlglot 未安装，AST 解析功能将不可用。安装：pip install sqlglot")


class SQLFieldValidator:
    """SQL 字段验证器"""

    def __init__(self, db_url: str):
        """
        初始化验证器

        Args:
            db_url: 数据库连接 URL
        """
        self.engine = create_engine(db_url)
        self.inspector = inspect(self.engine)
        self._table_cache: Dict[str, List[str]] = {}
        logger.info("SQL 字段验证器初始化完成")

    def close(self):
        self.engine.dispose()

    def get_table_columns(self, table_name: str) -> List[str]:
        """
        获取表的字段列表

        Args:
            table_name: 表名

        Returns:
            字段名列表
        """
        if table_name not in self._table_cache:
            try:
                columns = self.inspector.get_columns(table_name)
                self._table_cache[table_name] = [col['name'] for col in columns]
            except Exception as e:
                logger.warning(f"无法获取表 {table_name} 的字段：{e}")
                self._table_cache[table_name] = []

        return self._table_cache[table_name]

    def get_all_tables(self) -> List[str]:
        """获取所有表名"""
        return self.inspector.get_table_names()

    def extract_tables_from_sql(self, sql: str) -> Set[str]:
        """
        从 SQL 中提取表名（使用 sqlglot AST 解析）

        Args:
            sql: SQL 语句

        Returns:
            表名集合
        """
        if not SQLGLOT_AVAILABLE:
            logger.warning("sqlglot 不可用，使用降级处理")
            return set()

        tables = set()

        try:
            # 解析 SQL 为 AST
            parsed = parse(sql, read='mysql')

            for statement in parsed:
                if statement is None:
                    continue

                # 提取所有表引用（包括 FROM、JOIN、CTE 等）
                for table_node in statement.find_all(Table):
                    table_name = self._get_table_name(table_node)
                    if table_name:
                        tables.add(table_name.lower())

                # 处理 CTE (WITH 语句)
                for cte in statement.find_all(CTE):
                    if cte.alias:
                        # CTE 别名也作为"表"处理（用于后续字段验证）
                        tables.add(cte.alias.lower())

        except ParseError as e:
            logger.warning(f"SQL 解析失败：{e}")
        except Exception as e:
            logger.error(f"提取表名失败：{e}")

        logger.debug(f"从 SQL 中提取到表名：{tables}")
        return tables

    def _get_table_name(self, table_node: Table) -> Optional[str]:
        """
        从 Table 节点获取表名

        Args:
            table_node: sqlglot Table 节点

        Returns:
            表名
        """
        # 处理 schema.table 或 database.schema.table 格式
        parts = []

        # 获取 db (database)
        if table_node.db:
            parts.append(table_node.db.name)

        # 获取 schema
        if table_node.args.get('schema'):
            parts.append(table_node.args['schema'].name)

        # 获取 table name
        if table_node.name:
            parts.append(table_node.name)

        if parts:
            # 返回完整表名（包含 schema 前缀）或仅表名
            return parts[-1]  # 只返回表名用于验证

        return None

    def extract_columns_from_sql(self, sql: str) -> Set[str]:
        """
        从 SQL 中提取字段名（使用 sqlglot AST 解析）

        Args:
            sql: SQL 语句

        Returns:
            字段名集合
        """
        if not SQLGLOT_AVAILABLE:
            logger.warning("sqlglot 不可用，使用降级处理")
            return set()

        columns = set()

        try:
            # 解析 SQL 为 AST
            parsed = parse(sql, read='mysql')

            for statement in parsed:
                if statement is None:
                    continue

                # 提取所有列引用
                for col_node in statement.find_all(Column):
                    col_name = self._get_column_name(col_node)
                    if col_name:
                        columns.add(col_name.lower())

                # 提取 ORDER BY 中的列
                for order in statement.find_all(exp.Ordered):
                    if isinstance(order.this, Column):
                        col_name = self._get_column_name(order.this)
                        if col_name:
                            columns.add(col_name.lower())

                # 提取 GROUP BY 中的列
                for group in statement.find_all(exp.Group):
                    for expr in group.expressions:
                        if isinstance(expr, Column):
                            col_name = self._get_column_name(expr)
                            if col_name:
                                columns.add(col_name.lower())

                # 提取 WHERE 条件中的列
                for where in statement.find_all(exp.Where):
                    for col_node in where.find_all(Column):
                        col_name = self._get_column_name(col_node)
                        if col_name:
                            columns.add(col_name.lower())

                # 提取 HAVING 中的列
                for having in statement.find_all(exp.Having):
                    for col_node in having.find_all(Column):
                        col_name = self._get_column_name(col_node)
                        if col_name:
                            columns.add(col_name.lower())

        except ParseError as e:
            logger.warning(f"SQL 解析失败：{e}")
        except Exception as e:
            logger.error(f"提取字段名失败：{e}")

        logger.debug(f"从 SQL 中提取到字段名：{columns}")
        return columns

    def _get_column_name(self, col_node: Column) -> Optional[str]:
        """
        从 Column 节点获取字段名

        Args:
            col_node: sqlglot Column 节点

        Returns:
            字段名
        """
        # 获取列名（不包含表前缀）
        if col_node.name:
            return col_node.name

        return None

    def extract_table_alias_mapping(self, sql: str) -> Dict[str, str]:
        """
        提取 SQL 中的表别名映射

        Args:
            sql: SQL 语句

        Returns:
            {别名：实际表名} 的映射字典
        """
        if not SQLGLOT_AVAILABLE:
            return {}

        alias_mapping = {}

        try:
            parsed = parse(sql, read='mysql')

            for statement in parsed:
                if statement is None:
                    continue

                # 查找所有表及其别名
                for table_node in statement.find_all(Table):
                    table_name = self._get_table_name(table_node)
                    alias = table_node.alias

                    if table_name:
                        if alias:
                            alias_mapping[alias.lower()] = table_name.lower()
                        alias_mapping[table_name.lower()] = table_name.lower()

        except Exception as e:
            logger.error(f"提取表别名失败：{e}")

        return alias_mapping

    def extract_columns_with_table(self, sql: str) -> Dict[str, Set[str]]:
        """
        提取字段名并关联其所属的表

        Args:
            sql: SQL 语句

        Returns:
            {表名：字段名集合} 的字典
        """
        if not SQLGLOT_AVAILABLE:
            return {}

        table_columns: Dict[str, Set[str]] = {}
        table_alias_mapping = self.extract_table_alias_mapping(sql)

        try:
            parsed = parse(sql, read='mysql')

            for statement in parsed:
                if statement is None:
                    continue

                for col_node in statement.find_all(Column):
                    col_name = col_node.name.lower() if col_node.name else None

                    # 获取列的表引用
                    table_ref = None

                    # 如果列有表前缀 (如 t.customer_id)
                    if col_node.table:
                        table_ref = col_node.table.lower()
                        # 解析别名到实际表名
                        table_ref = table_alias_mapping.get(table_ref, table_ref)

                    # 如果没有表前缀，尝试从所有相关表中查找
                    if not table_ref:
                        # 对于无表前缀的列，标记为 unknown 待后续处理
                        table_ref = 'unknown'

                    if col_name and table_ref:
                        if table_ref not in table_columns:
                            table_columns[table_ref] = set()
                        table_columns[table_ref].add(col_name)

        except Exception as e:
            logger.error(f"提取带表字段失败：{e}")

        return table_columns

    def validate_sql(self, sql: str) -> Tuple[bool, List[str]]:
        """
        验证 SQL 语句

        Args:
            sql: SQL 语句

        Returns:
            (是否有效，错误列表)
        """
        errors = []

        # 使用 AST 解析提取表和字段
        tables = self.extract_tables_from_sql(sql)
        table_columns = self.extract_columns_with_table(sql)
        all_columns = set()
        for cols in table_columns.values():
            all_columns.update(cols)

        logger.info(f"验证 SQL: 发现表 {tables}, 字段 {all_columns}")

        # 获取表别名映射
        table_alias_mapping = self.extract_table_alias_mapping(sql)

        # 验证表名
        all_tables = set(t.lower() for t in self.get_all_tables())
        validated_tables = set()

        for table in tables:
            # 检查是否是别名映射到的表
            actual_table = table_alias_mapping.get(table, table)

            if actual_table not in all_tables:
                errors.append(f"表不存在：{table} (可用表：{', '.join(sorted(all_tables)[:10])})")
            else:
                validated_tables.add(actual_table)

        # 验证字段 - 按表分组验证
        for table_ref, columns in table_columns.items():
            # 解析表引用到实际表名
            actual_table = table_alias_mapping.get(table_ref.lower(), table_ref.lower())

            # 如果是 'unknown'，需要在所有表中查找
            if actual_table == 'unknown':
                for column in columns:
                    if column == '*':
                        continue
                    found = False
                    for tbl in validated_tables or all_tables:
                        table_cols = [c.lower() for c in self.get_table_columns(tbl)]
                        if column in table_cols:
                            found = True
                            break
                    if not found:
                        errors.append(f"字段不存在：{column}")
            else:
                # 验证特定表的字段
                if actual_table in validated_tables or actual_table in all_tables:
                    table_cols = [c.lower() for c in self.get_table_columns(actual_table)]
                    for column in columns:
                        if column == '*':
                            continue
                        if column not in table_cols:
                            errors.append(f"字段不存在：{actual_table}.{column} (可用字段：{', '.join(sorted(table_cols)[:10])})")

        is_valid = len(errors) == 0
        logger.info(f"SQL 验证结果：{'有效' if is_valid else '无效'} - {len(errors)} 个错误")

        return is_valid, errors

    def suggest_fix(self, sql: str, errors: List[str]) -> str:
        """
        提供修复建议

        Args:
            sql: 原始 SQL
            errors: 错误列表

        Returns:
            修复建议
        """
        if not errors:
            return "SQL 有效，无需修复"

        suggestions = ["发现以下问题:"]
        for error in errors:
            suggestions.append(f"- {error}")

        suggestions.append("\n建议:")
        suggestions.append("1. 检查表名是否正确")
        suggestions.append("2. 检查字段名是否与 schema 一致")
        suggestions.append("3. 确保 JOIN 条件使用正确的关联字段")

        return "\n".join(suggestions)

    def debug_parse(self, sql: str) -> Dict[str, Any]:
        """
        调试 SQL 解析结果

        Args:
            sql: SQL 语句

        Returns:
            解析结果字典
        """
        if not SQLGLOT_AVAILABLE:
            return {'error': 'sqlglot not available'}

        result = {
            'tables': [],
            'columns': [],
            'table_aliases': {},
            'table_columns': {},
            'ast_dump': ''
        }

        try:
            parsed = parse(sql, read='mysql')

            for statement in parsed:
                if statement is None:
                    continue

                # 提取表
                for table_node in statement.find_all(Table):
                    result['tables'].append({
                        'name': table_node.name,
                        'db': table_node.db.name if table_node.db else None,
                        'schema': table_node.args.get('schema').name if table_node.args.get('schema') else None,
                        'alias': table_node.alias
                    })

                # 提取列
                for col_node in statement.find_all(Column):
                    result['columns'].append({
                        'name': col_node.name,
                        'table': col_node.table,
                        'db': col_node.db.name if col_node.db else None
                    })

                # AST 转储（简化版）
                result['ast_dump'] = statement.sql(pretty=True)

        except Exception as e:
            result['error'] = str(e)

        return result


# ==================== 测试函数 ====================

def test_validator():
    """测试字段验证器 - AST 解析 vs 正则"""
    NEON_URL = "postgresql://neondb_owner:npg_oGspmF6zTY9w@ep-polished-snow-ak3wzf24.c-3.us-west-2.aws.neon.tech/neondb?sslmode=require"

    print("=" * 60)
    print("SQL 字段验证器测试 (AST 解析)")
    print("=" * 60)

    validator = SQLFieldValidator(NEON_URL)

    # 测试用例 - 包含复杂 SQL 场景
    test_cases = [
        {
            'sql': "SELECT customer_id, customer_name FROM bank_customer",
            'expected': True,
            'desc': '简单查询 - 应通过'
        },
        {
            'sql': "SELECT customer_id, nonexistent_field FROM bank_customer",
            'expected': False,
            'desc': '字段不存在 - 应失败'
        },
        {
            'sql': "SELECT bc.customer_id, bc.customer_name FROM bank_customer bc JOIN customer_account ca ON bc.customer_id = ca.customer_id",
            'expected': True,
            'desc': 'JOIN 查询带别名 - 应通过'
        },
        {
            'sql': "SELECT risk_level, AVG(loan_balance) FROM bank_customer",
            'expected': False,
            'desc': '字段不存在 (risk_level) - 应失败'
        },
        {
            'sql': """
                WITH active_customers AS (
                    SELECT customer_id, customer_name
                    FROM bank_customer
                    WHERE status = 'active'
                )
                SELECT ac.customer_id, ac.customer_name
                FROM active_customers ac
            """,
            'expected': True,
            'desc': 'CTE (WITH 语句) - AST 解析优势场景'
        },
        {
            'sql': """
                SELECT customer_id,
                       (SELECT MAX(amount) FROM transactions t WHERE t.customer_id = c.customer_id) as max_amount
                FROM customers c
            """,
            'expected': True,
            'desc': '子查询 - AST 解析优势场景'
        },
        {
            'sql': """
                SELECT customer_id,
                       ROW_NUMBER() OVER (PARTITION BY customer_type ORDER BY created_at) as rn
                FROM bank_customer
            """,
            'expected': True,
            'desc': 'Window 函数 - AST 解析优势场景'
        },
    ]

    passed = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\n【测试 {i}】{test['desc']}")
        print(f"SQL: {test['sql'][:80]}...")

        # 先展示 AST 解析结果
        debug_result = validator.debug_parse(test['sql'])
        if 'error' not in debug_result:
            print(f"   表：{[t['name'] for t in debug_result.get('tables', [])]}")
            print(f"   字段：{[c['name'] for c in debug_result.get('columns', [])]}")

        is_valid, errors = validator.validate_sql(test['sql'])

        if is_valid == test['expected']:
            print(f"✅ 通过")
            passed += 1
        else:
            print(f"❌ 失败")
            print(f"   期望：{'有效' if test['expected'] else '无效'}")
            print(f"   实际：{'有效' if is_valid else '无效'}")
            if errors:
                print(f"   错误：{errors[0]}")

    print(f"\n{'=' * 60}")
    print(f"测试结果：{passed}/{len(test_cases)} 通过")
    print(f"{'=' * 60}")

    validator.close()


if __name__ == '__main__':
    test_validator()
