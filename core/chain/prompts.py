"""
提示词模板模块
定义 SQL 生成相关的提示词模板
"""
from typing import Dict, List, Optional, Any
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger


# ==================== 系统提示词 ====================

SQL_GENERATION_SYSTEM_PROMPT = """你是一个专业的 SQL 生成专家。你的任务是根据用户的自然语言查询，结合提供的数据库 schema 和知识库信息，生成准确的 SQL 查询语句。

请遵循以下规则：
1. 先进行思考分析，然后在<thinking>标签中输出思考过程
2. 在<thinking>之后，在<sql>标签中输出最终的 SQL 语句
3. 确保 SQL 语法正确，符合 {dialect} 方言
4. 使用提供的表名和字段名，不要臆造
5. 如果信息不足以生成 SQL，在<thinking>中说明原因，并返回 "INSUFFICIENT_INFO"
6. 对于模糊查询，使用 LIKE 语句
7. 注意表之间的关联关系
8. 合理使用 WHERE、GROUP BY、ORDER BY 等子句
9. 涉及金额、数量等计算时，注意数据类型转换
10. 参考历史示例的写法和风格

思考过程应包含：
- 识别用户意图和关键实体
- 确定需要用到的表和字段
- 分析表之间的关联关系
- 规划 SQL 结构和计算逻辑

数据库 Schema:
{schema_info}

相关知识:
{knowledge_info}

历史相似查询示例:
{few_shot_examples}"""


# ==================== 提示词模板类 ====================

class SQLPromptTemplates:
    """SQL 生成提示词模板集合"""

    def __init__(self, dialect: str = "mysql"):
        """
        初始化提示词模板

        Args:
            dialect: SQL 方言
        """
        self.dialect = dialect

        # SQL 生成主模板
        self.sql_generation_template = PromptTemplate(
            input_variables=["user_query", "schema_info", "knowledge_info", "example_patterns"],
            template=SQL_GENERATION_SYSTEM_PROMPT + "\n\n用户查询：{user_query}\n\nSQL:"
        )

        # Schema 信息格式化模板
        self.schema_format_template = PromptTemplate(
            input_variables=["tables", "columns", "relationships"],
            template=self._get_schema_format()
        )

        # 知识信息格式化模板
        self.knowledge_format_template = PromptTemplate(
            input_variables=["matched_nodes"],
            template=self._get_knowledge_format()
        )

        # SQL 校验模板
        self.sql_validation_template = PromptTemplate(
            input_variables=["sql", "schema_info", "error_message"],
            template=self._get_validation_template()
        )

        # SQL 修复模板
        self.sql_fix_template = PromptTemplate(
            input_variables=["sql", "error", "schema_info"],
            template=self._get_fix_template()
        )

        # SQL 解释模板
        self.sql_explain_template = PromptTemplate(
            input_variables=["sql", "schema_info"],
            template=self._get_explain_template()
        )

        # 查询改写模板
        self.query_rewrite_template = PromptTemplate(
            input_variables=["user_query", "context"],
            template=self._get_rewrite_template()
        )

    def _get_schema_format(self) -> str:
        """获取 Schema 格式化模板"""
        return """表结构信息:
{tables}

字段详情:
{columns}

表关系:
{relationships}"""

    def _get_knowledge_format(self) -> str:
        """获取知识信息格式化模板"""
        return """匹配的知识节点:
{matched_nodes}

以上知识可能对理解用户查询有帮助。"""

    def _get_validation_template(self) -> str:
        """获取 SQL 校验模板"""
        return """请检查以下 SQL 语句是否正确:

SQL: {sql}

Schema: {schema_info}

{error_message}

如果 SQL 有错误，请指出问题并给出修正建议。
如果 SQL 正确，请回复 "VALID"。"""

    def _get_fix_template(self) -> str:
        """获取 SQL 修复模板"""
        return """以下 SQL 语句有错误，请修复:

错误 SQL: {sql}
错误信息: {error}

Schema: {schema_info}

请给出修正后的 SQL 语句，只输出 SQL，不要解释。"""

    def _get_explain_template(self) -> str:
        """获取 SQL 解释模板"""
        return """请解释以下 SQL 语句的作用:

SQL: {sql}

Schema: {schema_info}

请用简洁的中文解释这个 SQL 查询的目的和执行逻辑。"""

    def _get_rewrite_template(self) -> str:
        """获取查询改写模板"""
        return """用户查询可能不够清晰或完整，请根据上下文进行改写:

原始查询：{user_query}

上下文信息：{context}

请改写查询使其更清晰明确，保持原意。只输出改写后的查询。"""

    def format_schema_info(self, tables: List[Dict], columns: List[Dict],
                          relationships: Optional[List[Dict]] = None) -> str:
        """
        格式化 Schema 信息

        Args:
            tables: 表信息列表
            columns: 字段信息列表
            relationships: 关系列表

        Returns:
            str: 格式化的 Schema 字符串
        """
        # 格式化表信息
        table_lines = []
        for t in tables:
            line = f"- {t.get('table_name', 'unknown')}"
            if t.get('table_comment'):
                line += f" ({t['table_comment']})"
            table_lines.append(line)

        # 格式化字段信息
        column_lines = []
        current_table = None
        for c in columns:
            if c.get('table_name') != current_table:
                current_table = c.get('table_name')
                column_lines.append(f"\n表 [{current_table}]:")
            pk = "PRIMARY KEY, " if c.get('is_primary_key') else ""
            nullable = "NULL" if c.get('is_nullable') else "NOT NULL"
            line = f"  - {c.get('column_name')}: {c.get('column_type')} ({pk}{nullable}"
            if c.get('column_comment'):
                line += f", {c['column_comment']}"
            line += ")"
            column_lines.append(line)

        # 格式化关系
        rel_lines = []
        if relationships:
            for r in relationships:
                rel_lines.append(f"- {r.get('from_table')} -> {r.get('to_table')} ({r.get('type')})")

        return self.schema_format_template.format(
            tables="\n".join(table_lines),
            columns="\n".join(column_lines),
            relationships="\n".join(rel_lines) if rel_lines else "无明确关系定义"
        )

    def format_knowledge_info(self, matched_nodes: List[Dict]) -> str:
        """
        格式化知识信息

        Args:
            matched_nodes: 匹配的节点列表

        Returns:
            str: 格式化的知识字符串
        """
        if not matched_nodes:
            return "暂无相关知识"

        lines = []
        for node in matched_nodes:
            line = f"- [{node.get('node_type', 'unknown')}] {node.get('name', '')}"
            if node.get('description'):
                line += f": {node['description']}"
            lines.append(line)

        return "\n".join(lines)

    def create_sql_generation_messages(self, user_query: str, schema_info: str,
                                       knowledge_info: str,
                                       few_shot_examples: str = "") -> List:
        """
        创建 SQL 生成的消息列表

        Args:
            user_query: 用户查询
            schema_info: Schema 信息
            knowledge_info: 知识信息
            few_shot_examples: 动态 Few-Shot 示例

        Returns:
            List: 消息列表
        """
        system_content = SQL_GENERATION_SYSTEM_PROMPT.format(
            dialect=self.dialect,
            schema_info=schema_info,
            knowledge_info=knowledge_info,
            few_shot_examples=few_shot_examples or "暂无历史示例，请根据 schema 和知识自行推导"
        )

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=f"用户查询：{user_query}\n\n请生成对应的 SQL 语句：")
        ]

    def create_validation_messages(self, sql: str, schema_info: str,
                                  error_message: str = "") -> List:
        """
        创建 SQL 校验消息列表

        Args:
            sql: SQL 语句
            schema_info: Schema 信息
            error_message: 错误信息

        Returns:
            List: 消息列表
        """
        content = self.sql_validation_template.format(
            sql=sql,
            schema_info=schema_info,
            error_message=error_message or "请检查语法和逻辑是否正确"
        )

        return [
            SystemMessage(content="你是一个 SQL 校验专家，负责检查 SQL 语句的正确性。"),
            HumanMessage(content=content)
        ]

    def create_fix_messages(self, sql: str, error: str, schema_info: str) -> List:
        """
        创建 SQL 修复消息列表

        Args:
            sql: 原 SQL
            error: 错误信息
            schema_info: Schema 信息

        Returns:
            List: 消息列表
        """
        content = self.sql_fix_template.format(
            sql=sql,
            error=error,
            schema_info=schema_info
        )

        return [
            SystemMessage(content="你是一个 SQL 修复专家，负责修正有错误的 SQL 语句。只输出修正后的 SQL。"),
            HumanMessage(content=content)
        ]


# 全局模板实例
_default_templates: Optional[SQLPromptTemplates] = None


def get_prompt_templates(dialect: str = "mysql") -> SQLPromptTemplates:
    """
    获取提示词模板实例

    Args:
        dialect: SQL 方言

    Returns:
        SQLPromptTemplates: 模板实例
    """
    global _default_templates
    if _default_templates is None or _default_templates.dialect != dialect:
        _default_templates = SQLPromptTemplates(dialect)
    return _default_templates
