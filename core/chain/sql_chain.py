"""
LangChain 链式处理模块
实现 SQL 生成的完整链路（支持异步）
"""
from typing import Dict, List, Optional, Any, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from loguru import logger
import re
import asyncio

from .prompts import SQLPromptTemplates, get_prompt_templates


class SQLGenerationChain:
    """SQL 生成链 - 整合检索、提示词、LLM 调用"""

    def __init__(self, llm: BaseChatModel, dialect: str = "mysql",
                 retrieval_top_k: int = 5):
        """
        初始化 SQL 生成链

        Args:
            llm: LangChain LLM 实例
            dialect: SQL 方言
            retrieval_top_k: 检索返回数量
        """
        self.llm = llm
        self.dialect = dialect
        self.retrieval_top_k = retrieval_top_k
        self.prompt_templates = get_prompt_templates(dialect)

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0
        }

    def generate(self, user_query: str, schema_info: Dict,
                 knowledge_results: Optional[List[Tuple[Dict, float]]] = None,
                 session_id: Optional[str] = None,
                 few_shot_examples: Optional[str] = None) -> Dict[str, Any]:
        """
        生成 SQL

        Args:
            user_query: 用户自然语言查询
            schema_info: Schema 信息字典
            knowledge_results: 检索结果
            session_id: 会话 ID
            few_shot_examples: 动态 Few-Shot 示例字符串

        Returns:
            Dict: 生成结果，包含 sql、status、error 等字段
        """
        self.stats['total_requests'] += 1

        try:
            # 1. 格式化 Schema 信息
            formatted_schema = self._format_schema(schema_info)

            # 2. 格式化知识信息
            formatted_knowledge = self._format_knowledge(knowledge_results)

            # 3. 构建提示词（支持动态 Few-Shot 示例）
            messages = self.prompt_templates.create_sql_generation_messages(
                user_query=user_query,
                schema_info=formatted_schema,
                knowledge_info=formatted_knowledge,
                few_shot_examples=few_shot_examples or ""
            )

            # 4. 调用 LLM
            logger.info(f"调用 LLM 生成 SQL, query={user_query[:50]}...")
            response = self.llm.invoke(messages)

            # 5. 提取 SQL 和思考过程
            sql = self._extract_sql(response.content)
            thinking = self._extract_thinking(response.content)

            if sql == "INSUFFICIENT_INFO":
                logger.warning("LLM 判断信息不足")
                self.stats['failed_generations'] += 1
                return {
                    'sql': None,
                    'status': 'insufficient_info',
                    'error': '信息不足以生成 SQL',
                    'raw_response': response.content,
                    'thinking': thinking
                }

            if not sql:
                logger.error("未能从响应中提取 SQL")
                self.stats['failed_generations'] += 1
                return {
                    'sql': None,
                    'status': 'extraction_failed',
                    'error': '无法从响应中提取 SQL',
                    'raw_response': response.content,
                    'thinking': thinking
                }

            self.stats['successful_generations'] += 1
            if thinking:
                logger.info(f"思考过程：{thinking[:200]}...")
            logger.info(f"SQL 生成成功：{sql[:100]}...")

            return {
                'sql': sql,
                'status': 'success',
                'error': None,
                'raw_response': response.content,
                'thinking': thinking,
                'matched_knowledge': self._extract_matched_knowledge(knowledge_results),
                'retrieval_query': user_query
            }

        except Exception as e:
            logger.error(f"SQL 生成失败：{e}")
            self.stats['failed_generations'] += 1
            return {
                'sql': None,
                'status': 'error',
                'error': str(e),
                'raw_response': None
            }

    def _format_schema(self, schema_info: Dict) -> str:
        """格式化 Schema 信息"""
        tables = schema_info.get('tables', [])
        columns = schema_info.get('columns', [])
        relationships = schema_info.get('relationships', [])

        return self.prompt_templates.format_schema_info(
            tables=tables,
            columns=columns,
            relationships=relationships
        )

    def _format_knowledge(self, knowledge_results: Optional[List[Tuple[Dict, float]]]) -> str:
        """格式化知识信息"""
        if not knowledge_results:
            return "暂无相关知识信息"

        matched_nodes = [
            {
                'node_type': result[0].get('node_type', 'unknown'),
                'name': result[0].get('name', ''),
                'description': result[0].get('description', ''),
                'score': result[1]
            }
            for result in knowledge_results
        ]

        return self.prompt_templates.format_knowledge_info(matched_nodes)

    def _extract_sql(self, text: str) -> Optional[str]:
        """
        从文本中提取 SQL（支持 CoT 思维链格式）

        Args:
            text: LLM 响应文本

        Returns:
            Optional[str]: 提取的 SQL
        """
        if not text:
            return None

        # 检查是否明确表示信息不足
        if "INSUFFICIENT_INFO" in text.upper():
            return "INSUFFICIENT_INFO"

        # 优先提取<sql>标签内的内容（CoT 格式）
        sql_tag_pattern = r"<sql>\s*(.*?)\s*</sql>"
        matches = re.findall(sql_tag_pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip()
            # 移除末尾的分号（可选）
            sql = sql.rstrip(';')
            return sql

        # 尝试提取 SQL 代码块
        code_block_pattern = r"```(?:sql)?\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip()
            sql = sql.rstrip(';')
            return sql

        # 如果没有代码块，尝试查找 SQL 关键字
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE']
        lines = text.strip().split('\n')

        for line in lines:
            line_upper = line.upper().strip()
            for keyword in sql_keywords:
                if line_upper.startswith(keyword):
                    sql = line.strip().rstrip(';')
                    return sql

        # 如果还是没找到，返回整个文本（可能是纯 SQL）
        sql = text.strip().rstrip(';')
        if any(kw in sql.upper() for kw in sql_keywords):
            return sql

        return None

    def _extract_thinking(self, text: str) -> Optional[str]:
        """
        从文本中提取思考过程（CoT 思维链）

        Args:
            text: LLM 响应文本

        Returns:
            Optional[str]: 思考过程
        """
        if not text:
            return None

        # 提取<thinking>标签内的内容
        thinking_tag_pattern = r"<thinking>\s*(.*?)\s*</thinking>"
        matches = re.findall(thinking_tag_pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        return None

    def _extract_matched_knowledge(self, knowledge_results: Optional[List[Tuple[Dict, float]]]) -> List[Dict]:
        """提取匹配的知识用于记录"""
        if not knowledge_results:
            return []

        return [
            {
                'node_id': result[0].get('node_id'),
                'node_type': result[0].get('node_type'),
                'name': result[0].get('name'),
                'score': round(result[1], 4)
            }
            for result in knowledge_results[:self.retrieval_top_k]
        ]

    def fix_sql(self, sql: str, error: str, schema_info: Dict) -> Dict[str, Any]:
        """
        修复 SQL

        Args:
            sql: 原 SQL
            error: 错误信息
            schema_info: Schema 信息

        Returns:
            Dict: 修复结果
        """
        try:
            formatted_schema = self._format_schema(schema_info)
            messages = self.prompt_templates.create_fix_messages(
                sql=sql,
                error=error,
                schema_info=formatted_schema
            )

            logger.info(f"调用 LLM 修复 SQL, error={error[:50]}...")
            response = self.llm.invoke(messages)
            fixed_sql = self._extract_sql(response.content)

            if fixed_sql and fixed_sql != "INSUFFICIENT_INFO":
                logger.info(f"SQL 修复成功：{fixed_sql[:100]}...")
                return {
                    'sql': fixed_sql,
                    'status': 'success',
                    'error': None
                }
            else:
                return {
                    'sql': None,
                    'status': 'fix_failed',
                    'error': '无法修复 SQL'
                }

        except Exception as e:
            logger.error(f"SQL 修复失败：{e}")
            return {
                'sql': None,
                'status': 'error',
                'error': str(e)
            }

    def explain_sql(self, sql: str, schema_info: Dict) -> str:
        """
        解释 SQL

        Args:
            sql: SQL 语句
            schema_info: Schema 信息

        Returns:
            str: SQL 解释
        """
        formatted_schema = self._format_schema(schema_info)
        messages = self.prompt_templates.create_sql_generation_messages(
            user_query="请解释这个 SQL 的作用",
            schema_info=formatted_schema,
            knowledge_info="",
            example_patterns=sql
        )

        # 添加 SQL 到消息
        messages.append(HumanMessage(content=f"SQL: {sql}\n\n请解释这个查询的作用。"))

        response = self.llm.invoke(messages)
        return response.content

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()

    async def generate_async(self, user_query: str, schema_info: Dict,
                            knowledge_results: Optional[List[Tuple[Dict, float]]] = None,
                            session_id: Optional[str] = None,
                            few_shot_examples: Optional[str] = None) -> Dict[str, Any]:
        """
        生成 SQL（异步版本）

        Args:
            user_query: 用户自然语言查询
            schema_info: Schema 信息字典
            knowledge_results: 检索结果
            session_id: 会话 ID
            few_shot_examples: 动态 Few-Shot 示例字符串

        Returns:
            Dict: 生成结果
        """
        self.stats['total_requests'] += 1

        try:
            # 1. 格式化 Schema 信息
            formatted_schema = self._format_schema(schema_info)

            # 2. 格式化知识信息
            formatted_knowledge = self._format_knowledge(knowledge_results)

            # 3. 构建提示词
            messages = self.prompt_templates.create_sql_generation_messages(
                user_query=user_query,
                schema_info=formatted_schema,
                knowledge_info=formatted_knowledge,
                few_shot_examples=few_shot_examples or ""
            )

            # 4. 异步调用 LLM
            logger.info(f"异步调用 LLM 生成 SQL, query={user_query[:50]}...")
            response = await self.llm.ainvoke(messages)

            # 5. 提取 SQL 和思考过程
            sql = self._extract_sql(response.content)
            thinking = self._extract_thinking(response.content)

            if sql == "INSUFFICIENT_INFO":
                logger.warning("LLM 判断信息不足")
                self.stats['failed_generations'] += 1
                return {
                    'sql': None,
                    'status': 'insufficient_info',
                    'error': '信息不足以生成 SQL',
                    'raw_response': response.content,
                    'thinking': thinking
                }

            if not sql:
                logger.error("未能从响应中提取 SQL")
                self.stats['failed_generations'] += 1
                return {
                    'sql': None,
                    'status': 'extraction_failed',
                    'error': '无法从响应中提取 SQL',
                    'raw_response': response.content,
                    'thinking': thinking
                }

            self.stats['successful_generations'] += 1
            if thinking:
                logger.info(f"思考过程：{thinking[:200]}...")
            logger.info(f"SQL 生成成功：{sql[:100]}...")

            return {
                'sql': sql,
                'status': 'success',
                'error': None,
                'raw_response': response.content,
                'thinking': thinking,
                'matched_knowledge': self._extract_matched_knowledge(knowledge_results),
                'retrieval_query': user_query
            }

        except Exception as e:
            logger.error(f"SQL 生成失败：{e}")
            self.stats['failed_generations'] += 1
            return {
                'sql': None,
                'status': 'error',
                'error': str(e),
                'raw_response': None
            }

    async def fix_sql_async(self, sql: str, error: str, schema_info: Dict) -> Dict[str, Any]:
        """
        修复 SQL（异步版本）

        Args:
            sql: 原 SQL
            error: 错误信息
            schema_info: Schema 信息

        Returns:
            Dict: 修复结果
        """
        try:
            formatted_schema = self._format_schema(schema_info)
            messages = self.prompt_templates.create_fix_messages(
                sql=sql,
                error=error,
                schema_info=formatted_schema
            )

            logger.info(f"异步调用 LLM 修复 SQL, error={error[:50]}...")
            response = await self.llm.ainvoke(messages)
            fixed_sql = self._extract_sql(response.content)

            if fixed_sql and fixed_sql != "INSUFFICIENT_INFO":
                logger.info(f"SQL 修复成功：{fixed_sql[:100]}...")
                return {
                    'sql': fixed_sql,
                    'status': 'success',
                    'error': None
                }
            else:
                return {
                    'sql': None,
                    'status': 'fix_failed',
                    'error': '无法修复 SQL'
                }

        except Exception as e:
            logger.error(f"SQL 修复失败：{e}")
            return {
                'sql': None,
                'status': 'error',
                'error': str(e)
            }


class QueryRewriteChain:
    """查询改写链 - 用于优化用户查询（支持异步）"""

    def __init__(self, llm: BaseChatModel):
        """
        初始化查询改写链

        Args:
            llm: LangChain LLM 实例
        """
        self.llm = llm
        self.prompt_templates = get_prompt_templates()

    def rewrite(self, user_query: str, context: Optional[str] = None) -> str:
        """
        改写用户查询（同步）

        Args:
            user_query: 原始查询
            context: 上下文信息

        Returns:
            str: 改写后的查询
        """
        messages = [
            SystemMessage(content="你是一个查询优化专家，负责将用户模糊的查询改写成清晰明确的表达。"),
            HumanMessage(content=f"""原始查询：{user_query}

上下文：{context or '无'}

请改写查询使其更清晰明确，保持原意。只输出改写后的查询。""")
        ]

        response = self.llm.invoke(messages)
        return response.content.strip() or user_query

    async def rewrite_async(self, user_query: str, context: Optional[str] = None) -> str:
        """
        改写用户查询（异步）

        Args:
            user_query: 原始查询
            context: 上下文信息

        Returns:
            str: 改写后的查询
        """
        messages = [
            SystemMessage(content="你是一个查询优化专家，负责将用户模糊的查询改写成清晰明确的表达。"),
            HumanMessage(content=f"""原始查询：{user_query}

上下文：{context or '无'}

请改写查询使其更清晰明确，保持原意。只输出改写后的查询。""")
        ]

        response = await self.llm.ainvoke(messages)
        return response.content.strip() or user_query
