"""
基于阿里云 Qwen-Max 的 SQL 生成器
使用 Neo4j 知识库动态构建提示词
"""
import requests
import json
from typing import Dict, Any, Optional
from loguru import logger


class QwenMaxLLM:
    """阿里云 Qwen-Max LLM 调用封装"""

    def __init__(self, api_key: str, model: str = "qwen-max",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Qwen-Max LLM 初始化完成：{model}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        调用 Qwen-Max 生成 SQL

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            生成结果字典
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            logger.info(f"调用 Qwen-Max API, prompt_length={len(prompt)}")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                usage = result.get('usage', {})

                logger.info(f"Qwen-Max 响应成功，tokens={usage}")

                return {
                    'success': True,
                    'content': content,
                    'usage': usage,
                    'raw_response': result
                }
            else:
                error_msg = f"API 请求失败：{response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'status_code': response.status_code
                }

        except requests.exceptions.Timeout:
            error_msg = "API 请求超时"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"调用失败：{str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def extract_sql(self, content: str) -> Optional[str]:
        """
        从响应内容中提取 SQL

        Args:
            content: LLM 响应内容

        Returns:
            提取的 SQL 语句
        """
        import re

        # 检查是否信息不足
        if "INSUFFICIENT_INFO" in content.upper():
            return "INSUFFICIENT_INFO"

        # 尝试从代码块中提取
        code_block_pattern = r"```(?:sql)?\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, content, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip().rstrip(';')
            logger.info(f"从代码块提取 SQL: {sql[:50]}...")
            return sql

        # 尝试直接提取 SQL 语句
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE']
        lines = content.strip().split('\n')

        for line in lines:
            line_upper = line.upper().strip()
            for keyword in sql_keywords:
                if line_upper.startswith(keyword):
                    sql = line.strip().rstrip(';')
                    logger.info(f"直接提取 SQL: {sql[:50]}...")
                    return sql

        logger.warning("未能提取 SQL，返回原始内容")
        return content.strip()


def generate_sql_with_qwen(user_query: str, schema_prompt: str,
                           api_key: str, model: str = "qwen-max",
                           retry_on_error: bool = True) -> Dict[str, Any]:
    """
    使用 Qwen-Max 生成 SQL 的便捷函数

    Args:
        user_query: 用户查询
        schema_prompt: 包含 schema 的提示词
        api_key: API Key
        model: 模型名称
        retry_on_error: 是否自动重试修复

    Returns:
        生成结果
    """
    llm = QwenMaxLLM(api_key=api_key, model=model)

    system_prompt = """你是一个专业的 SQL 生成专家，专门处理银行金融领域的数据查询。

【重要规则 - PostgreSQL 语法】
1. 只输出 SQL 语句，不要包含任何解释
2. 确保 SQL 语法正确，符合 PostgreSQL 方言
3. 必须使用提供的表名和字段名，不要臆造！
4. 如果信息不足以生成 SQL，返回 "INSUFFICIENT_INFO"
5. 注意表之间的关联关系，使用正确的 JOIN 条件
6. 对于时间查询，使用 CURRENT_DATE、INTERVAL 等 PostgreSQL 函数
7. 涉及金额计算时，注意使用 DECIMAL 类型
8. 字段名必须与 schema 中完全一致
9. 使用表别名时，必须在 FROM 或 JOIN 子句中定义
10. ⚠️ 不要使用反引号 (`)！PostgreSQL 不支持反引号语法
11. 字段名和表名要么不加引号，要么用双引号 (")
12. 推荐：简单字段名不加引号，如 SELECT customer_id FROM bank_customer"""

    result = llm.generate(
        prompt=f"{schema_prompt}\n\n用户查询：{user_query}\n\n请生成 SQL 语句：",
        system_prompt=system_prompt,
        temperature=0.1,
        max_tokens=1000
    )

    if result['success']:
        sql = llm.extract_sql(result['content'])
        result['sql'] = sql
        result['is_insufficient'] = (sql == "INSUFFICIENT_INFO")

        # 如果需要重试且有错误
        if retry_on_error and result.get('error'):
            logger.info(f"尝试修复 SQL...")
            fix_prompt = f"""原始查询：{user_query}

生成的 SQL 有错误：{result['error']}

Schema:
{schema_prompt}

请修复 SQL，确保字段名和表名与上面 schema 完全一致。只输出 SQL，不要解释。"""

            fix_result = llm.generate(
                prompt=fix_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1000
            )

            if fix_result['success']:
                fixed_sql = llm.extract_sql(fix_result['content'])
                if fixed_sql and fixed_sql != "INSUFFICIENT_INFO":
                    result['sql'] = fixed_sql
                    result['was_fixed'] = True
                    logger.info(f"SQL 修复成功")

    return result


# ==================== 测试函数 ====================

def test_qwen_generation():
    """测试 Qwen-Max SQL 生成"""
    from config import LLM_API_KEY, LLM_MODEL
    from core.knowledge.neon_knowledge import Neo4jKnowledgeRetriever, build_dynamic_prompt
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    print("=" * 60)
    print("Qwen-Max SQL 生成测试")
    print("=" * 60)

    # 初始化 Neo4j 检索器
    neo4j_retriever = Neo4jKnowledgeRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # 测试查询
    test_queries = [
        "查询所有客户的基本信息",
        "查找月收入在 10000 元以上的客户",
        "统计每个城市的客户数量",
        "查询客户及其账户信息",
        "查找有贷款记录的客户",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n【测试 {i}/{len(test_queries)}】")
        print(f"❓ 查询：{query}")

        # 构建动态提示词
        prompt, context = build_dynamic_prompt(neo4j_retriever, query)

        # 调用 Qwen-Max
        result = generate_sql_with_qwen(
            user_query=query,
            schema_prompt=context['schema'],
            api_key=LLM_API_KEY,
            model=LLM_MODEL
        )

        if result['success']:
            print(f"✅ 生成成功")
            print(f"   SQL: {result.get('sql', 'N/A')[:100]}...")
            print(f"   Tokens: {result.get('usage', {})}")
        else:
            print(f"❌ 生成失败：{result.get('error', '未知错误')}")

    neo4j_retriever.close()
    print("\n✅ 测试完成")


if __name__ == '__main__':
    test_qwen_generation()
