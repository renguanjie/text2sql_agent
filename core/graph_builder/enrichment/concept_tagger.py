"""
业务概念生成器
"""
from typing import List
from loguru import logger
import json
import re

from ..models import SchemaExtractResult, BusinessConceptIR
from core.llm_factory import create_llm


PROMPT = """分析以下数据库表结构，为每个表生成 3-5 个业务标签（中文）。

表结构：
{schema_info}

以 JSON 格式输出：{{"table_name": {{"tags": ["标签 1", "标签 2"], "concept": "概念名"}}}}
只输出 JSON，不要有其他内容。
"""


class ConceptTagger:
    def __init__(self, llm_provider: str = "openai", model: str = "gpt-4", **kwargs):
        self.llm = create_llm(provider=llm_provider, model=model, **kwargs)

    def generate(self, result: SchemaExtractResult) -> List[BusinessConceptIR]:
        if not result.tables:
            return []

        schema_info = "\n".join([f"- {t.name}: {[c.name for c in t.columns[:5]]}" for t in result.tables])
        prompt = PROMPT.format(schema_info=schema_info)

        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                content = str(self.llm(prompt))

            logger.info(f"LLM 响应内容：{content[:200]}...")

            # 提取 JSON 内容（处理可能的 markdown 包裹）
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group()

            if not content.strip():
                logger.warning("LLM 返回空内容，使用降级方案")
                return self._fallback(result)

            data = json.loads(content)
            concepts = []
            for table_name, info in data.items():
                if isinstance(info, dict):
                    concept = BusinessConceptIR(
                        name=info.get("concept", table_name),
                        tags=info.get("tags", []),
                        mapped_tables=[table_name]
                    )
                    concepts.append(concept)
            return concepts

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败：{e}, 原始响应：{content[:500]}")
            return self._fallback(result)
        except Exception as e:
            logger.error(f"LLM 生成概念失败：{e}")
            return self._fallback(result)

    def _fallback(self, result: SchemaExtractResult) -> List[BusinessConceptIR]:
        """降级方案：为每个表创建基本概念"""
        concepts = []
        for table in result.tables:
            concept = BusinessConceptIR(
                name=table.name,
                description=table.description or f"{table.name} 表",
                tags=["通用"],
                mapped_tables=[table.name]
            )
            concepts.append(concept)
        logger.info(f"使用降级方案生成 {len(concepts)} 个业务概念")
        return concepts


def generate_concepts(result: SchemaExtractResult, **kwargs) -> List[BusinessConceptIR]:
    tagger = ConceptTagger(**kwargs)
    return tagger.generate(result)
