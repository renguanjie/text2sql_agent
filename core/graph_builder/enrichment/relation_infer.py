"""
关系推断器
"""
from typing import List
from loguru import logger

from ..models import SchemaExtractResult, RelationshipIR
from core.llm_factory import create_llm


PROMPT = """分析以下数据库表结构，推断表之间的主外键关系。

表结构：
{schema_info}

以 JSON 数组输出：[{"from_table": "表 A", "from_column": "字段", "to_table": "表 B", "to_column": "id"}]
"""


class RelationInferencer:
    def __init__(self, llm_provider: str = "openai", model: str = "gpt-4", **kwargs):
        self.llm = create_llm(provider=llm_provider, model=model, **kwargs)

    def infer(self, result: SchemaExtractResult) -> List[RelationshipIR]:
        if len(result.tables) < 2:
            return result.relationships

        if result.relationships:
            return result.relationships

        schema_info = "\n".join([
            f"- {t.name}: {[c.name + ('[PK]' if c.is_primary_key else '') for c in t.columns[:10]]}"
            for t in result.tables
        ])
        prompt = PROMPT.format(schema_info=schema_info)

        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                content = str(self.llm(prompt))

            import json
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            data = json.loads(json_match.group() if json_match else content)

            relationships = []
            for rel_data in data:
                from_table = rel_data.get("from_table")
                to_table = rel_data.get("to_table")
                from_tbl = next((t for t in result.tables if t.name == from_table), None)
                to_tbl = next((t for t in result.tables if t.name == to_table), None)

                if from_tbl and to_tbl:
                    rel = RelationshipIR(
                        from_database=from_tbl.database, from_table=from_table,
                        from_column=rel_data.get("from_column", "id"),
                        to_database=to_tbl.database, to_table=to_table,
                        to_column=rel_data.get("to_column", "id"),
                        relationship_type="foreign_key", join_type="LEFT JOIN"
                    )
                    relationships.append(rel)
            return relationships

        except Exception as e:
            logger.error(f"LLM 推断关系失败：{e}")
            return self._fallback(result)

    def _fallback(self, result: SchemaExtractResult) -> List[RelationshipIR]:
        relationships = []
        for table in result.tables:
            for col in table.columns:
                if col.name.endswith('_id') and not col.is_primary_key:
                    potential_ref = col.name[:-3]
                    for target in result.tables:
                        if potential_ref.lower() in target.name.lower():
                            rel = RelationshipIR(
                                from_database=table.database, from_table=table.name,
                                from_column=col.name,
                                to_database=table.database, to_table=target.name,
                                to_column='id',
                                relationship_type="foreign_key", join_type="LEFT JOIN"
                            )
                            relationships.append(rel)
                            break
        return relationships


def infer_relationships(result: SchemaExtractResult, **kwargs) -> List[RelationshipIR]:
    inferencer = RelationInferencer(**kwargs)
    return inferencer.infer(result)
