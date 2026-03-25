"""
Neon PostgreSQL + Neo4j 知识库集成模块
实现基于 Neo4j 知识库的动态提示词构建和 LLM SQL 生成
"""
from neo4j import GraphDatabase
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import json


class Neo4jKnowledgeRetriever:
    """从 Neo4j 动态检索知识库"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Neo4j 知识库检索器初始化完成：{uri}")

    def close(self):
        self.driver.close()

    def search_tables(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        根据用户查询搜索相关表

        Args:
            query: 用户查询
            top_k: 返回数量

        Returns:
            相关表列表
        """
        with self.driver.session() as session:
            # 使用中文和英文表名进行匹配
            result = session.run("""
                MATCH (t:Table)
                WHERE toLower(t.name) CONTAINS toLower($query)
                   OR toLower(t.name_cn) CONTAINS toLower($query)
                RETURN t.name as name, t.name_cn as name_cn, t.comment as comment
                LIMIT $top_k
            """, {'query': query, 'top_k': top_k})

            return [dict(record) for record in result]

    def get_table_schema(self, table_name: str) -> Dict:
        """
        获取指定表的完整 schema

        Args:
            table_name: 表名

        Returns:
            表 schema 信息
        """
        with self.driver.session() as session:
            # 获取表信息
            table_result = session.run("""
                MATCH (t:Table {name: $table_name})
                RETURN t.name as name, t.name_cn as name_cn, t.comment as comment
            """, {'table_name': table_name})

            table_info = table_result.single()
            if not table_info:
                return {}

            # 获取字段信息
            columns_result = session.run("""
                MATCH (t:Table {name: $table_name})-[:HAS_COLUMN]->(c:Column)
                RETURN c.name as name, c.name_cn as name_cn, c.type as type, c.nullable as nullable
                ORDER BY c.name
            """, {'table_name': table_name})

            columns = [dict(record) for record in columns_result]

            # 获取关联表
            relations_result = session.run("""
                MATCH (t:Table {name: $table_name})-[r:CONNECTS]->(to:Table)
                RETURN to.name as target_table, to.name_cn as target_table_cn,
                       r.relationship_type as relationship, r.on_column as on_column
            """, {'table_name': table_name})

            relations = [dict(record) for record in relations_result]

            return {
                'name': table_info['name'],
                'name_cn': table_info['name_cn'],
                'comment': table_info['comment'],
                'columns': columns,
                'relations': relations
            }

    def get_related_tables(self, table_names: List[str]) -> List[Dict]:
        """
        获取一组表的关联表（用于多表查询）

        Args:
            table_names: 表名列表

        Returns:
            关联表信息
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (from:Table)-[r:CONNECTS]->(to:Table)
                WHERE from.name IN $table_names OR to.name IN $table_names
                RETURN from.name as from_table, from.name_cn as from_table_cn,
                       to.name as to_table, to.name_cn as to_table_cn,
                       r.relationship_type as relationship, r.on_column as on_column
            """, {'table_names': table_names})

            return [dict(record) for record in result]

    def get_all_tables(self) -> List[Dict]:
        """获取所有表的基本信息"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Table)
                OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                RETURN t.name as name, t.name_cn as name_cn,
                       count(c) as column_count
                ORDER BY t.name_cn
            """)

            return [dict(record) for record in result]

    def get_schema_for_prompt(self, table_names: Optional[List[str]] = None,
                              include_relations: bool = True) -> str:
        """
        构建用于 LLM 提示词的 schema 字符串

        Args:
            table_names: 指定表名列表，None 则返回所有表
            include_relations: 是否包含关联关系

        Returns:
            格式化的 schema 字符串
        """
        if table_names:
            tables = []
            for name in table_names:
                schema = self.get_table_schema(name)
                if schema:
                    tables.append(schema)
        else:
            all_tables = self.get_all_tables()
            tables = [self.get_table_schema(t['name']) for t in all_tables]

        # 构建提示词 - 优化格式，更清晰展示字段信息
        lines = ["【重要】数据库 Schema - 请严格按照以下表名和字段名生成 SQL，不要臆造！"]
        lines.append("=" * 70)

        for table in tables:
            lines.append(f"\n### 表：{table['name']}")
            lines.append(f"中文名称：{table['name_cn']}")
            lines.append(f"说明：{table.get('comment', '')}")
            lines.append(f"\n字段列表 (必须使用以下 exact 字段名):")
            lines.append("| 字段名 | 中文名 | 类型 | 可空 |")
            lines.append("|--------|--------|------|------|")

            for col in table['columns']:
                nullable = "YES" if col['nullable'] else "NO"
                lines.append(f"| `{col['name']}` | {col['name_cn']} | {col['type']} | {nullable} |")

            if include_relations and table['relations']:
                lines.append(f"\n关联关系 (JOIN 时使用以下条件):")
                for rel in table['relations']:
                    lines.append(f"- `{table['name']}` → `{rel['target_table']}` (关联字段：{rel['on_column']})")

        lines.append("\n" + "=" * 70)
        lines.append("【重要规则 - PostgreSQL 语法】")
        lines.append("1. 只能使用上面列出的表名和字段名")
        lines.append("2. ⚠️ 不要使用反引号 (`)！PostgreSQL 不支持反引号")
        lines.append("3. 字段名和表名要么不加引号，要么用双引号 (\")")
        lines.append("4. 推荐：简单字段名不加引号，如 SELECT customer_id FROM bank_customer")
        lines.append("5. JOIN 时必须使用上面列出的关联字段")
        lines.append("6. 不要臆造任何字段名或表名")
        lines.append("=" * 70)
        
        return "\n".join(lines)

    def search_knowledge(self, query: str, top_k: int = 5) -> str:
        """
        搜索与查询相关的知识

        Args:
            query: 用户查询
            top_k: 返回数量

        Returns:
            格式化的知识字符串
        """
        # 提取查询中的关键词
        keywords = query.split()

        matched_tables = set()
        for keyword in keywords:
            if len(keyword) >= 2:  # 忽略太短的词
                results = self.search_tables(keyword, top_k=3)
                for r in results:
                    matched_tables.add(r['name'])

        if not matched_tables:
            return "未找到匹配的知识"

        # 获取匹配表的详细信息
        lines = ["相关知识:"]
        lines.append("=" * 60)

        for table_name in list(matched_tables)[:top_k]:
            schema = self.get_table_schema(table_name)
            if schema:
                lines.append(f"\n表：{schema['name']} ({schema['name_cn']})")
                col_names = ", ".join([f"{c['name']}({c['name_cn']})" for c in schema['columns'][:8]])
                lines.append(f"字段：{col_names}{'...' if len(schema['columns']) > 8 else ''}")

        lines.append("=" * 60)
        return "\n".join(lines)


def build_dynamic_prompt(neo4j_retriever: Neo4jKnowledgeRetriever,
                         user_query: str,
                         dialect: str = "postgresql",
                         top_k: int = 5) -> Tuple[str, Dict]:
    """
    基于 Neo4j 知识库动态构建提示词

    Args:
        neo4j_retriever: Neo4j 检索器
        user_query: 用户查询
        dialect: SQL 方言
        top_k: 检索数量

    Returns:
        (提示词，上下文信息)
    """
    # 1. 搜索相关表
    keywords = user_query.split()
    relevant_tables = []
    for keyword in keywords:
        if len(keyword) >= 2:
            results = neo4j_retriever.search_tables(keyword, top_k=2)
            for r in results:
                if r not in relevant_tables:
                    relevant_tables.append(r)

    table_names = [t['name'] for t in relevant_tables[:top_k]]

    # 2. 获取 schema
    schema_str = neo4j_retriever.get_schema_for_prompt(table_names)

    # 3. 搜索相关知识
    knowledge_str = neo4j_retriever.search_knowledge(user_query, top_k=top_k)

    # 4. 构建系统提示词
    system_prompt = f"""你是一个专业的 SQL 生成专家，专门处理银行金融领域的数据查询。

任务：根据用户的自然语言查询，生成准确的 {dialect} SQL 语句。

规则：
1. 只输出 SQL 语句，不要包含解释
2. 确保 SQL 语法正确，符合 {dialect} 方言
3. 使用提供的表名和字段名，不要臆造
4. 如果信息不足，返回 "INSUFFICIENT_INFO"
5. 注意表之间的关联关系，使用正确的 JOIN 条件
6. 涉及金额、数量等计算时，注意数据类型
7. 对于时间查询，使用 CURRENT_DATE、INTERVAL 等 {dialect} 函数

{schema_str}

{knowledge_str}

用户查询：{user_query}

请生成 SQL 语句："""

    context = {
        'relevant_tables': relevant_tables,
        'table_names': table_names,
        'schema': schema_str,
        'knowledge': knowledge_str
    }

    return system_prompt, context


# ==================== 测试函数 ====================

def test_neo4j_retriever():
    """测试 Neo4j 检索器"""
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    print("=" * 60)
    print("Neo4j 知识库检索器测试")
    print("=" * 60)

    retriever = Neo4jKnowledgeRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # 测试 1: 获取所有表
    print("\n📋 测试 1: 获取所有表")
    tables = retriever.get_all_tables()
    print(f"   共 {len(tables)} 个表:")
    for t in tables[:5]:
        print(f"   - {t['name']} ({t['name_cn']}) - {t['column_count']} 个字段")

    # 测试 2: 搜索表
    print("\n🔍 测试 2: 搜索表 '客户'")
    results = retriever.search_tables("客户", top_k=5)
    for r in results:
        print(f"   - {r['name']} ({r['name_cn']})")

    # 测试 3: 获取表 schema
    print("\n📊 测试 3: 获取 bank_customer 表 schema")
    schema = retriever.get_table_schema("bank_customer")
    print(f"   表：{schema['name_cn']}")
    print(f"   字段数：{len(schema['columns'])}")
    print(f"   关联数：{len(schema['relations'])}")
    print("   前 5 个字段:")
    for col in schema['columns'][:5]:
        print(f"     - {col['name']} ({col['name_cn']}): {col['type']}")

    # 测试 4: 构建提示词
    print("\n💬 测试 4: 构建动态提示词")
    query = "查询月收入 10000 以上的客户及其账户信息"
    prompt, context = build_dynamic_prompt(retriever, query)
    print(f"   查询：{query}")
    print(f"   相关表：{context['table_names']}")
    print(f"   提示词长度：{len(prompt)} 字符")
    print(f"   提示词预览：{prompt[:200]}...")

    retriever.close()
    print("\n✅ 测试完成")


if __name__ == '__main__':
    test_neo4j_retriever()
