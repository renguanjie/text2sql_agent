"""
Schema Agent CLI - AI Agent 专属的 Schema 检索工具

基于现有知识图谱和检索能力，提供命令行方式的 Schema 检索服务。
外部 AI Agent (如 Claude Code) 可通过此工具获取数据库表结构信息。

用法:
    # 基础用法 - 检索 Schema 并由 LLM 总结
    python scripts/02_schema_search_cli.py "查询用户订单信息"

    # 仅返回检索到的原始 Schema 信息（不加 LLM 总结）
    python scripts/02_schema_search_cli.py "查询用户订单信息" --raw

    # 指定返回结果数量
    python scripts/02_schema_search_cli.py "查询用户订单信息" --top-k 10

    # 指定数据库
    python scripts/02_schema_search_cli.py "查询用户订单信息" --database-id 2
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.knowledge.neo4j_client import Neo4jClient
from core.retrieval.bm25_tfidf import KnowledgeRetriever
from core.embedding_factory import EmbeddingFactory
from core.llm_factory import create_llm
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_KEY
)
from loguru import logger


class SchemaSearcher:
    """Schema 检索器 - 复用项目现有的检索能力"""

    def __init__(self, neo4j_client: Neo4jClient):
        """
        初始化检索器

        Args:
            neo4j_client: Neo4j 客户端实例
        """
        self.neo4j_client = neo4j_client
        self.retriever = None
        self._init_retriever()

    def _init_retriever(self):
        """初始化检索器 - 从 Neo4j 加载 Schema 信息构建索引"""
        logger.info("正在初始化 Schema 检索器...")

        # 从 Neo4j 获取所有表节点
        tables = self.neo4j_client.execute_query("""
            MATCH (t:Table)
            RETURN t.name as name, t.database as database, t.database_id as database_id,
                   t.description as description
        """)

        # 获取所有字段节点
        columns = self.neo4j_client.execute_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            RETURN t.name as table_name, c.name as name, c.data_type as data_type,
                   c.description as description
        """)

        # 构建文档列表和元数据
        documents = []
        metadata = []

        # 为每张表创建文档
        for table in tables:
            table_name = table.get('name', '')
            table_desc = table.get('description', '')
            database = table.get('database', '')
            database_id = table.get('database_id')

            # 构建文档内容（表名 + 描述）
            doc_content = f"{table_name} {table_desc}".strip()
            if doc_content:
                documents.append(doc_content)
                metadata.append({
                    'node_type': 'table',
                    'name': table_name,
                    'database': database,
                    'database_id': database_id,
                    'description': table_desc
                })

        # 为每个字段创建文档
        for col in columns:
            table_name = col.get('table_name', '')
            col_name = col.get('name', '')
            col_type = col.get('data_type', '')
            col_desc = col.get('description', '')

            # 构建文档内容（表名。字段名 类型 描述）
            doc_content = f"{table_name}.{col_name} {col_type} {col_desc}".strip()
            if doc_content:
                documents.append(doc_content)
                metadata.append({
                    'node_type': 'column',
                    'table_name': table_name,
                    'name': col_name,
                    'data_type': col_type,
                    'description': col_desc
                })

        # 创建 Embedding 工厂
        embedding_factory = EmbeddingFactory(
            provider=EMBEDDING_PROVIDER,
            model=EMBEDDING_MODEL,
            api_key=EMBEDDING_API_KEY
        )

        # 创建检索器
        self.retriever = KnowledgeRetriever(
            config={
                'bm25_k1': 1.5,
                'bm25_b': 0.75,
                'bm25_weight': 0.5,
                'dense_weight': 0.5,
                'embedding_dim': 1536,
                'index_type': 'flat'
            },
            embedding_factory=embedding_factory
        )

        # 构建索引
        if documents:
            self.retriever.retriever.index_documents(documents, metadata)
            logger.info(f"Schema 检索器初始化完成，共索引 {len(documents)} 个节点")
        else:
            logger.warning("Neo4j 中没有找到表或字段数据")

    def search(self, query: str, top_k: int = 5, database_id: Optional[int] = None) -> List[Dict]:
        """
        检索 Schema 信息

        Args:
            query: 自然语言查询
            top_k: 返回结果数量
            database_id: 数据库 ID（可选，用于过滤）

        Returns:
            List[Dict]: 检索结果列表
        """
        if not self.retriever:
            logger.error("检索器未初始化")
            return []

        # 执行检索
        results = self.retriever.retriever.search(query, top_k=top_k * 2)  # 多检索一些用于过滤

        # 过滤数据库
        filtered_results = []
        for result, score in results:
            result_db_id = result.get('database_id')
            if database_id is None or result_db_id == database_id:
                filtered_results.append({
                    'metadata': result,
                    'score': score
                })

        # 按分数排序并截取 top_k
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        return filtered_results[:top_k]

    def get_full_schema_info(self, results: List[Dict]) -> Dict[str, Any]:
        """
        根据检索结果获取完整的 Schema 信息

        Args:
            results: 检索结果列表

        Returns:
            Dict: 完整的 Schema 信息
        """
        # 收集涉及的表名
        table_names = set()
        for result in results:
            metadata = result.get('metadata', {})
            if metadata.get('node_type') == 'table':
                table_names.add(metadata.get('name'))
            elif metadata.get('node_type') == 'column':
                table_names.add(metadata.get('table_name'))

        if not table_names:
            return {'tables': [], 'columns': [], 'concepts': []}

        # 从 Neo4j 获取详细的表和字段信息
        tables_result = self.neo4j_client.execute_query("""
            MATCH (t:Table)
            WHERE t.name IN $table_names
            RETURN t.name as name, t.name_cn as name_cn, t.description as description,
                   t.database as database, t.create_statement as create_statement
        """, {'table_names': list(table_names)})

        columns_result = self.neo4j_client.execute_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE t.name IN $table_names
            RETURN t.name as table_name, c.name as name, c.data_type as data_type,
                   c.description as description, c.is_primary_key as is_primary_key,
                   c.is_nullable as is_nullable
        """, {'table_names': list(table_names)})

        # 获取相关的业务概念
        concepts_result = self.neo4j_client.execute_query("""
            MATCH (c:BusinessConcept)
            RETURN c.name as name, c.description as description, c.tags as tags,
                   c.mapped_tables as mapped_tables
        """)

        # 格式化结果
        tables = []
        for t in tables_result:
            tables.append({
                'name': t.get('name', ''),
                'name_cn': t.get('name_cn', ''),
                'description': t.get('description', ''),
                'database': t.get('database', ''),
                'create_statement': t.get('create_statement', '')
            })

        # 按表名组织字段
        columns_by_table = {}
        for c in columns_result:
            table_name = c.get('table_name', '')
            if table_name not in columns_by_table:
                columns_by_table[table_name] = []
            columns_by_table[table_name].append({
                'name': c.get('name', ''),
                'data_type': c.get('data_type', ''),
                'description': c.get('description', ''),
                'is_primary_key': c.get('is_primary_key', False),
                'is_nullable': c.get('is_nullable', True)
            })

        concepts = []
        for c in concepts_result:
            concepts.append({
                'name': c.get('name', ''),
                'description': c.get('description', ''),
                'tags': c.get('tags', []),
                'mapped_tables': c.get('mapped_tables', [])
            })

        return {
            'tables': tables,
            'columns_by_table': columns_by_table,
            'concepts': concepts,
            'matched_nodes': results
        }


def format_schema_context(schema_info: Dict[str, Any]) -> str:
    """
    格式化 Schema 信息为上下文文本

    Args:
        schema_info: Schema 信息字典

    Returns:
        str: 格式化后的上下文文本
    """
    lines = []
    lines.append("=== 检索到的数据库 Schema 信息 ===")
    lines.append("")

    # 表信息
    if schema_info.get('tables'):
        lines.append("【相关表】")
        for table in schema_info['tables']:
            lines.append(f"  表名：{table.get('name', 'N/A')}")
            if table.get('name_cn'):
                lines.append(f"  中文名：{table.get('name_cn', 'N/A')}")
            if table.get('description'):
                lines.append(f"  描述：{table.get('description', 'N/A')}")
            lines.append(f"  数据库：{table.get('database', 'N/A')}")
            lines.append("")

    # 字段信息
    if schema_info.get('columns_by_table'):
        lines.append("【相关字段】")
        for table_name, columns in schema_info['columns_by_table'].items():
            lines.append(f"  表 `{table_name}` 的字段:")
            for col in columns:
                pk_mark = "[主键]" if col.get('is_primary_key') else ""
                lines.append(
                    f"    - {col.get('name', 'N/A')} ({col.get('data_type', 'N/A')}) "
                    f"{pk_mark} {col.get('description', '')}"
                )
            lines.append("")

    # 业务概念
    if schema_info.get('concepts'):
        lines.append("【相关业务概念】")
        for concept in schema_info['concepts']:
            tags_str = ', '.join(concept.get('tags', [])) if concept.get('tags') else ''
            lines.append(f"  - {concept.get('name', 'N/A')}: {concept.get('description', 'N/A')}")
            if tags_str:
                lines.append(f"    标签：{tags_str}")
        lines.append("")

    return '\n'.join(lines)


def generate_llm_response(query: str, schema_context: str, llm) -> str:
    """
    使用 LLM 生成精炼的回答

    Args:
        query: 用户查询
        schema_context: Schema 上下文
        llm: LLM 实例

    Returns:
        str: LLM 生成的回答
    """
    prompt = f"""你是一个数据字典助手。根据以下检索到的数据库表结构信息，回答用户的业务需求。

你需要明确指出：
1. 建议使用的表名（用反引号包裹）
2. 需要关注的核心字段（用反引号包裹）
3. 相关的业务标签或逻辑
4. 如果涉及多表，说明表之间的关联关系

检索到的 Schema 信息:
{schema_context}

用户需求：{query}

请用简洁清晰的语言回答，使用 Markdown 格式。"""

    messages = [
        {"role": "system", "content": "你是一个专业的数据字典助手，帮助用户理解数据库结构。"},
        {"role": "user", "content": prompt}
    ]

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"LLM 调用失败：{e}")
        return f"[LLM 调用失败，直接返回 Schema 信息]\n\n{schema_context}"


def main():
    """主函数"""
    # 配置命令行参数
    parser = argparse.ArgumentParser(
        description='Schema Agent CLI - AI Agent 专属的 Schema 检索工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python scripts/02_schema_search_cli.py "查询用户订单信息"

  # 仅返回原始 Schema 信息
  python scripts/02_schema_search_cli.py "查询用户订单信息" --raw

  # 指定返回结果数量
  python scripts/02_schema_search_cli.py "查询用户订单信息" --top-k 10

  # 指定数据库
  python scripts/02_schema_search_cli.py "查询用户订单信息" --database-id 2
        """
    )

    parser.add_argument(
        'query',
        type=str,
        help='自然语言查询语句'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='返回结果数量 (默认：5)'
    )

    parser.add_argument(
        '--database-id',
        type=int,
        default=None,
        help='数据库 ID (可选，用于过滤)'
    )

    parser.add_argument(
        '--raw',
        action='store_true',
        help='仅返回原始 Schema 信息，不使用 LLM 总结'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='以 JSON 格式输出结果'
    )

    args = parser.parse_args()

    # 配置日志 - 只输出错误级别，避免干扰 stdout
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    # 初始化 Neo4j 客户端
    try:
        neo4j_client = Neo4jClient(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        neo4j_client.connect()
    except Exception as e:
        print(f"[错误] Neo4j 连接失败：{e}", file=sys.stderr)
        sys.exit(1)

    try:
        # 创建检索器
        searcher = SchemaSearcher(neo4j_client)

        # 执行检索
        results = searcher.search(
            query=args.query,
            top_k=args.top_k,
            database_id=args.database_id
        )

        if not results:
            print("未检索到相关的 Schema 信息。", file=sys.stdout)
            sys.exit(0)

        # 获取完整 Schema 信息
        schema_info = searcher.get_full_schema_info(results)

        if args.json:
            # JSON 格式输出
            print(json.dumps(schema_info, ensure_ascii=False, indent=2))
        elif args.raw:
            # 原始 Schema 信息输出
            print(format_schema_context(schema_info))
        else:
            # 使用 LLM 生成精炼回答
            llm = create_llm(
                provider=LLM_PROVIDER,
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL
            )

            schema_context = format_schema_context(schema_info)
            response = generate_llm_response(args.query, schema_context, llm)
            print(response)

    finally:
        # 关闭连接
        if neo4j_client.driver:
            neo4j_client.driver.close()


if __name__ == '__main__':
    main()
