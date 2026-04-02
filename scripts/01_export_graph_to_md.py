"""
元数据 Markdown 导出工具
从 Neo4j 知识图谱中导出所有 Database 和 Table 节点的 Schema 信息到 Markdown 文件

用法:
    python scripts/01_export_graph_to_md.py

输出:
    data/schemas/schema_export_YYYYMMDD_HHMMSS.md
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.knowledge.neo4j_client import Neo4jClient
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from loguru import logger


def export_schema_to_markdown(neo4j_client: Neo4jClient, output_path: str) -> str:
    """
    从 Neo4j 导出 Schema 信息到 Markdown 文件

    Args:
        neo4j_client: Neo4j 客户端实例
        output_path: 输出文件路径

    Returns:
        str: 输出的文件路径
    """
    logger.info("开始从 Neo4j 导出 Schema 信息...")

    # 获取所有数据库节点
    db_query = """
        MATCH (d:Database)
        RETURN d.database_id as database_id, d.name as name, d.db_type as db_type,
               d.db_language as db_language, d.description as description
        ORDER BY d.database_id
    """
    databases = neo4j_client.execute_query(db_query)

    # 获取所有表节点
    table_query = """
        MATCH (t:Table)
        RETURN t.name as name, t.database as database, t.database_id as database_id,
               t.name_cn as name_cn, t.description as description,
               t.create_statement as create_statement
        ORDER BY t.database, t.name
    """
    tables = neo4j_client.execute_query(table_query)

    # 获取所有字段节点（通过关系获取所属表）
    column_query = """
        MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
        RETURN t.name as table_name, c.name as name, c.data_type as data_type,
               c.description as description, c.is_primary_key as is_primary_key,
               c.is_nullable as is_nullable, c.default_value as default_value
        ORDER BY t.name, c.name
    """
    columns = neo4j_client.execute_query(column_query)

    # 获取业务概念节点
    concept_query = """
        MATCH (c:BusinessConcept)
        RETURN c.name as name, c.description as description, c.tags as tags,
               c.mapped_tables as mapped_tables
        ORDER BY c.name
    """
    concepts = neo4j_client.execute_query(concept_query)

    # 构建表到字段的映射
    table_columns_map = {}
    for col in columns:
        table_name = col.get('table_name', '')
        if table_name not in table_columns_map:
            table_columns_map[table_name] = []
        table_columns_map[table_name].append({
            'name': col.get('name', ''),
            'data_type': col.get('data_type', ''),
            'description': col.get('description', ''),
            'is_primary_key': col.get('is_primary_key', False),
            'is_nullable': col.get('is_nullable', True),
            'default_value': col.get('default_value', '')
        })

    # 构建数据库到表的映射
    db_tables_map = {}
    for table in tables:
        db_name = table.get('database', '')
        if db_name not in db_tables_map:
            db_tables_map[db_name] = []
        db_tables_map[db_name].append(table)

    # 生成 Markdown 内容
    md_content = generate_markdown_content(
        databases=databases,
        tables=tables,
        table_columns_map=table_columns_map,
        db_tables_map=db_tables_map,
        concepts=concepts
    )

    # 写入文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(md_content, encoding='utf-8')

    logger.info(f"Schema 信息已导出到：{output_path}")
    return output_path


def generate_markdown_content(
    databases: list,
    tables: list,
    table_columns_map: dict,
    db_tables_map: dict,
    concepts: list
) -> str:
    """
    生成 Markdown 格式的内容

    Args:
        databases: 数据库列表
        tables: 表列表
        table_columns_map: 表到字段的映射
        db_tables_map: 数据库到表的映射
        concepts: 业务概念列表

    Returns:
        str: Markdown 内容
    """
    md_lines = []

    # YAML 头部
    md_lines.append("---")
    md_lines.append("title: 数据库 Schema 文档")
    md_lines.append(f"generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"database_count: {len(databases)}")
    md_lines.append(f"table_count: {len(tables)}")
    md_lines.append(f"column_count: {sum(len(cols) for cols in table_columns_map.values())}")
    md_lines.append("---")
    md_lines.append("")

    # 目录
    md_lines.append("# 📚 数据库 Schema 文档")
    md_lines.append("")
    md_lines.append("## 目录")
    md_lines.append("")
    md_lines.append("1. [数据库概览](#数据库概览)")
    md_lines.append("2. [业务概念](#业务概念)")
    for db in databases:
        db_name = db.get('name', 'unknown')
        safe_db_name = db_name.replace(' ', '-').lower()
        md_lines.append(f"3. [{db_name}](#{safe_db_name})")
    md_lines.append("")

    # 数据库概览
    md_lines.append("## 数据库概览")
    md_lines.append("")
    md_lines.append("| 数据库 ID | 数据库名称 | 类型 | 语言 | 描述 |")
    md_lines.append("|----------|----------|------|------|------|")
    for db in databases:
        md_lines.append(
            f"| {db.get('database_id', 'N/A')} | {db.get('name', 'N/A')} | "
            f"{db.get('db_type', 'N/A')} | {db.get('db_language', 'N/A')} | "
            f"{db.get('description', 'N/A')} |"
        )
    md_lines.append("")

    # 业务概念
    md_lines.append("## 业务概念")
    md_lines.append("")
    if concepts:
        for concept in concepts:
            tags_str = ', '.join(concept.get('tags', [])) if concept.get('tags') else 'N/A'
            mapped_tables_str = ', '.join(concept.get('mapped_tables', [])) if concept.get('mapped_tables') else 'N/A'
            md_lines.append(f"### {concept.get('name', 'N/A')}")
            md_lines.append("")
            md_lines.append(f"- **描述**: {concept.get('description', 'N/A')}")
            md_lines.append(f"- **业务标签**: {tags_str}")
            md_lines.append(f"- **映射表**: {mapped_tables_str}")
            md_lines.append("")
    else:
        md_lines.append("暂无业务概念")
        md_lines.append("")

    # 每个数据库的详细信息
    for db in databases:
        db_name = db.get('name', 'unknown')
        safe_db_name = db_name.replace(' ', '-').lower()
        db_id = db.get('database_id', 'N/A')

        md_lines.append(f"---")
        md_lines.append("")
        md_lines.append(f"## {db_name}")
        md_lines.append("")
        md_lines.append(f"**数据库 ID**: {db_id}")
        md_lines.append(f"**类型**: {db.get('db_type', 'N/A')}")
        md_lines.append(f"**语言**: {db.get('db_language', 'N/A')}")
        md_lines.append(f"**描述**: {db.get('description', 'N/A')}")
        md_lines.append("")

        # 获取该数据库的所有表
        db_tables = db_tables_map.get(db_name, [])

        if not db_tables:
            md_lines.append("*暂无表数据*")
            md_lines.append("")
            continue

        md_lines.append(f"### 表列表 (共 {len(db_tables)} 张)")
        md_lines.append("")

        for table in db_tables:
            table_name = table.get('name', 'N/A')
            table_name_cn = table.get('name_cn', 'N/A')
            table_desc = table.get('description', 'N/A')
            create_stmt = table.get('create_statement', '')

            md_lines.append(f"#### `{table_name}`")
            if table_name_cn and table_name_cn != 'N/A':
                md_lines.append(f"**中文名**: {table_name_cn}")
            md_lines.append("")
            md_lines.append(f"**描述**: {table_desc}")
            md_lines.append("")

            # 字段列表
            columns = table_columns_map.get(table_name, [])
            if columns:
                md_lines.append("**字段定义**:")
                md_lines.append("")
                md_lines.append("| 字段名 | 数据类型 | 描述 | 主键 | 可空 | 默认值 |")
                md_lines.append("|--------|---------|------|------|------|--------|")
                for col in columns:
                    pk_mark = "✅" if col.get('is_primary_key') else ""
                    nullable_mark = "✅" if col.get('is_nullable') else ""
                    md_lines.append(
                        f"| `{col.get('name', 'N/A')}` | {col.get('data_type', 'N/A')} | "
                        f"{col.get('description', 'N/A')} | {pk_mark} | {nullable_mark} | "
                        f"{col.get('default_value', 'N/A')} |"
                    )
                md_lines.append("")
            else:
                md_lines.append("*暂无字段信息*")
                md_lines.append("")

            # 创建语句
            if create_stmt:
                md_lines.append("**创建语句**:")
                md_lines.append("")
                md_lines.append("```sql")
                md_lines.append(create_stmt)
                md_lines.append("```")
                md_lines.append("")

            md_lines.append("")

    # 页脚
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("*此文档由 Schema Export Tool 自动生成*")

    return '\n'.join(md_lines)


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("Schema Export Tool - 元数据 Markdown 导出工具")
    logger.info("=" * 50)

    # 初始化 Neo4j 客户端
    try:
        neo4j_client = Neo4jClient(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        neo4j_client.connect()
        logger.info("Neo4j 连接成功")
    except Exception as e:
        logger.error(f"Neo4j 连接失败：{e}")
        sys.exit(1)

    # 生成输出文件路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = project_root / 'data' / 'schemas' / f'schema_export_{timestamp}.md'

    # 执行导出
    try:
        export_schema_to_markdown(neo4j_client, str(output_path))
        logger.info(f"✅ 导出完成：{output_path}")
    except Exception as e:
        logger.error(f"导出失败：{e}")
        sys.exit(1)
    finally:
        # 关闭连接
        if neo4j_client.driver:
            neo4j_client.driver.close()


if __name__ == '__main__':
    main()
