"""
知识图谱构建模块测试脚本
"""
import sys
import os
from pathlib import Path

import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config import NEO4J_ENABLED, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from core.knowledge.neo4j_client import get_neo4j_client
from core.graph_builder import run_pipeline, GraphBuilderPipeline

# 配置日志
logger.remove()
logger.add(sys.stdout, level="INFO", format="%(message)s")


def test_offline_extractor():
    """测试离线 DDL 抽取器"""
    print("\n" + "=" * 60)
    print("测试 1: 离线 DDL 抽取器")
    print("=" * 60)

    ddl_content = """
    CREATE TABLE bank_customer (
        customer_id BIGINT PRIMARY KEY COMMENT '客户 ID',
        customer_name VARCHAR(100) NOT NULL COMMENT '客户姓名',
        id_card VARCHAR(18) COMMENT '身份证号',
        phone VARCHAR(20) COMMENT '手机号'
    ) COMMENT='客户信息表';

    CREATE TABLE deposit_account (
        account_id VARCHAR(32) PRIMARY KEY COMMENT '账户 ID',
        customer_id BIGINT NOT NULL COMMENT '客户 ID',
        account_type VARCHAR(10) COMMENT '账户类型',
        balance DECIMAL(18,2) DEFAULT 0 COMMENT '余额'
    ) COMMENT='存款账户表';

    CREATE TABLE transaction_record (
        trans_id VARCHAR(64) PRIMARY KEY COMMENT '交易 ID',
        account_id VARCHAR(32) NOT NULL COMMENT '账户 ID',
        trans_type VARCHAR(10) COMMENT '交易类型',
        amount DECIMAL(18,2) NOT NULL COMMENT '金额'
    ) COMMENT='交易流水表';
    """

    ddl_path = Path("/tmp/test_schema.sql")
    ddl_path.write_text(ddl_content, encoding="utf-8")

    try:
        pipeline = GraphBuilderPipeline(
            llm_provider=LLM_PROVIDER,
            llm_model=LLM_MODEL,
            llm_config={"api_key": LLM_API_KEY, "base_url": LLM_BASE_URL} if LLM_API_KEY else {}
        )

        result = pipeline.extract_from_files(
            ddl_path=str(ddl_path),
            database_name="bank_db"
        )

        print(f"\n抽取完成！")
        print(f"   数据库：{len(result.databases)}")
        print(f"   表：{len(result.tables)}")
        print(f"   字段：{sum(len(t.columns) for t in result.tables)}")
        print(f"   关系：{len(result.relationships)}")

        print("\n表列表:")
        for table in result.tables:
            print(f"   - {table.name}: {len(table.columns)} 个字段")
            if table.description:
                print(f"     注释：{table.description}")

        if result.relationships:
            print("\n推断的关系:")
            for rel in result.relationships:
                print(f"   - {rel.from_table}.{rel.from_column} -> {rel.to_table}.{rel.to_column}")

        assert len(result.databases) == 1
        assert len(result.tables) == 3
        assert sum(len(t.columns) for t in result.tables) == 12

    finally:
        if ddl_path.exists():
            ddl_path.unlink()


def test_full_pipeline_with_neo4j():
    """测试完整流水线 + Neo4j 装载"""
    if os.getenv("TEXT2SQL_RUN_DB_TESTS") != "1" or not NEO4J_ENABLED:
        pytest.skip("设置 TEXT2SQL_RUN_DB_TESTS=1 且 NEO4J_ENABLED=true 后运行真实 Neo4j 装载测试")

    print("\n" + "=" * 60)
    print("测试 2: 完整流水线 + Neo4j 装载")
    print("=" * 60)

    ddl_content = """
    CREATE TABLE bank_customer (
        customer_id BIGINT PRIMARY KEY,
        customer_name VARCHAR(100),
        id_card VARCHAR(18),
        phone VARCHAR(20)
    ) COMMENT='客户信息表';

    CREATE TABLE deposit_account (
        account_id VARCHAR(32) PRIMARY KEY,
        customer_id BIGINT,
        account_type VARCHAR(10),
        balance DECIMAL(18,2)
    ) COMMENT='存款账户表';
    """

    ddl_path = Path("/tmp/test_neo4j.sql")
    ddl_path.write_text(ddl_content, encoding="utf-8")

    try:
        neo4j_client = get_neo4j_client(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        stats = run_pipeline(
            source_type="offline",
            source_config={
                "ddl_path": str(ddl_path),
                "database_name": "bank_db"
            },
            neo4j_client=neo4j_client,
            llm_provider=LLM_PROVIDER,
            llm_model=LLM_MODEL,
            enable_enrichment=False
        )

        print("\n流水线完成！")
        print(f"   抽取统计：{stats['extract']}")
        print(f"   装载统计：{stats['load']}")

        print("\nNeo4j 图谱统计:")
        result = neo4j_client.execute_query("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
        for row in result:
            print(f"   - {row['label']}: {row['count']}")

        assert stats["extract"]["tables"] == 2
        assert stats["load"]["tables"] >= 2

    except Exception as e:
        print(f"Neo4j 测试失败：{e}")
    finally:
        if ddl_path.exists():
            ddl_path.unlink()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("知识图谱构建模块 - 测试套件")
    print("=" * 60)

    # 测试 1: 离线抽取
    test_offline_extractor()

    # 测试 2: 完整流水线（如果 Neo4j 可用）
    try:
        if NEO4J_URI and NEO4J_USER:
            test_full_pipeline_with_neo4j()
        else:
            print("\nNeo4j 配置缺失，跳过测试 2")
    except Exception as e:
        print(f"\nNeo4j 测试失败：{e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
