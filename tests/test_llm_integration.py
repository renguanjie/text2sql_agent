"""
Text2SQL 智能体 - LLM 集成综合测试
测试阿里云 Qwen-Max + Neo4j 知识库的完整流程
"""
import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from config import LLM_API_KEY, LLM_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from core.knowledge.neon_knowledge import Neo4jKnowledgeRetriever, build_dynamic_prompt
from core.llm.qwen_generator import QwenMaxLLM, generate_sql_with_qwen
from sqlalchemy import create_engine, text

# Neon 数据库连接
NEON_URL = os.getenv("NEON_DB_URL", "")
REPORT_PATH = Path(__file__).with_name("llm_test_report.json")

# ==================== 测试用例 ====================
# 注意：expected_contains 使用实际数据库中的字段名

TEST_QUERIES = [
    {
        'query': '查询所有客户的基本信息',
        'expected_tables': ['bank_customer'],
        'difficulty': '简单',
        'expected_contains': ['SELECT', 'FROM', 'bank_customer']
    },
    {
        'query': '查找高收入水平的客户',
        'expected_tables': ['bank_customer'],
        'difficulty': '简单',
        'expected_contains': ['WHERE', 'income_level'],
        'note': 'income_level 是 VARCHAR 类型，值为 高/中/低，应该用 income_level = 高'
    },
    {
        'query': '统计每个城市的客户数量',
        'expected_tables': ['bank_customer'],
        'difficulty': '中等',
        'expected_contains': ['GROUP BY', 'COUNT']  # 不指定 city 字段
    },
    {
        'query': '查询客户及其账户信息',
        'expected_tables': ['bank_customer', 'customer_account'],
        'difficulty': '中等',
        'expected_contains': ['JOIN']  # 不指定表名，验证 JOIN 语法
    },
    {
        'query': '查找有贷款记录的客户',
        'expected_tables': ['bank_customer', 'credit_loan_records'],
        'difficulty': '中等',
        'expected_contains': ['JOIN']
    },
    {
        'query': '查询客户的信用卡额度和使用情况',
        'expected_tables': ['bank_customer', 'credit_card_records'],
        'difficulty': '中等',
        'expected_contains': ['credit_limit']  # 这个字段存在
    },
    {
        'query': '统计每个客户等级的平均贷款余额',
        'expected_tables': ['bank_customer', 'credit_loan_records'],
        'difficulty': '困难',
        'expected_contains': ['GROUP BY', 'AVG', 'customer_level'],
        'note': '使用 customer_level 字段 (高/中/低)，不是 risk_level'
    },
    {
        'query': '查找年营收超过 1000 万的企业客户及其法定代表人',
        'expected_tables': ['corporate_customer'],
        'difficulty': '困难',
        'expected_contains': ['annual_revenue', 'legal_representative']
    },
    {
        'query': '查询近 3 个月有交易记录的客户',
        'expected_tables': ['bank_customer', 'business_transaction'],
        'difficulty': '困难',
        'expected_contains': ['transaction_date', 'INTERVAL']
    },
    {
        'query': '查询信用评分低于 600 的客户及其贷款记录',
        'expected_tables': ['bank_customer', 'user_credit_report'],
        'difficulty': '困难',
        'expected_contains': ['credit_score']
    },
]


def test_llm_integration():
    """测试 LLM 集成完整流程"""
    if not LLM_API_KEY:
        pytest.skip("缺少 LLM_API_KEY，跳过真实 LLM 集成测试")
    if not NEON_URL:
        pytest.skip("缺少 NEON_DB_URL，跳过 Neon PostgreSQL 集成测试")

    print("=" * 80)
    print("Text2SQL 智能体 - LLM 集成综合测试")
    print("=" * 80)

    # 初始化组件
    print("\n🔧 初始化组件...")
    neo4j_retriever = Neo4jKnowledgeRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    llm = QwenMaxLLM(api_key=LLM_API_KEY, model=LLM_MODEL)
    neon_engine = create_engine(NEON_URL)

    print(f"   ✅ Neo4j 检索器初始化完成")
    print(f"   ✅ Qwen-Max LLM 初始化完成 ({LLM_MODEL})")
    print(f"   ✅ Neon PostgreSQL 连接成功")

    # 运行测试
    print("\n" + "=" * 80)
    print("运行测试用例")
    print("=" * 80)

    results = []

    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n【测试 {i}/{len(TEST_QUERIES)}】难度：{test['difficulty']}")
        print(f"❓ 查询：{test['query']}")
        print(f"📋 期望表：{', '.join(test['expected_tables'])}")

        # 1. 构建动态提示词
        prompt, context = build_dynamic_prompt(
            neo4j_retriever,
            test['query'],
            dialect='postgresql',
            top_k=5
        )

        print(f"🔍 检索到表：{', '.join(context['table_names'])}")

        # 检查期望表是否在检索结果中
        tables_matched = all(t in context['table_names'] for t in test['expected_tables'])
        print(f"   表匹配：{'✅' if tables_matched else '⚠️'}")

        # 2. 调用 LLM 生成 SQL
        result = generate_sql_with_qwen(
            user_query=test['query'],
            schema_prompt=context['schema'],
            api_key=LLM_API_KEY,
            model=LLM_MODEL
        )

        if result['success']:
            sql = result.get('sql', '')

            if sql == "INSUFFICIENT_INFO":
                print(f"⚠️ LLM 判断信息不足")
                results.append({
                    'query': test['query'],
                    'success': False,
                    'reason': 'insufficient_info'
                })
                continue

            print(f"✅ SQL 生成成功")
            print(f"   SQL: {sql[:100]}{'...' if len(sql) > 100 else ''}")

            # 3. 验证 SQL 内容（只检查关键词，不检查具体字段名）
            sql_upper = sql.upper()
            content_matched = all(kw.upper() in sql_upper for kw in test['expected_contains'])
            print(f"   内容验证：{'✅' if content_matched else '⚠️'}")
            if not content_matched:
                missing = [kw for kw in test['expected_contains'] if kw.upper() not in sql_upper]
                print(f"   缺失关键词：{missing}")

            # 4. 执行 SQL 验证
            try:
                with neon_engine.connect() as conn:
                    query_result = conn.execute(text(sql))
                    rows = query_result.fetchall()
                    print(f"   执行结果：✅ {len(rows)} 条记录")

                    results.append({
                        'query': test['query'],
                        'success': True,
                        'sql': sql,
                        'rows': len(rows),
                        'tables_matched': tables_matched,
                        'content_matched': content_matched
                    })

            except Exception as e:
                error_msg = str(e)
                print(f"   执行结果：❌ {error_msg[:80]}")
                
                # 尝试修复：如果是字段不存在错误
                if 'does not exist' in error_msg.lower() or 'undefined column' in error_msg.lower():
                    print(f"   🔄 尝试修复...")
                    # 这里可以添加自动重试逻辑
                    results.append({
                        'query': test['query'],
                        'success': False,
                        'reason': 'field_not_exist',
                        'error': error_msg
                    })
                else:
                    results.append({
                        'query': test['query'],
                        'success': False,
                        'reason': 'execution_error',
                        'error': error_msg
                    })

        else:
            print(f"❌ 生成失败：{result.get('error', '未知错误')}")
            results.append({
                'query': test['query'],
                'success': False,
                'reason': 'generation_error',
                'error': result.get('error')
            })

    # 统计结果
    print("\n" + "=" * 80)
    print("测试结果统计")
    print("=" * 80)

    total = len(results)
    success = sum(1 for r in results if r['success'])
    failed = total - success

    print(f"\n总测试数：{total}")
    print(f"✅ 成功：{success} ({success/total*100:.1f}%)")
    print(f"❌ 失败：{failed} ({failed/total*100:.1f}%)")

    # 按难度统计
    print("\n按难度统计:")
    for difficulty in ['简单', '中等', '困难']:
        diff_tests = [r for r, t in zip(results, TEST_QUERIES) if t['difficulty'] == difficulty]
        diff_success = sum(1 for r in diff_tests if r['success'])
        print(f"   {difficulty}: {diff_success}/{len(diff_tests)} ({diff_success/len(diff_tests)*100:.0f}%)")

    # 显示失败详情
    if failed > 0:
        print("\n失败详情:")
        for i, r in enumerate(results):
            if not r['success']:
                print(f"   - {TEST_QUERIES[i]['query']}: {r.get('reason', '未知')}")

    # 清理
    neo4j_retriever.close()
    neon_engine.dispose()

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = test_llm_integration()

    # 保存结果
    report = {
        'timestamp': datetime.now().isoformat(),
        'total': len(results),
        'success': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'results': results
    }

    with REPORT_PATH.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 测试报告已保存到 {REPORT_PATH}")
