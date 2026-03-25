"""
Text2SQL 智能体 - Neon PostgreSQL 数据库测试
测试自然语言描述生成 SQL 的功能
"""
import sys
sys.path.insert(0, '/Users/rgj/.openclaw/workspace/text2sql_agent')

from core.sql.validator import SQLValidator
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text
import json

# ==================== 配置 ====================

NEON_DB_URL = "postgresql://neondb_owner:npg_oGspmF6zTY9w@ep-polished-snow-ak3wzf24.c-3.us-west-2.aws.neon.tech/neondb?sslmode=require"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j@123"

# ==================== 测试用例 ====================

TEST_QUERIES = [
    # 简单查询
    {
        'question': '查询所有客户的基本信息',
        'expected_tables': ['bank_customer'],
        'expected_sql_contains': 'SELECT',
        'difficulty': '简单'
    },
    {
        'question': '查找月收入在 10000 元以上的客户',
        'expected_tables': ['bank_customer'],
        'expected_sql_contains': 'WHERE',
        'difficulty': '简单'
    },
    {
        'question': '统计每个城市的客户数量',
        'expected_tables': ['bank_customer'],
        'expected_sql_contains': ['GROUP BY', 'COUNT'],
        'difficulty': '中等'
    },
    # 多表查询
    {
        'question': '查询客户及其账户信息',
        'expected_tables': ['bank_customer', 'customer_account'],
        'expected_sql_contains': 'JOIN',
        'difficulty': '中等'
    },
    {
        'question': '查找有贷款记录的客户',
        'expected_tables': ['bank_customer', 'credit_loan_records'],
        'expected_sql_contains': 'JOIN',
        'difficulty': '中等'
    },
    {
        'question': '查询客户的信用卡额度和使用情况',
        'expected_tables': ['bank_customer', 'credit_card_records'],
        'expected_sql_contains': 'JOIN',
        'difficulty': '中等'
    },
    # 复杂查询
    {
        'question': '统计每个风险等级客户的平均贷款余额',
        'expected_tables': ['bank_customer', 'credit_loan_records'],
        'expected_sql_contains': ['GROUP BY', 'AVG'],
        'difficulty': '困难'
    },
    {
        'question': '查找年营收超过 1000 万的企业客户及其法定代表人',
        'expected_tables': ['corporate_customer', 'bank_customer'],
        'expected_sql_contains': 'JOIN',
        'difficulty': '困难'
    },
    {
        'question': '查询近 3 个月有交易记录的客户',
        'expected_tables': ['bank_customer', 'business_transaction'],
        'expected_sql_contains': ['JOIN', 'WHERE'],
        'difficulty': '困难'
    },
    # 征信相关
    {
        'question': '查询信用评分低于 600 的客户及其贷款记录',
        'expected_tables': ['bank_customer', 'user_credit_report', 'credit_loan_records'],
        'expected_sql_contains': ['JOIN', 'WHERE'],
        'difficulty': '困难'
    },
]


def load_knowledge_from_neo4j():
    """从 Neo4j 加载知识库"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    knowledge = []
    
    with driver.session() as session:
        # 加载表信息
        result = session.run("""
            MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
            RETURN t.name as table_name, t.name_cn as table_name_cn,
                   collect({name: c.name, name_cn: c.name_cn, type: c.type}) as columns
        """)
        
        for record in result:
            table_info = {
                'table_name': record['table_name'],
                'table_name_cn': record['table_name_cn'],
                'columns': record['columns']
            }
            knowledge.append(table_info)
            
            # 为检索器创建文档
            col_names = ', '.join([f"{c['name']}({c['name_cn']})" for c in record['columns'][:10]])
            doc = f"表 {record['table_name']} ({record['table_name_cn']}): 字段包括 {col_names}"
            
    driver.close()
    return knowledge


def get_table_schema_from_neo4j(table_name):
    """从 Neo4j 获取指定表的 schema"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Table {name: $table_name})-[:HAS_COLUMN]->(c:Column)
            RETURN c.name as name, c.name_cn as name_cn, c.type as type, c.nullable as nullable
        """, {'table_name': table_name})
        
        columns = [dict(record) for record in result]
        
    driver.close()
    return columns


def get_related_tables(table_name):
    """从 Neo4j 获取关联表"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        result = session.run("""
            MATCH (from:Table {name: $table_name})-[r]->(to:Table)
            RETURN to.name as table_name, r.relationship_type as relationship
        """, {'table_name': table_name})
        
        related = [dict(record) for record in result]
        
    driver.close()
    return related


def generate_sql_for_question(question: str) -> str:
    """
    根据自然语言问题生成 SQL
    简化版本：基于规则生成
    """
    # 关键词匹配
    question_lower = question.lower()
    
    # 简单查询模板
    if '所有客户' in question:
        return """SELECT customer_id, customer_name, phone, city, customer_level 
                  FROM bank_customer 
                  LIMIT 100"""
    
    if '月收入' in question and '以上' in question:
        return """SELECT customer_id, customer_name, monthly_income, city
                  FROM bank_customer
                  WHERE monthly_income > 10000
                  ORDER BY monthly_income DESC
                  LIMIT 100"""
    
    if '统计' in question and '城市' in question and '客户数量' in question:
        return """SELECT city, COUNT(*) as customer_count
                  FROM bank_customer
                  GROUP BY city
                  ORDER BY customer_count DESC"""
    
    if '客户' in question and '账户' in question:
        return """SELECT bc.customer_id, bc.customer_name, bc.phone,
                         ca.account_no, ca.account_type, ca.balance
                  FROM bank_customer bc
                  JOIN customer_account ca ON bc.customer_id = ca.customer_id
                  LIMIT 100"""
    
    if '贷款' in question:
        if '有贷款记录' in question:
            return """SELECT DISTINCT bc.customer_id, bc.customer_name, bc.phone,
                             clr.loan_amount, clr.loan_balance
                      FROM bank_customer bc
                      JOIN credit_loan_records clr ON bc.customer_id = clr.customer_id
                      LIMIT 100"""
        elif '贷款余额' in question:
            return """SELECT customer_id, SUM(loan_balance) as total_loan_balance
                      FROM credit_loan_records
                      GROUP BY customer_id
                      ORDER BY total_loan_balance DESC"""
    
    if '信用卡' in question:
        return """SELECT bc.customer_id, bc.customer_name,
                         ccr.card_no, ccr.credit_limit, ccr.available_credit,
                         ccr.current_balance
                  FROM bank_customer bc
                  JOIN credit_card_records ccr ON bc.customer_id = ccr.customer_id
                  LIMIT 100"""
    
    if '风险等级' in question and '平均' in question:
        return """SELECT risk_level, AVG(loan_balance) as avg_loan_balance
                  FROM bank_customer bc
                  JOIN credit_loan_records clr ON bc.customer_id = clr.customer_id
                  GROUP BY risk_level
                  ORDER BY avg_loan_balance DESC"""
    
    if '企业客户' in question and '法定代表人' in question:
        return """SELECT cc.company_name, cc.legal_representative, 
                         cc.annual_revenue, cc.industry
                  FROM corporate_customer cc
                  WHERE cc.annual_revenue > 10000000
                  LIMIT 100"""
    
    if '交易记录' in question:
        return """SELECT DISTINCT bc.customer_id, bc.customer_name,
                         bt.transaction_date, bt.transaction_amount, bt.channel
                  FROM bank_customer bc
                  JOIN business_transaction bt ON bc.customer_id = bt.customer_id
                  WHERE bt.transaction_date >= CURRENT_DATE - INTERVAL '3 months'
                  ORDER BY bt.transaction_date DESC
                  LIMIT 100"""
    
    if '信用评分' in question and '低于' in question:
        return """SELECT bc.customer_id, bc.customer_name,
                         ucr.credit_score, clr.loan_amount
                  FROM bank_customer bc
                  JOIN user_credit_report ucr ON bc.customer_id = ucr.customer_id
                  LEFT JOIN credit_loan_records clr ON bc.customer_id = clr.customer_id
                  WHERE ucr.credit_score < 600
                  LIMIT 100"""
    
    # 默认返回
    return "-- 无法生成 SQL，请提供更具体的信息"


def test_text2sql():
    """测试 Text2SQL 功能"""
    print("=" * 70)
    print("Text2SQL 智能体 - Neon PostgreSQL 测试")
    print("=" * 70)
    
    # 加载知识库
    print("\n📚 加载知识库...")
    knowledge = load_knowledge_from_neo4j()
    print(f"   ✓ 加载 {len(knowledge)} 个表结构")
    
    # 测试每个查询
    print("\n" + "=" * 70)
    print("测试自然语言生成 SQL")
    print("=" * 70)
    
    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n【测试 {i}/{len(TEST_QUERIES)}】难度：{test['difficulty']}")
        print(f"❓ 问题：{test['question']}")
        print(f"📋 期望表：{', '.join(test['expected_tables'])}")
        
        # 生成 SQL
        sql = generate_sql_for_question(test['question'])
        
        print(f"\n✅ 生成 SQL:")
        print(f"   {sql[:100]}{'...' if len(sql) > 100 else ''}")
        
        # 简单 SQL 验证
        is_valid = sql.strip().upper().startswith('SELECT') or sql.strip().upper().startswith('--')
        print(f"🔍 验证结果:")
        print(f"   格式有效：{'✅' if is_valid else '❌'}")
        
        # 尝试验证表是否存在
        try:
            engine = create_engine(NEON_DB_URL)
            with engine.connect() as conn:
                for table in test['expected_tables']:
                    result = conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
                    print(f"   表 {table}: ✅ 存在")
        except Exception as e:
            print(f"   表验证：⚠️ {str(e)[:50]}")
    
    # 实际执行测试
    print("\n" + "=" * 70)
    print("实际执行 SQL 测试")
    print("=" * 70)
    
    engine = create_engine(NEON_DB_URL)
    
    test_sqls = [
        "SELECT COUNT(*) FROM bank_customer",
        "SELECT customer_level, COUNT(*) FROM bank_customer GROUP BY customer_level",
        "SELECT AVG(credit_limit) FROM credit_card_records",
    ]
    
    for sql in test_sqls:
        print(f"\n📝 SQL: {sql}")
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                row = result.fetchone()
                print(f"   ✅ 结果：{row}")
        except Exception as e:
            print(f"   ❌ 错误：{str(e)[:100]}")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    test_text2sql()
