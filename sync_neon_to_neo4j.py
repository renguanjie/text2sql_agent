"""
Neon PostgreSQL 表结构同步到 Neo4j 知识库
包含中文字段翻译和表关联关系构建
"""
import json
from neo4j import GraphDatabase
from sqlalchemy import create_engine, inspect, text

# ==================== 配置 ====================

NEON_DB_URL = "postgresql://neondb_owner:npg_oGspmF6zTY9w@ep-polished-snow-ak3wzf24.c-3.us-west-2.aws.neon.tech/neondb?sslmode=require"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j@123"

# ==================== 字段中文翻译映射 ====================

COLUMN_TRANSLATIONS = {
    # 通用字段
    'id': 'ID',
    'name': '名称',
    'username': '用户名',
    'email': '邮箱',
    'password': '密码',
    'hashed_password': '加密密码',
    'full_name': '全名',
    'is_active': '是否激活',
    'is_superuser': '是否超级管理员',
    'created_at': '创建时间',
    'updated_at': '更新时间',
    'deleted_at': '删除时间',
    'created_by': '创建人',
    'updated_by': '更新人',
    'status': '状态',
    'type': '类型',
    'remarks': '备注',
    
    # 员工相关
    'department': '部门',
    'position': '职位',
    'join_date': '入职日期',
    
    # 客户相关
    'customer_id': '客户 ID',
    'customer_no': '客户编号',
    'customer_name': '客户姓名',
    'customer_type': '客户类型',
    'phone': '电话',
    'mobile': '手机',
    'address': '地址',
    'city': '城市',
    'province': '省份',
    'country': '国家',
    'postal_code': '邮编',
    'id_card': '身份证号',
    'id_card_type': '证件类型',
    'gender': '性别',
    'birthday': '生日',
    'age': '年龄',
    'occupation': '职业',
    'company': '公司',
    'income': '收入',
    'education': '学历',
    'marital_status': '婚姻状况',
    
    # 企业客户相关
    'company_name': '公司名称',
    'unified_social_credit_code': '统一社会信用代码',
    'company_type': '公司类型',
    'industry': '行业',
    'sub_industry': '子行业',
    'registered_capital': '注册资本',
    'paid_in_capital': '实缴资本',
    'establishment_date': '成立日期',
    'registration_address': '注册地址',
    'business_address': '经营地址',
    'business_scope': '经营范围',
    'legal_representative': '法定代表人',
    'legal_rep_id_card': '法人身份证号',
    'legal_rep_phone': '法人电话',
    'financial_contact': '财务联系人',
    'financial_phone': '财务电话',
    'financial_email': '财务邮箱',
    'annual_revenue': '年营收',
    'annual_profit': '年利润',
    'total_assets': '总资产',
    'total_liabilities': '总负债',
    'employee_count': '员工人数',
    'basic_account_no': '基本账户号',
    'basic_account_bank': '开户行',
    'account_balance': '账户余额',
    'credit_line': '授信额度',
    'used_credit': '已用额度',
    'loan_balance': '贷款余额',
    'credit_rating': '信用评级',
    'internal_rating': '内部评级',
    'risk_level': '风险等级',
    'risk_score': '风险评分',
    'customer_level': '客户等级',
    'kyc_status': 'KYC 状态',
    'kyc_date': 'KYC 日期',
    
    # 账户相关
    'account_id': '账户 ID',
    'account_no': '账号',
    'account_type': '账户类型',
    'balance': '余额',
    'currency': '币种',
    'open_date': '开户日期',
    'close_date': '销户日期',
    
    # 产品相关
    'product_id': '产品 ID',
    'product_name': '产品名称',
    'product_type': '产品类型',
    'description': '描述',
    'price': '价格',
    'amount': '金额',
    'rate': '利率',
    'term': '期限',
    'start_date': '开始日期',
    'end_date': '结束日期',
    'maturity_date': '到期日',
    
    # 交易相关
    'transaction_id': '交易 ID',
    'transaction_type': '交易类型',
    'transaction_date': '交易日期',
    'transaction_amount': '交易金额',
    'channel': '渠道',
    
    # 信贷相关
    'loan_id': '贷款 ID',
    'loan_type': '贷款类型',
    'loan_amount': '贷款金额',
    'loan_balance': '贷款余额',
    'interest_rate': '利率',
    'term_months': '期限 (月)',
    'repayment_method': '还款方式',
    'disbursement_date': '放款日期',
    'first_repayment_date': '首次还款日',
    'last_repayment_date': '最后还款日',
    'overdue_days': '逾期天数',
    'overdue_amount': '逾期金额',
    
    # 信用卡相关
    'card_id': '卡片 ID',
    'card_no': '卡号',
    'card_type': '卡片类型',
    'credit_limit': '信用额度',
    'available_credit': '可用额度',
    'bill_date': '账单日',
    'due_date': '还款日',
    'min_payment': '最低还款额',
    'current_balance': '当前余额',
    
    # 征信相关
    'report_id': '报告 ID',
    'query_reason': '查询原因',
    'query_date': '查询日期',
    'query_institution': '查询机构',
    'credit_score': '信用评分',
    'overdue_count': '逾期次数',
    'loan_count': '贷款笔数',
    'card_count': '信用卡张数',
    'guarantee_count': '担保笔数',
    
    # 会话相关
    'session_id': '会话 ID',
    'session_start': '会话开始时间',
    'session_end': '会话结束时间',
    'duration': '时长',
    'page_views': '页面浏览数',
    'source': '来源',
    'device': '设备',
    'ip_address': 'IP 地址',
    
    # 文件相关
    'file_id': '文件 ID',
    'file_name': '文件名',
    'file_path': '文件路径',
    'file_size': '文件大小',
    'file_type': '文件类型',
    'upload_date': '上传日期',
    
    # 标签相关
    'tag_id': '标签 ID',
    'tag_name': '标签名称',
    'tag_type': '标签类型',
    'tag_value': '标签值',
    
    # 活动相关
    'activity_id': '活动 ID',
    'activity_name': '活动名称',
    'activity_type': '活动类型',
    'branch_code': '网点编码',
    'branch_name': '网点名称',
}

# ==================== 表中文翻译映射 ====================

TABLE_TRANSLATIONS = {
    'employees': '员工表',
    'users': '用户表',
    'uploaded_files': '上传文件表',
    'bank_customer': '银行客户表',
    'customer_interaction': '客户互动表',
    'business_transaction': '业务交易表',
    'financial_product': '金融产品表',
    'customer_account': '客户账户表',
    'customer_tag_mapping': '客户标签映射表',
    'bank_product': '银行产品表',
    'bank_branch_activity': '银行网点活动表',
    'user_credit_report': '用户征信报告表',
    'credit_loan_records': '信贷记录表',
    'credit_card_records': '信用卡记录表',
    'credit_query_records': '征信查询记录表',
    'credit_public_records': '征信公共记录表',
    'customer_sessions': '客户会话表',
    'bank_wechat_sop_tags_v2': '银行微信 SOP 标签表',
    'corporate_customer': '企业客户表',
}

# ==================== 表关联关系定义 ====================

TABLE_RELATIONSHIPS = [
    # 客户相关关系
    {
        'from_table': 'bank_customer',
        'to_table': 'customer_account',
        'relationship': 'HAS_ACCOUNT',
        'on': 'customer_id'
    },
    {
        'from_table': 'bank_customer',
        'to_table': 'customer_interaction',
        'relationship': 'HAS_INTERACTION',
        'on': 'customer_id'
    },
    {
        'from_table': 'bank_customer',
        'to_table': 'business_transaction',
        'relationship': 'HAS_TRANSACTION',
        'on': 'customer_id'
    },
    {
        'from_table': 'bank_customer',
        'to_table': 'customer_sessions',
        'relationship': 'HAS_SESSION',
        'on': 'customer_id'
    },
    {
        'from_table': 'bank_customer',
        'to_table': 'customer_tag_mapping',
        'relationship': 'HAS_TAG',
        'on': 'customer_id'
    },
    # 企业客户关系
    {
        'from_table': 'corporate_customer',
        'to_table': 'bank_customer',
        'relationship': 'LINKED_TO_INDIVIDUAL',
        'on': 'customer_id'
    },
    # 产品相关关系
    {
        'from_table': 'customer_account',
        'to_table': 'financial_product',
        'relationship': 'HOLDS_PRODUCT',
        'on': 'product_id'
    },
    {
        'from_table': 'business_transaction',
        'to_table': 'financial_product',
        'relationship': 'TRANSACTION_FOR_PRODUCT',
        'on': 'product_id'
    },
    # 信贷相关关系
    {
        'from_table': 'bank_customer',
        'to_table': 'credit_loan_records',
        'relationship': 'HAS_LOAN',
        'on': 'customer_id'
    },
    {
        'from_table': 'bank_customer',
        'to_table': 'credit_card_records',
        'relationship': 'HAS_CREDIT_CARD',
        'on': 'customer_id'
    },
    {
        'from_table': 'bank_customer',
        'to_table': 'user_credit_report',
        'relationship': 'HAS_CREDIT_REPORT',
        'on': 'customer_id'
    },
    # 活动相关关系
    {
        'from_table': 'bank_branch_activity',
        'to_table': 'customer_interaction',
        'relationship': 'GENERATES_INTERACTION',
        'on': 'activity_id'
    },
]


# ==================== Neo4j 客户端 ====================

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def clear_knowledge(self):
        """清空现有知识库"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ 已清空现有知识库")
        
    def create_table_node(self, table_name, table_comment):
        """创建表节点"""
        with self.driver.session() as session:
            session.run("""
                CREATE (t:Table {
                    name: $name,
                    name_cn: $name_cn,
                    comment: $comment
                })
            """, {
                'name': table_name,
                'name_cn': table_comment,
                'comment': f"表：{table_comment}"
            })
        print(f"  ✓ 创建表节点：{table_name} ({table_comment})")
        
    def create_column_node(self, table_name, column_info):
        """创建字段节点并关联到表"""
        col_name = column_info['name']
        col_type = column_info['type']
        col_cn = COLUMN_TRANSLATIONS.get(col_name, col_name)
        
        with self.driver.session() as session:
            session.run("""
                MATCH (t:Table {name: $table_name})
                CREATE (c:Column {
                    name: $name,
                    name_cn: $name_cn,
                    type: $type,
                    nullable: $nullable
                })
                CREATE (t)-[:HAS_COLUMN]->(c)
            """, {
                'table_name': table_name,
                'name': col_name,
                'name_cn': col_cn,
                'type': col_type,
                'nullable': column_info['nullable']
            })
            
    def create_relationship(self, from_table, to_table, relationship, on_column):
        """创建表关联关系"""
        # 使用动态关系类型需要 APOC，这里使用固定关系类型 + 属性
        with self.driver.session() as session:
            # 先删除已存在的关系
            session.run("""
                MATCH (from:Table {name: $from_table})
                MATCH (to:Table {name: $to_table})
                MATCH (from)-[r]->(to)
                DELETE r
            """, {
                'from_table': from_table,
                'to_table': to_table
            })
            # 创建新关系
            session.run("""
                MATCH (from:Table {name: $from_table})
                MATCH (to:Table {name: $to_table})
                CREATE (from)-[r:CONNECTS {
                    relationship_type: $relationship,
                    on_column: $on_column
                }]->(to)
            """, {
                'from_table': from_table,
                'to_table': to_table,
                'relationship': relationship,
                'on_column': on_column
            })


def sync_schema_to_neo4j():
    """同步表结构到 Neo4j"""
    print("=" * 60)
    print("Neon PostgreSQL → Neo4j 知识库同步")
    print("=" * 60)
    
    # 加载表结构
    with open('neon_schema.json', 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    # 连接 Neo4j
    print("\n📡 连接 Neo4j...")
    neo4j = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # 清空现有知识库
    print("\n🗑️  清空现有知识库...")
    neo4j.clear_knowledge()
    
    # 创建表节点和字段节点
    print("\n📊 创建表结构...")
    for table_name, table_info in schema.items():
        table_cn = TABLE_TRANSLATIONS.get(table_name, table_name)
        print(f"\n📋 表：{table_name} ({table_cn})")
        
        # 创建表节点
        neo4j.create_table_node(table_name, table_cn)
        
        # 创建字段节点
        for col in table_info['columns']:
            neo4j.create_column_node(table_name, col)
        
        # 创建外键关系
        for fk in table_info['foreign_keys']:
            for i, ref_col in enumerate(fk['referred_columns']):
                from_col = fk['constrained_columns'][i] if i < len(fk['constrained_columns']) else fk['constrained_columns'][0]
                print(f"  → 外键：{table_name}.{from_col} → {fk['referred_table']}.{ref_col}")
    
    # 创建预定义的关系
    print("\n🔗 创建表关联关系...")
    for rel in TABLE_RELATIONSHIPS:
        neo4j.create_relationship(
            rel['from_table'],
            rel['to_table'],
            rel['relationship'],
            rel['on']
        )
        print(f"  ✓ {rel['from_table']} -[{rel['relationship']}]-> {rel['to_table']}")
    
    # 统计
    with neo4j.driver.session() as session:
        table_count = session.run("MATCH (t:Table) RETURN count(t) as count").single()['count']
        column_count = session.run("MATCH (c:Column) RETURN count(c) as count").single()['count']
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
    
    print(f"\n{'=' * 60}")
    print(f"✅ 同步完成!")
    print(f"   表：{table_count} 个")
    print(f"   字段：{column_count} 个")
    print(f"   关系：{rel_count} 个")
    print(f"{'=' * 60}")
    
    neo4j.close()


def test_neo4j_query():
    """测试 Neo4j 查询"""
    print("\n🧪 测试 Neo4j 查询...")
    
    neo4j = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    with neo4j.driver.session() as session:
        # 查询所有表
        result = session.run("MATCH (t:Table) RETURN t.name as name, t.name_cn as name_cn LIMIT 5")
        print("\n📋 表列表 (前 5 个):")
        for record in result:
            print(f"   - {record['name']} ({record['name_cn']})")
        
        # 查询某个表的字段
        result = session.run("""
            MATCH (t:Table {name: 'bank_customer'})-[:HAS_COLUMN]->(c:Column)
            RETURN c.name as name, c.name_cn as name_cn, c.type as type
            LIMIT 10
        """)
        print("\n📝 bank_customer 表字段 (前 10 个):")
        for record in result:
            print(f"   - {record['name']} ({record['name_cn']}): {record['type']}")
        
        # 查询表关系
        result = session.run("""
            MATCH (from:Table)-[r]->(to:Table)
            RETURN from.name as from_table, type(r) as rel_type, to.name as to_table
            LIMIT 5
        """)
        print("\n🔗 表关系 (前 5 个):")
        for record in result:
            print(f"   - {record['from_table']} -[{record['rel_type']}]-> {record['to_table']}")
    
    neo4j.close()


if __name__ == '__main__':
    sync_schema_to_neo4j()
    test_neo4j_query()
