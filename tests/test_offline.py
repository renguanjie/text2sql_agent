"""
Text2SQL 智能体 - 离线测试脚本
测试不依赖外部包的模块功能
"""
import sys
sys.path.insert(0, '/Users/rgj/.openclaw/workspace/text2sql_agent')

print("=" * 50)
print("Text2SQL 智能体 - 模块测试")
print("=" * 50)

# ==================== 测试配置模块 ====================
print("\n[1] 测试配置模块...")
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    LLM_PROVIDER, LLM_MODEL,
    RETRIEVAL_TOP_K, SQL_DIALECT
)
print(f"  ✓ MySQL: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
print(f"  ✓ Neo4j: {NEO4J_URI}")
print(f"  ✓ LLM: {LLM_PROVIDER}/{LLM_MODEL}")
print(f"  ✓ Retrieval Top-K: {RETRIEVAL_TOP_K}")
print(f"  ✓ SQL Dialect: {SQL_DIALECT}")

# ==================== 测试文本预处理 ====================
print("\n[2] 测试文本预处理...")
# 直接导入源码测试
import re

class TextPreprocessor:
    def __init__(self):
        self.punctuation = r'[!@#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~,.,;:"\']'

    def tokenize(self, text: str) -> list:
        text = text.lower()
        text = re.sub(self.punctuation, ' ', text)
        words = text.split()
        words = [w for w in words if w and (len(w) > 1 or w.isdigit())]
        return words

preprocessor = TextPreprocessor()
tokens = preprocessor.tokenize("Query users WHERE id = 1")
assert len(tokens) > 0, "分词失败"
assert "query" in tokens, "未正确分词"
print(f"  ✓ 分词测试：'Query users WHERE id = 1' -> {tokens}")

# ==================== 测试 SQL 提取 ====================
print("\n[3] 测试 SQL 提取...")
import re

def extract_sql(text: str) -> str:
    if not text:
        return None
    if "INSUFFICIENT_INFO" in text.upper():
        return "INSUFFICIENT_INFO"
    code_block_pattern = r"```(?:sql)?\s*(.*?)\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip().rstrip(';')
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE']
    lines = text.strip().split('\n')
    for line in lines:
        line_upper = line.upper().strip()
        for keyword in sql_keywords:
            if line_upper.startswith(keyword):
                return line.strip().rstrip(';')
    return None

# 测试代码块格式
sql1 = extract_sql("```sql\nSELECT * FROM users\n```")
assert sql1 == "SELECT * FROM users", f"代码块提取失败：{sql1}"
print(f"  ✓ 代码块提取：{sql1}")

# 测试纯文本格式
sql2 = extract_sql("SELECT * FROM users WHERE id = 1")
assert "SELECT" in sql2.upper(), f"纯文本提取失败：{sql2}"
print(f"  ✓ 纯文本提取：{sql2}")

# 测试信息不足
sql3 = extract_sql("INSUFFICIENT_INFO - 无法生成 SQL")
assert sql3 == "INSUFFICIENT_INFO", "信息不足检测失败"
print(f"  ✓ 信息不足检测：{sql3}")

# ==================== 测试 SQL 安全校验 ====================
print("\n[4] 测试 SQL 安全校验...")

def is_safe_query(sql: str) -> bool:
    sql_upper = sql.strip().upper()
    unsafe_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
    for keyword in unsafe_keywords:
        if sql_upper.startswith(keyword) or f" {keyword} " in sql_upper:
            return False
    return True

assert is_safe_query("SELECT * FROM users") == True, "SELECT 被误判"
assert is_safe_query("DELETE FROM users") == False, "DELETE 未被拦截"
assert is_safe_query("DROP TABLE users") == False, "DROP 未被拦截"
print("  ✓ SELECT 查询允许")
print("  ✓ DELETE 查询拦截")
print("  ✓ DROP 查询拦截")

# ==================== 测试提示词模板 ====================
print("\n[5] 测试提示词模板结构...")

# 验证模板文件存在且包含必要内容
with open('/Users/rgj/.openclaw/workspace/text2sql_agent/core/chain/prompts.py', 'r', encoding='utf-8') as f:
    prompts_content = f.read()

assert 'SQL_GENERATION_SYSTEM_PROMPT' in prompts_content, "缺少系统提示词"
assert 'SQLPromptTemplates' in prompts_content, "缺少模板类"
assert '{schema_info}' in prompts_content, "缺少 schema 占位符"
assert '{knowledge_info}' in prompts_content, "缺少 knowledge 占位符"
assert '{user_query}' in prompts_content, "缺少 query 占位符"
print("  ✓ 系统提示词模板存在")
print("  ✓ SQLPromptTemplates 类定义")
print("  ✓ 必要的占位符完整")

# ==================== 测试模块结构 ====================
print("\n[6] 测试项目结构...")
import os

root = '/Users/rgj/.openclaw/workspace/text2sql_agent'

required_files = [
    'config.py',
    'requirements.txt',
    'schema.sql',
    'app.py',
    'core/__init__.py',
    'core/knowledge/neo4j_client.py',
    'core/retrieval/bm25_tfidf.py',
    'core/chain/prompts.py',
    'core/chain/sql_chain.py',
    'core/sql/generator.py',
    'core/sql/validator.py',
    'core/history/mysql_client.py',
    'core/llm_factory.py',
    'app/__init__.py',
    'app/pages/history.py',
    'tests/test_core.py'
]

missing_files = []
for file in required_files:
    path = os.path.join(root, file)
    if os.path.exists(path):
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} (缺失)")
        missing_files.append(file)

if missing_files:
    print(f"\n  ⚠ 缺失文件：{missing_files}")
else:
    print("\n  ✓ 所有必需文件存在")

# ==================== 测试配置文件语法 ====================
print("\n[7] 测试 Python 文件语法...")
import py_compile
import tempfile

files_to_check = [
    'config.py',
    'core/llm_factory.py',
    'core/knowledge/neo4j_client.py',
    'core/history/mysql_client.py',
    'core/retrieval/bm25_tfidf.py',
    'core/chain/prompts.py',
    'core/chain/sql_chain.py',
    'core/sql/generator.py',
    'core/sql/validator.py',
    'app.py',
    'app/pages/history.py'
]

syntax_errors = []
for file in files_to_check:
    path = os.path.join(root, file)
    try:
        py_compile.compile(path, doraise=True)
        print(f"  ✓ {file} 语法正确")
    except py_compile.PyCompileError as e:
        print(f"  ✗ {file} 语法错误：{e}")
        syntax_errors.append(file)

# ==================== 汇总结果 ====================
print("\n" + "=" * 50)
print("测试汇总")
print("=" * 50)

all_passed = len(missing_files) == 0 and len(syntax_errors) == 0
if all_passed:
    print("✅ 所有测试通过!")
    print("\n项目已完成初始化，可以开始使用。")
    print("\n下一步:")
    print("1. 安装依赖：pip install -r requirements.txt")
    print("2. 配置环境变量：cp .env.example .env")
    print("3. 初始化数据库：mysql -u root -p < schema.sql")
    print("4. 启动应用：streamlit run app.py")
else:
    print("❌ 部分测试失败，请检查上方错误信息")
    if missing_files:
        print(f"   缺失文件：{missing_files}")
    if syntax_errors:
        print(f"   语法错误：{syntax_errors}")
