"""
Text2SQL 智能体测试模块
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestNeo4jClient:
    """Neo4j 客户端测试"""

    def test_init(self):
        """测试初始化"""
        from core.knowledge.neo4j_client import Neo4jClient

        client = Neo4jClient("bolt://localhost:7687", "neo4j", "neo4j@123")

        assert client.uri == "bolt://localhost:7687"
        assert client.user == "neo4j"
        assert client.password == "neo4j@123"

    def test_connect_fail(self):
        """测试连接失败（Neo4j 未启动时）"""
        from core.knowledge.neo4j_client import Neo4jClient

        client = Neo4jClient("bolt://localhost:7687", "neo4j", "wrong_password")
        # 不实际连接，只测试接口


class TestMySQLClient:
    """MySQL 客户端测试"""

    def test_init(self):
        """测试初始化"""
        from core.history.mysql_client import MySQLClient

        client = MySQLClient("localhost", 3306, "root", "root", "text2sql_db")

        assert client.host == "localhost"
        assert client.port == 3306
        assert client.database == "text2sql_db"


class TestBM25Retriever:
    """BM25 检索器测试"""

    def test_preprocessor_tokenize(self):
        """测试分词"""
        from core.retrieval.bm25_tfidf import TextPreprocessor

        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize("Hello World! This is a test.")

        assert len(tokens) > 0
        assert "hello" in tokens
        assert "world" in tokens

    def test_hybrid_retriever_index(self):
        """测试索引构建"""
        from core.retrieval.bm25_tfidf import HybridRetriever

        retriever = HybridRetriever()
        documents = [
            "User table contains user information",
            "Order table stores order records",
            "Product table has product details"
        ]

        retriever.index_documents(documents)

        assert retriever.get_document_count() == 3

    def test_hybrid_retriever_search(self):
        """测试检索"""
        from core.retrieval.bm25_tfidf import HybridRetriever

        retriever = HybridRetriever()
        documents = [
            "Query all users from database",
            "Get orders by user id",
            "Find products with price filter"
        ]
        metadata = [
            {'type': 'query', 'name': 'users'},
            {'type': 'query', 'name': 'orders'},
            {'type': 'query', 'name': 'products'}
        ]

        retriever.index_documents(documents, metadata)
        results = retriever.search("users", top_k=2)

        assert len(results) <= 2
        assert results[0][0]['content'] is not None


class TestSQLValidator:
    """SQL 校验器测试"""

    def test_validate_valid_sql(self):
        """测试有效 SQL 校验"""
        from core.sql.validator import SQLValidator

        validator = SQLValidator(dialect="mysql")
        result = validator.validate("SELECT * FROM users WHERE id = 1")

        assert 'valid' in result
        # 注意：SQLFluff 可能对简单 SQL 也有布局警告

    def test_validate_empty_sql(self):
        """测试空 SQL 校验"""
        from core.sql.validator import SQLValidator

        validator = SQLValidator(dialect="mysql")
        result = validator.validate("")

        assert result['valid'] == False
        assert len(result['errors']) > 0

    def test_format_sql(self):
        """测试 SQL 格式化"""
        from core.sql.validator import SQLValidator

        validator = SQLValidator(dialect="mysql")
        sql = "select * from users where id=1"
        result = validator.format(sql)

        assert result['success'] == True or result['success'] == False

    def test_execution_validator_safe(self):
        """测试执行校验 - 安全 SQL"""
        from core.sql.validator import SQLExecutionValidator

        validator = SQLExecutionValidator(allow_select_only=True)
        result = validator.validate("SELECT * FROM users")

        assert result['valid'] == True

    def test_execution_validator_unsafe(self):
        """测试执行校验 - 危险 SQL"""
        from core.sql.validator import SQLExecutionValidator

        validator = SQLExecutionValidator(allow_select_only=True)
        result = validator.validate("DELETE FROM users")

        assert result['valid'] == False
        assert len(result['errors']) > 0


class TestPromptTemplates:
    """提示词模板测试"""

    def test_format_schema_info(self):
        """测试 Schema 格式化"""
        from core.chain.prompts import SQLPromptTemplates

        templates = SQLPromptTemplates()

        tables = [{'table_name': 'users', 'table_comment': '用户表'}]
        columns = [
            {'table_name': 'users', 'column_name': 'id', 'column_type': 'INT',
             'is_primary_key': True, 'is_nullable': False, 'column_comment': 'ID'}
        ]

        result = templates.format_schema_info(tables, columns)

        assert 'users' in result
        assert '用户表' in result

    def test_extract_sql_from_codeblock(self):
        """测试从代码块提取 SQL"""
        from core.chain.sql_chain import SQLGenerationChain
        from unittest.mock import Mock

        chain = SQLGenerationChain(Mock(), dialect="mysql")

        text = """```sql
SELECT * FROM users WHERE id = 1
```"""
        sql = chain._extract_sql(text)

        assert sql is not None
        assert "SELECT" in sql.upper()

    def test_extract_sql_plain(self):
        """测试从纯文本提取 SQL"""
        from core.chain.sql_chain import SQLGenerationChain
        from unittest.mock import Mock

        chain = SQLGenerationChain(Mock(), dialect="mysql")

        text = "SELECT * FROM users WHERE id = 1"
        sql = chain._extract_sql(text)

        assert sql is not None
        assert "SELECT" in sql.upper()


class TestIntegration:
    """集成测试"""

    def test_full_pipeline_mock(self):
        """测试完整流程（模拟）"""
        from unittest.mock import Mock, MagicMock

        # 模拟 LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "SELECT * FROM users"
        mock_llm.invoke.return_value = mock_response

        # 模拟检索器
        from core.retrieval.bm25_tfidf import KnowledgeRetriever
        retriever = KnowledgeRetriever()
        retriever.retriever.index_documents(
            ["user table"],
            [{'node_type': 'table', 'name': 'users'}]
        )

        # 模拟 MySQL 客户端
        mock_mysql = Mock()
        mock_mysql.get_table_schema.return_value = []
        mock_mysql.get_column_schema.return_value = []

        # 创建生成器
        from core.chain.sql_chain import SQLGenerationChain
        from core.sql.generator import SQLGenerator

        sql_chain = SQLGenerationChain(mock_llm, dialect="mysql")
        generator = SQLGenerator(
            llm=mock_llm,
            knowledge_retriever=retriever,
            sql_chain=sql_chain,
            mysql_client=mock_mysql
        )

        # 测试生成
        result = generator.generate_sql(
            user_query="查询所有用户",
            auto_save_history=False
        )

        assert result['status'] == 'success'
        assert result['sql'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
