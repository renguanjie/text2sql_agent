"""
知识图谱构建模块 - 统一入口
"""
from typing import Optional, Dict, Any, Literal
from loguru import logger

from .models import SchemaExtractResult
from .extractors import LiveDBExtractor, OfflineExtractor, OfflineJSONExtractor
from .enrichment import generate_concepts, infer_relationships
from .compiler import compile_join_sqls
from .loader import Neo4jGraphBuilder


class GraphBuilderPipeline:
    """知识图谱构建流水线"""

    def __init__(self, neo4j_client=None, llm_provider: str = "openai",
                 llm_model: str = "gpt-4", llm_config: Optional[Dict] = None,
                 dialect: str = "mysql"):
        self.neo4j_client = neo4j_client
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_config = llm_config or {}
        self.dialect = dialect
        self.result: Optional[SchemaExtractResult] = None
        logger.info(f"流水线初始化 - LLM: {llm_provider}/{llm_model}")

    def extract_from_db(self, db_uri: str, db_type: str = "mysql",
                        schema: Optional[str] = None) -> SchemaExtractResult:
        extractor = LiveDBExtractor(db_uri=db_uri, db_type=db_type, schema=schema)
        try:
            self.result = extractor.extract()
        finally:
            extractor.close()
        return self.result

    def extract_from_files(self, ddl_path: Optional[str] = None,
                           database_name: str = "default",
                           db_type: str = "mysql") -> SchemaExtractResult:
        if not ddl_path:
            raise ValueError("必须提供 DDL 文件路径")
        extractor = OfflineExtractor(
            ddl_path=ddl_path,
            database_name=database_name,
            db_type=db_type,
            neo4j_client=self.neo4j_client
        )
        self.result = extractor.extract()
        if not self.result.tables:
            raise ValueError("DDL 文件中未解析到任何表")
        return self.result

    def extract_from_json(self, json_path: Optional[str] = None) -> SchemaExtractResult:
        if not json_path:
            raise ValueError("必须提供 JSON 文件路径")
        extractor = OfflineJSONExtractor(json_path=json_path)
        self.result = extractor.extract()
        if not self.result.tables:
            raise ValueError("JSON 文件中未解析到任何表")
        return self.result

    def enrich(self, enable_concepts: bool = True, enable_relations: bool = True):
        if not self.result:
            raise ValueError("请先执行 extract 操作")

        if enable_concepts:
            logger.info("开始生成业务概念...")
            self.result.concepts = generate_concepts(
                self.result, llm_provider=self.llm_provider,
                model=self.llm_model, **self.llm_config
            )

        if enable_relations and not self.result.relationships:
            logger.info("开始推断表关系...")
            self.result.relationships = infer_relationships(
                self.result, llm_provider=self.llm_provider,
                model=self.llm_model, **self.llm_config
            )
        return self

    def compile(self):
        if not self.result:
            raise ValueError("请先执行 extract 操作")
        logger.info("开始预编译 JOIN SQL...")
        self.result = compile_join_sqls(self.result, dialect=self.dialect)
        return self

    def load(self, neo4j_client=None) -> Dict[str, int]:
        if not self.result:
            raise ValueError("请先执行 extract 操作")
        client = neo4j_client or self.neo4j_client
        if not client:
            raise ValueError("必须提供 Neo4j 客户端")
        if not getattr(client, 'driver', None):
            raise ConnectionError("Neo4j 客户端未连接，无法装载知识图谱")

        logger.info("开始装载到 Neo4j...")
        loader = Neo4jGraphBuilder(client)
        stats = loader.build(self.result)
        expected_counts = {
            'databases': len(self.result.databases),
            'tables': len(self.result.tables),
            'columns': sum(len(table.columns) for table in self.result.tables),
            'relationships': len(self.result.relationships),
        }
        failed_entities = [
            name for name, expected in expected_counts.items()
            if expected > 0 and stats.get(name, 0) == 0
        ]
        if failed_entities:
            raise RuntimeError(f"Neo4j 装载失败：{', '.join(failed_entities)} 未写入")
        if self.result.concepts:
            loader.link_concepts_to_tables(self.result.concepts)
        return {**stats, **loader.get_graph_stats()}

    def run_full_pipeline(self, source_type: Literal["live_db", "offline"] = "live_db",
                          source_config: Optional[Dict] = None,
                          enable_enrichment: bool = True,
                          neo4j_client=None) -> Dict[str, Any]:
        source_config = source_config or {}

        logger.info("=" * 50)
        logger.info("阶段 1: 数据抽取 (Extractor)")
        if source_type == "live_db":
            self.extract_from_db(**source_config)
        elif source_type == "offline":
            self.extract_from_files(**source_config)
        else:
            raise ValueError(f"不支持的数据源类型：{source_type}")
        logger.info(f"抽取完成 - 表：{len(self.result.tables)}")

        if enable_enrichment:
            logger.info("=" * 50)
            logger.info("阶段 2: AI 语义增强 (Enrichment)")
            self.enrich(enable_concepts=True, enable_relations=True)
            logger.info(f"增强完成 - 概念：{len(self.result.concepts)}, 关系：{len(self.result.relationships)}")

        logger.info("=" * 50)
        logger.info("阶段 3: SQL 预编译 (Compiler)")
        self.compile()
        logger.info(f"预编译完成 - JOIN SQL: {len(self.result.relationships)}")

        logger.info("=" * 50)
        logger.info("阶段 4: Neo4j 装载 (Loader)")
        stats = self.load(neo4j_client=neo4j_client)

        logger.info("=" * 50)
        logger.info("✅ 知识图谱全链路构建完成！")
        logger.info(f"图谱统计：{stats}")
        return {
            "extract": {"databases": len(self.result.databases), "tables": len(self.result.tables),
                       "columns": sum(len(t.columns) for t in self.result.tables)},
            "enrichment": {"concepts": len(self.result.concepts), "relationships": len(self.result.relationships)},
            "load": stats
        }


def run_pipeline(source_type: Literal["live_db", "offline"] = "live_db",
                 source_config: Optional[Dict] = None, neo4j_client=None,
                 llm_provider: str = "openai", llm_model: str = "gpt-4",
                 enable_enrichment: bool = True, dialect: str = "mysql") -> Dict[str, Any]:
    """运行知识图谱构建流水线（便捷函数）"""
    pipeline = GraphBuilderPipeline(
        neo4j_client=neo4j_client, llm_provider=llm_provider,
        llm_model=llm_model, dialect=dialect
    )
    return pipeline.run_full_pipeline(
        source_type=source_type, source_config=source_config,
        enable_enrichment=enable_enrichment, neo4j_client=neo4j_client
    )
