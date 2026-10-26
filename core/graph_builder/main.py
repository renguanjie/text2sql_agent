"""
知识图谱构建模块 - 统一入口
"""
from typing import Optional, Dict, Any, Literal
from loguru import logger

from .models import SchemaExtractResult
from .extractors import LiveDBExtractor, OfflineExtractor, OfflineJSONExtractor
from .enrichment import generate_concepts, infer_relationships
from .compiler import compile_join_sqls
from .loader import Neo4jGraphBuilder, normalize_relationships


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
        extractor = OfflineJSONExtractor(json_path=json_path, neo4j_client=self.neo4j_client)
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
        # 全部写操作收集为原子批次统一提交；任一步失败则整批不生效
        with client.atomic_writes():
            stats = loader.build(self.result)
            if self.result.concepts:
                loader.link_concepts_to_tables(self.result.concepts)

        # 批次提交后用真实图数据校验装载结果
        stats.update(self._verify_load(client))

        # 权值规范化依赖写入后的图状态，在批次提交后单独执行
        stats["weight_normalized"] = normalize_relationships(client)
        return {**stats, **loader.get_graph_stats()}

    def _verify_load(self, client) -> Dict[str, int]:
        """提交后按 database_id 范围核对实际写入数量，发现缺写立即报错。"""
        expected = {
            'databases': len(self.result.databases),
            'tables': len(self.result.tables),
            'columns': sum(len(t.columns) for t in self.result.tables),
            'relationships': len(self.result.relationships),
        }
        db_ids = sorted({
            d.database_id if d.database_id is not None else d.id
            for d in self.result.databases
        })
        db_ids = [x for x in db_ids if x is not None]
        if not db_ids or not any(expected.values()):
            return {k: 0 for k in expected}

        rows = client.execute_query("""
            MATCH (d:Database) WHERE d.database_id IN $db_ids
            WITH count(DISTINCT d) as databases, collect(DISTINCT d.database_id) as ids
            MATCH (t:Table) WHERE t.database_id IN ids
            WITH databases, count(DISTINCT t) as tables, ids
            MATCH (c:Column) WHERE c.database_id IN ids
            WITH databases, tables, count(DISTINCT c) as columns, ids
            MATCH (a:Table)-[r:CONNECTS]->(b:Table) WHERE a.database_id IN ids
            RETURN databases, tables, columns, count(DISTINCT r) as relationships
        """, {"db_ids": db_ids})
        actual = rows[0] if rows else {k: 0 for k in expected}
        failed = [
            name for name, exp in expected.items()
            if exp > 0 and actual.get(name, 0) < exp
        ]
        if failed:
            raise RuntimeError(
                f"Neo4j 装载校验失败：{', '.join(failed)} 写入数量不足"
                f"（预期 {expected}，实际 {dict(actual)}）"
            )
        return {name: actual.get(name, 0) for name in expected}

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
