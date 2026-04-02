"""
Neo4j 图谱装载器 - 使用 UNWIND + MERGE 模式批量装载
支持数据库 ID 自增和完整的数据库元数据
"""
from typing import Dict, List
from loguru import logger

from ..models import SchemaExtractResult


class Neo4jGraphBuilder:
    def __init__(self, neo4j_client):
        self.client = neo4j_client
        logger.info("Neo4j 图谱装载器初始化完成")

    def build(self, result: SchemaExtractResult) -> Dict[str, int]:
        """
        执行完整的图谱构建

        装载顺序：
        1. Database - 数据库节点
        2. BusinessConcept - 业务概念节点
        3. Table - 表节点
        4. Column - 字段节点
        5. Relationship - 关系边
        """
        stats = {
            "databases": self.batch_upsert_databases(result.databases),
            "concepts": self.batch_upsert_concepts(result.concepts),
            "tables": self.batch_upsert_tables(result.tables),
            "columns": self.batch_upsert_columns(result.tables),
            "relationships": self.batch_upsert_relationships(result.relationships)
        }
        logger.info(f"图谱构建完成：{stats}")
        return stats

    def batch_upsert_databases(self, databases: List) -> int:
        """
        批量装载数据库节点

        节点属性：
        - database_id: 数据库 ID（自增）
        - name: 数据库名称
        - description: 数据库描述
        - db_type: 数据库类型
        - db_language: 数据库语言
        - create_statement: 数据库创建语句
        """
        if not databases:
            return 0

        cypher = """
        UNWIND $databases AS db
        MERGE (d:Database {database_id: COALESCE(db.database_id, db.id)})
        ON CREATE SET
            d.name = db.name,
            d.description = db.description,
            d.db_type = db.db_type,
            d.db_language = db.db_language,
            d.create_statement = db.create_statement,
            d.created_at = datetime()
        ON MATCH SET
            d.name = db.name,
            d.description = db.description,
            d.db_type = db.db_type,
            d.db_language = db.db_language,
            d.create_statement = db.create_statement,
            d.updated_at = datetime()
        RETURN count(d) as count
        """

        params = {"databases": [db.model_dump() for db in databases]}
        result = self.client.execute_query(cypher, params)
        count = result[0]["count"] if result and result[0] else 0
        logger.info(f"装载 {count} 个数据库节点")
        return count

    def batch_upsert_concepts(self, concepts: List) -> int:
        """
        批量装载业务概念节点

        节点属性：
        - name: 概念名称
        - description: 概念描述
        - tags: 业务标签列表
        - mapped_tables: 映射的表列表
        """
        if not concepts:
            return 0

        cypher = """
        UNWIND $concepts AS concept
        MERGE (c:BusinessConcept {name: concept.name})
        SET c.description = concept.description,
            c.tags = concept.tags,
            c.mapped_tables = concept.mapped_tables,
            c.updated_at = datetime()
        RETURN count(c) as count
        """

        params = {"concepts": [c.model_dump() for c in concepts]}
        result = self.client.execute_query(cypher, params)
        count = result[0]["count"] if result and result[0] else 0
        logger.info(f"装载 {count} 个业务概念节点")
        return count

    def batch_upsert_tables(self, tables: List) -> int:
        """
        批量装载表节点

        节点属性：
        - name: 表名
        - database: 所属数据库名称
        - database_id: 所属数据库 ID
        - name_cn: 表中文名
        - description: 表描述
        - is_view: 是否视图
        - create_statement: 表创建语句
        """
        if not tables:
            return 0

        cypher = """
        UNWIND $tables AS tbl
        // 确保数据库节点存在（使用 database_id 作为键）
        MERGE (d:Database {database_id: tbl.database_id})
        ON CREATE SET d.name = tbl.database, d.created_at = datetime()

        // 创建或更新表节点（使用 name + database_id 作为唯一键）
        MERGE (t:Table {name: tbl.name, database_id: tbl.database_id})
        SET t.name_cn = tbl.name_cn,
            t.description = tbl.description,
            t.is_view = tbl.is_view,
            t.create_statement = tbl.create_statement,
            t.database = tbl.database,
            t.updated_at = datetime()

        // 建立数据库 - 表关系
        MERGE (d)-[:HAS_TABLE]->(t)

        RETURN count(t) as count
        """

        params = {"tables": [t.model_dump() for t in tables]}
        result = self.client.execute_query(cypher, params)
        count = result[0]["count"] if result and result[0] else 0
        logger.info(f"装载 {count} 个表节点")
        return count

    def batch_upsert_columns(self, tables: List) -> int:
        """
        批量装载字段节点

        节点属性：
        - name: 字段名
        - table_name: 所属表名
        - database_id: 所属数据库 ID
        - data_type: 数据类型
        - description: 字段描述
        - is_primary_key: 是否主键
        - is_nullable: 是否可空
        - is_partition: 是否分区键
        - default_value: 默认值
        """
        if not tables:
            return 0

        # 扁平化所有字段
        all_columns = []
        for table in tables:
            for col in table.columns:
                col_data = col.model_dump()
                col_data["table_name"] = table.name
                col_data["database_id"] = table.database_id
                col_data["database"] = table.database
                all_columns.append(col_data)

        if not all_columns:
            return 0

        cypher = """
        UNWIND $columns AS col
        // 确保表节点存在
        MERGE (t:Table {name: col.table_name, database_id: col.database_id})

        // 创建或更新字段节点
        MERGE (c:Column {
            name: col.name,
            table_name: col.table_name,
            database_id: col.database_id
        })
        SET c.data_type = col.data_type,
            c.description = col.description,
            c.is_primary_key = col.is_primary_key,
            c.is_nullable = col.is_nullable,
            c.is_partition = col.is_partition,
            c.default_value = col.default_value,
            c.name_cn = col.name_cn,
            c.updated_at = datetime()

        // 建立表 - 字段关系
        MERGE (t)-[:HAS_COLUMN]->(c)

        RETURN count(c) as count
        """

        params = {"columns": all_columns}
        result = self.client.execute_query(cypher, params)
        count = result[0]["count"] if result and result[0] else 0
        logger.info(f"装载 {count} 个字段节点")
        return count

    def batch_upsert_relationships(self, relationships: List) -> int:
        """
        批量装载表关系

        关系属性：
        - relationship_type: 关系类型
        - join_type: JOIN 类型
        - join_sql: 预编译的 JOIN SQL
        - extra_condition: 额外条件
        - from_column: 源字段
        - to_column: 目标字段
        """
        if not relationships:
            return 0

        cypher = """
        UNWIND $relationships AS rel
        // 找到源表和目标表（优先使用 database_id，没有则使用 database 名称）
        MATCH (from:Table {name: rel.from_table})
        MATCH (to:Table {name: rel.to_table})
        WHERE (rel.from_database_id IS NULL OR from.database_id = rel.from_database_id)
          AND (rel.to_database_id IS NULL OR to.database_id = rel.to_database_id)

        // 创建或更新关系
        MERGE (from)-[r:CONNECTS {
            from_column: rel.from_column,
            to_column: rel.to_column
        }]->(to)
        SET r.relationship_type = rel.relationship_type,
            r.join_type = rel.join_type,
            r.join_sql = rel.join_sql,
            r.extra_condition = rel.extra_condition,
            r.updated_at = datetime()

        RETURN count(r) as count
        """

        params = {"relationships": [r.model_dump() for r in relationships]}
        result = self.client.execute_query(cypher, params)
        count = result[0]["count"] if result and result[0] else 0
        logger.info(f"装载 {count} 个关系边")
        return count

    def link_concepts_to_tables(self, concepts: List) -> int:
        """
        建立业务概念与表的关联

        关系：BusinessConcept-[:MAPPED_TO]->Table
        """
        if not concepts:
            return 0

        count = 0
        for concept in concepts:
            for table_name in concept.mapped_tables:
                cypher = """
                MATCH (c:BusinessConcept {name: $concept_name})
                MATCH (t:Table {name: $table_name})
                MERGE (c)-[:MAPPED_TO]->(t)
                """
                self.client.execute_query(cypher, {
                    "concept_name": concept.name,
                    "table_name": table_name
                })
                count += 1

        logger.info(f"建立 {count} 个概念 - 表关联")
        return count

    def clear_graph(self):
        """清空图谱（危险操作，谨慎使用）"""
        cypher = "MATCH (n) DETACH DELETE n"
        self.client.execute_query(cypher)
        logger.warning("图谱已清空")

    def get_graph_stats(self) -> Dict[str, int]:
        """获取图谱统计信息"""
        stats = {}

        # 按标签统计节点
        labels = ["Database", "Table", "Column", "BusinessConcept"]
        for label in labels:
            result = self.client.execute_query(f"MATCH (n:{label}) RETURN count(n) as count")
            stats[label] = result[0]["count"] if result and result[0] else 0

        # 统计关系边
        result = self.client.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
        stats["Relationships"] = result[0]["count"] if result and result[0] else 0

        return stats

    def get_database_list(self) -> List[Dict]:
        """获取所有数据库列表"""
        cypher = """
        MATCH (d:Database)
        RETURN d.id as id, d.name as name, d.db_type as db_type,
               d.db_language as db_language, d.description as description
        ORDER BY d.id
        """
        result = self.client.execute_query(cypher)
        return result if result else []

    def get_table_details(self, table_name: str, database_id: int = None) -> Dict:
        """获取表的详细信息"""
        params = {"table_name": table_name}
        if database_id:
            params["database_id"] = database_id
            cypher = """
            MATCH (t:Table {name: $table_name, database_id: $database_id})
            OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
            OPTIONAL MATCH (from:Table)-[:CONNECTS]->(t)
            OPTIONAL MATCH (t)-[:CONNECTS]->(to:Table)
            RETURN t,
                   collect(DISTINCT c) as columns,
                   collect(DISTINCT from) as referenced_by,
                   collect(DISTINCT to) as references
            """
        else:
            cypher = """
            MATCH (t:Table {name: $table_name})
            OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
            OPTIONAL MATCH (from:Table)-[:CONNECTS]->(t)
            OPTIONAL MATCH (t)-[:CONNECTS]->(to:Table)
            RETURN t,
                   collect(DISTINCT c) as columns,
                   collect(DISTINCT from) as referenced_by,
                   collect(DISTINCT to) as references
            """
        result = self.client.execute_query(cypher, params)
        return result[0] if result and result[0] else None
