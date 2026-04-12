"""
Neo4j 图数据库客户端
负责知识图谱的连接、查询和元数据同步
支持异步查询和连接池配置
"""
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase, Result, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from loguru import logger
import json
import asyncio
from datetime import datetime


class Neo4jClient:
    """Neo4j 图数据库客户端 - 支持连接池和异步查询"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                 max_connection_pool_size: int = 50, connection_timeout: int = 30):
        """
        初始化 Neo4j 客户端

        Args:
            uri: Neo4j 连接 URI (e.g., bolt://localhost:7687)
            user: 用户名
            password: 密码
            database: 数据库名称
            max_connection_pool_size: 最大连接池大小，默认 50
            connection_timeout: 连接超时时间（秒），默认 30
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_timeout = connection_timeout
        self.driver = None
        self.async_driver = None

    def connect(self) -> bool:
        """
        连接到 Neo4j 数据库（同步 + 异步驱动）

        Returns:
            bool: 连接是否成功
        """
        try:
            # 同步驱动（带连接池配置）
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=self.max_connection_pool_size,
                connection_timeout=self.connection_timeout
            )
            # 验证连接
            self.driver.verify_connectivity()

            # 异步驱动（用于高并发场景）
            try:
                self.async_driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    max_connection_pool_size=self.max_connection_pool_size,
                    connection_timeout=self.connection_timeout
                )
                logger.info(f"Neo4j 同步 + 异步驱动初始化成功，pool_size={self.max_connection_pool_size}")
            except Exception as e:
                logger.warning(f"Neo4j 异步驱动初始化失败：{e}，将使用同步模式")
                self.async_driver = None

            logger.info(f"成功连接到 Neo4j: {self.uri}")
            return True
        except ServiceUnavailable as e:
            logger.error(f"Neo4j 连接失败 - 服务不可用：{e}")
            return False
        except AuthError as e:
            logger.error(f"Neo4j 认证失败：{e}")
            return False
        except Exception as e:
            logger.error(f"Neo4j 连接失败：{e}")
            return False

    def close(self):
        """关闭连接池"""
        if self.driver:
            self.driver.close()
        if self.async_driver:
            asyncio.run(self.async_driver.close())
        logger.info("Neo4j 连接池已关闭")

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        执行 Cypher 查询（同步）

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            List[Dict]: 查询结果列表
        """
        if not self.driver:
            logger.error("Neo4j 未连接")
            return []

        try:
            with self.driver.session(database=self.database) as session:
                result: Result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j 查询执行失败：{e}, query={query}")
            return []

    async def execute_query_async(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        执行 Cypher 查询（异步）

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            List[Dict]: 查询结果列表
        """
        if not self.async_driver:
            logger.warning("Neo4j 异步驱动未初始化，使用同步模式")
            return self.execute_query(query, parameters)

        try:
            async with self.async_driver.session(database=self.database) as session:
                result = await session.run(query, parameters or {})
                return [await record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j 异步查询执行失败：{e}, query={query}")
            return []

    def get_all_nodes(self, label: Optional[str] = None) -> List[Dict]:
        """
        获取所有节点

        Args:
            label: 节点标签过滤（可选）

        Returns:
            List[Dict]: 节点列表
        """
        if label:
            query = f"MATCH (n:`{label}`) RETURN n"
        else:
            query = "MATCH (n) RETURN n"

        results = self.execute_query(query)
        nodes = []
        for record in results:
            node_data = record.get('n', {})
            nodes.append({
                'id': node_data.id if hasattr(node_data, 'id') else None,
                'element_id': node_data.element_id if hasattr(node_data, 'element_id') else None,
                'labels': list(node_data.labels) if hasattr(node_data, 'labels') else [],
                'properties': dict(node_data)
            })
        return nodes

    def get_table_nodes(self) -> List[Dict]:
        """
        获取所有表节点

        Returns:
            List[Dict]: 表节点列表
        """
        query = """
        MATCH (t:Table)
        RETURN t.name AS name, t.description AS description, t.schema AS schema
        """
        return self.execute_query(query)

    def get_database_list(self) -> List[Dict]:
        """
        获取所有数据库列表

        Returns:
            List[Dict]: 数据库列表
        """
        query = """
            MATCH (d:Database)
            RETURN d.database_id as database_id, d.name as name, d.db_type as db_type,
                   d.db_language as db_language, d.description as description
            ORDER BY d.database_id
        """
        results = self.execute_query(query)
        return [
            {
                "database_id": r["database_id"],
                "name": r["name"],
                "db_type": r["db_type"],
                "db_language": r.get("db_language", "SQL"),
                "description": r.get("description", "")
            }
            for r in results
        ]

    def get_column_nodes(self, table_name: Optional[str] = None) -> List[Dict]:
        """
        获取字段节点

        Args:
            table_name: 表名过滤（可选）

        Returns:
            List[Dict]: 字段节点列表
        """
        if table_name:
            query = """
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE t.name = $table_name
            RETURN c.name AS name, c.type AS type, c.description AS description,
                   c.is_primary_key AS is_primary_key, c.is_nullable AS is_nullable
            """
            return self.execute_query(query, {"table_name": table_name})
        else:
            query = """
            MATCH (c:Column)
            RETURN c.name AS name, c.type AS type, c.description AS description,
                   c.is_primary_key AS is_primary_key, c.is_nullable AS is_nullable
            """
            return self.execute_query(query)

    def get_relationships(self, start_node: Optional[str] = None,
                         rel_type: Optional[str] = None) -> List[Dict]:
        """
        获取关系

        Args:
            start_node: 起始节点 ID
            rel_type: 关系类型

        Returns:
            List[Dict]: 关系列表
        """
        conditions = []
        params = {}

        if start_node:
            conditions.append("start_node.name = $start_node")
            params["start_node"] = start_node

        if rel_type:
            conditions.append("type(rel) = $rel_type")
            params["rel_type"] = rel_type

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        query = f"""
        MATCH (start_node)-[rel]-(end_node)
        {where_clause}
        RETURN start_node.name AS start_name, type(rel) AS rel_type,
               end_node.name AS end_name, end_node.labels AS end_labels
        """
        return self.execute_query(query, params)

    def get_schema_metadata(self) -> Dict[str, Any]:
        """
        获取完整的数据库元数据（表 + 字段 + 关系）

        Returns:
            Dict: 元数据字典
        """
        tables = self.get_table_nodes()
        columns = self.get_column_nodes()
        relationships = self.get_relationships()

        # 构建表到字段的映射
        table_columns = {}
        for table in tables:
            table_name = table.get('name')
            if table_name:
                table_columns[table_name] = []

        # 通过关系获取每个表的字段
        for rel in relationships:
            if rel.get('rel_type') == 'BELONGS_TO':
                table_name = rel.get('end_name')
                column_name = rel.get('start_name')
                if table_name and column_name:
                    if table_name not in table_columns:
                        table_columns[table_name] = []
                    table_columns[table_name].append(column_name)

        return {
            'tables': tables,
            'columns': columns,
            'relationships': relationships,
            'table_columns': table_columns
        }

    def search_knowledge(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        搜索知识库

        Args:
            query_text: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Dict]: 匹配的节点列表
        """
        # 使用全文搜索（需要 Neo4j 全文索引）
        # 这里使用简单的模糊匹配作为备选
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $query OR n.description CONTAINS $query
        RETURN n,
               CASE
                 WHEN n.name CONTAINS $query THEN 2
                 WHEN n.description CONTAINS $query THEN 1
                 ELSE 0
               END AS score
        ORDER BY score DESC
        LIMIT $limit
        """
        results = self.execute_query(query, {"query": query_text, "limit": top_k})
        return results

    def get_node_by_name(self, name: str, label: Optional[str] = None) -> Optional[Dict]:
        """
        根据名称获取节点

        Args:
            name: 节点名称
            label: 节点标签（可选）

        Returns:
            Optional[Dict]: 节点信息
        """
        if label:
            query = """
            MATCH (n:`{label}`)
            WHERE n.name = $name
            RETURN n
            """.format(label=label)
        else:
            query = """
            MATCH (n)
            WHERE n.name = $name
            RETURN n
            """

        results = self.execute_query(query, {"name": name})
        if results:
            return results[0].get('n')
        return None

    def export_knowledge_graph(self) -> Dict[str, Any]:
        """
        导出知识图谱为 JSON 格式

        Returns:
            Dict: 包含 databases, tables, columns, relationships, concepts 的字典
        """
        if not self.driver:
            logger.error("Neo4j 未连接")
            return {}

        try:
            # 导出数据库
            databases = self.execute_query("""
                MATCH (d:Database)
                RETURN d.database_id as database_id, d.name as name, d.db_type as db_type,
                       d.db_language as db_language, d.description as description,
                       d.host as host, d.port as port, d.version as version
            """)

            # 导出表
            tables = self.execute_query("""
                MATCH (t:Table)
                OPTIONAL MATCH (d:Database)-[:HAS_TABLE]->(t)
                RETURN t.name as name, t.description as description, t.schema as schema,
                       t.database_id as database_id, d.name as database_name
            """)

            # 导出字段
            columns = self.execute_query("""
                MATCH (c:Column)
                RETURN c.name as name, c.type as type, c.description as description,
                       c.is_primary_key as is_primary_key, c.is_nullable as is_nullable,
                       c.order as order
            """)

            # 导出关系
            relationships = self.execute_query("""
                MATCH (start:Table)-[r:HAS_COLUMN|BELONGS_TO|REFERENCES]->(end)
                RETURN start.name as from_table, type(r) as relationship_type, end.name as to_table,
                       r.description as description
            """)

            # 导出业务概念
            concepts = self.execute_query("""
                MATCH (c:BusinessConcept)
                OPTIONAL MATCH (t:Table)-[:RELATED_TO]->(c)
                RETURN c.name as name, c.description as description, c.category as category,
                       collect(t.name) as related_tables
            """)

            export_data = {
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "databases": databases,
                "tables": tables,
                "columns": columns,
                "relationships": relationships,
                "concepts": concepts
            }

            logger.info(f"知识图谱导出完成：{len(databases)} 个数据库，{len(tables)} 个表，{len(columns)} 个字段")
            return export_data

        except Exception as e:
            logger.error(f"导出知识图谱失败：{e}")
            return {}

    def import_knowledge_graph(self, import_data: Dict[str, Any]) -> Dict[str, int]:
        """
        从 JSON 导入知识图谱（支持两种格式：标准格式和 graph_builder 格式）

        Args:
            import_data: 包含 databases, tables, columns, relationships, concepts 的字典

        Returns:
            Dict: 导入的节点数和关系数
        """
        if not self.driver:
            logger.error("Neo4j 未连接")
            return {"nodes": 0, "relationships": 0}

        try:
            stats = {"nodes": 0, "relationships": 0, "databases": 0, "tables": 0, "columns": 0, "concepts": 0}

            # 1. 导入数据库
            for db in import_data.get("databases", []):
                self.execute_query("""
                    MERGE (d:Database {database_id: $database_id})
                    SET d.name = $name, d.db_type = $db_type, d.db_language = $db_language,
                        d.description = $description, d.host = $host, d.port = $port, d.version = $version
                """, {
                    "database_id": db.get("database_id"),
                    "name": db.get("name", ""),
                    "db_type": db.get("db_type", "mysql"),
                    "db_language": db.get("db_language", "SQL"),
                    "description": db.get("description", ""),
                    "host": db.get("host", ""),
                    "port": db.get("port", ""),
                    "version": db.get("version", "")
                })
                stats["databases"] += 1

            # 如果没有数据库但有关系，使用默认数据库 ID
            default_db_id = 1
            if not import_data.get("databases") and import_data.get("relationships"):
                default_db_id = None

            # 2. 导入表（如果 JSON 中有 tables 数组）
            table_names_from_json = set()
            for table in import_data.get("tables", []):
                table_name = table.get("name", "")
                if table_name:
                    table_names_from_json.add(table_name)
                    self.execute_query("""
                        MERGE (t:Table {name: $name})
                        SET t.description = $description, t.schema = $schema, t.database_id = $database_id
                    """, {
                        "name": table_name,
                        "description": table.get("description", ""),
                        "schema": table.get("schema", ""),
                        "database_id": table.get("database_id", default_db_id)
                    })
                    stats["tables"] += 1

                    # 建立与数据库的关系
                    db_name = table.get("database_name")
                    if db_name:
                        self.execute_query("""
                            MATCH (d:Database {name: $db_name})
                            MATCH (t:Table {name: $table_name})
                            MERGE (d)-[:HAS_TABLE]->(t)
                        """, {"db_name": db_name, "table_name": table_name})

            # 2b. 从关系中推断表（如果 JSON 中没有 tables 数组）
            inferred_tables = set()
            for rel in import_data.get("relationships", []):
                from_table = rel.get("from_table", "")
                to_table = rel.get("to_table", "")
                if from_table and from_table not in table_names_from_json:
                    inferred_tables.add(from_table)
                if to_table and to_table not in table_names_from_json:
                    inferred_tables.add(to_table)

            for table_name in inferred_tables:
                self.execute_query("""
                    MERGE (t:Table {name: $name})
                    SET t.database_id = $database_id, t.description = COALESCE(t.description, '从关系推断')
                """, {"name": table_name, "database_id": default_db_id})
                stats["tables"] += 1
                logger.info(f"从关系中推断表：{table_name}")

            # 3. 导入字段（如果 JSON 中有 columns 数组）
            column_names_from_json = set()
            for col in import_data.get("columns", []):
                col_name = col.get("name", "")
                if col_name:
                    column_names_from_json.add(col_name)
                    self.execute_query("""
                        MERGE (c:Column {name: $name})
                        SET c.type = $type, c.description = $description,
                            c.is_primary_key = $is_primary_key, c.is_nullable = $is_nullable,
                            c.`order` = $order
                    """, {
                        "name": col_name,
                        "type": col.get("type", "VARCHAR"),
                        "description": col.get("description", ""),
                        "is_primary_key": col.get("is_primary_key", False),
                        "is_nullable": col.get("is_nullable", True),
                        "order": col.get("order")
                    })
                    stats["columns"] += 1

            # 4. 导入关系
            for rel in import_data.get("relationships", []):
                rel_type = rel.get("relationship_type", "HAS_COLUMN")
                from_table = rel.get("from_table", "")
                to_table = rel.get("to_table", "")
                description = rel.get("description", "")

                if rel_type == "BELONGS_TO":
                    # 字段属于表（如果 to_table 是字段名）或 表属于数据库
                    if to_table in column_names_from_json:
                        self.execute_query("""
                            MATCH (c:Column {name: $to_table})
                            MATCH (t:Table {name: $from_table})
                            MERGE (c)-[r:BELONGS_TO {description: $description}]->(t)
                        """, {"to_table": to_table, "from_table": from_table, "description": description})
                    else:
                        # 表属于数据库（简化处理）
                        self.execute_query("""
                            MATCH (t:Table {name: $from_table})
                            MATCH (d:Database)
                            MERGE (t)-[r:BELONGS_TO {description: $description}]->(d)
                        """, {"from_table": from_table, "description": description})
                elif rel_type == "REFERENCES":
                    # 表间引用关系
                    self.execute_query("""
                        MATCH (t1:Table {name: $from_table})
                        MATCH (t2:Table {name: $to_table})
                        MERGE (t1)-[r:REFERENCES {description: $description}]->(t2)
                    """, {"from_table": from_table, "to_table": to_table, "description": description})
                else:  # HAS_COLUMN 或其他
                    # 表有字段
                    self.execute_query("""
                        MATCH (t:Table {name: $from_table})
                        MATCH (c:Column {name: $to_table})
                        MERGE (t)-[r:HAS_COLUMN {description: $description}]->(c)
                    """, {"from_table": from_table, "to_table": to_table, "description": description})
                stats["relationships"] += 1

            # 5. 导入业务概念
            for concept in import_data.get("concepts", []):
                self.execute_query("""
                    MERGE (c:BusinessConcept {name: $name})
                    SET c.description = $description, c.category = $category
                """, {
                    "name": concept.get("name", ""),
                    "description": concept.get("description", ""),
                    "category": concept.get("category", "")
                })
                stats["concepts"] += 1

                # 建立与表的关系
                for table_name in concept.get("related_tables", []):
                    self.execute_query("""
                        MATCH (t:Table {name: $table_name})
                        MATCH (c:BusinessConcept {name: $concept_name})
                        MERGE (t)-[:RELATED_TO]->(c)
                    """, {"table_name": table_name, "concept_name": concept.get("name", "")})

            stats["nodes"] = stats["databases"] + stats["tables"] + stats["columns"] + stats["concepts"]
            logger.info(f"知识图谱导入完成：{stats['nodes']} 个节点，{stats['relationships']} 个关系")
            return stats

        except Exception as e:
            logger.error(f"导入知识图谱失败：{e}")
            return {"nodes": 0, "relationships": 0, "error": str(e)}

    def clear_knowledge_graph(self) -> Dict[str, int]:
        """
        清除知识图谱中的所有节点和关系

        Returns:
            Dict[str, int]: 删除的节点数和关系数
        """
        if not self.driver:
            logger.error("Neo4j 未连接")
            return {"nodes": 0, "relationships": 0}

        try:
            total_deleted_nodes = 0
            total_deleted_rels = 0
            batch_size = 10000

            # 循环删除所有关系，直到没有剩余
            while True:
                result = self.execute_query(f"""
                    MATCH ()-[r]-()
                    WITH r LIMIT {batch_size}
                    DELETE r
                    RETURN count(r) as deleted_rels
                """)
                deleted_rels = sum(r.get("deleted_rels", 0) for r in result)
                total_deleted_rels += deleted_rels
                logger.info(f"删除关系批次：{deleted_rels}")
                if deleted_rels < batch_size:
                    break

            # 循环删除所有节点，直到没有剩余
            while True:
                result = self.execute_query(f"""
                    MATCH (n)
                    WITH n LIMIT {batch_size}
                    DELETE n
                    RETURN count(n) as deleted_nodes
                """)
                deleted_nodes = sum(r.get("deleted_nodes", 0) for r in result)
                total_deleted_nodes += deleted_nodes
                logger.info(f"删除节点批次：{deleted_nodes}")
                if deleted_nodes < batch_size:
                    break

            logger.info(f"清除知识图谱完成：删除了 {total_deleted_nodes} 个节点，{total_deleted_rels} 个关系")
            return {"nodes": total_deleted_nodes, "relationships": total_deleted_rels}

        except Exception as e:
            logger.error(f"清除知识图谱失败：{e}")
            return {"nodes": 0, "relationships": 0, "error": str(e)}

    def sync_metadata_to_mysql(self, mysql_client) -> int:
        """
        同步元数据到 MySQL

        Args:
            mysql_client: MySQL 客户端实例

        Returns:
            int: 同步的记录数
        """
        metadata = self.get_schema_metadata()
        count = 0

        # 同步表信息
        for table in metadata['tables']:
            table_name = table.get('name')
            if table_name:
                mysql_client.upsert_table_schema(
                    table_name=table_name,
                    table_comment=table.get('description', ''),
                    properties=table
                )
                count += 1

        # 同步字段信息
        for column in metadata['columns']:
            # 需要通过关系找到对应的表
            for rel in metadata['relationships']:
                if rel.get('rel_type') == 'BELONGS_TO' and rel.get('start_name') == column.get('name'):
                    mysql_client.upsert_column_schema(
                        table_name=rel.get('end_name'),
                        column_name=column.get('name'),
                        column_type=column.get('type', 'VARCHAR'),
                        column_comment=column.get('description', '')
                    )
                    break

        logger.info(f"同步了 {count} 条元数据到 MySQL")
        return count


# ==================== 向后兼容（已废弃，请使用 ApplicationContext） ====================
# 保留全局单例以兼容旧代码，新功能应使用 ApplicationContext

_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client(uri: str, user: str, password: str,
                     database: str = "neo4j",
                     max_connection_pool_size: int = 50,
                     connection_timeout: int = 30) -> Neo4jClient:
    """
    获取或创建 Neo4j 客户端单例（带连接池配置）

    .. deprecated:: 请使用 ApplicationContext 管理组件生命周期

    Args:
        uri: Neo4j URI
        user: 用户名
        password: 密码
        database: 数据库名
        max_connection_pool_size: 最大连接池大小，默认 50
        connection_timeout: 连接超时时间（秒），默认 30

    Returns:
        Neo4jClient: 客户端实例
    """
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient(uri, user, password, database,
                                    max_connection_pool_size, connection_timeout)
        _neo4j_client.connect()
    return _neo4j_client
