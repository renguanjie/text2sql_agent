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
from neo4j.time import DateTime, Date, Time, Duration


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
            nodes = self._export_graph_nodes()
            relationships = self._export_graph_relationships()
            databases = [
                node["properties"]
                for node in nodes
                if node["label"] == "Database"
            ]
            tables = [
                node["properties"]
                for node in nodes
                if node["label"] == "Table"
            ]
            columns = [
                node["properties"]
                for node in nodes
                if node["label"] == "Column"
            ]
            concepts = [
                {
                    **node["properties"],
                    "related_tables": [
                        rel["start"]["properties"].get("name")
                        for rel in relationships
                        if rel["type"] in {"RELATED_TO", "MAPPED_TO"}
                        and rel["end"]["key"] == node["key"]
                        and rel["start"]["label"] == "Table"
                    ]
                }
                for node in nodes
                if node["label"] == "BusinessConcept"
            ]

            export_data = {
                "version": "2.0",
                "exported_at": datetime.now().isoformat(),
                "format": "text2sql_knowledge_graph",
                "nodes": nodes,
                "edges": relationships,
                "databases": databases,
                "tables": tables,
                "columns": columns,
                "relationships": [
                    {
                        "relationship_type": rel["type"],
                        "from_table": rel["start"]["properties"].get("name"),
                        "from_database_id": rel["start"]["properties"].get("database_id"),
                        "from_column": rel["properties"].get("from_column"),
                        "to_table": rel["end"]["properties"].get("name"),
                        "to_database_id": rel["end"]["properties"].get("database_id"),
                        "to_column": rel["properties"].get("to_column"),
                        **rel["properties"],
                    }
                    for rel in relationships
                    if rel["start"]["label"] == "Table" and rel["end"]["label"] == "Table"
                ],
                "concepts": concepts
            }

            logger.info(f"知识图谱导出完成：{len(databases)} 个数据库，{len(tables)} 个表，{len(columns)} 个字段")
            return export_data

        except Exception as e:
            logger.error(f"导出知识图谱失败：{e}")
            return {}

    def validate_knowledge_graph_json(self, import_data: Dict[str, Any]) -> Dict[str, Any]:
        """校验知识图谱 JSON 格式，返回标准化校验结果。"""
        errors = []
        if not isinstance(import_data, dict):
            return {"valid": False, "errors": ["JSON 顶层必须是对象"], "database_refs": []}

        has_v2_graph = isinstance(import_data.get("nodes"), list) and isinstance(import_data.get("edges"), list)
        has_legacy_graph = any(isinstance(import_data.get(key), list) for key in ("databases", "tables", "columns", "relationships", "concepts"))

        if not has_v2_graph and not has_legacy_graph:
            errors.append("缺少 nodes/edges 或 databases/tables/columns/relationships/concepts 数组")

        if has_v2_graph:
            for idx, node in enumerate(import_data.get("nodes", [])):
                if not isinstance(node, dict):
                    errors.append(f"nodes[{idx}] 必须是对象")
                    continue
                if not node.get("label") or not isinstance(node.get("properties"), dict):
                    errors.append(f"nodes[{idx}] 缺少 label 或 properties")
            for idx, edge in enumerate(import_data.get("edges", [])):
                if not isinstance(edge, dict):
                    errors.append(f"edges[{idx}] 必须是对象")
                    continue
                if not edge.get("type") or not isinstance(edge.get("start"), dict) or not isinstance(edge.get("end"), dict):
                    errors.append(f"edges[{idx}] 缺少 type/start/end")

        for key in ("databases", "tables", "columns", "relationships", "concepts"):
            value = import_data.get(key)
            if value is not None and not isinstance(value, list):
                errors.append(f"{key} 必须是数组")

        return {
            "valid": not errors,
            "errors": errors,
            "database_refs": self.get_import_database_refs(import_data)
        }

    def get_import_database_refs(self, import_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从导入 JSON 中提取数据库标识，兼容多数据库配置。"""
        refs = {}

        def add_ref(database_id=None, name=None):
            if database_id is None and not name:
                return
            key = f"id:{database_id}" if database_id is not None else f"name:{name}"
            refs[key] = {"database_id": database_id, "name": name}

        for db in import_data.get("databases", []) or []:
            if isinstance(db, dict):
                add_ref(db.get("database_id") or db.get("id"), db.get("name"))

        for node in import_data.get("nodes", []) or []:
            if isinstance(node, dict) and node.get("label") == "Database":
                props = node.get("properties") or {}
                add_ref(props.get("database_id") or props.get("id"), props.get("name"))

        for table in import_data.get("tables", []) or []:
            if isinstance(table, dict):
                add_ref(table.get("database_id"), table.get("database") or table.get("database_name"))

        return list(refs.values())

    def get_existing_import_databases(self, import_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查询导入 JSON 涉及的数据库在当前图谱中是否已存在。"""
        existing = []
        for ref in self.get_import_database_refs(import_data):
            result = self.execute_query("""
                MATCH (d:Database)
                WHERE ($database_id IS NOT NULL AND d.database_id = $database_id)
                   OR ($name IS NOT NULL AND d.name = $name)
                RETURN d.database_id as database_id, d.name as name
                LIMIT 1
            """, ref)
            if result:
                existing.append(result[0])
        return existing

    def import_knowledge_graph(self, import_data: Dict[str, Any], overwrite: bool = False) -> Dict[str, int]:
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
            validation = self.validate_knowledge_graph_json(import_data)
            if not validation["valid"]:
                return {"nodes": 0, "relationships": 0, "error": "；".join(validation["errors"])}

            if overwrite:
                self.clear_databases_from_import(import_data)

            if import_data.get("nodes") and import_data.get("edges"):
                return self._import_v2_knowledge_graph(import_data)

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

    def clear_databases_from_import(self, import_data: Dict[str, Any]) -> Dict[str, int]:
        """删除导入 JSON 涉及的数据库子图，用于覆盖式导入。"""
        total = {"nodes": 0, "relationships": 0}
        refs = self.get_import_database_refs(import_data)
        for ref in refs:
            result = self.execute_query("""
                MATCH (d:Database)
                WHERE ($database_id IS NOT NULL AND d.database_id = $database_id)
                   OR ($name IS NOT NULL AND d.name = $name)
                OPTIONAL MATCH (d)-[:HAS_TABLE]->(t:Table)
                OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                WITH collect(DISTINCT d) + collect(DISTINCT t) + collect(DISTINCT c) AS nodes
                UNWIND nodes AS n
                WITH DISTINCT n WHERE n IS NOT NULL
                DETACH DELETE n
                RETURN count(n) as deleted_nodes
            """, ref)
            total["nodes"] += result[0].get("deleted_nodes", 0) if result else 0

        # 清理被覆盖数据库遗留下来的孤立业务概念
        result = self.execute_query("""
            MATCH (c:BusinessConcept)
            WHERE NOT (c)--()
            DELETE c
            RETURN count(c) as deleted_nodes
        """)
        total["nodes"] += result[0].get("deleted_nodes", 0) if result else 0
        return total

    def _export_graph_nodes(self) -> List[Dict[str, Any]]:
        records = self.execute_query("""
            MATCH (n)
            WHERE n:Database OR n:Table OR n:Column OR n:BusinessConcept
            RETURN labels(n) as labels, properties(n) as properties
            ORDER BY labels(n)[0], coalesce(n.database_id, 0), coalesce(n.table_name, ''), coalesce(n.name, '')
        """)
        nodes = []
        for record in records:
            labels = record.get("labels") or []
            label = self._primary_export_label(labels)
            properties = self._json_safe(record.get("properties") or {})
            nodes.append({
                "label": label,
                "labels": labels,
                "key": self._node_key(label, properties),
                "properties": properties,
            })
        return nodes

    def _export_graph_relationships(self) -> List[Dict[str, Any]]:
        records = self.execute_query("""
            MATCH (start)-[r]->(end)
            WHERE (start:Database OR start:Table OR start:Column OR start:BusinessConcept)
              AND (end:Database OR end:Table OR end:Column OR end:BusinessConcept)
            RETURN labels(start) as start_labels, properties(start) as start_properties,
                   type(r) as type, properties(r) as properties,
                   labels(end) as end_labels, properties(end) as end_properties
            ORDER BY type(r)
        """)
        edges = []
        for record in records:
            start_label = self._primary_export_label(record.get("start_labels") or [])
            end_label = self._primary_export_label(record.get("end_labels") or [])
            start_props = self._json_safe(record.get("start_properties") or {})
            end_props = self._json_safe(record.get("end_properties") or {})
            edges.append({
                "type": record.get("type"),
                "start": {
                    "label": start_label,
                    "key": self._node_key(start_label, start_props),
                    "properties": self._node_ref_properties(start_label, start_props),
                },
                "end": {
                    "label": end_label,
                    "key": self._node_key(end_label, end_props),
                    "properties": self._node_ref_properties(end_label, end_props),
                },
                "properties": self._json_safe(record.get("properties") or {}),
            })
        return edges

    def _import_v2_knowledge_graph(self, import_data: Dict[str, Any]) -> Dict[str, int]:
        stats = {"nodes": 0, "relationships": 0, "databases": 0, "tables": 0, "columns": 0, "concepts": 0}
        label_counts = {
            "Database": "databases",
            "Table": "tables",
            "Column": "columns",
            "BusinessConcept": "concepts",
        }

        for node in import_data.get("nodes", []):
            label = node.get("label")
            properties = self._neo4j_property_map(node.get("properties") or {})
            if label not in label_counts:
                continue
            self._merge_node(label, properties)
            stats[label_counts[label]] += 1

        for edge in import_data.get("edges", []):
            rel_type = self._safe_relationship_type(edge.get("type"))
            if not rel_type:
                continue
            start = edge.get("start") or {}
            end = edge.get("end") or {}
            self._merge_relationship(
                rel_type=rel_type,
                start_label=start.get("label"),
                start_props=start.get("properties") or {},
                end_label=end.get("label"),
                end_props=end.get("properties") or {},
                properties=edge.get("properties") or {},
            )
            stats["relationships"] += 1

        stats["nodes"] = stats["databases"] + stats["tables"] + stats["columns"] + stats["concepts"]
        logger.info(f"知识图谱 v2 导入完成：{stats}")
        return stats

    def _merge_node(self, label: str, properties: Dict[str, Any]):
        key_props = self._node_ref_properties(label, properties)
        safe_props = self._neo4j_property_map(properties)
        if label == "Database":
            query = """
                MERGE (n:Database {database_id: $key.database_id})
                SET n += $props, n.updated_at = datetime()
            """ if key_props.get("database_id") is not None else """
                MERGE (n:Database {name: $key.name})
                SET n += $props, n.updated_at = datetime()
            """
        elif label == "Table":
            query = """
                MERGE (n:Table {name: $key.name, database_id: $key.database_id})
                SET n += $props, n.updated_at = datetime()
            """
        elif label == "Column":
            query = """
                MERGE (n:Column {name: $key.name, table_name: $key.table_name, database_id: $key.database_id})
                SET n += $props, n.updated_at = datetime()
            """
        elif label == "BusinessConcept":
            query = """
                MERGE (n:BusinessConcept {name: $key.name})
                SET n += $props, n.updated_at = datetime()
            """
        else:
            return
        self.execute_query(query, {"key": key_props, "props": safe_props})

    def _merge_relationship(self, rel_type: str, start_label: str, start_props: Dict[str, Any],
                            end_label: str, end_props: Dict[str, Any], properties: Dict[str, Any]):
        if start_label not in {"Database", "Table", "Column", "BusinessConcept"}:
            return
        if end_label not in {"Database", "Table", "Column", "BusinessConcept"}:
            return

        start_match = self._node_match_clause("start", start_label, "start_key")
        end_match = self._node_match_clause("end", end_label, "end_key")
        rel_key = self._relationship_key(rel_type, properties)
        merge_clause = (
            f"MERGE (start)-[r:{rel_type} {{relationship_key: $rel_key.relationship_key}}]->(end)"
            if rel_key.get("relationship_key") else
            f"MERGE (start)-[r:{rel_type}]->(end)"
        )
        query = f"""
            {start_match}
            {end_match}
            {merge_clause}
            SET r += $props, r.updated_at = datetime()
        """
        self.execute_query(query, {
            "start_key": self._node_ref_properties(start_label, start_props),
            "end_key": self._node_ref_properties(end_label, end_props),
            "rel_key": rel_key,
            "props": self._neo4j_property_map(properties),
        })

    def _node_match_clause(self, variable: str, label: str, key_name: str) -> str:
        if label == "Database":
            return f"""
                MATCH ({variable}:Database)
                WHERE (${key_name}.database_id IS NOT NULL AND {variable}.database_id = ${key_name}.database_id)
                   OR (${key_name}.database_id IS NULL AND {variable}.name = ${key_name}.name)
            """
        if label == "Table":
            return f"MATCH ({variable}:Table {{name: ${key_name}.name, database_id: ${key_name}.database_id}})"
        if label == "Column":
            return f"MATCH ({variable}:Column {{name: ${key_name}.name, table_name: ${key_name}.table_name, database_id: ${key_name}.database_id}})"
        return f"MATCH ({variable}:BusinessConcept {{name: ${key_name}.name}})"

    def _node_ref_properties(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        if label == "Database":
            return {
                "database_id": properties.get("database_id") or properties.get("id"),
                "name": properties.get("name"),
            }
        if label == "Table":
            return {
                "database_id": properties.get("database_id"),
                "name": properties.get("name"),
            }
        if label == "Column":
            return {
                "database_id": properties.get("database_id"),
                "table_name": properties.get("table_name"),
                "name": properties.get("name"),
            }
        return {"name": properties.get("name")}

    def _node_key(self, label: str, properties: Dict[str, Any]) -> str:
        ref = self._node_ref_properties(label, properties)
        if label == "Database":
            return f"Database:{ref.get('database_id') or ref.get('name')}"
        if label == "Table":
            return f"Table:{ref.get('database_id')}:{ref.get('name')}"
        if label == "Column":
            return f"Column:{ref.get('database_id')}:{ref.get('table_name')}:{ref.get('name')}"
        return f"BusinessConcept:{ref.get('name')}"

    def _primary_export_label(self, labels: List[str]) -> str:
        for label in ("Database", "Table", "Column", "BusinessConcept"):
            if label in labels:
                return label
        return labels[0] if labels else ""

    def _safe_relationship_type(self, rel_type: str) -> Optional[str]:
        if not rel_type:
            return None
        rel_type = str(rel_type).upper()
        if not rel_type.replace("_", "").isalnum():
            return None
        return rel_type

    def _relationship_key(self, rel_type: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        if rel_type in {"CONNECTS", "REFERENCES"}:
            key = "|".join([
                rel_type,
                str(properties.get("relationship_type") or ""),
                str(properties.get("from_column") or ""),
                str(properties.get("to_column") or ""),
                str(properties.get("source") or ""),
            ])
            return {"relationship_key": key}
        return {"relationship_key": None}

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (DateTime, Date, Time, Duration)):
            return str(value)
        if hasattr(value, "iso_format"):
            return value.iso_format()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    def _neo4j_property_map(self, value: Dict[str, Any]) -> Dict[str, Any]:
        safe = {}
        for key, item in (value or {}).items():
            if item is None:
                safe[key] = None
            elif isinstance(item, (str, int, float, bool)):
                safe[key] = item
            elif isinstance(item, list):
                safe[key] = [
                    i if isinstance(i, (str, int, float, bool)) else json.dumps(i, ensure_ascii=False, default=str)
                    for i in item
                ]
            else:
                safe[key] = json.dumps(item, ensure_ascii=False, default=str)
        return safe

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
