"""
权值初始化与重平衡模块
管理 Neo4j CONNECTS 关系的权重分配
"""
from typing import List, Dict, Optional
from loguru import logger


def normalize_relationships(neo4j_client) -> int:
    """
    规范化所有 CONNECTS 关系的权重
    对同一对表 (from_table -> to_table) 之间的多条关系，平分权重

    例：A->B 有 3 条关系，每条 weight = 1/3

    Returns:
        int: 更新的关系数量
    """
    # 1. 查询所有 CONNECTS 关系
    result = neo4j_client.execute_query("""
        MATCH (from:Table)-[r:CONNECTS]->(to:Table)
        RETURN from.name as from_table, to.name as to_table,
               r.from_column as from_column, r.to_column as to_column,
               r.weight as current_weight, r.occurrence_count as occurrence_count,
               r.source as source, id(r) as rel_id
    """)

    if not result:
        logger.info("没有需要规范化的 CONNECTS 关系")
        return 0

    # 2. 按 (from_table, to_table) 分组
    table_pairs: Dict[tuple, List[Dict]] = {}
    for row in result:
        key = (row['from_table'], row['to_table'])
        if key not in table_pairs:
            table_pairs[key] = []
        table_pairs[key].append(row)

    # 3. 对每组平分权重
    updated_count = 0
    for (from_table, to_table), rels in table_pairs.items():
        n = len(rels)
        if n == 0:
            continue
        weight = 1.0 / n
        for rel in rels:
            neo4j_client.execute_query(
                """
                MATCH ()-[r:CONNECTS]->()
                WHERE id(r) = $rel_id
                SET r.weight = $weight
                """,
                {"rel_id": rel['rel_id'], "weight": weight}
            )
            updated_count += 1

    logger.info(f"权值规范化完成：{updated_count} 条关系已更新，共 {len(table_pairs)} 个表对")
    return updated_count


def rebalance_with_history(history_joins: List[Dict], neo4j_client,
                          eta: float = 0.1) -> Dict[str, int]:
    """
    根据历史提取的 JOIN 关系，重平衡 Neo4j 中的权重

    算法：
    - 对于已存在的关系：increment occurrence_count, new_weight = old_weight * (1 - eta) + eta
    - 对于新关系：创建 weight = 1 / (existing_count + 1)，旧关系按比例收缩

    Args:
        history_joins: 从历史 SQL 中提取的 JOIN 信息列表
        neo4j_client: Neo4j 客户端
        eta: 学习率 (0-1)，控制权重增量速度

    Returns:
        Dict: {updated: 更新数, created: 新建数, unchanged: 未变数}
    """
    stats = {"updated": 0, "created": 0, "unchanged": 0}

    for join_info in history_joins:
        from_table = join_info.get("left_table", "")
        to_table = join_info.get("right_table", "")
        from_column = join_info.get("from_column", "")
        to_column = join_info.get("to_column", "")
        join_type = join_info.get("join_type", "LEFT JOIN")

        if not (from_table and to_table and from_column and to_column):
            continue

        # 检查该关系是否已存在
        existing = neo4j_client.execute_query(
            """
            MATCH (from:Table {name: $from_table})-[r:CONNECTS {
                from_column: $from_column, to_column: $to_column
            }]->(to:Table {name: $to_table})
            RETURN r.weight as weight, r.occurrence_count as occurrence_count,
                   r.join_type as join_type
            """,
            {
                "from_table": from_table,
                "to_table": to_table,
                "from_column": from_column,
                "to_column": to_column
            }
        )

        if existing and len(existing) > 0:
            # 关系已存在：平滑增加权重
            old_weight = existing[0].get("weight", 1.0)
            old_count = existing[0].get("occurrence_count", 1)
            new_weight = old_weight * (1 - eta) + eta
            new_count = old_count + 1

            neo4j_client.execute_query(
                """
                MATCH (from:Table {name: $from_table})-[r:CONNECTS {
                    from_column: $from_column, to_column: $to_column
                }]->(to:Table {name: $to_table})
                SET r.weight = $new_weight,
                    r.occurrence_count = $new_count,
                    r.source = 'history_extracted',
                    r.updated_at = datetime()
                """,
                {
                    "from_table": from_table,
                    "to_table": to_table,
                    "from_column": from_column,
                    "to_column": to_column,
                    "new_weight": min(new_weight, 1.0),
                    "new_count": new_count
                }
            )
            stats["updated"] += 1
            logger.debug(f"更新关系权重: {from_table}.{from_column} -> {to_table}.{to_column}, "
                        f"{old_weight:.3f} -> {new_weight:.3f}")
        else:
            # 新关系：获取当前表对的关系数
            pair_count_result = neo4j_client.execute_query(
                """
                MATCH (:Table {name: $from_table})-[:CONNECTS]->(:Table {name: $to_table})
                RETURN count(*) as count
                """,
                {"from_table": from_table, "to_table": to_table}
            )
            existing_count = pair_count_result[0]["count"] if pair_count_result else 0
            new_weight = 1.0 / (existing_count + 1)

            # 收缩已有关系的权重
            if existing_count > 0:
                neo4j_client.execute_query(
                    """
                    MATCH (from:Table {name: $from_table})-[r:CONNECTS]->(to:Table {name: $to_table})
                    SET r.weight = r.weight * (1 - $new_weight)
                    """,
                    {"from_table": from_table, "to_table": to_table, "new_weight": new_weight}
                )

            # 创建新关系
            neo4j_client.execute_query(
                """
                MATCH (from:Table {name: $from_table})
                MATCH (to:Table {name: $to_table})
                MERGE (from)-[r:CONNECTS {
                    from_column: $from_column, to_column: $to_column
                }]->(to)
                SET r.relationship_type = 'foreign_key',
                    r.join_type = $join_type,
                    r.weight = $new_weight,
                    r.occurrence_count = 1,
                    r.source = 'history_extracted',
                    r.updated_at = datetime()
                """,
                {
                    "from_table": from_table,
                    "to_table": to_table,
                    "from_column": from_column,
                    "to_column": to_column,
                    "join_type": join_type,
                    "new_weight": new_weight
                }
            )
            stats["created"] += 1
            logger.info(f"创建新关系: {from_table}.{from_column} -> {to_table}.{to_column}, "
                       f"weight={new_weight:.3f}")

    logger.info(f"权重重平衡完成：{stats}")
    return stats
