"""Publish an activated compiled semantic model to Neo4j."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from core.knowledge.neo4j_client import Neo4jClient
from core.semantic.publisher import SemanticNeo4jPublisher
from core.semantic.registry import SemanticRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish an activated compiled semantic snapshot to Neo4j."
    )
    parser.add_argument("datasource")
    parser.add_argument(
        "--compiled-root", type=Path, default=Path("semantic_models/.compiled")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = SemanticRegistry(args.compiled_root, args.datasource)
    snapshot = registry.load_active()
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
        max_connection_pool_size=config.NEO4J_MAX_POOL_SIZE,
        connection_timeout=config.NEO4J_CONNECTION_TIMEOUT,
    )
    if not client.connect():
        raise RuntimeError("Neo4j 连接失败")
    try:
        result = SemanticNeo4jPublisher(client).publish(snapshot)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
