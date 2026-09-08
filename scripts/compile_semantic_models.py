"""Command-line compiler for governed semantic-model YAML files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.semantic.compiler import SemanticCompiler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("semantic_models/.compiled"))
    args = parser.parse_args()
    compiler = SemanticCompiler()
    for source in args.sources:
        snapshot = compiler.compile_file(source, args.output)
        print(f"{snapshot.datasource}@{snapshot.model_version} {snapshot.content_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
