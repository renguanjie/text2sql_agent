"""Evaluate pre-generated Text2SQL results without invoking an LLM or database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation import (  # noqa: E402
    GoldenConfigurationError,
    GoldenEvaluator,
    get_git_commit,
    load_golden_results,
    load_golden_suite,
    parse_model_parameters,
)
from core.semantic.registry import SemanticRegistry, SemanticRegistryError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线评估已有 Text2SQL 结果；不会调用 LLM 或数据库。"
    )
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--semantic-version", required=True)
    parser.add_argument(
        "--compiled-root",
        type=Path,
        default=Path("semantic_models/.compiled"),
    )
    parser.add_argument(
        "--model-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="可重复；仅接受非敏感 allowlist 参数。",
    )
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="显式本地调试 opt-in：报告将包含完整问题和 SQL。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        suite = load_golden_suite(args.cases)
        results = load_golden_results(args.results)
        if args.semantic_version != suite.semantic_version:
            raise GoldenConfigurationError("SNAPSHOT_MISMATCH")
        try:
            snapshot = SemanticRegistry(args.compiled_root, suite.datasource).load_active()
        except SemanticRegistryError as exc:
            raise GoldenConfigurationError("SNAPSHOT_INVALID") from exc
        parameters = parse_model_parameters(args.model_param)
        evaluator = GoldenEvaluator(
            snapshot,
            model=args.model,
            model_parameters=parameters,
            git_commit=get_git_commit(PROJECT_ROOT),
            include_content=args.include_content,
        )
        evaluator.validate_inputs(suite, results)
        by_id = {result.id: result.model_dump(mode="json") for result in results.results}
        report = evaluator.evaluate_results(
            [(case, by_id[case.id]) for case in suite.cases]
        )
        evaluator.write_report(report, args.output)
        print(
            json.dumps(
                {
                    "failed": report.failed,
                    "output": str(args.output),
                    "pass_rate": report.pass_rate,
                    "passed": report.passed,
                    "total": report.total,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if report.failed == 0 else 1
    except GoldenConfigurationError as exc:
        print(f"CONFIG_ERROR: {exc.code}", file=sys.stderr)
        return 2
    except Exception:
        print("CONFIG_ERROR: EVALUATION_IO_ERROR", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
