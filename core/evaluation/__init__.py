"""Offline, deterministic evaluation APIs."""

from .golden import (
    CaseResult,
    EvaluationReport,
    GoldenCase,
    GoldenConfigurationError,
    GoldenEvaluator,
    GoldenGeneratedResult,
    GoldenResults,
    GoldenSuite,
    ModelParameter,
    RuleRate,
    get_git_commit,
    load_golden_results,
    load_golden_suite,
    parse_model_parameters,
)

__all__ = [
    "CaseResult",
    "EvaluationReport",
    "GoldenCase",
    "GoldenConfigurationError",
    "GoldenEvaluator",
    "GoldenGeneratedResult",
    "GoldenResults",
    "GoldenSuite",
    "ModelParameter",
    "RuleRate",
    "get_git_commit",
    "load_golden_results",
    "load_golden_suite",
    "parse_model_parameters",
]
